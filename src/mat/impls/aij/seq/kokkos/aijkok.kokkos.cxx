#include <petsc/private/petscimpl.h>
#include <petscsystypes.h>
#include <petscerror.h>
#include <petscvec_kokkos.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <../src/mat/impls/aij/seq/aij.h>

#include <../src/mat/impls/aij/seq/kokkos/aijkokkosimpl.hpp>

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat); /* Forward declaration */

static PetscErrorCode MatAssemblyEnd_SeqAIJKokkos(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  A->offloadmask = PETSC_OFFLOAD_CPU;
  /* Don't build (or update) the Mat_SeqAIJKokkos struct. We delay it to the very last moment until we need it. */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosSyncDevice(Mat A)
{
  Mat_SeqAIJ                *aijseq = static_cast<Mat_SeqAIJ*>(A->data);
  Mat_SeqAIJKokkos          *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscInt                  nrows   = A->rmap->n,ncols = A->cmap->n,nnz = aijseq->nz;

  PetscFunctionBegin;
  /* If aijkok is not built yet OR the nonzero pattern on CPU has changed, build aijkok from the scratch */
  if (!aijkok || aijkok->nonzerostate != A->nonzerostate) {
    delete aijkok;
    aijkok               = new Mat_SeqAIJKokkos(nrows,ncols,nnz,aijseq->i,aijseq->j,aijseq->a);
    aijkok->nonzerostate = A->nonzerostate;
    A->spptr             = aijkok;
  } else if (A->offloadmask == PETSC_OFFLOAD_CPU) { /* Copy values only */
    // Kokkos::deep_copy(aijkok->a_d,aijkok->a_h);
    aijkok->a_dual.modify_host(); /* Mark host is modified */
    aijkok->a_dual.sync_device(); /* Sync the device */
    aijkok->need_sync_At_values = PETSC_TRUE; /* Mark matT and matH need to sync, but don't sync them now */
    aijkok->need_sync_Ah_values = PETSC_TRUE;
  }
  A->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosGenerateTranspose(Mat A)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (!aijkok->At) { /* Generate At for the first time */
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&aijkok->At);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosSyncDevice(aijkok->At);CHKERRQ(ierr);
  } else if (aijkok->need_sync_At_values) { /* Only update At values */
    ierr = MatTranspose(A,MAT_REUSE_MATRIX,&aijkok->At);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosSyncDevice(aijkok->At);CHKERRQ(ierr);
  }
  aijkok->need_sync_At_values = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosGenerateTransposeHermitian(Mat A)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (!aijkok->Ah) { /* Generate Ah for the first time */
    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&aijkok->Ah);CHKERRQ(ierr);
    ierr = MatConjugate(aijkok->Ah);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosSyncDevice(aijkok->Ah);CHKERRQ(ierr);
  } else if (aijkok->need_sync_Ah_values) { /* Only update Ah values */
    ierr = MatTranspose(A,MAT_REUSE_MATRIX,&aijkok->Ah);CHKERRQ(ierr);
    ierr = MatConjugate(aijkok->Ah);CHKERRQ(ierr);
    ierr = MatSeqAIJKokkosSyncDevice(aijkok->Ah);CHKERRQ(ierr);
  }
  aijkok->need_sync_Ah_values = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* y = A x */
static PetscErrorCode MatMult_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;

  PetscFunctionBegin;
  ierr   = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr   = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr   = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csrmat,xv,0.0/*beta*/,yv); /* y = alpha A x + beta y */
  ierr   = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr   = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr   = WaitForKokkos();CHKERRQ(ierr);
  /* 2.0*aijkok->csrmat.nnz()-aijkok->csrmat.numRows() seems more accurate here but assumes there are no zero-rows. So a little sloopy here. */
  ierr   = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^T x */
static PetscErrorCode MatMultTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJ                       *aijseq = static_cast<Mat_SeqAIJ*>(A->data);
  Mat_SeqAIJKokkos                 *aijkok;
  Mat                              B;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  if (aijseq->matmult_explicit_transpose) {
    ierr = MatSeqAIJKokkosGenerateTranspose(A);CHKERRQ(ierr);
    B    = static_cast<Mat_SeqAIJKokkos*>(A->spptr)->At;
    mode = "N";
  } else {
    B    = A;
    mode = "T";
  }
  aijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  KokkosSparse::spmv(mode,1.0/*alpha*/,aijkok->csrmat,xv,0.0/*beta*/,yv); /* y = alpha A^T x + beta y */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^H x */
static PetscErrorCode MatMultHermitianTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJ                       *aijseq = static_cast<Mat_SeqAIJ*>(A->data);
  Mat_SeqAIJKokkos                 *aijkok;
  Mat                              B;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv;
  PetscScalarKokkosView            yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  if (aijseq->matmult_explicit_transpose) {
    ierr = MatSeqAIJKokkosGenerateTransposeHermitian(A);CHKERRQ(ierr);
    B    = static_cast<Mat_SeqAIJKokkos*>(A->spptr)->Ah;
    mode = "N";
  } else {
    B    = A;
    mode = "C";
  }
  aijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  KokkosSparse::spmv(mode,1.0/*alpha*/,aijkok->csrmat,xv,0.0/*beta*/,yv); /* y = alpha A^H x + beta y */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A x + y */
static PetscErrorCode MatMultAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csrmat,xv,1.0/*beta*/,zv); /* z = alpha A x + beta z */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^T x + y */
static PetscErrorCode MatMultTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJ                       *aijseq = static_cast<Mat_SeqAIJ*>(A->data);
  Mat_SeqAIJKokkos                 *aijkok;
  Mat                              B;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  if (aijseq->matmult_explicit_transpose) {
    ierr = MatSeqAIJKokkosGenerateTranspose(A);CHKERRQ(ierr);
    B    = static_cast<Mat_SeqAIJKokkos*>(A->spptr)->At;
    mode = "N";
  } else {
    B    = A;
    mode = "T";
  }
  aijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  KokkosSparse::spmv(mode,1.0/*alpha*/,aijkok->csrmat,xv,1.0/*beta*/,zv); /* z = alpha A^T x + beta z */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^H x + y */
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJ                       *aijseq = static_cast<Mat_SeqAIJ*>(A->data);
  Mat_SeqAIJKokkos                 *aijkok;
  Mat                              B;
  const char                       *mode;
  ConstPetscScalarKokkosView       xv,yv;
  PetscScalarKokkosView            zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecGetKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecGetKokkosView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  if (aijseq->matmult_explicit_transpose) {
    ierr = MatSeqAIJKokkosGenerateTransposeHermitian(A);CHKERRQ(ierr);
    B    = static_cast<Mat_SeqAIJKokkos*>(A->spptr)->Ah;
    mode = "N";
  } else {
    B    = A;
    mode = "C";
  }
  aijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  KokkosSparse::spmv(mode,1.0/*alpha*/,aijkok->csrmat,xv,1.0/*beta*/,zv); /* z = alpha A^H x + beta z */
  ierr = VecRestoreKokkosView(xx,&xv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(yy,&yv);CHKERRQ(ierr);
  ierr = VecRestoreKokkosView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csrmat.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatSeqAIJKokkosSetGenerateTranspose - Sets the flag to explicitly generate the tranpose matrix before calling MatMultTranspose

   Collective on mat

   Input Parameters:
+  A - Matrix of type SEQAIJKOKKOS
-  transgen - the boolean flag

   Level: intermediate

.seealso: MATSEQAIJKOKKOS
@*/
PetscErrorCode MatSeqAIJKokkosSetGenerateTranspose(Mat A,PetscBool transgen)
{
  PetscErrorCode            ierr;
  PetscBool                 flg;
  Mat_SeqAIJ                *aijseq = static_cast<Mat_SeqAIJ*>(A->data);
  Mat_SeqAIJKokkos          *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare(((PetscObject)A),MATSEQAIJKOKKOS,&flg);CHKERRQ(ierr);
  if (flg) {
    if (A->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
    aijseq->matmult_explicit_transpose = transgen;
     /* Destroy the transpose matrix if present to prevent from logic errors if transgen is set to true later */
    if (!transgen && aijkok) {ierr = aijkok->DestroyMatTranspose();CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_SeqAIJKokkos(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode                   ierr;
  PetscBool                        flg,transgen;
  Mat_SeqAIJ                       *aijseq = static_cast<Mat_SeqAIJ*>(A->data);

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqAIJKokkos options");CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    transgen = aijseq->matmult_explicit_transpose;
    ierr     = PetscOptionsBool("-matmult_explicit_transpose","Generate explicit transpose for MatMultTranspose and varients","MatSeqAIJKokkosSetGenerateTranspose",transgen,&transgen,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatSeqAIJKokkosSetGenerateTranspose(A,transgen);CHKERRQ(ierr);}
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode   ierr;
  Mat              B;
  Mat_SeqAIJ       *aij;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) { /* Build a new mat */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) { /* Reuse the mat created before */
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  B    = *newmat;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECKOKKOS,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSetOps_SeqAIJKokkos(B);CHKERRQ(ierr);
  /* TODO: see ViennaCL and CUSPARSE once we have a BindToCPU? */
  aij  = (Mat_SeqAIJ*)B->data;
  aij->inode.use = PETSC_FALSE;

  B->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqAIJKokkos(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,cpvalues,B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJKokkos(*B,MATSEQAIJKOKKOS,MAT_INPLACE_MATRIX,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqAIJKokkos(Mat A)
{
  PetscErrorCode        ierr;
  Mat_SeqAIJKokkos      *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  delete aijkok;
  A->spptr = NULL;
  ierr     = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJKokkos(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_SeqAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJKokkos(A,MATSEQAIJKOKKOS,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)A);CHKERRQ(ierr);
  ierr = MatSetFromOptions_SeqAIJKokkos(PetscOptionsObject,A);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  A->ops->assemblyend               = MatAssemblyEnd_SeqAIJKokkos;
  A->ops->destroy                   = MatDestroy_SeqAIJKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqAIJKokkos;

  A->ops->mult                      = MatMult_SeqAIJKokkos;
  A->ops->multadd                   = MatMultAdd_SeqAIJKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqAIJKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJKokkos;
  A->ops->setfromoptions            = MatSetFromOptions_SeqAIJKokkos;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*@C
   MatCreateSeqAIJKokkos - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format). This matrix will ultimately be handled by
   Kokkos for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATSeqAIJKokkos, MATAIJKOKKOS
@*/
PetscErrorCode  MatCreateSeqAIJKokkos(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
