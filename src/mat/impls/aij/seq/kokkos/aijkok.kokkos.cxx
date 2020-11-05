#include "petsc/private/petscimpl.h"
#include <petscsystypes.h>
#include <petscerror.h>
#include <petscveckokkos.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
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
  Mat_SeqAIJ                *aijseq = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJKokkos          *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscInt                  nrows   = A->rmap->n,ncols = A->cmap->n,nnz = aijseq->nz;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  /* If aijkok is not built yet or the nonzero pattern on CPU has changed, build aijkok from the scratch */
  if (!aijkok || aijkok->nonzerostate != A->nonzerostate) {
    delete aijkok;
    aijkok               = new Mat_SeqAIJKokkos(nrows,ncols,nnz,aijseq->i,aijseq->j,aijseq->a);
    aijkok->nonzerostate = A->nonzerostate;
    A->spptr             = aijkok;
  } else if (A->offloadmask == PETSC_OFFLOAD_CPU) { /* Copy values only */
    Kokkos::deep_copy(aijkok->a_d,aijkok->a_h);
  }
  A->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

/* y = A x */
static PetscErrorCode MatMult_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr   = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr   = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr   = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A x + beta y */
  ierr   = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr   = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  /* 2.0*aijkok->csr.nnz()-aijkok->csr.numRows() seems more accurate here but assumes there are no zero-rows. So a little sloopy here. */
  ierr   = WaitForKokkos();CHKERRQ(ierr);
  ierr   = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^T x */
static PetscErrorCode MatMultTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("T",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A^T x + beta y */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^H x */
static PetscErrorCode MatMultHermitianTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("C",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A^H x + beta y */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A x + y */
static PetscErrorCode MatMultAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^T x + y */
static PetscErrorCode MatMultTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("T",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A^T x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^H x + y */
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("C",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A^H x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode ierr;
  Mat            B;
  Mat_SeqAIJ     *aij;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
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
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatKernelHandleDestroy_Private(void* data)
{
  MatMatKernelHandle_t *kh = static_cast<MatMatKernelHandle_t *>(data);

  PetscFunctionBegin;
  delete kh;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product          *product = C->product;
  Mat                  A,B;
  MatProductType       ptype;
  Mat_SeqAIJKokkos     *akok,*bkok,*ckok;
  bool                 tA,tB;
  PetscErrorCode       ierr;
  MatMatKernelHandle_t *kh;
  Mat_SeqAIJ           *c;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  A = product->A;
  B = product->B;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = MatSeqAIJKokkosSyncDevice(B);CHKERRQ(ierr);
  akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  ckok = static_cast<Mat_SeqAIJKokkos*>(C->spptr);
  kh   = static_cast<MatMatKernelHandle_t*>(C->product->data);
  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  if (B->symmetric && ptype == MATPRODUCT_ABt) ptype = MATPRODUCT_AB;
  switch (ptype) {
  case MATPRODUCT_AB:
    tA = false;
    tB = false;
    break;
  case MATPRODUCT_AtB:
    tA = true;
    tB = false;
    break;
  case MATPRODUCT_ABt:
    tA = false;
    tB = true;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }

  KokkosSparse::spgemm_numeric(*kh, akok->csr, tA, bkok->csr, tB, ckok->csr);
  C->offloadmask = PETSC_OFFLOAD_GPU;
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  c = (Mat_SeqAIJ*)C->data;
  ierr = PetscInfo3(C,"Matrix size: %D X %D; storage space: 0 unneeded,%D used\n",C->rmap->n,C->cmap->n,c->nz);CHKERRQ(ierr);
  ierr = PetscInfo(C,"Number of mallocs during MatSetValues() is 0\n");CHKERRQ(ierr);
  ierr = PetscInfo1(C,"Maximum nonzeros in any row is %D\n",c->rmax);CHKERRQ(ierr);
  c->reallocs         = 0;
  C->info.mallocs    += 0;
  C->info.nz_unneeded = 0;
  C->assembled = C->was_assembled = PETSC_TRUE;
  C->num_ass++;
  /* we can remove these calls when MatSeqAIJGetArray operations are used everywhere! */
  // TODO JZ, copy from device to host since most of Petsc code for AIJ matrices does not use MatSeqAIJGetArray()
  C->offloadmask = PETSC_OFFLOAD_BOTH;
  // Also, we should add support to copy back from device to host
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product          *product = C->product;
  Mat                  A,B;
  MatProductType       ptype;
  Mat_SeqAIJKokkos     *akok,*bkok,*ckok;
  PetscInt             m,n,k;
  bool                 tA,tB;
  PetscErrorCode       ierr;
  Mat_SeqAIJ           *c;
  MatMatKernelHandle_t *kh;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A = product->A;
  B = product->B;
  // TODO only copy the i,j data, not the values
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = MatSeqAIJKokkosSyncDevice(B);CHKERRQ(ierr);
  akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  if (B->symmetric && ptype == MATPRODUCT_ABt) ptype = MATPRODUCT_AB;
  switch (ptype) {
  case MATPRODUCT_AB:
    tA = false;
    tB = false;
    m = A->rmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    tA = true;
    tB = false;
    m = A->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_ABt:
    tA = false;
    tB = true;
    m = A->rmap->n;
    n = B->rmap->n;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  ierr = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  c = (Mat_SeqAIJ*)C->data;

  kh = new MatMatKernelHandle_t;
  // TODO SZ: ADD RUNTIME SELECTION OF THESE
  kh->set_team_work_size(16);
  kh->set_dynamic_scheduling(true);
  // Select an spgemm algorithm, limited by configuration at compile-time and
  // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
  // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
  std::string myalg("SPGEMM_KK_MEMORY");
  kh->create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(myalg));

  /////////////////////////////////////
  // TODO JZ
  ckok = NULL; //new Mat_SeqAIJKokkos();
  C->spptr = ckok;
  KokkosCsrMatrix_t ccsr; // here only to have the code compile
  KokkosSparse::spgemm_symbolic(*kh, akok->csr, tA, bkok->csr, tB, ccsr);
  //cerr = WaitForKOKKOS();CHKERRCUDA(cerr);
  //c->nz = get_nnz_from_ccsr
  //////////////////////////////////////
  c->singlemalloc = PETSC_FALSE;
  c->free_a       = PETSC_TRUE;
  c->free_ij      = PETSC_TRUE;
  ierr = PetscMalloc1(m+1,&c->i);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->j);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->a);CHKERRQ(ierr);
  ////////////////////////////////////
  // TODO JZ copy from device to c->i and c->j
  ////////////////////////////////////
  ierr = PetscMalloc1(m,&c->ilen);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&c->imax);CHKERRQ(ierr);
  c->maxnz = c->nz;
  c->nonzerorowcnt = 0;
  c->rmax = 0;
  for (k = 0; k < m; k++) {
    const PetscInt nn = c->i[k+1] - c->i[k];
    c->ilen[k] = c->imax[k] = nn;
    c->nonzerorowcnt += (PetscInt)!!nn;
    c->rmax = PetscMax(c->rmax,nn);
  }

  C->nonzerostate++;
  ierr = PetscLayoutSetUp(C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(C->cmap);CHKERRQ(ierr);
  ierr = MatMarkDiagonal_SeqAIJ(C);CHKERRQ(ierr);
  ckok->nonzerostate = C->nonzerostate;
  C->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  C->preallocated  = PETSC_TRUE;
  C->assembled     = PETSC_FALSE;
  C->was_assembled = PETSC_FALSE;

  C->ops->productnumeric = MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos;
  C->product->data = kh;
  C->product->destroy = MatMatKernelHandleDestroy_Private;
  PetscFunctionReturn(0);
}

/* handles sparse matrix matrix ops */
static PetscErrorCode MatProductSetFromOptions_SeqAIJKokkos(Mat mat)
{
  Mat_Product    *product = mat->product;
  PetscErrorCode ierr;
  PetscBool      Biskok = PETSC_FALSE,Ciskok = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  ierr = PetscObjectTypeCompare((PetscObject)product->B,MATSEQAIJKOKKOS,&Biskok);CHKERRQ(ierr);
  if (product->type == MATPRODUCT_ABC) {
    ierr = PetscObjectTypeCompare((PetscObject)product->C,MATSEQAIJKOKKOS,&Ciskok);CHKERRQ(ierr);
  }
  if (Biskok && Ciskok) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_ABt:
      mat->ops->productsymbolic = MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos;
      break;
    case MATPRODUCT_PtAP:
    case MATPRODUCT_RARt:
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    default:
      break;
    }
  } else { /* fallback for AIJ */
    ierr = MatProductSetFromOptions_SeqAIJ(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJKokkos(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_SeqAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJKokkos(A,MATSEQAIJKOKKOS,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  A->ops->assemblyend               = MatAssemblyEnd_SeqAIJKokkos;
  A->ops->destroy                   = MatDestroy_SeqAIJKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqAIJKokkos;
  // TODO: enable when code is ready
  //A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJKokkos;
  A->ops->mult                      = MatMult_SeqAIJKokkos;
  A->ops->multadd                   = MatMultAdd_SeqAIJKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqAIJKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJKokkos;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*C
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
