#include <petscconf.h>
#include <../src/mat/impls/sell/mpi/mpisell.h>   /*I "petscmat.h" I*/
//#include <../src/mat/impls/sell/seq/seqcuda/cudamatimpl.h>
//#include <../src/mat/impls/sell/mpi/mpicuda/mpicudamatimpl.h>

PetscErrorCode MatMPISELLSetPreallocation_MPISELLCUDA(Mat B,PetscInt d_rlenmax,const PetscInt d_rlen[],PetscInt o_rlenmax,const PetscInt o_rlen[])
{
  Mat_MPISELL    *b = (Mat_MPISELL*)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  
  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQSELLCUDA matrices. */
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatPinToCPU(b->A,B->pinnedtocpu);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQSELLCUDA);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatPinToCPU(b->B,B->pinnedtocpu);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQSELLCUDA);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);
  }
  ierr = MatSeqSELLSetPreallocation(b->A,d_rlenmax,d_rlen);CHKERRQ(ierr);
  ierr = MatSeqSELLSetPreallocation(b->B,o_rlenmax,o_rlen);CHKERRQ(ierr);
  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPISELLCUDA(Mat A,Vec xx,Vec yy)
{
  /*
     This multiplication sequence is different sequence
     than the CPU version. In particular, the diagonal block
     multiplication kernel is launched in one stream. Then,
     in a separate stream, the data transfers from DeviceToHost
     (with MPI messaging in between), then HostToDevice are
     launched. Once the data transfer stream is synchronized,
     to ensure messaging is complete, the MatMultAdd kernel
     is launched in the original (MatMult) stream to protect
     against race conditions.
  */
  Mat_MPISELL    *a = (Mat_MPISELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPISELLCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  /*
     This multiplication sequence is different sequence
     than the CPU version. In particular, the diagonal block
     multiplication kernel is launched in one stream. Then,
     in a separate stream, the data transfers from DeviceToHost
     (with MPI messaging in between), then HostToDevice are
     launched. Once the data transfer stream is synchronized,
     to ensure messaging is complete, the MatMultAdd kernel
     is launched in the original (MatMult) stream to protect
     against race conditions.
  */
  Mat_MPISELL    *a = (Mat_MPISELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPISELLCUDA(Mat A,Vec xx,Vec yy)
{
  /* This multiplication sequence is different sequence
     than the CPU version. In particular, the diagonal block
     multiplication kernel is launched in one stream. Then,
     in a separate stream, the data transfers from DeviceToHost
     (with MPI messaging in between), then HostToDevice are
     launched. Once the data transfer stream is synchronized,
     to ensure messaging is complete, the MatMultAdd kernel
     is launched in the original (MatMult) stream to protect
     against race conditions.

     This sequence should only be called for GPU computation. */
  Mat_MPISELL    *a = (Mat_MPISELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->rmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_MPISELLCUDA(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MPISELLCUDA options");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPISELLCUDA(Mat A,MatAssemblyType mode)
{
  Mat_MPISELL    *mpisell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mpisell = (Mat_MPISELL*)A->data;
  ierr = MatAssemblyEnd_MPISELL(A,mode);CHKERRQ(ierr);
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = VecSetType(mpisell->lvec,VECSEQCUDA);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPISELLCUDA(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy_MPISELL(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPISELLCUDA(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_MPISELL(A);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPISELLSetPreallocation_C",MatMPISELLSetPreallocation_MPISELLCUDA);CHKERRQ(ierr);
  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);

  A->ops->assemblyend    = MatAssemblyEnd_MPISELLCUDA;
  A->ops->mult           = MatMult_MPISELLCUDA;
  A->ops->multadd        = MatMultAdd_MPISELLCUDA;
  A->ops->multtranspose  = MatMultTranspose_MPISELLCUDA;
  A->ops->setfromoptions = MatSetFromOptions_MPISELLCUDA;
  A->ops->destroy        = MatDestroy_MPISELLCUDA;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPISELLCUDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreateSELLCUDA - Creates a sparse matrix in SELL (compressed row) format.
   This matrix will ultimately pushed down to NVidia GPUs. For good matrix
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
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqSELLSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   Level: intermediate

.seealso: MatCreate(), MatCreateSELL(), MatSetValues(), MATMPISELLCUDA, MATSELLCUDA
@*/
PetscErrorCode  MatCreateSELLCUDA(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPISELLCUDA);CHKERRQ(ierr);
    ierr = MatMPISELLSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQSELLCUDA);CHKERRQ(ierr);
    ierr = MatSeqSELLSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATSELLCUDA - MATMPISELLCUDA = "sellcuda" = "mpisellcuda" - A matrix type to be used for sparse matrices.

   Sliced ELLPACK matrix type whose data resides on Nvidia GPUs.

   This matrix type is identical to MATSEQSELLCUDA when constructed with a single process communicator,
   and MATMPISELLCUDA otherwise.  As a result, for single process communicators,
   MatSeqSELLSetPreallocation is supported, and similarly MatMPISELLSetPreallocation is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpisellcuda - sets the matrix type to "mpisellcuda" during a call to MatSetFromOptions()

  Level: beginner

 .seealso: MatCreateSELLCUDA(), MATSEQSELLCUDA, MatCreateSeqSELLCUDA(), MatCUDAFormatOperation
M
M*/
