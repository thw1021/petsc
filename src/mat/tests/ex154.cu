static char help[] = "Example showing concurrent solves for PetscDeviceContext\n";

#include <petsc/private/vecimpl.h>
#include <../src/mat/impls/dense/seq/dense.h>
#include <cuda_profiler_api.h>
#include <../src/sys/objects/device/impls/cupm/contextcupm.hpp>

using namespace Petsc; /* needed for PetscDeviceContext_(IMPLS) macro to work */

typedef struct {
  PetscScalar *d_v; /* pointer to the matrix on the GPU */
  PetscBool   user_alloc;
  PetscScalar *unplacedarray; /* if one called MatCUDADensePlaceArray(), this is where it stashed the original */
  PetscBool   unplaced_user_alloc;
  /* factorization support */
  PetscCuBLASInt *d_fact_ipiv; /* device pivots */
  PetscScalar *d_fact_tau;  /* device QR tau vector */
  PetscScalar *d_fact_work; /* device workspace */
  PetscCuBLASInt fact_lwork;
  PetscCuBLASInt *d_fact_info; /* device info */
  /* workspace */
  Vec         workvec;
} Mat_SeqDenseCUDA;

PetscErrorCode MatSeqDenseCUDACopyToGPU_IMPLS(Mat A)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscBool        copy;
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSECUDA);
  if (A->boundtocpu) PetscFunctionReturn(0);
  copy = (PetscBool)(A->offloadmask == PETSC_OFFLOAD_CPU || A->offloadmask == PETSC_OFFLOAD_UNALLOCATED);
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",copy ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (copy) {
    if (!dA->d_v) { /* Allocate GPU memory if not present */
      ierr = MatSeqDenseCUDASetPreallocation(A,NULL);CHKERRQ(ierr);
    }
    ierr = PetscLogEventBegin(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (cA->lda > A->rmap->n) {
      PetscInt n = A->cmap->n,m = A->rmap->n;

      cerr = cudaMemcpy2D(dA->d_v,cA->lda*sizeof(PetscScalar),cA->v,cA->lda*sizeof(PetscScalar),m*sizeof(PetscScalar),n,cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    } else {
      cerr = cudaMemcpy(dA->d_v,cA->v,cA->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    }
    ierr = PetscLogCpuToGpu(cA->lda*sizeof(PetscScalar)*A->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqDenseCUDAGetArrayReadAsync(Mat A, const PetscScalar **arr)
{
  Mat_SeqDense     *cA = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA *dA = (Mat_SeqDenseCUDA*)A->spptr;
  PetscBool        copy;
  PetscErrorCode   ierr;
  cudaError_t      cerr;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQDENSECUDA);
  if (A->boundtocpu) PetscFunctionReturn(0);
  copy = (PetscBool)(A->offloadmask == PETSC_OFFLOAD_CPU || A->offloadmask == PETSC_OFFLOAD_UNALLOCATED);
  ierr = PetscInfo3(A,"%s matrix %d x %d\n",copy ? "Copy" : "Reusing",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  if (copy) {
    PetscDeviceContext_(CUDA) *dcu = (PetscDeviceContext_(CUDA)*)dctx->data;

    if (!dA->d_v) { /* Allocate GPU memory if not present */
      ierr = MatSeqDenseCUDASetPreallocation(A,NULL);CHKERRQ(ierr);
    }
    ierr = PetscLogEventBegin(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (cA->lda > A->rmap->n) {
      PetscInt n = A->cmap->n,m = A->rmap->n;

      cerr = cudaMemcpy2DAsync(dA->d_v,cA->lda*sizeof(PetscScalar),cA->v,cA->lda*sizeof(PetscScalar),m*sizeof(PetscScalar),n,cudaMemcpyHostToDevice,dcu->stream);CHKERRCUDA(cerr);
    } else {
      cerr = cudaMemcpyAsync(dA->d_v,cA->v,cA->lda*sizeof(PetscScalar)*A->cmap->n,cudaMemcpyHostToDevice,dcu->stream);CHKERRCUDA(cerr);
    }
    ierr = PetscLogCpuToGpu(cA->lda*sizeof(PetscScalar)*A->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_DenseCopyToGPU,A,0,0,0);CHKERRQ(ierr);

    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  *arr = dA->d_v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Internal_LU_Async(Mat A, PetscScalar *x, PetscCuBLASInt ldx, PetscCuBLASInt m, PetscCuBLASInt nrhs, PetscCuBLASInt k, PetscBool T)
{
  Mat_SeqDense       *mat = (Mat_SeqDense*)A->data;
  Mat_SeqDenseCUDA   *dA = (Mat_SeqDenseCUDA*)A->spptr;
  const PetscScalar  *da;
  PetscCuBLASInt     lda;
  //cusolverDnHandle_t handle;
  cudaError_t        ccer;
  cusolverStatus_t   cerr;
  int                info;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscCuBLASIntCast(mat->lda,&lda);CHKERRQ(ierr);
  ierr = MatSeqDenseCUDAGetArrayReadAsync(A,&da);CHKERRQ(ierr);
  ierr = PetscInfo2(A,"LU ASYNC solve %d x %d on backend\n",m,k);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  cerr = cusolverDnDgetrs(((PetscDeviceContext_(CUDA)*)dctx->data)->solver,T ? CUBLAS_OP_T : CUBLAS_OP_N,m,nrhs,da,lda,dA->d_fact_ipiv,x,ldx,dA->d_fact_info);CHKERRCUSOLVER(cerr);
  if (PetscDefined(USE_DEBUG)) {
    ccer = cudaDeviceSynchronize();CHKERRCUDA(ccer);
    ccer = cudaMemcpy(&info, dA->d_fact_info, sizeof(PetscCuBLASInt), cudaMemcpyDeviceToHost);CHKERRCUDA(ccer);
    if (info > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %d",info-1);
    else if (info < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong argument to cuSolver %d",-info);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDenseCUDA_Async(Mat A, Vec xx, Vec yy)
{
  PetscScalar      *y;
  PetscCuBLASInt   m=0, k=0;
  PetscErrorCode   ierr;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(A->factortype == MAT_FACTOR_NONE)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = PetscCuBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  {
    cudaError_t cerr;
    const PetscScalar *x;
    PetscBool xishost = PETSC_TRUE;
    PetscDeviceContext_(CUDA) *dcu = (PetscDeviceContext_(CUDA)*)dctx->data;

    /* The logic here is to try to minimize the amount of memory copying:
       if we call VecCUDAGetArrayRead(X,&x) every time xiscuda and the
       data is not offloaded to the GPU yet, then the data is copied to the
       GPU.  But we are only trying to get the data in order to copy it into the y
       array.  So the array x will be wherever the data already is so that
       only one memcpy is performed */
    if (xx->offloadmask & PETSC_OFFLOAD_GPU) {
      ierr = VecCUDAGetArrayRead(xx, &x);CHKERRQ(ierr);
      xishost =  PETSC_FALSE;
    } else {
      ierr = VecGetArrayRead(xx, &x);CHKERRQ(ierr);
    }
    ierr = VecCUDAGetArrayWrite(yy,&y);CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(y,x,m*sizeof(PetscScalar),xishost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice,dcu->stream);CHKERRCUDA(cerr);
  }
  ierr = MatSolve_SeqDenseCUDA_Internal_LU_Async(A,y,m,m,1,k,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode VecClose(Vec vref, Vec vtest)
{
  PetscErrorCode    ierr;
  PetscBool         equal;
  PetscInt          n;
  const PetscScalar *arrRef, *arrTest;

  PetscFunctionBegin;
  ierr = VecEqual(vref,vtest,&equal);CHKERRQ(ierr);
  if (equal) PetscFunctionReturn(0);
  ierr = VecGetLocalSize(vref,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vref,&arrRef);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vtest,&arrTest);CHKERRQ(ierr);
  for (PetscInt i = 0; i < n; ++i) {
    const PetscReal realRef = PetscRealPart(arrRef[i]), realTest = PetscRealPart(arrTest[i]);
    if (!PetscIsCloseAtTol(realRef,realTest,1e-7,1e-7)) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Vectors don't match. refVector[%D]: %.10g != testVector[%D]: %.10g",i,(double)realRef,i,(double)realTest);
    }
  }
  ierr = VecRestoreArrayRead(vref,&arrRef);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vtest,&arrTest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode HostDeviceBarrier(void)
{
  cudaError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Mat            *mats,*matsref;
  Vec            *xx,*xxref,*yy,*yyref;
  PetscInt       nMat = 10,matSize = 2000;
  PetscDeviceContext dctx;
  PetscDeviceContext *subCtx;
  cudaError_t cerr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextFork(dctx,nMat,&subCtx);CHKERRQ(ierr);

  ierr = PetscMalloc6(nMat,&mats,nMat,&matsref,nMat,&xx,nMat,&xxref,nMat,&yy,nMat,&yyref);CHKERRQ(ierr);
  for (PetscInt i = 0; i < nMat; ++i) {
    PetscScalar *dummy;

    ierr = MatCreate(comm,mats+i);CHKERRQ(ierr);
    ierr = MatSetSizes(mats[i],matSize,matSize,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(mats[i],MATSEQDENSECUDA);CHKERRQ(ierr);
    ierr = MatSetUp(mats[i]);CHKERRQ(ierr);
    ierr = MatSetRandom(mats[i],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(mats[i],xx+i,yy+i);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(mats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatConvert(mats[i],MATSAME,MAT_INITIAL_MATRIX,matsref+i);CHKERRQ(ierr);
    ierr = MatLUFactor(mats[i],NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatLUFactor(matsref[i],NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatSetOperation(mats[i],MATOP_SOLVE,(void(*)(void))MatSolve_SeqDenseCUDA_Async);CHKERRQ(ierr);
    ierr = VecZeroEntries(yy[i]);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(xx[i]);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(xx[i]);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(yy[i]);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(yy[i]);CHKERRQ(ierr);
    ierr = VecDuplicate(xx[i],xxref+i);CHKERRQ(ierr);
    ierr = VecDuplicate(yy[i],yyref+i);CHKERRQ(ierr);
    ierr = VecCopy(xx[i],xxref[i]);CHKERRQ(ierr);
    ierr = VecCopy(yy[i],yyref[i]);CHKERRQ(ierr);
    // Move any data down onto the GPU now
    ierr = MatSeqDenseCUDACopyToGPU_IMPLS(mats[i]);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(xx[i],&dummy);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(xx[i],&dummy);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(yy[i],&dummy);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(yy[i],&dummy);CHKERRQ(ierr);
  }
  cerr = cudaProfilerStart();CHKERRCUDA(cerr);
  ierr = HostDeviceBarrier();CHKERRQ(ierr);

  for (PetscInt i = 0; i < nMat; ++i) {
    ierr = PetscDeviceContextSetCurrentContext(subCtx[i]);CHKERRQ(ierr);
    ierr = MatSolve(mats[i],xx[i],yy[i]);CHKERRQ(ierr);
    // uncomment below to get the current synchronous version
    //ierr = HostDeviceBarrier();CHKERRQ(ierr);
  }

  ierr = HostDeviceBarrier();CHKERRQ(ierr);
  cerr = cudaProfilerStop();CHKERRCUDA(cerr);
  ierr = PetscDeviceContextJoin(dctx,PETSC_FALSE,PETSC_TRUE,nMat,&subCtx);CHKERRQ(ierr);
  ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetCurrentContext(dctx);CHKERRQ(ierr);

  for (PetscInt i = 0; i < nMat; ++i) {
    // reference solve goes here so that it does not interfere with async solves, we want
    // those to overlap as much as possible
    ierr = MatSolve(matsref[i],xxref[i],yyref[i]);CHKERRQ(ierr);
    ierr = VecClose(yyref[i],yy[i]);CHKERRQ(ierr);
    ierr = VecClose(xxref[i],xx[i]);CHKERRQ(ierr);
    ierr = MatDestroy(mats+i);CHKERRQ(ierr);
    ierr = MatDestroy(matsref+i);CHKERRQ(ierr);
    ierr = VecDestroy(xx+i);CHKERRQ(ierr);
    ierr = VecDestroy(xxref+i);CHKERRQ(ierr);
    ierr = VecDestroy(yy+i);CHKERRQ(ierr);
    ierr = VecDestroy(yyref+i);CHKERRQ(ierr);
  }
  ierr = PetscFree6(mats,matsref,xx,xxref,yy,yyref);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"All tests successful\n");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
