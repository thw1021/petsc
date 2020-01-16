#include <cuda_runtime.h>

#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>
#include <../src/mat/impls/sell/seq/sell.h>  /*I   "petscmat.h"  I*/

typedef struct {
  PetscInt  *colidx;           /* column index */
  MatScalar *val;
  PetscInt  *sliidx;
  PetscInt  nonzerostate;
} Mat_SeqSELLCUDA;

static PetscErrorCode MatSeqSELLCUDA_Destroy(Mat_SeqSELLCUDA **cudastruct)
{
  cudaError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (*cudastruct) {
    if ((*cudastruct)->colidx) {
      cerr = cudaFree((*cudastruct)->colidx);CHKERRCUDA(cerr);
    }
    if ((*cudastruct)->val) {
      cerr = cudaFree((*cudastruct)->val);CHKERRCUDA(cerr);
    }
    if ((*cudastruct)->sliidx) {
      cerr = cudaFree((*cudastruct)->sliidx);CHKERRCUDA(cerr);
    }
    ierr = PetscFree(*cudastruct);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqSELLCUDACopyToGPU(Mat A)
{
  Mat_SeqSELLCUDA  *cudastruct = (Mat_SeqSELLCUDA*)A->spptr;
  Mat_SeqSELL      *a = (Mat_SeqSELL*)A->data;
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    ierr = PetscLogEventBegin(MAT_CUDACopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (!A->assembled || A->nonzerostate != cudastruct->nonzerostate) {
      if (cudastruct->colidx) {
        cerr = cudaFree(cudastruct->colidx);CHKERRCUDA(cerr);
      }
      if (cudastruct->val) {
        cerr = cudaFree(cudastruct->val);CHKERRCUDA(cerr);
      }
      if (cudastruct->sliidx) {
        cerr = cudaFree(cudastruct->sliidx);CHKERRCUDA(cerr);
      }
      cerr = cudaMalloc((void **)&(cudastruct->colidx),a->maxallocmat*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = cudaMalloc((void **)&(cudastruct->val),a->maxallocmat*sizeof(MatScalar));CHKERRCUDA(cerr);
      cerr = cudaMalloc((void **)&(cudastruct->sliidx),(a->totalslices+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
    }
    /* copy values, nz or maxallocmat? */
    cerr = cudaMemcpy(cudastruct->colidx,a->colidx,a->sliidx[a->totalslices]*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(cudastruct->val,a->val,a->sliidx[a->totalslices]*sizeof(MatScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(cudastruct->sliidx,a->sliidx,(a->totalslices+1)*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    ierr = PetscLogCpuToGpu(a->sliidx[a->totalslices]*(sizeof(MatScalar)+sizeof(PetscInt))+(a->totalslices+1)*sizeof(PetscInt));CHKERRQ(ierr);
    cudastruct->nonzerostate = A->nonzerostate;
    cerr  = WaitForGPU();CHKERRCUDA(cerr);
    A->offloadmask = PETSC_OFFLOAD_BOTH;
    ierr = PetscLogEventEnd(MAT_CUDACopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

__global__ void matmult_seqell_kernel(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  PetscInt       i,row,slice_id,row_in_slice,block_start,block_inc,block_end;
  MatScalar      sum;
  /* one thread per row. Multiple rows may be assigned to one thread if block_inc is greater than the number of threads per block (blockDim.x) */
  block_inc   = blockDim.x*(nrows+gridDim.x*blockDim.x-1)/(gridDim.x*blockDim.x);
  block_start = blockIdx.x*block_inc;
  block_end   = min((blockIdx.x+1)*block_inc,nrows);

  for (row=block_start+threadIdx.x; row<block_end; row+=blockDim.x) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;
    if (slice_id < totalslices) {
      sum = 0.0;
      for (i=sliidx[slice_id]+row_in_slice; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT) sum += aval[i] * x[acolidx[i]];
      if (slice_id == totalslices-1 && nrows%SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows%SLICE_HEIGHT)) y[row] = sum;
      } else {
        y[row] = sum;
      }
    }
  }
}

__global__ void matmultadd_seqell_kernel(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  PetscInt       i,row,slice_id,row_in_slice,block_start,block_inc,block_end;
  MatScalar      sum;

  /* one thread per row. Multiple rows may be assigned to one thread if block_inc is greater than the number of threads per block (blockDim.x) */
  block_inc   = blockDim.x*(nrows+gridDim.x*blockDim.x-1)/(gridDim.x*blockDim.x);
  block_start = blockIdx.x*block_inc;
  block_end   = min((blockIdx.x+1)*block_inc,nrows);

  for (row=block_start+threadIdx.x; row<block_end; row+=blockDim.x) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;
    if (slice_id < totalslices) {
      sum = 0.0;
      for (i=sliidx[slice_id]+row_in_slice; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT) sum += aval[i] * x[acolidx[i]];
      if (slice_id == totalslices-1 && nrows%SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows%SLICE_HEIGHT)) z[row] = y[row] + sum;
      } else {
        z[row] = y[row] + sum;
      }
    }
  }
}

PetscErrorCode MatMult_SeqSELLCUDA(Mat A,Vec xx,Vec yy)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  Mat_SeqSELLCUDA   *cudastruct = (Mat_SeqSELLCUDA*)A->spptr;
  PetscScalar       *y;
  const PetscScalar *x;
  PetscInt          totalslices=a->totalslices,nrows = A->rmap->n;
  MatScalar         *aval = cudastruct->val;
  PetscInt          *acolidx = cudastruct->colidx;
  PetscInt          *sliidx = cudastruct->sliidx;
  PetscErrorCode    ierr;
  cudaError_t       cerr;

  PetscFunctionBegin;
  ierr = MatSeqSELLCUDACopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(yy,&y);CHKERRQ(ierr);

  matmult_seqell_kernel<<<512,256>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
  
  ierr = VecCUDARestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(yy,&y);CHKERRQ(ierr);
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*a->nz-a->nonzerorowcnt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqSELLCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  Mat_SeqSELLCUDA   *cudastruct = (Mat_SeqSELLCUDA*)A->spptr;
  PetscScalar       *z;
  const PetscScalar *y,*x;
  PetscInt          totalslices=a->totalslices,nrows = A->rmap->n;
  MatScalar         *aval=cudastruct->val;
  PetscInt          *acolidx = cudastruct->colidx;
  PetscInt          *sliidx = cudastruct->sliidx;
  PetscErrorCode    ierr;
  cudaError_t       cerr;

  PetscFunctionBegin;
  ierr = MatSeqSELLCUDACopyToGPU(A);CHKERRQ(ierr);
  if (a->nz) {
    ierr = VecCUDAGetArrayRead(xx,&x);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yy,&y);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(zz,&z);CHKERRQ(ierr);

    matmultadd_seqell_kernel<<<512,256>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);

    ierr = VecCUDARestoreArrayRead(xx,&x);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yy,&y);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(zz,&z);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_SeqSELLCUDA(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqSELLCUDA options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

static PetscErrorCode MatAssemblyEnd_SeqSELLCUDA(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqSELL(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  if (A->factortype == MAT_FACTOR_NONE) {
    ierr = MatSeqSELLCUDACopyToGPU(A);CHKERRQ(ierr);
  }
  A->ops->mult    = MatMult_SeqSELLCUDA;
  A->ops->multadd = MatMultAdd_SeqSELLCUDA;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqSELLCUDA(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->factortype==MAT_FACTOR_NONE) {
    if (A->offloadmask != PETSC_OFFLOAD_UNALLOCATED) {
      ierr = MatSeqSELLCUDA_Destroy((Mat_SeqSELLCUDA**)&A->spptr);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy_SeqSELL(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqSELLCUDA(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat             C;
  Mat_SeqSELLCUDA *cudastruct;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqSELL(A,cpvalues,B);CHKERRQ(ierr);
  C    = *B;
  ierr = PetscFree(C->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&C->defaultvectype);CHKERRQ(ierr);

  /* inject CUSPARSE-specific stuff */
  if (C->factortype==MAT_FACTOR_NONE) {
    ierr = PetscNew(&cudastruct);CHKERRQ(ierr);
    C->spptr = cudastruct;
  }

  C->ops->assemblyend    = MatAssemblyEnd_SeqSELLCUDA;
  C->ops->destroy        = MatDestroy_SeqSELLCUDA;
  C->ops->setfromoptions = MatSetFromOptions_SeqSELLCUDA;
  C->ops->mult           = MatMult_SeqSELLCUDA;
  C->ops->multadd        = MatMultAdd_SeqSELLCUDA;
  C->ops->duplicate      = MatDuplicate_SeqSELLCUDA;

  ierr = PetscObjectChangeTypeName((PetscObject)C,MATSEQSELLCUDA);CHKERRQ(ierr);
  C->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_SeqSELL_SeqSELLCUDA(Mat B)
{
  Mat_SeqSELLCUDA *cudastruct;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&B->defaultvectype);CHKERRQ(ierr);

  /* inject CUSPARSE-specific stuff */
  if (B->factortype==MAT_FACTOR_NONE) {
    ierr = PetscNew(&cudastruct);CHKERRQ(ierr);
    B->spptr = cudastruct;
  }

  B->ops->assemblyend    = MatAssemblyEnd_SeqSELLCUDA;
  B->ops->destroy        = MatDestroy_SeqSELLCUDA;
  B->ops->setfromoptions = MatSetFromOptions_SeqSELLCUDA;
  B->ops->mult           = MatMult_SeqSELLCUDA;
  B->ops->multadd        = MatMultAdd_SeqSELLCUDA;
  B->ops->duplicate      = MatDuplicate_SeqSELLCUDA;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQSELLCUDA);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqSELLCUDA(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqSELL(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqSELL_SeqSELLCUDA(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
