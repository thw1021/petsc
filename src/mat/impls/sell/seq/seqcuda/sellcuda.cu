#include <cuda_runtime.h>

#include <petsc/private/cudavecimpl.h>
#include <../src/mat/impls/sell/seq/sell.h>  /*I   "petscmat.h"  I*/

#define SLICE_HEIGHT 16

typedef struct {
  PetscInt  *colidx;           /* column index */
  MatScalar *val;
  PetscInt  *sliidx;
  PetscInt  *sliperm;          /* permutation for slices */
  PetscInt  *blockidx;
  PetscInt  *block_row_map;
  PetscInt  nonzerostate;
  PetscInt  kernelchoice;
  PetscBool perm;
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
    if ((*cudastruct)->sliperm) {
      cerr = cudaFree((*cudastruct)->sliperm);CHKERRCUDA(cerr);
    }
    if ((*cudastruct)->blockidx) {
      cerr = cudaFree((*cudastruct)->blockidx);CHKERRCUDA(cerr);
    }
    if ((*cudastruct)->block_row_map) {
      cerr = cudaFree((*cudastruct)->block_row_map);CHKERRCUDA(cerr);
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
    if (A->assembled && A->nonzerostate == cudastruct->nonzerostate) {
      /* copy values only */
      cerr = cudaMemcpy(cudastruct->val,a->val,a->sliidx[a->totalslices]*sizeof(MatScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
      ierr = PetscLogCpuToGpu(a->sliidx[a->totalslices]*(sizeof(MatScalar)));CHKERRQ(ierr);
    } else {
      if (cudastruct->colidx) {
        cerr = cudaFree(cudastruct->colidx);CHKERRCUDA(cerr);
      }
      if (cudastruct->val) {
        cerr = cudaFree(cudastruct->val);CHKERRCUDA(cerr);
      }
      if (cudastruct->sliidx) {
        cerr = cudaFree(cudastruct->sliidx);CHKERRCUDA(cerr);
      }
      if (cudastruct->sliperm) {
        cerr = cudaFree(cudastruct->sliperm);CHKERRCUDA(cerr);
      }
      if (cudastruct->blockidx) {
        cerr = cudaFree(cudastruct->blockidx);CHKERRCUDA(cerr);
      }
      if (cudastruct->block_row_map) {
        cerr = cudaFree(cudastruct->block_row_map);CHKERRCUDA(cerr);
      }
      cerr = cudaMalloc((void **)&(cudastruct->colidx),a->maxallocmat*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = cudaMalloc((void **)&(cudastruct->val),a->maxallocmat*sizeof(MatScalar));CHKERRCUDA(cerr);
      /* copy values, nz or maxallocmat? */
      cerr = cudaMemcpy(cudastruct->colidx,a->colidx,a->sliidx[a->totalslices]*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
      cerr = cudaMemcpy(cudastruct->val,a->val,a->sliidx[a->totalslices]*sizeof(MatScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

      if (a->totalblocks && !cudastruct->kernelchoice) { /* For cases with wide slices, we will use column blocking, so blockidx is used instead of sliceidx. */
        cerr = cudaMalloc((void **)&(cudastruct->blockidx),(a->totalblocks+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
        cerr = cudaMalloc((void **)&(cudastruct->block_row_map),a->totalblocks*sizeof(PetscInt));CHKERRCUDA(cerr);
        cerr = cudaMemcpy(cudastruct->blockidx,a->blockidx,(a->totalblocks+1)*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
        cerr = cudaMemcpy(cudastruct->block_row_map,a->block_row_map,a->totalblocks*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
        ierr = PetscLogCpuToGpu(a->sliidx[a->totalslices]*(sizeof(MatScalar)+sizeof(PetscInt))+(2*a->totalblocks+1)*sizeof(PetscInt));CHKERRQ(ierr);
      } else {
        cerr = cudaMalloc((void **)&(cudastruct->sliidx),(a->totalslices+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
        cerr = cudaMemcpy(cudastruct->sliidx,a->sliidx,(a->totalslices+1)*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
        if (a->sliperm) {
          cerr = cudaMalloc((void **)&(cudastruct->sliperm),(a->totalslices+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
          cerr = cudaMemcpy(cudastruct->sliperm,a->sliperm,a->totalslices*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
          ierr = PetscLogCpuToGpu(a->sliidx[a->totalslices]*(sizeof(MatScalar)+sizeof(PetscInt))+(2*a->totalslices+1)*sizeof(PetscInt));CHKERRQ(ierr);
        } else {
          ierr = PetscLogCpuToGpu(a->sliidx[a->totalslices]*(sizeof(MatScalar)+sizeof(PetscInt))+(a->totalslices+1)*sizeof(PetscInt));CHKERRQ(ierr);
        }
      }
      cudastruct->nonzerostate = A->nonzerostate;
    }
    cerr = WaitForCUDA();CHKERRCUDA(cerr);
    ierr = PetscLogEventEnd(MAT_CUDACopyToGPU,A,0,0,0);CHKERRQ(ierr);
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

__global__ void matmult_seqsell_basic_kernel(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  PetscInt  i,row,slice_id,row_in_slice;
  MatScalar sum;
  /* one thread per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;
    sum = 0.0;
    for (i=sliidx[slice_id]+row_in_slice; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT) sum += aval[i] * x[acolidx[i]];
    y[row] = sum;
  }
}

__global__ void matmultadd_seqsell_basic_kernel(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  PetscInt  i,row,slice_id,row_in_slice;
  MatScalar sum;
  /* one thread per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;
    sum = 0.0;
    for (i=sliidx[slice_id]+row_in_slice; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT) sum += aval[i] * x[acolidx[i]];
    z[row] = y[row] + sum;
  }
}

__global__ void matmult_seqsell_tiled_kernelx(PetscInt nrows,PetscInt totalblocks,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *blockidx,const PetscInt *block_row_map,const PetscScalar *x,PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,row_in_slice;
  /* multiple blocks per slice. */
  row = block_row_map[blockIdx.x] + threadIdx.x;
  if (row < nrows) {
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=blockidx[blockIdx.x]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<blockidx[blockIdx.x+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 16) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+16)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 8) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+8)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 4) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      atomicAdd(&y[row],shared[threadIdx.x]);
    }
  }
}

__global__ void matmult_seqsell_tiled_kernel6(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 16) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+16)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 8) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+8)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 4) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

__global__ void matmult_seqsell_tiled_kernel5(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 8) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+8)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 4) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

__global__ void matmult_seqsell_tiled_kernel4(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 4) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

__global__ void matmult_seqsell_tiled_kernel3(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

__global__ void matmult_seqsell_tiled_perm_kernel3(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscInt *sliperm,const PetscScalar *x,PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  slice_id     = sliperm[(blockIdx.x*blockDim.x+threadIdx.x)/SLICE_HEIGHT];
  row_in_slice = (blockIdx.x*blockDim.x+threadIdx.x)%SLICE_HEIGHT;
  row          = slice_id*SLICE_HEIGHT+row_in_slice;
  if (row < nrows) {
    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

__global__ void matmult_seqsell_tiled_kernel2(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

__global__ void matmultadd_seqsell_tiled_kernelx(PetscInt nrows,PetscInt totalblocks,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *blockidx,const PetscInt *block_row_map,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,row_in_slice;
  /* multiple blocks per slice. */
  row = block_row_map[blockIdx.x]+threadIdx.x;
  if (row < nrows) {
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=blockidx[blockIdx.x]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<blockidx[blockIdx.x+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 16) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+16)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 8) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+8)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 4) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

__global__ void matmultadd_seqsell_tiled_kernel6(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 16) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+16)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 8) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+8)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 4) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

__global__ void matmultadd_seqsell_tiled_kernel5(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 8) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+8)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 4) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

__global__ void matmultadd_seqsell_tiled_kernel4(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 4) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

__global__ void matmultadd_seqsell_tiled_kernel3(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

__global__ void matmultadd_seqsell_tiled_perm_kernel3(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscInt *sliperm,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  slice_id     = sliperm[(blockIdx.x*blockDim.x+threadIdx.x)/SLICE_HEIGHT];
  row_in_slice = (blockIdx.x*blockDim.x+threadIdx.x)%SLICE_HEIGHT;
  row          = slice_id*SLICE_HEIGHT+row_in_slice;
  if (row < nrows) {
    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 2) {
      shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

__global__ void matmultadd_seqsell_tiled_kernel2(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt   i,row,slice_id,row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
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
  MatScalar         *aval;
  PetscInt          *acolidx;
  PetscInt          *sliidx,*sliperm;
  PetscErrorCode    ierr;
  cudaError_t       cerr;
  PetscInt          nblocks,blocksize = 512; /* blocksize must be multiple of SLICE_HEIGHT*32 */
  dim3              block2(256,2),block4(128,4),block8(64,8),block16(32,16),block32(16,32);

  PetscFunctionBegin;
  if (a->sliceheight !=16) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"The kernel requires a slice height of 16, but the input matrix has a slice height of %D\n",a->sliceheight);
  ierr = MatSeqSELLCUDACopyToGPU(A);CHKERRQ(ierr);
  /* cudastruct may not be available until MatSeqSELLCUDACopyToGPU() is called */ 
  aval    = cudastruct->val;
  acolidx = cudastruct->colidx;
  sliidx  = cudastruct->sliidx;
  sliperm = cudastruct->sliperm;

  ierr = VecCUDAGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);

  switch(cudastruct->kernelchoice) {
    case 6:
      nblocks = 1+(nrows-1)/(blocksize/32); /* 1 slice per block if blocksize=512 */
      matmult_seqsell_tiled_kernel6<<<nblocks,block32>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
      break;
    case 5:
      nblocks = 1+(nrows-1)/(blocksize/16); /* 2 slices per block if blocksize=512*/
      matmult_seqsell_tiled_kernel5<<<nblocks,block16>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
      break;
    case 4:
      nblocks = 1+(nrows-1)/(blocksize/8); /* 4 slices per block if blocksize=512 */
      matmult_seqsell_tiled_kernel4<<<nblocks,block8>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
      break;
    case 3:
      nblocks = 1+(nrows-1)/(blocksize/4); /* 8 slices per block if blocksize=512 */
      matmult_seqsell_tiled_kernel3<<<nblocks,block4>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
      break;
    case 2: /* 16 slices per block if blocksize=512 */
      nblocks = 1+(nrows-1)/(blocksize/2);
      matmult_seqsell_tiled_kernel2<<<nblocks,block2>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
      break;
    case 1: /* 32 slices per block if blocksize=512 */
      nblocks = 1+(nrows-1)/blocksize;
      matmult_seqsell_basic_kernel<<<nblocks,blocksize>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
      break;
    case 0:
      if (a->avgslicewidth > 64 && a->maxslicewidth > 4000) {
        matmult_seqsell_tiled_kernelx<<<a->totalblocks,block32>>>(nrows,a->totalblocks,acolidx,aval,cudastruct->blockidx,cudastruct->block_row_map,x,y);
      } else {
        if (a->maxslicewidth <= 33 || (a->maxslicewidth <= 200 && a->avgslicewidth<=15.0)) {
          nblocks = 1+(nrows-1)/(blocksize/4);
          if (sliperm && cudastruct->perm) matmult_seqsell_tiled_perm_kernel3<<<nblocks,block4>>>(nrows,totalslices,acolidx,aval,sliidx,sliperm,x,y);
          else matmult_seqsell_tiled_kernel3<<<nblocks,block4>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
        } else {
          nblocks = 1+(nrows-1)/(blocksize/32);
          matmult_seqsell_tiled_kernel6<<<nblocks,block32>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
        }
      }
  }
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(yy,&y);CHKERRQ(ierr);
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
  PetscInt          *sliidx = cudastruct->sliidx,*sliperm = cudastruct->sliperm;
  PetscErrorCode    ierr;
  cudaError_t       cerr;

  PetscFunctionBegin;
  if (a->sliceheight !=16) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"The kernel requires a slice height of 16, but the input matrix has a slice height of %D\n",a->sliceheight);
  ierr = MatSeqSELLCUDACopyToGPU(A);CHKERRQ(ierr);
  if (a->nz) {
    PetscInt nblocks,blocksize = 512;
    dim3     block2(256,2),block4(128,4),block8(64,8),block16(32,16),block32(16,32);
    ierr = VecCUDAGetArrayRead(xx,&x);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yy,&y);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(zz,&z);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);

    switch(cudastruct->kernelchoice) {
      case 6:
        nblocks = 1+(nrows-1)/(blocksize/32);
        matmultadd_seqsell_tiled_kernel6<<<nblocks,block32>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
        break;
      case 5:
        nblocks = 1+(nrows-1)/(blocksize/16);
        matmultadd_seqsell_tiled_kernel5<<<nblocks,block16>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
        break;
      case 4:
        nblocks = 1+(nrows-1)/(blocksize/8);
        matmultadd_seqsell_tiled_kernel4<<<nblocks,block8>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
        break;
      case 3:
        nblocks = 1+(nrows-1)/(blocksize/4);
        matmultadd_seqsell_tiled_kernel3<<<nblocks,block4>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
        break;
      case 2:
        nblocks = 1+(nrows-1)/(blocksize/2);
        matmultadd_seqsell_tiled_kernel2<<<nblocks,block2>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
        break;
      case 1:
        nblocks = 1+(nrows-1)/blocksize;
        matmultadd_seqsell_basic_kernel<<<nblocks,blocksize>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
        break;
      case 0:
        if (a->avgslicewidth > 64 && a->maxslicewidth > 4000) {
          matmultadd_seqsell_tiled_kernelx<<<a->totalblocks,block32>>>(nrows,a->totalblocks,acolidx,aval,a->blockidx,a->block_row_map,x,y,z);
        } else {
          if (a->maxslicewidth <= 33 || (a->maxslicewidth <= 200 && a->avgslicewidth<=15.0)) {
            nblocks = 1+(nrows-1)/(blocksize/4);
            if (sliperm) matmultadd_seqsell_tiled_perm_kernel3<<<nblocks,block4>>>(nrows,totalslices,acolidx,aval,sliidx,sliperm,x,y,z);
            else matmultadd_seqsell_tiled_kernel3<<<nblocks,block4>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
          } else {
            nblocks = 1+(nrows-1)/(blocksize/32);
            matmultadd_seqsell_tiled_kernel6<<<nblocks,block32>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);

          }
        }
    }
    cerr = WaitForCUDA();CHKERRCUDA(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xx,&x);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yy,&y);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(zz,&z);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*a->nz);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_SeqSELLCUDA(PetscOptionItems *PetscOptionsObject,Mat A)
{
  Mat_SeqSELLCUDA *cudastruct = (Mat_SeqSELLCUDA*)A->spptr;
  PetscInt        kernel;
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqSELLCUDA options");CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_sell_spmv_cuda_kernel",&kernel,&flg);
  if (flg) {
    if (kernel< 0 || kernel >6) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Wrong kernel choice: %D it should be in [0,6]",kernel); 
    cudastruct->kernelchoice = kernel;
  }
  ierr = PetscOptionsGetBool(NULL,NULL,"-mat_sell_spmv_cuda_kernel_perm",&cudastruct->perm,&flg);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatAssemblyEnd_SpMV_Preprocessing_Private(Mat A)
{
  Mat_SeqSELL     *a = (Mat_SeqSELL*)A->data;
  Mat_SeqSELLCUDA *cudastruct = (Mat_SeqSELLCUDA*)A->spptr;
  PetscErrorCode  ierr;

  ierr = MatSeqSELLGetAvgSliceWidth(A,&a->avgslicewidth);CHKERRQ(ierr);
  ierr = MatSeqSELLGetMaxSliceWidth(A,&a->maxslicewidth);CHKERRQ(ierr);
  ierr = MatSeqSELLGetFillRatio(A,&a->fillratio);CHKERRQ(ierr);

  if (a->avgslicewidth > 64 && a->maxslicewidth > 4000) { /* Set up column blocking for fast SpMV */
    PetscInt i,j,bidx = 0;

    a->totalblocks = 0;
    for (i=0; i<a->totalslices; ++i) a->totalblocks += (a->sliidx[i+1]-a->sliidx[i]+a->sliceheight*32-1)/(a->sliceheight*32);
    ierr = PetscMalloc2(a->totalblocks+1,&a->blockidx,a->totalblocks,&a->block_row_map);CHKERRQ(ierr);
    for (i=0; i<a->totalslices; ++i)
      for (j=0; j<(a->sliidx[i+1]-a->sliidx[i]+a->sliceheight*32-1)/(a->sliceheight*32); ++j) {
        a->blockidx[bidx]      = a->sliidx[i]+j*a->sliceheight*32;
        a->block_row_map[bidx] = i*a->sliceheight;
        bidx++;
      }
    a->blockidx[bidx] = a->sliidx[a->totalslices];
  } else {
    if (cudastruct->perm) {
      if (!a->totalblocks) {
        ierr = PetscFree2(a->blockidx,a->block_row_map);CHKERRQ(ierr);
      }
      if (a->maxslicewidth <= 33 || (a->maxslicewidth <= 200 && a->avgslicewidth<=15.0)) {
        PetscInt i,*sliwidth;
        ierr = PetscMalloc1(a->totalslices,&sliwidth);CHKERRQ(ierr);
        ierr = PetscMalloc1(a->totalslices,&a->sliperm);CHKERRQ(ierr);
        for (i=0; i<a->totalslices; ++i) {
          sliwidth[i] = (a->sliidx[i+1]-a->sliidx[i])/a->sliceheight;
          a->sliperm[i] = i;
        }
        ierr = PetscSortIntWithPermutation(a->totalslices,sliwidth,a->sliperm);CHKERRQ(ierr);
        ierr = PetscFree(sliwidth);CHKERRQ(ierr);
      } else {
        ierr = PetscFree(a->sliperm);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqSELLCUDA(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqSELL(A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SpMV_Preprocessing_Private(A);CHKERRQ(ierr);
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

PETSC_EXTERN PetscErrorCode MatConvert_SeqSELL_SeqSELLCUDA(Mat B)
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

  /* No need to assemble SeqSELL, but need to do the preprocessing for SpMV */
  ierr = MatAssemblyEnd_SpMV_Preprocessing_Private(B);CHKERRQ(ierr);

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
