/*
   Implements the sequential cuda vectors.
*/

#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/cudavecimpl.h>
#include <petsc/private/deviceimpl.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscCUDAFlag for the vector
    Does NOT zero the CUDA array

 */
PetscErrorCode VecCUDAAllocateCheck(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUDA       *veccuda;
  PetscBool      option_set;

  PetscFunctionBegin;
  if (!v->event) {
    const char *prefix;

    ierr = PetscEventCreate(&v->event);CHKERRQ(ierr);
    ierr = PetscEventSetType(v->event,PETSCSTREAMCUDA);CHKERRQ(ierr);
    ierr = PetscObjectGetOptionsPrefix((PetscObject)v,&prefix);CHKERRQ(ierr);
    ierr = PetscEventSetFromOptions(PetscObjectComm((PetscObject)v),prefix,v->event);CHKERRQ(ierr);
  }
  if (!v->spptr) {
    PetscReal pinned_memory_min;
    ierr = PetscCalloc(sizeof(Vec_CUDA),&v->spptr);CHKERRQ(ierr);
    veccuda = (Vec_CUDA*)v->spptr;
    err = cudaMalloc((void**)&veccuda->GPUarray_allocated,sizeof(PetscScalar)*((PetscBLASInt)v->map->n));CHKERRCUDA(err);
    veccuda->GPUarray = veccuda->GPUarray_allocated;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      if (v->data && ((Vec_Seq*)v->data)->array) {
        v->offloadmask = PETSC_OFFLOAD_CPU;
      } else {
        v->offloadmask = PETSC_OFFLOAD_GPU;
      }
    }
    pinned_memory_min = 0;

    /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
       Note: This same code duplicated in VecCreate_SeqCUDA_Private() and VecCreate_MPICUDA_Private(). Is there a good way to avoid this? */
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)v),((PetscObject)v)->prefix,"VECCUDA Options","Vec");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&option_set);CHKERRQ(ierr);
    if (option_set) v->minimum_bytes_pinned_memory = pinned_memory_min;
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecCUDACopyToGPU(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUDA       *veccuda;
  PetscScalar    *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    ierr           = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    veccuda        = (Vec_CUDA*)v->spptr;
    varray         = veccuda->GPUarray;
    err            = cudaMemcpy(varray,((Vec_Seq*)v->data)->array,v->map->n*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
    ierr           = PetscLogCpuToGpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr           = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCUDACopyToGPUAsync(Vec v,PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscValidStreamTypeSpecific(pstream,2,PETSCSTREAMCUDA);
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    Vec_CUDA     *veccuda;
    PetscScalar  *varray;
    cudaStream_t cstream;
    cudaError_t  cerr;

    ierr           = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    veccuda        = (Vec_CUDA*)v->spptr;
    varray         = veccuda->GPUarray;
    ierr           = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    cerr           = cudaMemcpyAsync(varray,((Vec_Seq*)v->data)->array,v->map->n*sizeof(PetscScalar),cudaMemcpyHostToDevice,cstream);CHKERRCUDA(cerr);
    ierr           = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    ierr           = PetscLogCpuToGpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr           = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    ierr           = PetscStreamRecordEvent(pstream,v->event);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/*
     VecCUDACopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecCUDACopyFromGPU(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUDA       *veccuda;
  PetscScalar    *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDAAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    ierr           = PetscLogEventBegin(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    veccuda        = (Vec_CUDA*)v->spptr;
    varray         = veccuda->GPUarray;
    err            = cudaMemcpy(((Vec_Seq*)v->data)->array,varray,v->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    ierr           = PetscLogGpuToCpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr           = PetscLogEventEnd(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCUDACopyFromGPUAsync(Vec v,PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscValidStreamTypeSpecific(pstream,2,PETSCSTREAMCUDA);
  ierr = VecCUDAAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    Vec_CUDA     *veccuda;
    PetscScalar  *varray;
    cudaStream_t cstream;
    cudaError_t  cerr;

    ierr           = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    veccuda        = (Vec_CUDA*)v->spptr;
    varray         = veccuda->GPUarray;
    ierr           = PetscLogEventBegin(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    cerr           = cudaMemcpyAsync(((Vec_Seq*)v->data)->array,varray,v->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost,cstream);CHKERRCUDA(cerr);
    ierr           = PetscLogEventEnd(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    ierr           = PetscLogGpuToCpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr           = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    ierr           = PetscStreamRecordEvent(pstream,v->event);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/*MC
   VECSEQCUDA - VECSEQCUDA = "seqcuda" - The basic sequential vector, modified to use CUDA

   Options Database Keys:
. -vec_type seqcuda - sets the vector type to VECSEQCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecAYPX_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  PetscScalar       sone = 1.0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    err = cudaMemcpy(yarray,xarray,bn*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
  } else if (alpha == (PetscScalar)1.0) {
    /* ensure cublas uses NULL stream */
    ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuFlops(1.0*yin->map->n);CHKERRQ(ierr);
  } else {
    /* ensure cublas uses NULL stream */
    ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,&alpha,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,&sone,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAYPX_SeqCUDAAsync(Vec yin,PetscStreamScalar pscalalpha,Vec xin,PetscStream pstream)
{
  PetscBool         isZero,isOne;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  PetscScalar       sone = 1.0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaStream_t      cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,4,PETSCSTREAMCUDA);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetInfo(pscalalpha,PSS_ZERO,PETSC_FALSE,&isZero,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetInfo(pscalalpha,PSS_ONE,PETSC_FALSE,&isOne,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (isZero) {
    cudaError_t cerr;
    cerr = cudaMemcpyAsync(yarray,xarray,bn*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,cstream);CHKERRCUDA(cerr);
  } else if (isOne) {
    const PetscScalar *alpha;

    ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscalalpha,&alpha,pstream);CHKERRQ(ierr);
    cberr = cublasXaxpy(cublasv2handle,bn,alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscalalpha,&alpha,pstream);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(1.0*yin->map->n);CHKERRQ(ierr);
  } else {
    const PetscScalar *alpha;

    ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscalalpha,&alpha,pstream);CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,alpha,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,&sone,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscalalpha,&alpha,pstream);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode ierr;
  PetscBool      xiscuda;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)xin,&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) PetscFunctionReturn(0);
  if (xiscuda) {
    PetscBLASInt      one = 1,bn = 0;
    const PetscScalar *xarray;
    PetscScalar       *yarray;
    cublasHandle_t    cublasv2handle;
    cublasStatus_t    cberr;

    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
    /* ensure cublas is on NULL stream */
    ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqCUDAAsync(Vec yin,PetscStreamScalar pscal,Vec xin,PetscStream pstream)
{
  PetscErrorCode ierr;
  PetscBool      xiscuda,isZero;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,4,PETSCSTREAMCUDA);
  ierr = PetscStreamScalarGetInfo(pscal,PSS_ZERO,PETSC_FALSE,&isZero,pstream);CHKERRQ(ierr);
  if (isZero) PetscFunctionReturn(0);
  ierr = PetscObjectTypeCompareAny((PetscObject)xin,&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (xiscuda) {
    const PetscScalar *xarray;
    PetscScalar       *yarray;
    cublasHandle_t    cublasv2handle;
    cublasStatus_t    cberr;
    PetscBLASInt      bn = 0, one = 1;
    const PetscScalar *dptr;
    cudaStream_t      cstream;

    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscal,&dptr,pstream);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXaxpy(cublasv2handle,bn,dptr,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscal,&dptr,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    PetscScalar alpha;

    /* Do on host */
    ierr = PetscStreamScalarAwait(pscal,&alpha,pstream);CHKERRQ(ierr);
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_SeqCUDA(Vec win, Vec xin, Vec yin)
{
  PetscInt                              n = xin->map->n;
  const PetscScalar                     *xarray=NULL,*yarray=NULL;
  PetscScalar                           *warray=NULL;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayWrite(win,&warray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_SeqCUDAAsync(Vec win, Vec xin, Vec yin,PetscStream pstream)
{
  PetscInt                              n = xin->map->n;
  const PetscScalar                     *xarray=NULL,*yarray=NULL;
  PetscScalar                           *warray=NULL;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;
  PetscErrorCode                        ierr;
  cudaStream_t                          cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,4,PETSCSTREAMCUDA);
  /* Could make 2 temp streams here but current PCIE only allows 1 transfer per direction,
     maybe in the future? */
  ierr = VecCUDAGetArrayWriteAsync(win,&warray,pstream);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(thrust::cuda::par.on(cstream),xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWriteAsync(win,&warray,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_SeqCUDA(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecCopy_SeqCUDA(yin,win);CHKERRQ(ierr);
  } else {
    const PetscScalar *xarray=NULL,*yarray=NULL;
    PetscScalar       *warray=NULL;
    cublasHandle_t    cublasv2handle;
    cublasStatus_t    cberr;
    cudaError_t       cerr;
    cudaStream_t      stream;
    PetscBLASInt      one = 1.0,bn=0;

    ierr = PetscBLASIntCast(win->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(win,&warray);CHKERRQ(ierr);
    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    /* ensure cublas is NULL stream */
    ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasGetStream(cublasv2handle,&stream);CHKERRCUBLAS(cberr);
    cerr = cudaMemcpyAsync(warray,yarray,win->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,stream);CHKERRCUDA(cerr);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,warray,one);CHKERRCUBLAS(cberr);
    cerr = WaitForCUDA();CHKERRCUDA(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*win->map->n);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_SeqCUDAAsync(Vec win,PetscStreamScalar pscal,Vec xin, Vec yin,PetscStream pstream)
{
  PetscBool      isZero;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,5,PETSCSTREAMCUDA);
  ierr = PetscStreamScalarGetInfo(pscal,PSS_ZERO,PETSC_FALSE,&isZero,pstream);CHKERRQ(ierr);
  if (isZero) {
    ierr = VecCopy_SeqCUDAAsync(yin,win,pstream);CHKERRQ(ierr);
  } else {
    const PetscScalar *xarray=NULL,*yarray=NULL;
    PetscScalar       *warray=NULL;
    const PetscScalar *dptr;
    cudaStream_t      cstream;
    cublasHandle_t    cublasv2handle;
    cublasStatus_t    cberr;
    cudaError_t       err;
    PetscBLASInt      one = 1,bn = 0;

    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(win->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWriteAsync(win,&warray,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscal,&dptr,pstream);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    err = cudaMemcpyAsync(warray,yarray,win->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,cstream);CHKERRCUDA(err);
    cberr = cublasXaxpy(cublasv2handle,bn,dptr,xarray,one,warray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*win->map->n);CHKERRQ(ierr);
    ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscal,&dptr,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWriteAsync(win,&warray,pstream);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_SeqCUDA(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,j;
  PetscScalar       *xarray;
  const PetscScalar *yarray;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  ierr = PetscLogGpuFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpu(nv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
  ierr = VecCUDAGetArray(xin,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  for (j=0; j<nv; j++) {
    ierr = VecCUDAGetArrayRead(y[j],&yarray);CHKERRQ(ierr);
    cberr = cublasXaxpy(cublasv2handle,bn,alpha+j,yarray,one,xarray,one);CHKERRCUBLAS(cberr);
    ierr = VecCUDARestoreArrayRead(y[j],&yarray);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_SeqCUDAAsync(Vec xin, PetscInt nv,PetscStreamScalar *pscal,Vec *y,PetscStream pstream)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,j;
  PetscScalar       *xarray;
  const PetscScalar *yarray;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaStream_t      cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,5,PETSCSTREAMCUDA);
  ierr = PetscLogGpuFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpu(nv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  for (j=0; j<nv; j++) {
    const PetscScalar *dptr;

    ierr = VecCUDAGetArrayReadAsync(y[j],&yarray,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscal[j],&dptr,pstream);CHKERRQ(ierr);
    cberr = cublasXaxpy(cublasv2handle,bn,dptr,yarray,one,xarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscal[j],&dptr,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadAsync(y[j],&yarray,pstream);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the second */
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cublasXdot(cublasv2handle,bn,yarray,one,xarray,one,z);CHKERRCUBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (xin->map->n >0) {
    ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuToCpu(sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqCUDAAsync(Vec xin,Vec yin,PetscStreamScalar pscal,PetscStream pstream)
{
  const PetscScalar *xarray,*yarray;
  PetscScalar       *dptr;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaStream_t      cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,4,PETSCSTREAMCUDA);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetDeviceWrite(pscal,&dptr,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the second */
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cublasXdot(cublasv2handle,bn,yarray,one,xarray,one,dptr);CHKERRCUBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  if (xin->map->n >0) {ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);}
  ierr = PetscStreamScalarRestoreDeviceWrite(pscal,&dptr,pstream);CHKERRQ(ierr);
  ierr = PetscLogGpuToCpu(sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//
// CUDA kernels for MDot to follow
//

// set work group size to be a power of 2 (128 is usually a good compromise between portability and speed)
#define MDOT_WORKGROUP_SIZE 128
#define MDOT_WORKGROUP_NUM  128

#if !defined(PETSC_USE_COMPLEX)
// M = 2:
__global__ void VecMDot_SeqCUDA_kernel2(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[2*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
  }
  tmp_buffer[threadIdx.x]                       = group_sum0;
  tmp_buffer[threadIdx.x + MDOT_WORKGROUP_SIZE] = group_sum1;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                      ] += tmp_buffer[threadIdx.x+stride                      ];
      tmp_buffer[threadIdx.x + MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x]             = tmp_buffer[0];
    group_results[blockIdx.x + gridDim.x] = tmp_buffer[MDOT_WORKGROUP_SIZE];
  }
}

// M = 3:
__global__ void VecMDot_SeqCUDA_kernel3(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[3*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
  }
}

// M = 4:
__global__ void VecMDot_SeqCUDA_kernel4(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,const PetscScalar *y3,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[4*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  PetscScalar group_sum3 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
    group_sum3 += entry_x * y3[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;
  tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] = group_sum3;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 3 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 3 * gridDim.x] = tmp_buffer[3 * MDOT_WORKGROUP_SIZE];
  }
}

// M = 8:
__global__ void VecMDot_SeqCUDA_kernel8(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,const PetscScalar *y3,
                                          const PetscScalar *y4,const PetscScalar *y5,const PetscScalar *y6,const PetscScalar *y7,
                                          PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[8*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  PetscScalar group_sum3 = 0;
  PetscScalar group_sum4 = 0;
  PetscScalar group_sum5 = 0;
  PetscScalar group_sum6 = 0;
  PetscScalar group_sum7 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
    group_sum3 += entry_x * y3[i];
    group_sum4 += entry_x * y4[i];
    group_sum5 += entry_x * y5[i];
    group_sum6 += entry_x * y6[i];
    group_sum7 += entry_x * y7[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;
  tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] = group_sum3;
  tmp_buffer[threadIdx.x + 4 * MDOT_WORKGROUP_SIZE] = group_sum4;
  tmp_buffer[threadIdx.x + 5 * MDOT_WORKGROUP_SIZE] = group_sum5;
  tmp_buffer[threadIdx.x + 6 * MDOT_WORKGROUP_SIZE] = group_sum6;
  tmp_buffer[threadIdx.x + 7 * MDOT_WORKGROUP_SIZE] = group_sum7;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 3 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 4 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 4 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 5 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 5 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 6 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 6 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 7 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 7 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 3 * gridDim.x] = tmp_buffer[3 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 4 * gridDim.x] = tmp_buffer[4 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 5 * gridDim.x] = tmp_buffer[5 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 6 * gridDim.x] = tmp_buffer[6 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 7 * gridDim.x] = tmp_buffer[7 * MDOT_WORKGROUP_SIZE];
  }
}
#endif /* !defined(PETSC_USE_COMPLEX) */

PetscErrorCode VecMDot_SeqCUDA(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,current_y_index = 0;
  const PetscScalar *xptr,*y0ptr,*y1ptr,*y2ptr,*y3ptr,*y4ptr,*y5ptr,*y6ptr,*y7ptr;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt          nv1 = ((nv % 4) == 1) ? nv-1: nv,j;
  PetscScalar       *group_results_gpu,*group_results_cpu;
  cudaError_t       cerr;
#endif
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (nv <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUDA not positive.");
  /* Handle the case of local size zero first */
  if (!xin->map->n) {
    ierr = PetscArrayzero(z,nv);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

#if !defined(PETSC_USE_COMPLEX)
  // allocate scratchpad memory for the results of individual work groups:
  cerr = cudaMalloc((void**)&group_results_gpu, nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM);CHKERRCUDA(cerr);
  ierr = PetscMalloc1(nv1*MDOT_WORKGROUP_NUM,&group_results_cpu);CHKERRQ(ierr);
#else
  ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
#endif
  ierr = VecCUDAGetArrayRead(xin,&xptr);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);

  while (current_y_index < nv)
  {
    switch (nv - current_y_index) {

      case 7:
      case 6:
      case 5:
      case 4:
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]);CHKERRCUBLAS(cberr);
#else
        VecMDot_SeqCUDA_kernel4<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        current_y_index += 4;
        break;

      case 3:
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
#else
        VecMDot_SeqCUDA_kernel3<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        current_y_index += 3;
        break;

      case 2:
        ierr = VecCUDAGetArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
#else
        VecMDot_SeqCUDA_kernel2<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        current_y_index += 2;
        break;

      case 1:
        ierr = VecCUDAGetArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        current_y_index += 1;
        break;

      default: // 8 or more vectors left
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+4],&y4ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+5],&y5ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+6],&y6ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+7],&y7ptr);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y4ptr,one,xptr,one,&z[current_y_index+4]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y5ptr,one,xptr,one,&z[current_y_index+5]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y6ptr,one,xptr,one,&z[current_y_index+6]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y7ptr,one,xptr,one,&z[current_y_index+7]);CHKERRCUBLAS(cberr);
#else
        VecMDot_SeqCUDA_kernel8<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,y4ptr,y5ptr,y6ptr,y7ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+4],&y4ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+5],&y5ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+6],&y6ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+7],&y7ptr);CHKERRQ(ierr);
        current_y_index += 8;
        break;
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xptr);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscLogGpuToCpu(nv*sizeof(PetscScalar));CHKERRQ(ierr);
#else
  // copy results to CPU
  cerr = cudaMemcpy(group_results_cpu,group_results_gpu,nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM,cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);

  // sum group results into z
  for (j=0; j<nv1; ++j) {
    z[j] = 0;
    for (PetscInt i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[j] += group_results_cpu[i];
  }
  ierr = PetscLogFlops(nv1*MDOT_WORKGROUP_NUM);CHKERRQ(ierr);
  cerr = cudaFree(group_results_gpu);CHKERRCUDA(cerr);
  ierr = PetscFree(group_results_cpu);CHKERRQ(ierr);
  ierr = PetscLogGpuToCpu(nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM);CHKERRQ(ierr);
#endif
  ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_SeqCUDAAsync(Vec xin,PetscInt nv,const Vec yin[],PetscStreamScalar *pscal, PetscStream pstream)
{
  PetscErrorCode    ierr;
  PetscInt          i,n = xin->map->n,current_y_index = 0;
  const PetscScalar *xptr,*y0ptr,*y1ptr,*y2ptr,*y3ptr,*y4ptr,*y5ptr,*y6ptr,*y7ptr;
  PetscScalar       **dptr;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt          nv1 = ((nv % 4) == 1) ? nv-1: nv,j;
  PetscScalar       *group_results_gpu,group_results_cpu[nv1*MDOT_WORKGROUP_NUM];
  cudaError_t       cerr;
#endif
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaStream_t      cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,5,PETSCSTREAMCUDA);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (nv <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUDA not positive.");
  /* Handle the case of local size zero first */
  if (!xin->map->n) {
    for (i = 0; i < nv; ++i) {
      ierr = PetscStreamScalarSetValue(pscal[i],NULL,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  ierr = PetscMalloc1(nv, &dptr);CHKERRQ(ierr);
  for (i = 0; i < nv; ++i) {
    ierr = PetscStreamScalarGetDeviceWrite(pscal[i],&dptr[i],pstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
#else
  // allocate scratchpad memory for the results of individual work groups:
  cerr = cudaMalloc((void**)&group_results_gpu, nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM);CHKERRCUDA(cerr);
#endif
  ierr = VecCUDAGetArrayReadAsync(xin,&xptr,pstream);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);

  while (current_y_index < nv)
  {
    switch (nv - current_y_index) {

      case 7:
      case 6:
      case 5:
      case 4:
        /* Has __anyone__ ever heard of a for loop around these parts? */
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index  ],&y0ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+1],&y1ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+2],&y2ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+3],&y3ptr,pstream);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,dptr[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,dptr[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,dptr[current_y_index+2]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,dptr[current_y_index+3]);CHKERRCUBLAS(cberr);
#else
        VecMDot_SeqCUDA_kernel4<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE,0,cstream>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index  ],&y0ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+1],&y1ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+2],&y2ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+3],&y3ptr,pstream);CHKERRQ(ierr);
        current_y_index += 4;
        break;

      case 3:
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index  ],&y0ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+1],&y1ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+2],&y2ptr,pstream);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,dptr[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,dptr[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,dptr[current_y_index+2]);CHKERRCUBLAS(cberr);
#else
        VecMDot_SeqCUDA_kernel3<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE,0,cstream>>>(xptr,y0ptr,y1ptr,y2ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index  ],&y0ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+1],&y1ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+2],&y2ptr,pstream);CHKERRQ(ierr);
        current_y_index += 3;
        break;

      case 2:
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index],&y0ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+1],&y1ptr,pstream);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,dptr[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,dptr[current_y_index+1]);CHKERRCUBLAS(cberr);
#else
        VecMDot_SeqCUDA_kernel2<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE,0,cstream>>>(xptr,y0ptr,y1ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index],&y0ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+1],&y1ptr,pstream);CHKERRQ(ierr);
        current_y_index += 2;
        break;

      case 1:
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index],&y0ptr,pstream);CHKERRQ(ierr);
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,dptr[current_y_index]);CHKERRCUBLAS(cberr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index],&y0ptr,pstream);CHKERRQ(ierr);
        current_y_index += 1;
        break;

      default: // 8 or more vectors left
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index  ],&y0ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+1],&y1ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+2],&y2ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+3],&y3ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+4],&y4ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+5],&y5ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+6],&y6ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayReadAsync(yin[current_y_index+7],&y7ptr,pstream);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,dptr[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,dptr[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,dptr[current_y_index+2]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,dptr[current_y_index+3]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y4ptr,one,xptr,one,dptr[current_y_index+4]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y5ptr,one,xptr,one,dptr[current_y_index+5]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y6ptr,one,xptr,one,dptr[current_y_index+6]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y7ptr,one,xptr,one,dptr[current_y_index+7]);CHKERRCUBLAS(cberr);
#else
        VecMDot_SeqCUDA_kernel8<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE,0,cstream>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,y4ptr,y5ptr,y6ptr,y7ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index  ],&y0ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+1],&y1ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+2],&y2ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+3],&y3ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+4],&y4ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+5],&y5ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+6],&y6ptr,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(yin[current_y_index+7],&y7ptr,pstream);CHKERRQ(ierr);
        current_y_index += 8;
        break;
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(xin,&xptr,pstream);CHKERRQ(ierr);
  for (i = 0; i < nv; ++i) {
    ierr = PetscStreamScalarRestoreDeviceWrite(pscal[i],&dptr[i],pstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamRecordEvent(pstream,xin->event);CHKERRQ(ierr);
  ierr = PetscFree(dptr);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscLogGpuToCpu(nv*sizeof(PetscScalar));CHKERRQ(ierr);
#else
  // copy results to CPU, arguably this step should be performed on GPU for maximum pipelineability
  cerr = cudaMemcpyAsync(group_results_cpu,group_results_gpu,nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM,cudaMemcpyDeviceToHost,cstream);CHKERRCUDA(cerr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);

  // sum group results into z, it is safe to access pscal->host memory directly here since
  // we just synchronized
  for (j=0; j<nv1; ++j) {
    *pscal[j]->host = 0;
    for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) *(pscal[j]->host) += group_results_cpu[i];
  }
  /* Tell the scalars their GPU data is out of date */
  for (i = 0; i < nv; ++i) pscal[i]->omask = PETSC_OFFLOAD_CPU;
  ierr = PetscLogFlops(nv1*MDOT_WORKGROUP_NUM);CHKERRQ(ierr);
  cerr = cudaFree(group_results_gpu);CHKERRCUDA(cerr);
  ierr = PetscLogGpuToCpu(nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM);CHKERRQ(ierr);
#endif
  ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_SIZE
#undef MDOT_WORKGROUP_NUM

PetscErrorCode VecSet_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscInt                        n = xin->map->n;
  PetscScalar                     *xarray = NULL;
  thrust::device_ptr<PetscScalar> xptr;
  PetscErrorCode                  ierr;
  cudaError_t                     err;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayWrite(xin,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    err = cudaMemset(xarray,0,n*sizeof(PetscScalar));CHKERRCUDA(err);
  } else {
    try {
      xptr = thrust::device_pointer_cast(xarray);
      thrust::fill(xptr,xptr+n,alpha);
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct PetscScalarReciprocal
{
  __host__ __device__
  PetscScalar operator()(PetscScalar s)
  {
    return (s != (PetscScalar)0.0) ? (PetscScalar)1.0/s : 0.0;
  }
};

PetscErrorCode VecReciprocal_SeqCUDA(Vec v)
{
  cudaError_t    cerr;
  PetscErrorCode ierr;
  PetscInt       n;
  PetscScalar    *x;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = VecCUDAGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto xptr = thrust::device_pointer_cast(x);
    thrust::transform(xptr,xptr+n,xptr,PetscScalarReciprocal());
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

namespace PetscSeqCUDA {
static const PetscInt maxThreadsPerBlock = 512;

  /* Because thrust does not have a fill routine using device pointers */
__global__ __launch_bounds__(PetscSeqCUDA::maxThreadsPerBlock)
void vecSetAsync(PetscScalar *__restrict__ inout, const PetscScalar *__restrict__ vecSetVal, PetscInt size)
{
  const PetscInt    tx  = threadIdx.x, bx = blockIdx.x;
  const PetscInt    bdx = blockDim.x, gdx = PetscSeqCUDA::maxThreadsPerBlock;
  const PetscScalar val = *vecSetVal; /* Hope this becomes a LDG load op */

#pragma unroll
  for (PetscInt i = bx*bdx+tx; i < size; i += gdx*bdx) inout[i] = val;
  return;
}

  /* Round operand up to nearest multiple of multiple, ONLY works if multiple is a power
     of 2!!! */
__host__ __device__
PETSC_STATIC_INLINE PetscInt roundUpToPow2(PetscInt operand, PetscInt multiple)
{
  return (operand+multiple-1) & -multiple;
}
}

PetscErrorCode VecSet_SeqCUDAAsync(Vec xin,PetscStreamScalar pscal,PetscStream pstream)
{
  PetscBool       isZero;
  PetscInt        n       = xin->map->n;
  PetscScalar     *xarray = NULL;
  PetscErrorCode  ierr;
  cudaError_t     cerr;
  cudaStream_t    cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,3,PETSCSTREAMCUDA);
  ierr = VecCUDAGetArrayWriteAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetInfo(pscal,PSS_ZERO,PETSC_FALSE,&isZero,pstream);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (isZero) {
    cerr = cudaMemsetAsync(xarray,0,n*sizeof(PetscScalar),cstream);CHKERRCUDA(cerr);
  } else {
    PetscBool         isOne;
    const PetscScalar *alpha;

    ierr = PetscStreamScalarGetDeviceRead(pscal,&alpha,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetInfo(pscal,PSS_ONE,PETSC_FALSE,&isOne,pstream);CHKERRQ(ierr);
    if (isOne) {
      thrust::device_ptr<PetscScalar> xptr;

      try {
        xptr = thrust::device_pointer_cast(xarray);
        thrust::fill(thrust::cuda::par.on(cstream),xptr,xptr+n,1);
      } catch (char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
      }
    } else {
      const PetscInt nt = std::min(PetscSeqCUDA::roundUpToPow2(n,PetscInt(32)),PetscSeqCUDA::maxThreadsPerBlock);
      const dim3     dimGrid(std::max(nt/PetscSeqCUDA::maxThreadsPerBlock,PetscInt(1))), dimBlock(nt);

      PetscSeqCUDA::vecSetAsync<<<dimGrid,dimBlock,0,cstream>>>(xarray,alpha,n);
    }
    ierr = PetscStreamScalarRestoreDeviceRead(pscal,&alpha,pstream);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscStreamRecordEvent(pstream,xin->event);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWriteAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecSet_SeqCUDA(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != (PetscScalar)1.0) {
    PetscScalar    *xarray;
    PetscBLASInt   one = 1,bn = 0;
    cublasHandle_t cublasv2handle;
    cublasStatus_t cberr;

    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
    ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,&alpha,xarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqCUDAAsync(Vec xin,PetscStreamScalar pscal,PetscStream pstream)
{
  PetscBool      isZero,isOne;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,3,PETSCSTREAMCUDA);
  ierr = PetscStreamScalarGetInfo(pscal,PSS_ZERO,PETSC_FALSE,&isZero,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetInfo(pscal,PSS_ONE,PETSC_FALSE,&isOne,pstream);CHKERRQ(ierr);
  if (isZero) {
    ierr = VecSet_SeqCUDAAsync(xin,pscal,pstream);CHKERRQ(ierr);
  } else if (!isOne) {
    PetscScalar       *xarray;
    const PetscScalar *dptr;
    PetscBLASInt      one = 1,bn = 0;
    cublasHandle_t    cublasv2handle;
    cublasStatus_t    cberr;
    cudaStream_t      cstream;

    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscal,&dptr,pstream);CHKERRQ(ierr);
    ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,dptr,xarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscal,&dptr,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cublasXdotu(cublasv2handle,bn,xarray,one,yarray,one,z);CHKERRCUBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuToCpu(sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqCUDAAsync(Vec xin,Vec yin,PetscStreamScalar pscal,PetscStream pstream)
{
  const PetscScalar *xarray,*yarray;
  PetscScalar       *dptr;
  PetscErrorCode    ierr;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaStream_t      cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,4,PETSCSTREAMCUDA);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetDeviceWrite(pscal,&dptr,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cublasXdotu(cublasv2handle,bn,xarray,one,yarray,one,dptr);CHKERRCUBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuToCpu(sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscStreamScalarRestoreDeviceWrite(pscal,&dptr,pstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqCUDA(Vec xin,Vec yin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  cudaError_t       err;

  PetscFunctionBegin;
  if (xin != yin) {
    if (xin->offloadmask == PETSC_OFFLOAD_GPU) {
      PetscBool    yiscuda;

      ierr = PetscObjectTypeCompareAny((PetscObject)yin,&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
      ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
      if (yiscuda) {
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
        ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      }
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      if (yiscuda) {
        err = cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream);CHKERRCUDA(err);
      } else {
        err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
      if (yiscuda) {
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
        ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      }
    } else if (xin->offloadmask == PETSC_OFFLOAD_CPU) {
      /* copy in CPU if we are on the CPU */
      ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
    } else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->offloadmask == PETSC_OFFLOAD_CPU) {
        /* copy in CPU */
        ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
      } else if (yin->offloadmask == PETSC_OFFLOAD_GPU) {
        /* copy in GPU */
        ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        err  = cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream);CHKERRCUDA(err);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else if (yin->offloadmask == PETSC_OFFLOAD_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        err  = cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream);CHKERRCUDA(err);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
        ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqCUDAAsync(Vec xin,Vec yin,PetscStream pstream)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  cudaError_t       err;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,3,PETSCSTREAMCUDA);
  if (xin != yin) {
    cudaStream_t cstream;

    if (xin->offloadmask == PETSC_OFFLOAD_GPU) {
      PetscBool yiscuda;

      ierr = PetscObjectTypeCompareAny((PetscObject)yin,&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
      ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
      if (yiscuda) {
        ierr = VecCUDAGetArrayWriteAsync(yin,&yarray,pstream);CHKERRQ(ierr);
      } else {
        ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      }
      ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      if (yiscuda) {
        err = cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,cstream);CHKERRCUDA(err);
      } else {
        err = cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost,cstream);CHKERRCUDA(err);
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
      ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
      if (yiscuda) {
        ierr = VecCUDARestoreArrayWriteAsync(yin,&yarray,pstream);CHKERRQ(ierr);
      } else {
        ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      }
    } else if (xin->offloadmask == PETSC_OFFLOAD_CPU) {
      /* copy in CPU if we are on the CPU, note using synchronous version, so must
         synchronize the stream first */
      ierr = PetscEventSynchronize(xin->event);CHKERRQ(ierr);
      /* dont know if yin is cuda */
      if (yin->event) {ierr = PetscEventSynchronize(yin->event);CHKERRQ(ierr);}
      ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
    } else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->offloadmask == PETSC_OFFLOAD_CPU) {
        /* copy in CPU, again must sync first */
        ierr = PetscEventSynchronize(xin->event);CHKERRQ(ierr);
        /* don't know if yin is cuda */
        if (yin->event) {ierr = PetscEventSynchronize(yin->event);CHKERRQ(ierr);}
        ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
      } else if (yin->offloadmask == PETSC_OFFLOAD_GPU) {
        /* copy in GPU */
        ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWriteAsync(yin,&yarray,pstream);CHKERRQ(ierr);
        ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        err  = cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,cstream);CHKERRCUDA(err);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWriteAsync(yin,&yarray,pstream);CHKERRQ(ierr);
      } else if (yin->offloadmask == PETSC_OFFLOAD_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWriteAsync(yin,&yarray,pstream);CHKERRQ(ierr);
        ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        err  = cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,cstream);CHKERRCUDA(err);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWriteAsync(yin,&yarray,pstream);CHKERRQ(ierr);
      } else {
        ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_SeqCUDA(Vec xin,Vec yin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    PetscBLASInt   one = 1,bn = 0;
    PetscScalar    *xarray,*yarray;
    cublasHandle_t cublasv2handle;
    cublasStatus_t cberr;

    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXswap(cublasv2handle,bn,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_SeqCUDAAsync(Vec xin,Vec yin,PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,3,PETSCSTREAMCUDA);
  if (xin != yin) {
    PetscBLASInt   one = 1,bn = 0;
    PetscScalar    *xarray,*yarray;
    cublasHandle_t cublasv2handle;
    cublasStatus_t cberr;
    cudaStream_t   cstream;

    ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
    ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXswap(cublasv2handle,bn,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_SeqCUDA(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  PetscScalar       a = alpha,b = beta;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1, bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  if (a == (PetscScalar)0.0) {
    ierr = VecScale_SeqCUDA(yin,beta);CHKERRQ(ierr);
  } else if (b == (PetscScalar)1.0) {
    ierr = VecAXPY_SeqCUDA(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == (PetscScalar)1.0) {
    ierr = VecAYPX_SeqCUDA(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == (PetscScalar)0.0) {
    ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
    cberr = cublasXscal(cublasv2handle,bn,&alpha,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
  } else {
    ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,&beta,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(3.0*xin->map->n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(2*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_SeqCUDAAsync(Vec yin,PetscStreamScalar pscala,PetscStreamScalar pscalb,Vec xin,PetscStream pstream)
{
  PetscErrorCode    ierr;
  PetscBool         isAZero,isAOne,isBZero,isBOne;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1, bn = 0;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,5,PETSCSTREAMCUDA);
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetInfo(pscala,PSS_ZERO,PETSC_FALSE,&isAZero,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetInfo(pscala,PSS_ONE,PETSC_FALSE,&isAOne,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetInfo(pscalb,PSS_ZERO,PETSC_FALSE,&isBZero,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetInfo(pscalb,PSS_ONE,PETSC_FALSE,&isBOne,pstream);CHKERRQ(ierr);
  if (isAZero) {
    ierr = VecScale_SeqCUDAAsync(yin,pscalb,pstream);CHKERRQ(ierr);
  } else if (isBOne) {
    ierr = VecAXPY_SeqCUDAAsync(yin,pscala,xin,pstream);CHKERRQ(ierr);
  } else if (isAOne) {
    ierr = VecAYPX_SeqCUDAAsync(yin,pscalb,xin,pstream);CHKERRQ(ierr);
  } else if (isBZero) {
    const PetscScalar *dptr;
    cudaStream_t      cstream;
    cudaError_t       cerr;

    ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscala,&dptr,pstream);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,cstream);CHKERRCUDA(cerr);
    cberr = cublasXscal(cublasv2handle,bn,dptr,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscala,&dptr,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  } else {
    const PetscScalar *dptra, *dptrb;
    cudaStream_t      cstream;

    ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscala,&dptra,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetDeviceRead(pscalb,&dptrb,pstream);CHKERRQ(ierr);
    ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,dptrb,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,dptra,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscala,&dptra,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarRestoreDeviceRead(pscalb,&dptrb,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayAsync(yin,&yarray,pstream);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(3.0*xin->map->n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(2*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_SeqCUDA(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = zin->map->n;

  PetscFunctionBegin;
  if (gamma == (PetscScalar)1.0) {
    /* z = ax + b*y + z */
    ierr = VecAXPY_SeqCUDA(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(4.0*n);CHKERRQ(ierr);
  } else {
    /* z = a*x + b*y + c*z */
    ierr = VecScale_SeqCUDA(zin,gamma);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(5.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_SeqCUDAAsync(Vec zin,PetscStreamScalar pscala,PetscStreamScalar pscalb,PetscStreamScalar pscalg,Vec xin,Vec yin,PetscStream pstream)
{
  PetscErrorCode ierr;
  PetscBool      isGOne;
  PetscInt       n = zin->map->n;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,7,PETSCSTREAMCUDA);
  ierr = PetscStreamScalarGetInfo(pscalg,PSS_ONE,PETSC_FALSE,&isGOne,pstream);CHKERRQ(ierr);
  if (isGOne) {
    /* z = ax + b*y + z */
    ierr = VecAXPY_SeqCUDAAsync(zin,pscala,xin,pstream);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDAAsync(zin,pscalb,yin,pstream);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(4.0*n);CHKERRQ(ierr);
  } else {
    /* z = a*x + b*y + c*z */
    ierr = VecScale_SeqCUDAAsync(zin,pscalg,pstream);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDAAsync(zin,pscala,xin,pstream);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDAAsync(zin,pscalb,yin,pstream);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(5.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqCUDA(Vec win,Vec xin,Vec yin)
{
  PetscInt                              n = win->map->n;
  const PetscScalar                     *xarray,*yarray;
  PetscScalar                           *warray;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  ierr = VecCUDAGetArray(win,&warray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(xptr,xptr+n,yptr,wptr,thrust::multiplies<PetscScalar>());
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqCUDAAsync(Vec win,Vec xin,Vec yin,PetscStream pstream)
{
  PetscInt                              n = win->map->n;
  const PetscScalar                     *xarray,*yarray;
  PetscScalar                           *warray;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;
  PetscErrorCode                        ierr;
  cudaError_t                           err;
  cudaStream_t                          cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,4,PETSCSTREAMCUDA);
  ierr = VecCUDAGetArrayAsync(win,&warray,pstream);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(thrust::cuda::par.on(cstream),xptr,xptr+n,yptr,wptr,thrust::multiplies<PetscScalar>());
    err = cudaGetLastError();CHKERRCUDA(err);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadAsync(yin,&yarray,pstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayAsync(win,&warray,pstream);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* should do infinity norm in cuda */

PetscErrorCode VecNorm_SeqCUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn = 0;
  const PetscScalar *xarray;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = PetscCUBLASSetHost_Internal(cublasv2handle);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXnrm2(cublasv2handle,bn,xarray,one,z);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    int  i;
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasIXamax(cublasv2handle,bn,xarray,one,&i);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    if (bn) {
      PetscScalar zs;
      err = cudaMemcpy(&zs,xarray+i-1,sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
      *z = PetscAbsScalar(zs);
    } else *z = 0.0;
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXasum(cublasv2handle,bn,xarray,one,z);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_SeqCUDA(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUDA(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  ierr = PetscLogGpuToCpu(sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

namespace PetscSeqCUDA {
struct abs {
  __device__ __forceinline__
  PetscScalar operator()(PetscScalar x)
  {
#if defined(PETSC_USE_COMPLEX)
    return thrust::abs(x);
#else
    return fabs(x);
#endif
  }
};
}

PetscErrorCode VecNorm_SeqCUDAAsync(Vec xin,NormType type,PetscStreamScalar *pscal,PetscStream pstream)
{
  PetscErrorCode    ierr;
  PetscBool         restore = PETSC_TRUE;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn = 0;
  const PetscScalar *xarray;
  PetscScalar       *dptr;
  cublasHandle_t    cublasv2handle;
  cublasStatus_t    cberr;
  cudaStream_t      cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,4,PETSCSTREAMCUDA);
  if (type == NORM_1_AND_2) {
    ierr = VecNorm_SeqCUDAAsync(xin,NORM_1,pscal,pstream);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUDAAsync(xin,NORM_2,pscal+1,pstream);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscCUBLASGetHandle(&cublasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarGetDeviceWrite(*pscal,&dptr,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscCUBLASSetStream_Internal(cublasv2handle,cstream);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    /*
      reinterpret_cast() ~seems~ to be safe to do here. This is supported behavior in
      std::complex, and as per thrusts documentation: "[thrust::complex] is functionally
      identical to [std::complex], but [thrust::complex] can also be used in device code
      which [std::complex] currently cannot."

      Whether or not "functionally identical" includes compatibility with C's complex
      types is operative here though.
    */
    cberr = cublasXnrm2(cublasv2handle,bn,xarray,one,reinterpret_cast<PetscReal*>(dptr));CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    int  i;
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasIXamax(cublasv2handle,bn,xarray,one,&i);CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    if (bn) {
      try {
        auto zptr = thrust::device_pointer_cast(xarray+i-1);

        thrust::transform(thrust::cuda::par.on(cstream),zptr,zptr+1,dptr,PetscSeqCUDA::abs());
      } catch (char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
      }
    } else {
      ierr = PetscStreamScalarRestoreDeviceWrite(*pscal,&dptr,pstream);CHKERRQ(ierr);
      ierr = PetscStreamScalarSetValue(*pscal,NULL,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
      restore = PETSC_FALSE;
    }
  } else if (type == NORM_1) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cublasXasum(cublasv2handle,bn,xarray,one,reinterpret_cast<PetscReal*>(dptr));CHKERRCUBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  }
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  if (restore) {ierr = PetscStreamScalarRestoreDeviceWrite(*pscal,&dptr,pstream);CHKERRQ(ierr);}
  ierr = VecCUDARestoreArrayReadAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = PetscLogGpuToCpu(sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_SeqCUDA(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqCUDA(s,t,dp);CHKERRQ(ierr);
  ierr = VecDot_SeqCUDA(t,t,nm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_SeqCUDAAsync(Vec s, Vec t, PetscStreamScalar pscaldp, PetscStreamScalar pscalnm, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,5,PETSCSTREAMCUDA);
  ierr = VecDot_SeqCUDAAsync(s,t,pscaldp,pstream);CHKERRQ(ierr);
  ierr = VecDot_SeqCUDAAsync(t,t,pscalnm,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUDA(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  Vec_CUDA       *veccuda = (Vec_CUDA*)v->spptr;

  PetscFunctionBegin;
  if (v->spptr) {
    if (veccuda->GPUarray_allocated) {
     #if defined(PETSC_HAVE_NVSHMEM)
      if (veccuda->nvshmem) {
        ierr = PetscNvshmemFree(veccuda->GPUarray_allocated);CHKERRQ(ierr);
        veccuda->nvshmem = PETSC_FALSE;
      }
      else
     #endif
      {cerr = cudaFree(veccuda->GPUarray_allocated);CHKERRCUDA(cerr);}
      veccuda->GPUarray_allocated = NULL;
    }
    if (veccuda->stream) {
      cerr = cudaStreamDestroy(veccuda->stream);CHKERRCUDA(cerr);
    }
  }
  ierr = VecDestroy_SeqCUDA_Private(v);CHKERRQ(ierr);
  ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
struct conjugate
{
  __host__ __device__
    PetscScalar operator()(PetscScalar x)
    {
      return PetscConj(x);
    }
};
#endif

PetscErrorCode VecConjugate_SeqCUDA(Vec xin)
{
#if defined(PETSC_USE_COMPLEX)
  PetscScalar                     *xarray;
  PetscErrorCode                  ierr;
  PetscInt                        n = xin->map->n;
  thrust::device_ptr<PetscScalar> xptr;
  cudaError_t                     err;

  PetscFunctionBegin;
  ierr = VecCUDAGetArray(xin,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    xptr = thrust::device_pointer_cast(xarray);
    thrust::transform(xptr,xptr+n,xptr,conjugate());
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(xin,&xarray);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecConjugate_SeqCUDAAsync(Vec xin,PetscStream pstream)
{
#if defined(PETSC_USE_COMPLEX)
  PetscScalar                     *xarray;
  PetscErrorCode                  ierr;
  PetscInt                        n = xin->map->n;
  thrust::device_ptr<PetscScalar> xptr;
  cudaStream_t                    cstream;

  PetscFunctionBegin;
  PetscValidStreamTypeSpecific(pstream,2,PETSCSTREAMCUDA);
  ierr = VecCUDAGetArrayAsync(xin,&xarray,pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    xptr = thrust::device_pointer_cast(xarray);
    thrust::transform(thrust::cuda::par.on(cstream),xptr,xptr+n,xptr,conjugate());
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscStreamRestoreStream(pstream,&cstream);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayAsync(xin,&xarray,pstream);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeName(w,VECSEQCUDA);
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  if (w->event) {ierr = PetscEventSynchronize(w->event);CHKERRQ(ierr);}
  if (w->data) {
    if (((Vec_Seq*)w->data)->array_allocated) {
      if (w->pinned_memory) {
        ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
      }
      ierr = PetscFree(((Vec_Seq*)w->data)->array_allocated);CHKERRQ(ierr);
      if (w->pinned_memory) {
        ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
        w->pinned_memory = PETSC_FALSE;
      }
    }
    ((Vec_Seq*)w->data)->array = NULL;
    ((Vec_Seq*)w->data)->unplacedarray = NULL;
  }
  if (w->spptr) {
    if (((Vec_CUDA*)w->spptr)->GPUarray) {
      err = cudaFree(((Vec_CUDA*)w->spptr)->GPUarray);CHKERRCUDA(err);
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
    }
    ierr = PetscFree(w->spptr);CHKERRQ(ierr);
  }

  ierr = PetscEventSynchronize(v->event);CHKERRQ(ierr);
  if (v->petscnative) {
    ierr = PetscFree(w->data);CHKERRQ(ierr);
    w->data = v->data;
    w->offloadmask = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr = v->spptr;
    ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(v,&((Vec_Seq*)w->data)->array);CHKERRQ(ierr);
    w->offloadmask = PETSC_OFFLOAD_CPU;
    ierr = VecCUDAAllocateCheck(w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscCheckTypeName(w,VECSEQCUDA);
  if (v->petscnative) {
    v->data = w->data;
    v->offloadmask = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr = w->spptr;
    w->data = 0;
    w->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
    w->spptr = 0;
  } else {
    ierr = VecRestoreArray(v,&((Vec_Seq*)w->data)->array);CHKERRQ(ierr);
    if ((Vec_CUDA*)w->spptr) {
      err = cudaFree(((Vec_CUDA*)w->spptr)->GPUarray);CHKERRCUDA(err);
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
      ierr = PetscFree(w->spptr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

struct petscrealpart : public thrust::unary_function<PetscScalar,PetscReal>
{
  __host__ __device__
  PetscReal operator()(PetscScalar x) {
    return PetscRealPart(x);
  }
};

struct petscrealparti : public thrust::unary_function<thrust::tuple<PetscScalar, PetscInt>,thrust::tuple<PetscReal, PetscInt>>
{
  __host__ __device__
  thrust::tuple<PetscReal, PetscInt> operator()(thrust::tuple<PetscScalar, PetscInt> x) {
    return thrust::make_tuple(PetscRealPart(x.get<0>()), x.get<1>());
  }
};

struct petscmax : public thrust::binary_function<PetscReal,PetscReal,PetscReal>
{
  __host__ __device__
  PetscReal operator()(PetscReal x, PetscReal y) {
    return x < y ? y : x;
  }
};

struct petscmaxi : public thrust::binary_function<thrust::tuple<PetscReal, PetscInt>,thrust::tuple<PetscReal, PetscInt>,thrust::tuple<PetscReal, PetscInt>>
{
  __host__ __device__
  thrust::tuple<PetscReal, PetscInt> operator()(thrust::tuple<PetscReal, PetscInt> x, thrust::tuple<PetscReal, PetscInt> y) {
    return x.get<0>() < y.get<0>() ? thrust::make_tuple(y.get<0>(), y.get<1>()) :
           (x.get<0>() != y.get<0>() ? thrust::make_tuple(x.get<0>(), x.get<1>()) :
           (x.get<1>() < y.get<1>() ? thrust::make_tuple(x.get<0>(), x.get<1>()) : thrust::make_tuple(y.get<0>(), y.get<1>())));
  }
};

struct petscmin : public thrust::binary_function<PetscReal,PetscReal,PetscReal>
{
  __host__ __device__
  PetscReal operator()(PetscReal x, PetscReal y) {
    return x < y ? x : y;
  }
};

struct petscmini : public thrust::binary_function<thrust::tuple<PetscReal, PetscInt>,thrust::tuple<PetscReal, PetscInt>,thrust::tuple<PetscReal, PetscInt>>
{
  __host__ __device__
  thrust::tuple<PetscReal, PetscInt> operator()(thrust::tuple<PetscReal, PetscInt> x, thrust::tuple<PetscReal, PetscInt> y) {
    return x.get<0>() > y.get<0>() ? thrust::make_tuple(y.get<0>(), y.get<1>()) :
           (x.get<0>() != y.get<0>() ? thrust::make_tuple(x.get<0>(), x.get<1>()) :
           (x.get<1>() < y.get<1>() ? thrust::make_tuple(x.get<0>(), x.get<1>()) : thrust::make_tuple(y.get<0>(), y.get<1>())));
  }
};

PetscErrorCode VecMax_SeqCUDA(Vec v, PetscInt *p, PetscReal *m)
{
  PetscErrorCode                        ierr;
  PetscInt                              n = v->map->n;
  const PetscScalar                     *av;
  thrust::device_ptr<const PetscScalar> avpt;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  if (!n) {
    *m = PETSC_MIN_REAL;
    if (p) *p = -1;
    PetscFunctionReturn(0);
  }
  ierr = VecCUDAGetArrayRead(v,&av);CHKERRQ(ierr);
  avpt = thrust::device_pointer_cast(av);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (p) {
    thrust::tuple<PetscReal,PetscInt> res(PETSC_MIN_REAL,-1);
    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(avpt,thrust::counting_iterator<PetscInt>(0)));
    try {
#if defined(PETSC_USE_COMPLEX)
      res = thrust::transform_reduce(zibit,zibit+n,petscrealparti(),res,petscmaxi());
#else
      res = thrust::reduce(zibit,zibit+n,res,petscmaxi());
#endif
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
    *m = res.get<0>();
    *p = res.get<1>();
  } else {
    try {
#if defined(PETSC_USE_COMPLEX)
      *m = thrust::transform_reduce(avpt,avpt+n,petscrealpart(),PETSC_MIN_REAL,petscmax());
#else
      *m = thrust::reduce(avpt,avpt+n,PETSC_MIN_REAL,petscmax());
#endif
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(v,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_SeqCUDA(Vec v, PetscInt *p, PetscReal *m)
{
  PetscErrorCode                        ierr;
  PetscInt                              n = v->map->n;
  const PetscScalar                     *av;
  thrust::device_ptr<const PetscScalar> avpt;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  if (!n) {
    *m = PETSC_MAX_REAL;
    if (p) *p = -1;
    PetscFunctionReturn(0);
  }
  ierr = VecCUDAGetArrayRead(v,&av);CHKERRQ(ierr);
  avpt = thrust::device_pointer_cast(av);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (p) {
    thrust::tuple<PetscReal,PetscInt> res(PETSC_MAX_REAL,-1);
    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(avpt,thrust::counting_iterator<PetscInt>(0)));
    try {
#if defined(PETSC_USE_COMPLEX)
      res = thrust::transform_reduce(zibit,zibit+n,petscrealparti(),res,petscmini());
#else
      res = thrust::reduce(zibit,zibit+n,res,petscmini());
#endif
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
    *m = res.get<0>();
    *p = res.get<1>();
  } else {
    try {
#if defined(PETSC_USE_COMPLEX)
      *m = thrust::transform_reduce(avpt,avpt+n,petscrealpart(),PETSC_MAX_REAL,petscmin());
#else
      *m = thrust::reduce(avpt,avpt+n,PETSC_MAX_REAL,petscmin());
#endif
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(v,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#if defined(PETSC_HAVE_NVSHMEM)
/* Free old CUDA array and re-allocate a new one from nvshmem symmetric heap.
   New array does not retain values in the old array. The offload mask is not changed.

   Note: the function is only meant to be used in MatAssemblyEnd_MPIAIJCUSPARSE.
 */
PetscErrorCode  VecAllocateNVSHMEM_SeqCUDA(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  Vec_CUDA       *veccuda = (Vec_CUDA*)v->spptr;
  PetscInt       n;

  PetscFunctionBegin;
  cerr = cudaFree(veccuda->GPUarray_allocated);CHKERRCUDA(cerr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&n,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  ierr = PetscNvshmemMalloc(n*sizeof(PetscScalar),(void**)&veccuda->GPUarray_allocated);CHKERRQ(ierr);
  veccuda->GPUarray = veccuda->GPUarray_allocated;
  veccuda->nvshmem  = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#endif
