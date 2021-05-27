#include "contextcuda.hpp" /*I "petscdevice.h" I*/

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextCreateCUBLASHandle_Internal(cublasHandle_t *handle)
{
  cublasStatus_t cberr;

  PetscFunctionBegin;
  for (int i = 0; i < 3; ++i) {
    cberr = cublasCreate(handle);
    if (cberr == CUBLAS_STATUS_SUCCESS) break;
    if (cberr != CUBLAS_STATUS_ALLOC_FAILED && cberr != CUBLAS_STATUS_NOT_INITIALIZED) CHKERRCUBLAS(cberr);
    if (i < 2) {
      PetscErrorCode ierr;
      ierr = PetscSleep(3);CHKERRQ(ierr);
    }
  }
  if (cberr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS");
  PetscFunctionReturn(0);
}

/*
  TODO: the pool can be dynamically managed, however this requires some sort of pool
  manager, since you can't realloc the array of context pointers unless __ALL__ are
  checked back in.

  One idea I had would be to have a set of "linked arrays" which allocate n contexts at a
  time but have a "next" member such that they can act as linked lists. Once all members
  of a particular link in the chain are checked back in, they can merge previous links in.
*/
static const PetscInt     numHandles = 64;
static PetscBool          cublasHandleOwned[numHandles],cusolverHandleOwned[numHandles];
static cublasHandle_t     cublasV2HandleList[numHandles];
static cusolverDnHandle_t cusolverDnHandlesList[numHandles];

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetCUBLASHandle_Internal(cublasHandle_t *handle, PetscInt *handleId)
{
  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  PetscValidIntPointer(handleId,2);
  if (PetscUnlikelyDebug(*handleId != PETSC_DEFAULT || *handle)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Trying to get cuBLAS handle for object which seeming has already checked out a handle with ID %D",*handleId);
  *handle   = NULL;
  *handleId = PETSC_DEFAULT;
  for (PetscInt i = 0; i < numHandles; ++i) {
    if (!cublasHandledOwned[i]) {
      /* handle isn't currently owned by anyone, first check it exists though */
      if (!cublasV2Handlelist[i]) {
	PetscErrorCode ierr;
        /* doesn't exist, initialize it */
        ierr = PetscDeviceContextCreateCUBLASHandle_Internal(cublasV2Handlelist+i);CHKERRQ(ierr);
      }
      /* pass the handle back and log the fact that it's checked out */
      *handle   = cublasV2Handlelist[i];
      *handleId = i;
      cublasHandledOwned[i] = PETSC_TRUE;
      break;
    }
  }
  /* can also make this command line option */
  if (PetscUnlikelyDebug(*handleId == PETSC_DEFAULT)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MEM,"Cannot create more than %D cuBLAS handles",numHandles);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextRestoreCUBLASHandle_Internal(cublasHandle_t *handle, PetscInt *handleId)
{
  PetscFunctionBegin;
  PetscValidPointer(handle,1);
  PetscValidIntPointer(handleId,2);
  if (PetscUnlikelyDebug(*handleId >= numHandles)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Returned handle with ID %D > max ID %D",*handleId,numHandles-1);
  if (PetscUnlikelyDebug(!cublasHandleOwned[*handleId])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Restored handle with ID %D has seemingly already been restored",*handleId);
  if (PetscUnlikelyDebug(handle != cublasV2HandleList[*handleId])) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore the same cuBLAS handle as was checked out in PetscDeviceContextSetUp_CUDA()");
  cublasHandleOwned[*handleId] = PETSC_FALSE;
  *handleId = PETSC_DEFAULT;
  *handle   = NULL;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroy_CCUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  PetscErrorCode          ierr;
  cudaError_t             cerr;

  PetscFunctionBegin;
  if (dcu->stream) {cerr = cudaStreamDestroy(dcu->stream);CHKERRCUDA(cerr);}
  if (dcu->event)  {cerr = cudaEventDestroy(dcu->event);CHKERRCUDA(cerr);}
  if (dcu->cublasv2handle) {
    ierr = PetscDeviceContextRestoreCUBLASHandle_Internal(&dcu->cublasv2handle,&dcu->blasHandleId);CHKERRQ(ierr);
  }
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextSetUp_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  cudaError_t             cerr;

  PetscFunctionBegin;
  cerr = cudaStreamCreate(&dcu->stream);CHKERRCUDA(cerr);
  cerr = cudaEventCreate(&dcu->event);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetStream_CUDA(PetscDeviceContext dctx, void *dstrm)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  *((cudaStream_t *)dstrm) = dcu->stream;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextRestoreStream_CUDA(PetscDeviceContext dctx, void *dstrm)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(*((cudaStream_t *)dstrm) != dcu->stream)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"CUDA stream is not the same as the one that was checked out");
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetBLASHandle_CUDA(PetscDeviceContext dctx, void *handle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  if (!dcu->cublasv2handle) {
    cublasStatus_t cberr;
    cudaStream_t   blasStream;
    PetscErrorCode ierr;

    ierr = PetscDeviceContextGetCUBLASHandle_Internal(&dcu->cublasv2handle,&dcu->blasHandleId);CHKERRQ(ierr);
    cberr = cublasGetStream(dcu->cublasv2handle,&blasStream);CHKERRCUBLAS(cberr);
    /* do this check since cublasSetStream UNCONDITIONALLY clears the workspace on
       setStream, something we want to avoid */
    if (blasStream != dcu->stream) {
      cberr = cublasSetStream(dcu->cublasv2handle,dcu->stream);CHKERRCUBLAS(cberr);
    }
  }
  *((cublasHandle_t *)handle) = dcu->cublasv2handle;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextRestoreBLASHandle_CUDA(PetscDeviceContext dctxx, void *handle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(*((cublasHandle_t *)handle) != dcu->cublasv2handle)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"cuBLAS handle is not the same as the one that was checked out");
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextQuery_CUDA(PetscDeviceContext dctx, PetscBool *idle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  *idle = cudaStreamQuery(dcu->stream) == cudaErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

static cudaEvent_t waitEvent   = NULL;
static PetscBool   waitCreated = PETSC_FALSE;

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyWaitEvent_CUDA_Internal(void)
{
  cudaError_t cerr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!waitCreated)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContextDestroyWaitEvent_CUDA_Internal() has been called (or registered at PetscFinalize()) but the wait event was never created");
  cerr = cudaEventDestroy(waitEvent);CHKERRCUDA(cerr);
  waitEvent   = NULL;
  waitCreated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextWaitForContext_CUDA(PetscDeviceContext dctxa, PetscDeviceContext dctxb)
{
  PetscDeviceContext_CUDA *dcua = (PetscDeviceContext_CUDA *)dctxa->data;
  PetscDeviceContext_CUDA *dcub = (PetscDeviceContext_CUDA *)dctxb->data;
  cudaError_t             cerr;

  PetscFunctionBegin;
  if (!waitCreated) {
    PetscErrorCode ierr;

    cerr = cudaEventCreateWithFlags(&waitEvent,cudaEventDisableTiming);CHKERRCUDA(cerr);
    ierr = PetscRegisterFinalize(PetscDeviceContextDestroyWaitEvent_CUDA_Internal);CHKERRQ(ierr);
    waitCreated = PETSC_TRUE;
  }
  cerr = cudaEventRecord(waitEvent,dcub->stream);CHKERRCUDA(cerr);
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11011) /* 11.1.1 */
  cerr = cudaStreamWaitEvent(dcua->stream,waitEvent,cudaEventWaitDefault);CHKERRCUDA(cerr);
#else
  cerr = cudaStreamWaitEvent(dcua->stream,waitEvent,0);CHKERRCUDA(cerr);
#endif /* 11.1.1 */
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextJoin_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  cudaError_t             cerr;

  PetscFunctionBegin;
  /* in case anything was queued on the event */
  cerr = cudaStreamWaitEvent(dcu->stream,dcu->event);CHKERRCUDA(cerr);
  cerr = cudaStreamSynchronize(dcu->stream);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static const struct _DeviceOps cupos = {
  PetscDeviceContextCreate_CUDA,
  PetscDeviceContextDestroy_CUDA,
  PetscDeviceContextSetUp_CUDA,
  PetscDeviceContextGetStream_CUDA,
  PetscDeviceContextRestoreStream_CUDA,
  PetscDeviceContextGetBLASHandle_CUDA,
  PetscDeviceContextRestoreBLASHandle_CUDA,
  PetscDeviceContextQuery_CUDA,
  PetscDeviceContextWaitForContext_CUDA,
  PetscDeviceContextJoin_CUDA,
};

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dcu);CHKERRQ(ierr);
  dcu->blasHandleId   = PETSC_DEFAULT;
  dcu->solverHandleId = PETSC_DEFAULT;
  dctx->data = (void *)dcu;
  ierr = PetscMemcpy(dctx->ops,&cuops,sizeof(cuops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
