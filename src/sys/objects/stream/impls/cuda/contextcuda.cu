#include "contextcuda.hpp" /*I "petscdevice.h" I*/

static cublasHandle_t     cublasv2handle   = NULL;
static cusolverDnHandle_t cusolverdnhandle = NULL;

/* cublas */
static PetscErrorCode PetscCUBLASDestroyHandle_Internal(void)
{
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (cublasv2handle) {
    cberr          = cublasDestroy(cublasv2handle);CHKERRCUBLAS(cberr);
    cublasv2handle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

/* cusolver */
static PetscErrorCode PetscCUSOLVERDnDestroyHandle_Internal(void)
{
  cusolverStatus_t cerr;

  PetscFunctionBegin;
  if (cusolverdnhandle) {
    cerr             = cusolverDnDestroy(cusolverdnhandle);CHKERRCUSOLVER(cerr);
    cusolverdnhandle = NULL;  /* Ensures proper reinitialization */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCUBLASGetHandle_Internal(PetscDeviceContext_CUDA &dcu)
{
  cudaStream_t   cublasStream;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (!cublasv2handle) {
    PetscErrorCode ierr;

    for (int i=0; i<3; i++) {
      cberr = cublasCreate(&cublasv2handle);
      if (cberr == CUBLAS_STATUS_SUCCESS) break;
      if (cberr != CUBLAS_STATUS_ALLOC_FAILED && cberr != CUBLAS_STATUS_NOT_INITIALIZED) CHKERRCUBLAS(cberr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (cberr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS");
    /* Make sure that the handle will be destroyed properly */
    ierr = PetscRegisterFinalize(PetscCUBLASDestroyHandle_Internal);CHKERRQ(ierr);
  }
  cberr = cublasGetStream(cublasv2handle,&cublasStream);CHKERRCUBLAS(cberr);
  if (cublasStream != dcu->stream) {
    cberr = cublasSetStream(cublasv2handle,dcu->stream);CHKERRCUBLAS(cberr);
  }
  dcu->blas = cublasv2handle;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCUSOLVERDnGetHandle_Internal(PetscDeviceContext_CUDA &dcu)
{
  cudaStream_t     cusolverStream;
  cusolverStatus_t cerr;

  PetscFunctionBegin;
  if (!cusolverdnhandle) {
    PetscErrorCode ierr;

    for (int i=0; i<3; i++) {
      cerr = cusolverDnCreate(&cusolverdnhandle);
      if (cerr == CUSOLVER_STATUS_SUCCESS) break;
      if (cerr != CUSOLVER_STATUS_ALLOC_FAILED) CHKERRCUSOLVER(cerr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (cerr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuSolverDn");
    ierr = PetscRegisterFinalize(PetscCUSOLVERDnDestroyHandle_Internal);CHKERRQ(ierr);
  }
  cerr = cusolverDnGetStream(cusolverdnhandle,&cusolverStream);CHKERRCUSOLVER(cerr);
  if (cusolverStream != dcu->stream) {
    cerr = cusolverDnSetStream(cusolverdnhandle,dcu->stream);CHKERRCUSOLVER(cerr);
  }
  dcu->solver = cusolverdnhandle;
  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroy_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  PetscErrorCode          ierr;
  cudaError_t             cerr;

  PetscFunctionBegin;
  if (dcu->stream) {cerr = cudaStreamDestroy(dcu->stream);CHKERRCUDA(cerr);}
  if (dcu->event)  {cerr = cudaEventDestroy(dcu->event);CHKERRCUDA(cerr);}
  /* don't need to do anything to the handles, they live on without us */
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* don't also create solver contexts here since they aren't always used, and given the limited real estate frugality is prudent */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextSetUp_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  cudaError_t             cerr;

  PetscFunctionBegin;
  switch (dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* don't create a stream for global blocking */
    dcu->stream = NULL;
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
    cerr = cudaStreamCreate(&dcu->stream);CHKERRCUDA(cerr);
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    cerr = cudaStreamCreateWithFlags(&dcu->stream,cudaStreamNonBlocking);CHKERRCUDA(cerr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %D",(PetscInt)dctx->streamType);
    break;
  }
  cerr = cudaEventCreate(&dcu->event);CHKERRCUDA(cerr);
  ierr = PetscCUBLASGetHandle_Internal(dcu);CHKERRQ(ierr);
  ierr = PetscCUSOLVERDnGetHandle_Internal(dcu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextQuery_CUDA(PetscDeviceContext dctx, PetscBool *idle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  *idle = cudaStreamQuery(dcu->stream) == cudaErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextWaitForContext_CUDA(PetscDeviceContext dctxa, PetscDeviceContext dctxb)
{
  PetscDeviceContext_CUDA *dcua = (PetscDeviceContext_CUDA *)dctxa->data;
  PetscDeviceContext_CUDA *dcub = (PetscDeviceContext_CUDA *)dctxb->data;
  cudaError_t             cerr;

  PetscFunctionBegin;
  cerr = cudaEventRecord(dcub->event,dcub->stream);CHKERRCUDA(cerr);
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11011) /* 11.1.1 */
  cerr = cudaStreamWaitEvent(dcua->stream,dcub->event,cudaEventWaitDefault);CHKERRCUDA(cerr);
#else
  cerr = cudaStreamWaitEvent(dcua->stream,dcub->event,0);CHKERRCUDA(cerr);
#endif /* 11.1.1 */
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextSynchronize_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;
  cudaError_t             cerr;

  PetscFunctionBegin;
  /* in case anything was queued on the event */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11011) /* 11.1.1 */
  cerr = cudaStreamWaitEvent(dcu->stream,dcu->event,cudaEventWaitDefault);CHKERRCUDA(cerr);
#else
  cerr = cudaStreamWaitEvent(dcu->stream,dcu->event,0);CHKERRCUDA(cerr);
#endif /* 11.1.1 */
  cerr = cudaStreamSynchronize(dcu->stream);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static const struct _DeviceContextOps cuops = {
  PetscDeviceContextCreate_CUDA,
  PetscDeviceContextDestroy_CUDA,
  PetscDeviceContextSetUp_CUDA,
  PetscDeviceContextQuery_CUDA,
  PetscDeviceContextWaitForContext_CUDA,
  PetscDeviceContextSynchronize_CUDA,
};

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_CUDA *dcu;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dcu);CHKERRQ(ierr);
  dctx->data = (void *)dcu;
  ierr = PetscMemcpy(dctx->ops,&cuops,sizeof(cuops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
