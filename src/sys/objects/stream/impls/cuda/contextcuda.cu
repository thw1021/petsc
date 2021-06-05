#include "contextcuda.hpp" /*I "petscdevice.h" I*/
#include <stack>

template <typename handleT> struct errorType;
template <> struct errorType<cublasHandle_t> { typedef cublasStatus_t errorT; };
template <> struct errorType<cusolverDnHandle_t> { typedef cusolverStatus_t errorT; };

template <typename handleT>
struct handleAllocator : errorType<handleT> {
  using errorType<handleT>::errorT;
  [[nodiscard]] PetscErrorCode create(handleT*) noexcept;
  [[nodiscard]] PetscErrorCode destroy(handleT&) noexcept;
};

template <>
PetscErrorCode handleAllocator<cublasHandle_t>::create(cublasHandle_t *handle) noexcept
{
  PetscErrorCode ierr;
  errorT         err;

  PetscFunctionBegin;
  for (int i = 0; i < 3; ++i) {
    err = cublasCreate(handle);
    if (err == CUBLAS_STATUS_SUCCESS) break;
    if (err != CUBLAS_STATUS_ALLOC_FAILED && err != CUBLAS_STATUS_NOT_INITIALIZED) CHKERRCUBLAS(err);
    if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
  }
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS");
  PetscFunctionReturn(0);
}

template <>
PetscErrorCode handleAllocator<cusolverDnHandle_t>::create(cusolverDnHandle_t *handle) noexcept
{
  PetscErrorCode ierr;
  errorT         err;

  PetscFunctionBegin;
  for (int i = 0; i < 3; ++i) {
    err = cusolverDnCreate(handle);
    if (err == CUSOLVER_STATUS_SUCCESS) break;
    if (err != CUSOLVER_STATUS_ALLOC_FAILED) CHKERRCUSOLVER(err);
    if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
  }
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuSolverDn");
  PetscFunctionReturn(0);
}

template <>
PetscErrorCode handleAllocator<cublasHandle_t>::destroy(cublasHandle_t &handle) noexcept
{
  errorT err;

  PetscFunctionBegin;
  err    = cublasDestroy(handle);CHKERRCUBLAS(err);
  handle = NULL;
  PetscFunctionReturn(0);
}

template <>
PetscErrorCode handleAllocator<cusolverDnHandle_t>::destroy(cusolverDnHandle_t &handle) noexcept
{
  errorT err;

  PetscFunctionBegin;
  err    = cusolverDnDestroy(handle);CHKERRCUSOLVER(err);
  handle = NULL;
  PetscFunctionReturn(0);
}

template <typename handleT>
struct handlePool : handleAllocator<handleT> {
private:
  std::stack<handleT> _stack;
  PetscBool           _registered;

public:
  using handleAllocator<handleT>::create;
  using handleAllocator<handleT>::destroy;

  constexpr handlePool() noexcept : _registered(PETSC_FALSE) {}

  [[nodiscard]] PetscErrorCode get(handleT &handle) noexcept;
  [[nodiscard]] PetscErrorCode reclaim(handleT &handle) noexcept;
  [[nodiscard]] PetscErrorCode finalize(void) noexcept;
};

static handlePool<cublasHandle_t>     cublasHandlePool;
static handlePool<cusolverDnHandle_t> cusolverHandlePool;

/* exists purely to be an extern "C" wrapper to pass to PetscRegisterFinalize() */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyCUBLASHandles_Internal(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = cublasHandlePool.finalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyCUSOLVERHandles_Internal(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = cusolverHandlePool.finalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <typename handleT>
PetscErrorCode handlePool<handleT>::get(handleT &handle) noexcept
{
  PetscFunctionBegin;
  if (!this->_registered) {
    PetscErrorCode ierr;

    /* this is really stupid... */
    if (std::is_same<handleT,cublasHandle_t>::value) {
      ierr = PetscRegisterFinalize(PetscDeviceContextDestroyCUBLASHandles_Internal);CHKERRQ(ierr);
    } else if (std::is_same<handleT,cusolverDnHandle_t>::value) {
      ierr = PetscRegisterFinalize(PetscDeviceContextDestroyCUSOLVERHandles_Internal);CHKERRQ(ierr);
    }
    this->_registered = PETSC_TRUE;
  }
  try {
    if (this->_stack.empty()) {
      PetscErrorCode ierr;

      ierr = this->create(&handle);CHKERRQ(ierr);
    } else {
      handle = this->_stack.top();
      this->_stack.pop();
    }
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  PetscFunctionReturn(0);
}

template <typename handleT>
PetscErrorCode handlePool<handleT>::reclaim(handleT &handle) noexcept
{
  PetscFunctionBegin;
  try {
    this->_stack.push(handle);
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  handle = NULL;
  PetscFunctionReturn(0);
}

template <typename handleT>
PetscErrorCode handlePool<handleT>::finalize(void) noexcept
{
  PetscFunctionBegin;
  try {
    while (!this->_stack.empty()) {
      PetscErrorCode ierr;

      ierr = this->destroy(this->_stack.top());CHKERRQ(ierr);
      this->_stack.pop();
    }
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  this->_registered = PETSC_FALSE;
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
  if (dcu->cublasv2handle) {
    ierr = cublasHandlePool.reclaim(dcu->cublasv2handle);CHKERRQ(ierr);
  }
  if (dcu->cusolverdnhandle) {
    ierr = cusolverHandlePool.reclaim(dcu->cusolverdnhandle);CHKERRQ(ierr);
  }
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
  PetscFunctionReturn(0);
}

/* cublas handle created here */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetCUBLASHandle_CUDA(PetscDeviceContext dctx, void *handle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  if (!dcu->cublasv2handle) {
    cublasStatus_t cberr;
    cudaStream_t   blasStream;
    PetscErrorCode ierr;

    ierr = cublasHandlePool.get(dcu->cublasv2handle);CHKERRQ(ierr);
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

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextRestoreCUBLASHandle_CUDA(PetscDeviceContext dctx, void *handle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(*((cublasHandle_t *)handle) != dcu->cublasv2handle)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"cuBLAS handle is not the same as the one that was checked out");
  }
  PetscFunctionReturn(0);
}

/* cusolver handle created here */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextGetCUSOLVERHandle_CUDA(PetscDeviceContext dctx, void *handle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  if (!dcu->cusolverdnhandle) {
    cusolverStatus_t cserr;
    PetscErrorCode   ierr;

    ierr = cusolverHandlePool.get(dcu->cusolverdnhandle);CHKERRQ(ierr);
    /* no need to do the checks as with blas, docs make no mention of workspace reset so
       we take their (implicit) word for it */
    cserr = cusolverDnSetStream(dcu->cusolverdnhandle,dcu->stream);CHKERRCUSOLVER(cserr);
  }
  *((cusolverDnHandle_t *)handle) = dcu->cusolverdnhandle;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextRestoreCUSOLVERHandle_CUDA(PetscDeviceContext dctx, void *handle)
{
  PetscDeviceContext_CUDA *dcu = (PetscDeviceContext_CUDA *)dctx->data;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(*((cusolverDnHandle_t *)handle) != dcu->cusolverdnhandle)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"cusolver handle is not the same as the one that was checked out");
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
  PetscDeviceContextGetCUBLASHandle_CUDA,
  PetscDeviceContextRestoreCUBLASHandle_CUDA,
  PetscDeviceContextGetCUSOLVERHandle_CUDA,
  PetscDeviceContextRestoreCUSOLVERHandle_CUDA,
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
