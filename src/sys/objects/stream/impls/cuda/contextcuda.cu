#include "contextcuda.hpp" /*I "petscdevice.h" I*/
#include <stack>
#include <functional>

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

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextCreateCUSOLVERHandle_Internal(cusolverDnHandle_t *handle)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyCUBLASHandle_Internal(cublasHandle_t &handle)
{
  cublasStatus_t cberr;

  PetscFunctionBegin;
  cberr = cublasDestroy(handle);CHKERRCUBLAS(cberr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyCUSOLVERHandle_Internal(cusolverDnHandle_t &handle)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

template <typename handleT> struct errorType;
template <> struct errorType<cublasHandle_t> { typedef cublasStatus_t errorT; };
template <> struct errorType<cusolverDnHandle_t> { typedef cusolverStatus_t errorT; };

template <typename handleT>
struct handleAllocator : errorType<handleT> {
  using errorType<handleT>::errorT;
  const std::function<PetscErrorCode(handleT*)> create;
  const std::function<PetscErrorCode(handleT&)> destroy;
  handleAllocator() noexcept {}
};

template <>
handleAllocator<cublasHandle_t>::handleAllocator() noexcept :
  create(PetscDeviceContextCreateCUBLASHandle_Internal),
  destroy(PetscDeviceContextDestroyCUBLASHandle_Internal)
{}

template <>
handleAllocator<cusolverDnHandle_t>::handleAllocator() noexcept :
  create(PetscDeviceContextCreateCUSOLVERHandle_Internal),
  destroy(PetscDeviceContextDestroyCUSOLVERHandle_Internal)
{}

template <typename handleT>
struct handleStack : handleAllocator<handleT> {
private:
  std::stack<handleT> _stack;
  PetscBool           _registered;

protected:
  using handleAllocator<handleT>::create;
  using handleAllocator<handleT>::destroy;

public:
  constexpr handleStack() noexcept : _registered(PETSC_FALSE) {}

  [[nodiscard]] inline PetscErrorCode get(handleT &handle) noexcept;
  [[nodiscard]] inline PetscErrorCode reclaim(handleT &handle) noexcept;
  [[nodiscard]] inline PetscErrorCode empty(void) noexcept;
};

static handleStack<cublasHandle_t>     cublasHandleStack;
static handleStack<cusolverDnHandle_t> cusolverHandleStack;

/* exists purely to be an extern "C" wrapper to pass to PetscRegisterFinalize() */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyCUBLASHandles_Internal(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = cublasHandleStack.empty();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyCUSOLVERHandles_Internal(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = cusolverHandleStack.empty();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <typename handleT>
PetscErrorCode handleStack<handleT>::get(handleT &handle) noexcept
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
PetscErrorCode handleStack<handleT>::reclaim(handleT &handle) noexcept
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
PetscErrorCode handleStack<handleT>::empty(void) noexcept
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
    ierr = cublasHandleStack.reclaim(dcu->cublasv2handle);CHKERRQ(ierr);
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
  /* don't also create solver contexts here since they aren't always used, and given the
     limited real estate frugality is prudent */
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

    ierr = cublasHandleStack.get(dcu->cublasv2handle);CHKERRQ(ierr);
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

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextRestoreBLASHandle_CUDA(PetscDeviceContext dctx, void *handle)
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

static const struct _DeviceContextOps cuops = {
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
