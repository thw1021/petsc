#include "contextcuda.hpp" /*I "petscdevice.h" I*/
#include <stack>


#if defined(SETERRXX)
/* this shouldn't exist anyways, but there are certain .seealso's that reference this
   phantom macro so we guard just in case */
#error "SETERRX already defined"
#else
/*
  Not really kosher to throw an "exception" in a dtor since we may inadvertently
  clobber another petsc error on the way. Instead we build everything as if we were
  about to burn the house down __except__ throw the match.
*/
#define SETERRXX(comm,ierr,s) do {                                      \
    PetscBool      _finalized_xx_;                                      \
    PetscErrorCode _ierr_xx_ = reinterpret_cast<PetscErrorCode>(ierr);  \
    _ierr_xx_ = PetscFinalized(&_finalized_xx_);CHKERRQ(_ierr_xx_);     \
    if (_finalized_xx_) {                                               \
      /* were in deep trouble now */                                    \
      printf(s);                                                        \
    } else {                                                            \
      _ierr_xx_ = PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_INITIAL,s);CHKERRCONTINUE(_ierr_xx_); \
    }                                                                   \
  } while (0);
#endif

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

static PetscBool                      setupCublasHandles   = PETSC_FALSE;
static PetscBool                      setupCusolverHandles = PETSC_FALSE;
static std::stack<cublasHandle_t>     cublasHandleStack;
static std::stack<cusolverDnHandle_t> cusolverHandleStack;

/* allow getting around nodiscard */
template <typename T> void discard(const T&) {}

template <typename handleT>
struct handleStack {
private:
  PetscBool           _registered;
  std::stack<handleT> _stack;

  [[nodiscard]] inline PetscErrorCode _finalize(void)
  {
    PetscFunctionBegin;
    try {
      while (!this->_stack.empty()) {
        cublasStatus_t cberr;
        cublasHandle_t handle = this->_stack.top();

        cberr = cublasDestroy(handle);CHKERRCUBLAS(cberr);
        this->_stack.pop();
      }
    } catch (std::exception const &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error finalizing handles: %s",ex.what());
    }
    this->_registered = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

public:
  handleStack() : _registered(PETSC_FALSE) noexcept {}

  ~handleStack() noexcept
  {
    PetscErrorCode ierr;

    if (PetscUnlikelyDebug(this->_registered)) SETERRXX(PETSC_COM_SELF,PETSC_ERR_PLIB,"stack destructor called before PetscFinalize()");
    discard(this->_finalize());
  }

  /* better pop semantics */
  [[nodiscard]] inline PetscErrorCode pop(handleT &handle)
  {
    PetscFunctionBegin;
    if (!this->_registered) {
      PetscErrorCode ierr;
      ierr = PetscRegisterFinalize(this->_finalize);CHKERRQ(ierr);
      this->_registered = PETSC_TRUE;
    }
    try {
      handle = this->_stack.top();
      this->_stack.pop();
    } catch (std::exception const &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error popping handle: %s",ex.what());
    }
    PetscFunctionReturn(0);
  }

  [[nodiscard]] inline PetscErrorCode push(handleT &&handle)
  {
    PetscFunctionBegin;
    try {
      _stack.push(handle);
    } catch (std::exception const &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error pushing handle: %s",ex.what());
    }
    PetscFunctionReturn(0);
  }

  [[nodiscard]] inline PetscErrorCode push(const handleT &handle)
  {
    PetscFunctionBegin;
    try {
      _stack.push(handle);
    } catch (std::exception const &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error pushing handle: %s",ex.what());
    }
    PetscFunctionReturn(0);
  }
};

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyCUSOLVERHandles_Internal(void)
{
  PetscFunctionBegin;
  try {
    while (!cublasHandleStack.empty()) {
      cusolverStatus_t   cserr;
      cusolverDnHandle_t handle = cusolverHandleStack.top();

      cserr = cusolverDnDestroy(handle);CHKERRCUSOLVER(cserr);
      cusolverHandleStack.pop();
    }
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error finalizing cuSolver handles: %s",ex.what());
  }
  setupCusolverHandles = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyCUBLASHandles_Internal(void)
{
  PetscFunctionBegin;
  try {
    while (!cublasHandleStack.empty()) {
      cublasStatus_t cberr;
      cublasHandle_t handle = cublasHandleStack.top();

      cberr = cublasDestroy(handle);CHKERRCUBLAS(cberr);
      cublasHandleStack.pop();
    }
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error finalizing cuBLAS handles: %s",ex.what());
  }
  setupCublasHandles = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextAcquireCUBLASHandle_Internal(cublasHandle_t *handle)
{
  PetscFunctionBegin;
  if (!setupCublasHandles) {
    ierr = PetscRegisterFinalize(PetscDeviceContextDestroyCUBLASHandles_Internal);CHKERRQ(ierr);
    setupCublasHandles = PETSC_TRUE;
  }
  if (cublasHandleStack.empty()) {
    PetscErrorCode ierr;
    /* stack is empty, need to create a handle */
    ierr = PetscDeviceContextCreateCUBLASHandle_Internal(handle);CHKERRQ(ierr);
  } else {
    /* stuff on the stack, pop from it */
    try {
      *handle = cublasHandleStack.top();
      cublasHandleStack.pop();
    } catch (std::exception const &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error popping from std::stack: %s",ex.what());
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextReleaseCUBLASHandle_Internal(cublasHandle_t *handle)
{
  PetscFunctionBegin;
  try {
    /* release the handle so it may be recycled */
    cublasHandleStack.push(*handle);
  } catch (std::exception const &ex) {
    /* or not */
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error pushing to std::stack: %s",ex.what());
  }
  *handle = NULL;
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
    ierr = PetscDeviceContextReleaseCUBLASHandle_Internal(dcu);CHKERRQ(ierr);
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

    ierr  = PetscDeviceContextAcquireCUBLASHandle_Internal(&dcu->cublasv2handle);CHKERRQ(ierr);
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
