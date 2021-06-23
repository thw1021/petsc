#include "contextcupm.hpp" /*I "petscdevice.h" I*/
#include <type_traits>
#define CHKERRCUPM(cerr)                                                \
  do {                                                                  \
    if (PetscUnlikely(cerr)) {                                          \
      const char *name  = cupmGetErrorName(cerr);                       \
      const char *descr = cupmGetErrorString(cerr);                     \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",cupmName,(int)cerr,name,descr); \
    }                                                                   \
  } while (0)

enum class PetscDeviceBackends {CUDA, HIP};

template <PetscDeviceBackends T> struct cupmTypeTraits;

template <class Fn, Fn func> struct wrapper;
template <class Ret, class... Args,Ret (*func)(Args...)>
struct wrapper<Ret (*)(Args...),func>
{
  static Ret wrap(Args... args) { return func(args...);}
};

#define WRAP_FUNCTION(func_) wrapper<decltype(&func_),func_>::wrap

#if PetscDefined(HAVE_CUDA)
template <>
struct cupmTypeTraits<PetscDeviceBackends::CUDA>
{
  static constexpr char cupmName[] = "CUDA";

  typedef cudaError_t        cupmError_t;
  typedef cudaEvent_t        cupmEvent_t;
  typedef cudaStream_t       cupmStream_t;
  typedef cublasHandle_t     cupmBlasHandle_t;
  typedef cublasStatus_t     cupmBlasError_t;
  typedef cusolverDnHandle_t cupmSolverHandle_t;
  typedef cusolverStatus_t   cupmSolverError_t;

  // Error functions
  static constexpr auto cupmGetErrorName      = WRAP_FUNCTION(cudaGetErrorName);
  static constexpr auto cupmGetErrorString    = WRAP_FUNCTION(cudaGetErrorString);

  // Values
  static constexpr auto cupmErrorNotReady     = cudaErrorNotReady;
  static constexpr auto cupmStreamNonBlocking = cudaStreamNonBlocking;

  // Regular functions
  static cupmError_t cupmEventCreate(cupmEvent_t *e) { return cudaEventCreate(e);}
  static constexpr auto cupmEventDestroy          = WRAP_FUNCTION(cudaEventDestroy);
  static constexpr auto cupmEventRecord           = WRAP_FUNCTION(cudaEventRecord);
  static constexpr auto cupmStreamCreate          = WRAP_FUNCTION(cudaStreamCreate);
  static constexpr auto cupmStreamCreateWithFlags = WRAP_FUNCTION(cudaStreamCreateWithFlags);
  static constexpr auto cupmStreamDestroy         = WRAP_FUNCTION(cudaStreamDestroy);
  static constexpr auto cupmStreamWaitEvent       = WRAP_FUNCTION(cudaStreamWaitEvent);
  static constexpr auto cupmStreamQuery           = WRAP_FUNCTION(cudaStreamQuery);
  static constexpr auto cupmStreamSynchronize     = WRAP_FUNCTION(cudaStreamSynchronize);

  // There isn't a good way to auto-template this stuff between the cublas handle and
  // cusolver handle, not in the least because CHKERRCUBLAS and CHKERRCUSOLVER (not to
  // mention their hip counterparts) do ~slightly~ different things. So we just overload
  // and accept the bloat.
  static PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (!handle) {
      cupmBlasError_t cberr;

      for (int i=0; i<3; ++i) {
        PetscErrorCode ierr;

        cberr = cublasCreate(&handle);
        if (!cberr) break;
        if (cberr != CUBLAS_STATUS_ALLOC_FAILED && cberr != CUBLAS_STATUS_NOT_INITIALIZED) CHKERRCUBLAS(cberr);
        if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
      }
      if (cberr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS");
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (!handle) {
      cupmSolverError_t cerr;

      for (int i=0; i<3; i++) {
        PetscErrorCode ierr;

        cerr = cusolverDnCreate(&handle);
        if (!cerr) break;
        if (cerr != CUSOLVER_STATUS_ALLOC_FAILED) CHKERRCUSOLVER(cerr);
        if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
      }
      if (cerr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuSolverDn");
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode SetHandleStream(cupmBlasHandle_t &handle, cupmStream_t &stream) PETSC_NOEXCEPT
  {
    cupmStream_t    cupmStream;
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    cberr = cublasGetStream(handle,&cupmStream);CHKERRCUBLAS(cberr);
    if (cupmStream != stream) {
      cberr = cublasSetStream(handle,stream);CHKERRCUBLAS(cberr);
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) PETSC_NOEXCEPT
  {
    cupmStream_t      cupmStream;
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    cerr = cusolverDnGetStream(handle,&cupmStream);CHKERRCUSOLVER(cerr);
    if (cupmStream != stream) {
      cerr = cusolverDnSetStream(handle,stream);CHKERRCUSOLVER(cerr);
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode DestroyHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (handle) {
      cupmBlasError_t cberr;

      cberr  = cublasDestroy(handle);CHKERRCUBLAS(cberr);
      handle = NULL;
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (handle) {
      cupmSolverError_t cerr;

      cerr  = cusolverDnDestroy(handle);CHKERRCUSOLVER(cerr);
      handle = NULL;
    }
    PetscFunctionReturn(0);
  }
};
#endif

#if PetscDefined(HAVE_HIP)
template <>
struct cupmTypeTraits<PetscDeviceBackends::HIP>
{
  static constexpr char name[] = "HIP";

  typedef hipError_t        cupmError_t;
  typedef hipEvent_t        cupmEvent_t;
  typedef hipStream_t       cupmStream_t;
  typedef rocblas_handle    cupmBlasHandle_t;
  typedef roblas_status     cupmBlasError_t;
  typedef hipsolverHandle_t cupmSolverHandle_t;
  typedef hipsolverStatus_t cupmSolverError_t;

  static constexpr auto cupmGetErrorName   = hipGetErrorName;
  static constexpr auto cupmGetErrorString = hipGetErrorString;
  static constexpr auto cupmErrorNotReady  = hipErrorNotReady;

  static constexpr auto cupmEventCreate     = hipEventCreate;
  static constexpr auto cupmEventDestroy    = hipEventDestroy;
  static constexpr auto cupmEventRecord     = hipEventRecord;
  static constexpr auto cupmStreamCreate    = hipStreamCreate;
  static constexpr auto cupmStreamDestroy   = hipStreamDestroy;
  static constexpr auto cupmStreamWaitEvent = hipStreamWaitEvent;


  static PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (!handle) {
      cupmBlasError_t cberr;
      cberr = hipblasCreate(&handle);CHKERRHIPBLAS(cberr);
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (!handle) {
      cupmSolverError_t cerr;
      cerr = hipsolverCreate(&handle);CHKERRHIPSOLVER(cerr);
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode SetHandleStream(cupmBlasHandle_t &handle, cupmStream_t &stream) PETSC_NOEXCEPT
  {
    cupmStream_t    cupmStream;
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    cberr = hiplasGetStream(handle,&cupmStream);CHKERRHIPBLAS(cberr);
    if (cupmStream != stream) {
      cberr = hipblasSetStream(handle,stream);CHKERRHIPBLAS(cberr);
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) PETSC_NOEXCEPT
  {
    cupmStream_t      cupmStream;
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    cerr = hipsolverGetStream(handle,&cupmStream);CHKERRHIPSOLVER(cerr);
    if (cupmStream != stream) {
      cerr = hipsolverSetStream(handle,stream);CHKERRHIPSOLVER(cerr);
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode DestroyHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (handle) {
      cupmBlasError_t cberr;

      cberr  = hipblasDestroy(handle);CHKERRHIPBLAS(cberr);
      handle = NULL;
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (handle) {
      cupmSolverError_t cerr;

      cerr  = hipsolverDestroy(handle);CHKERRHIPSOLVER(cerr);
      handle = NULL;
    }
    PetscFunctionReturn(0);
  }
};
#endif

template <PetscDeviceBackends T>
class cupmContext : cupmTypeTraits<T>
{
public:
  using cupmType_t = cupmTypeTraits<T>;

  // types
  using typename cupmType_t::cupmError_t;
  using typename cupmType_t::cupmEvent_t;
  using typename cupmType_t::cupmStream_t;
  using typename cupmType_t::cupmBlasError_t;
  using typename cupmType_t::cupmSolverError_t;
  using typename cupmType_t::cupmBlasHandle_t;
  using typename cupmType_t::cupmSolverHandle_t;

  // vars
  using cupmType_t::cupmName;
  using cupmType_t::cupmErrorNotReady;
  using cupmType_t::cupmStreamNonBlocking;

  // functions
  using cupmType_t::cupmGetErrorName;
  using cupmType_t::cupmGetErrorString;
  using cupmType_t::cupmEventCreate;
  using cupmType_t::cupmEventDestroy;
  using cupmType_t::cupmEventRecord;
  using cupmType_t::cupmStreamCreate;
  using cupmType_t::cupmStreamCreateWithFlags;
  using cupmType_t::cupmStreamDestroy;
  using cupmType_t::cupmStreamWaitEvent;
  using cupmType_t::cupmStreamQuery;
  using cupmType_t::cupmStreamSynchronize;

  struct PetscDeviceContext_IMPLS
  {
    typename cupmType_t::cupmStream_t       stream;
    typename cupmType_t::cupmEvent_t        event;
    typename cupmType_t::cupmBlasHandle_t   blas;
    typename cupmType_t::cupmSolverHandle_t solver;
  };
protected:
  using cupmType_t::InitializeHandle;
  using cupmType_t::SetHandleStream;
  using cupmType_t::DestroyHandle;

  static cupmBlasHandle_t   _blashandle;
  static cupmSolverHandle_t _solverhandle;

  static PETSC_NODISCARD PetscErrorCode FinalizeBLASHandle(void) PETSC_NOEXCEPT { return DestroyHandle(_blashandle);}
  static PETSC_NODISCARD PetscErrorCode FinalizeSOLVERHandle(void) PETSC_NOEXCEPT { return DestroyHandle(_solverhandle);}

  static PETSC_NODISCARD PetscErrorCode GetHandles(PetscDeviceContext_IMPLS *dci) PETSC_NOEXCEPT
  {
    PetscErrorCode  ierr;

    PetscFunctionBegin;
    if (!_blashandle) {
      ierr = InitializeHandle(_blashandle);CHKERRQ(ierr);
      ierr = PetscRegisterFinalize(FinalizeBLASHandle);CHKERRQ(ierr);
    }
    if (!_solverhandle) {
      ierr = InitializeHandle(_solverhandle);CHKERRQ(ierr);
      ierr = PetscRegisterFinalize(FinalizeSOLVERHandle);CHKERRQ(ierr);
    }
    ierr = SetHandleStream(_blashandle,dci->stream);CHKERRQ(ierr);
    ierr = SetHandleStream(_solverhandle,dci->stream);CHKERRQ(ierr);
    dci->blas   = _blashandle;
    dci->solver = _solverhandle;
    PetscFunctionReturn(0);
  }

public:
  const struct _DeviceContextOps ops;

  explicit PETSC_CONSTEXPR cupmContext(PetscErrorCode (*create)(PetscDeviceContext)) PETSC_NOEXCEPT
    : ops{create, destroy, setUp, query, waitForContext, synchronize} {}

  static PETSC_NODISCARD PetscErrorCode destroy(PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode setUp(PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode query(PetscDeviceContext,PetscBool*) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode waitForContext(PetscDeviceContext,PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode synchronize(PetscDeviceContext) PETSC_NOEXCEPT;
};

template <PetscDeviceBackends T>
PETSC_NODISCARD PetscErrorCode cupmContext<T>::destroy(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;
  cupmError_t              cerr;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (dci->stream) {cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);}
  if (dci->event)  {cerr = cupmEventDestroy(dci->event);CHKERRCUPM(cerr);}
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <PetscDeviceBackends T>
PETSC_NODISCARD PetscErrorCode cupmContext<T>::setUp(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;
  PetscErrorCode           ierr;
  cupmError_t              cerr;

  PetscFunctionBegin;
  switch (dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* don't create a stream for global blocking */
    dci->stream = NULL;
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
    cerr = cupmStreamCreate(&dci->stream);CHKERRCUPM(cerr);
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    cerr = cupmStreamCreateWithFlags(&dci->stream,cupmStreamNonBlocking);CHKERRCUPM(cerr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %D",(PetscInt)dctx->streamType);
    break;
  }
  cerr = cupmEventCreate(&dci->event);CHKERRCUPM(cerr);
  ierr = GetHandles(dci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <PetscDeviceBackends T>
PETSC_NODISCARD PetscErrorCode cupmContext<T>::query(PetscDeviceContext dctx, PetscBool *idle) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;

  PetscFunctionBegin;
  *idle = cupmStreamQuery(dci->stream) == cupmErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

template <PetscDeviceBackends T>
PETSC_NODISCARD PetscErrorCode cupmContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dcia = (PetscDeviceContext_IMPLS *)dctxa->data;
  PetscDeviceContext_IMPLS *dcib = (PetscDeviceContext_IMPLS *)dctxb->data;
  cupmError_t               cerr;

  PetscFunctionBegin;
  cerr = cupmEventRecord(dcib->event,dcib->stream);CHKERRCUPM(cerr);
  cerr = cupmStreamWaitEvent(dcia->stream,dcib->event,0);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <PetscDeviceBackends T>
PETSC_NODISCARD PetscErrorCode cupmContext<T>::synchronize(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;
  cupmError_t               cerr;

  PetscFunctionBegin;
  /* in case anything was queued on the event */
  cerr = cupmStreamWaitEvent(dci->stream,dci->event,0);CHKERRCUPM(cerr);
  cerr = cupmStreamSynchronize(dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

static const cupmContext<PetscDeviceBackends::CUDA> contextCuda(PetscDeviceContextCreate_CUDAM);

#define PetscDeviceContext_(impls_) struct cupmContext<PetscDeviceBackends::impls_>::PetscDeviceContext_IMPLS

PetscErrorCode PetscDeviceContextCreate_CUDAM(PetscDeviceContext dctx)
{
  PetscDeviceContext_(CUDA) *dci;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = (void *)dci;
  ierr = PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
