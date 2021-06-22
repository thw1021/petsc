#include "contextcupm.hpp" /*I "petscdevice.h" I*/

#define CHKERRCUPM(cerr)                                                \
  do {                                                                  \
    if (PetscUnlikely(cerr)) {                                          \
      const char *name  = cupmGetErrorName(cerr);                       \
      const char *descr = cupmGetErrorString(cerr);                     \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",cupmName,(int)cerr,name,descr); \
    }                                                                   \
  } while (0)

enum class PetscDeviceBackends {CUDA, HIP};

template <PetscDeviceBackends T> struct cupmTypeTraits;

template <>
struct cupmTypeTraits<PetscDeviceBackends::CUDA>
{
  static constexpr char name[] = "HIP";

  typedef cudaError_t        cupmError_t;
  typedef cudaEvent_t        cupmEvent_t;
  typedef cudaStream_t       cupmStream_t;
  typedef cublasHandle_t     cupmBlasHandle_t;
  typedef cublasStatus_t     cupmBlasError_t;
  typedef cusolverDnHandle_t cupmSolverHandle_t;
  typedef cusolverStatus_t   cupmSolverError_t;

  static constexpr auto cupmGetErrorName   = cudaGetErrorName;
  static constexpr auto cupmGetErrorString = cudaGetErrorString;
  static constexpr auto cupmErrorNotReady  = cudaErrorNotReady;

  static constexpr auto cupmEventCreate     = cudaEventCreate;
  static constexpr auto cupmEventDestroy    = cudaEventDestroy;
  static constexpr auto cupmEventRecord     = cudaEventRecord;
  static constexpr auto cupmStreamCreate    = cudaStreamCreate;
  static constexpr auto cupmStreamDestroy   = cudaStreamDestroy;
  static constexpr auto cupmStreamWaitEvent = cudaStreamWaitEvent;

  // There isn't a good way to auto-template this stuff between the cublas handle and
  // cusolver handle, not in the least because CHKERRCUBLAS and CHKERRCUSOLVER (not to
  // mention their hip counterparts) do ~slightly~ different things. So we just overload
  // and accept the bloat.
  static PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (!handle) {
      cupmBlasStatus_t cberr;

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
      cupmSolverStatus_t cerr;

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
      cupmBlasStatus_t cberr;
      cberr = hipblasCreate(&handle);CHKERRHIPBLAS(cberr);
    }
    PetscFunctionReturn(0);
  }

  static PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (!handle) {
      cupmSolverStatus_t cerr;
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

template <PetscDeviceBackends T>
class cupmContext : cupmTypeTraits<T>
{
public:
  using cupmType_t = cupmTypeTraits<T>;
  using cupmName   = cupmType_t::name;

  // types
  using cupmType_t::cupmError_t;
  using cupmType_t::cupmEvent_t;
  using cupmType_t::cupmStream_t;
  using cupmType_t::cupmBlasError_t;
  using cupmType_t::cupmSolverError_t;
  using cupmType_t::cupmBlasHandle_t;
  using cupmType_t::cupmSolverHandle_t;

  // vars
  using cupmType_t::cupmErrorNotReady;

  // functions
  using cupmType_t::cupmGetErrorName;
  using cupmType_t::cupmGetErrorString;
  using cupmType_t::cupmEventCreate;
  using cupmType_t::cupmEventDestroy;
  using cupmType_t::cupmEventRecord;
  using cupmType_t::cupmStreamCreate;
  using cupmType_t::cupmStreamDestroy;
  using cupmType_t::cupmStreamWaitEvent;

protected:
  using cupmType_t::InitializeHandle;
  using cupmType_t::SetHandleStream;
  using cupmType_t::DestroyBLASHandle;
  using cupmType_t::DestroySOLVERHandle;

  static cupmBlasHandle_t   _blashandle;
  static cupmSolverHandle_t _solverhandle;

  static inline PETSC_NODISCARD PetscErrorCode FinalizeBLASHandle(void) PETSC_NOEXCEPT { return DestroyHandle(_blashandle);}
  static inline PETSC_NODISCARD PetscErrorCode FinalizeSOLVERHandle(void) PETSC_NOEXCEPT { return DestroyHandle(_solverhandle);}

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
    ierr = SetStream(_blashandle,dci->stream);CHKERRQ(ierr);
    ierr = SetStream(_solverhandle,dci->stream);CHKERRQ(ierr);
    dci->blas   = _blashandle;
    dci->solver = _solverhandle;
    PetscFunctionReturn(0);
  }

public:
  struct PetscDeviceContext_IMPLS
  {
    cupmStream_t       stream;
    cupmEvent_t        event;
    cupmBlasHandle_t   blas;
    cupmSolverHandle_t solver;
  };

  static const struct _DeviceContextOps ops;

  explicit PETSC_CONSTEXPR cupmContext(PetscErrorCode (*create)(PetscDeviceContext)) PETSC_NOEXCEPT
    : _blashandle(NULL), _solverhandle(NULL), ops{create, destroy, setUp, query, waitForContext, synchronize} {}

  static PETSC_NODISCARD PetscErrorCode destroy(PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode setUp(PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode query(PetscDeviceContext,PetscBool*) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode waitForContext(PetscDeviceContext,PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode synchronize(PetscDeviceContext) PETSC_NOEXCEPT;
};

static PETSC_NODISCARD PetscErrorCode cupmContext::destroy(PetscDeviceContext dctx) PETSC_NOEXCEPT
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

static PETSC_NODISCARD PetscErrorCode cupmContext::setUp(PetscDeviceContext dctx) PETSC_NOEXCEPT
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

static PETSC_NODISCARD PetscErrorCode cupmContext::query(PetscDeviceContext dctx, PetscBool *idle) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;

  PetscFunctionBegin;
  *idle = cupmStreamQuery(dci->stream) == cupmErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PETSC_NODISCARD PetscErrorCode cupmContext::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dcia = (PetscDeviceContext_IMPLS *)dctxa->data;
  PetscDeviceContext_IMPLS *dcib = (PetscDeviceContext_IMPLS *)dctxb->data;
  cupmError_t               cerr;

  PetscFunctionBegin;
  cerr = cupmEventRecord(dcub->event,dcub->stream);CHKERRCUPM(cerr);
  cerr = cupmStreamWaitEvent(dcia->stream,dcib->event);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

static PETSC_NODISCARD PetscErrorCode cupmContext::synchronize(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;
  cupmError_t               cerr;

  PetscFunctionBegin;
  /* in case anything was queued on the event */
  cerr = cupmStreamWaitEvent(dci->stream,dci->event);CHKERRCUPM(cerr);
  cerr = cupmStreamSynchronize(dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

static const cupmContext<CUDA> contextCuda(PetscDeviceContextCreate_CUDA);

// static const struct _DeviceContextOps cuops = {
//   PetscDeviceContextCreate_CUDA,
//   contextCuda.destroy,
//   contextCuda.setUp,
//   contextCuda.query,
//   contextCuda.waitForContext,
//   contextCuda.synchronize,
// };

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  contextCuda::PetscDeviceContext_IMPLS *dci;
  PetscErrorCode                         ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = (void *)dci;
  ierr = PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
