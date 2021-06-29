#if !defined(PETSCDEVICECONTEXTCUPM_HPP)
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

#if !defined(PETSC_HAVE_CXX_DIALECT_CXX11)
#error PetscDeviceContext backends for CUDA and HIP requires C++11
#endif

#define NUM_BACKENDS 5
PETSC_INTERN const char *const PetscDeviceContextBackends[];

// A useful template to serve as a function wrapper factory. Given a NON-OVERLOADED
// function "foo" which you'd like to thinly wrap as "bar", simply doing:
//
// ALIAS_FUNCTION(bar,foo);
//
// essentially creates
//
// returnType bar(argType1 arg1, argType2 arg2, ..., argTypeN argn)
// { return foo(arg1,arg2,...,argn);}
//
// for you. You may then call bar exactly as you would foo.
#if !defined(PETSC_HAVE_CXX_DIALECT_CXX14)
// decltype(auto) is c++14
#define ALIAS_FUNCTION(Alias_,Original_)                                \
  template <typename... Args> decltype(auto) Alias_(Args&&... args)     \
  { return Original_(std::forward<Args>(args)...);}
#else
#define ALIAS_FUNCTION(Alias_,Original_)                                \
  template <typename... Args>                                           \
  auto Alias_(Args&&... args) -> decltype(Original_(std::forward<Args>(args)...)) \
  { return Original_(std::forward<Args>(args)...);}
#endif

namespace Petsc {

// Available PetscDeviceContext backend implementations
enum class PetscDeviceContextBackend : int {CUDA, HIP};

#if defined(CHKERRCUPM)
#error "Invalid redefinition of CHKERRCUPM, perhaps change order of header-file includes"
#endif
// A backend agnostic CHKERR() function, this will only work inside the member functions
// of cupmContext
#define CHKERRCUPM(cerr)                                                \
  do {                                                                  \
    if (PetscUnlikely(cerr)) {                                          \
      const char *name    = cupmGetErrorName(cerr);                     \
      const char *descr   = cupmGetErrorString(cerr);                   \
      const char *backend = cupmName();                                 \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",backend,static_cast<int>(cerr),name,descr); \
    }                                                                   \
  } while (0)

// Forward declare
template <PetscDeviceContextBackend T> struct cupmTypeTraits;

#if PetscDefined(HAVE_CUDA)
template <>
struct cupmTypeTraits<PetscDeviceContextBackend::CUDA>
{
  PETSC_STATIC_INLINE PETSC_NODISCARD PETSC_CONSTEXPR const char* cupmName(void)
  { return PetscDeviceContextBackends[static_cast<int>(PetscDeviceContextBackend::CUDA)];}

  typedef cudaError_t        cupmError_t;
  typedef cudaEvent_t        cupmEvent_t;
  typedef cudaStream_t       cupmStream_t;
  typedef cublasHandle_t     cupmBlasHandle_t;
  typedef cublasStatus_t     cupmBlasError_t;
  typedef cusolverDnHandle_t cupmSolverHandle_t;
  typedef cusolverStatus_t   cupmSolverError_t;

  // Error functions
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmGetErrorName,cudaGetErrorName);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmGetErrorString,cudaGetErrorString);

  // Values
  static PETSC_CONSTEXPR const auto cupmErrorNotReady     = cudaErrorNotReady;
  static PETSC_CONSTEXPR const auto cupmStreamNonBlocking = cudaStreamNonBlocking;

  // Regular functions
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmEventCreate,cudaEventCreate);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmEventDestroy,cudaEventDestroy);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmEventRecord,cudaEventRecord);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamCreate,cudaStreamCreate);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamCreateWithFlags,cudaStreamCreateWithFlags);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamDestroy,cudaStreamDestroy);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamWaitEvent,cudaStreamWaitEvent);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamQuery,cudaStreamQuery);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamSynchronize,cudaStreamSynchronize);

  // There isn't a good way to auto-template this stuff between the cublas handle and
  // cusolver handle, not in the least because CHKERRCUBLAS and CHKERRCUSOLVER (not to
  // mention their hip counterparts) do ~slightly~ different things. So we just overload
  // and accept the bloat.
  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
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

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
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

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode SetHandleStream(cupmBlasHandle_t &handle, cupmStream_t &stream) PETSC_NOEXCEPT
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

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) PETSC_NOEXCEPT
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

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode DestroyHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (handle) {
      cupmBlasError_t cberr;

      cberr  = cublasDestroy(handle);CHKERRCUBLAS(cberr);
#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
      handle = nullptr;
#else
      handle = NULL;
#endif
    }
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (handle) {
      cupmSolverError_t cerr;

      cerr  = cusolverDnDestroy(handle);CHKERRCUSOLVER(cerr);
#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
      handle = nullptr;
#else
      handle = NULL;
#endif
    }
    PetscFunctionReturn(0);
  }
};
#endif

#if PetscDefined(HAVE_HIP)
template <>
struct cupmTypeTraits<PetscDeviceContextBackend::HIP>
{
  static PETSC_NODISCARD PETSC_CONSTEXPR const char* cupmName(void)
  { return PetscDeviceContextBackends[static_cast<int>(PetscDeviceContextBackend::HIP)];}

  typedef hipError_t        cupmError_t;
  typedef hipEvent_t        cupmEvent_t;
  typedef hipStream_t       cupmStream_t;
  typedef hipblasHandle_t   cupmBlasHandle_t;
  typedef hipblasStatus_t   cupmBlasError_t;
  typedef hipsolverHandle_t cupmSolverHandle_t;
  typedef hipsolverStatus_t cupmSolverError_t;

  // Error functions
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmGetErrorName,hipGetErrorName);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmGetErrorString,hipGetErrorString);

  // Values
  static PETSC_CONSTEXPR const auto cupmErrorNotReady     = hipErrorNotReady;
  static PETSC_CONSTEXPR const auto cupmStreamNonBlocking = hipStreamNonBlocking;

  // Functions
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmEventCreate,hipEventCreate);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmEventDestroy,hipEventDestroy);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmEventRecord,hipEventRecord);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamCreate,hipStreamCreate);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamCreateWithFlags,hipStreamCreateWithFlags);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamDestroy,hipStreamDestroy);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamWaitEvent,hipStreamWaitEvent);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamQuery,hipStreamQuery);
  ALIAS_FUNCTION(static PETSC_CONSTEXPR cupmStreamSynchronize,hipStreamSynchronize);

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (!handle) {
      cupmBlasError_t cberr;
      cberr = hipblasCreate(&handle);CHKERRHIPBLAS(cberr);
    }
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (!handle) {
      cupmSolverError_t cerr;
      cerr = hipsolverCreate(&handle);CHKERRHIPSOLVER(cerr);
    }
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode SetHandleStream(cupmBlasHandle_t &handle, cupmStream_t &stream) PETSC_NOEXCEPT
  {
    cupmStream_t    cupmStream;
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    cberr = hipblasGetStream(handle,&cupmStream);CHKERRHIPBLAS(cberr);
    if (cupmStream != stream) {
      cberr = hipblasSetStream(handle,stream);CHKERRHIPBLAS(cberr);
    }
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) PETSC_NOEXCEPT
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

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode DestroyHandle(cupmBlasHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (handle) {
      cupmBlasError_t cberr;

      cberr  = hipblasDestroy(handle);CHKERRHIPBLAS(cberr);
#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
      handle = nullptr;
#else
      handle = NULL;
#endif
    }
    PetscFunctionReturn(0);
  }

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle) PETSC_NOEXCEPT
  {
    PetscFunctionBegin;
    if (handle) {
      cupmSolverError_t cerr;

      cerr   = hipsolverDestroy(handle);CHKERRHIPSOLVER(cerr);
#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
      handle = nullptr;
#else
      handle = NULL;
#endif
    }
    PetscFunctionReturn(0);
  }
};
#endif

// Forward declare
template <PetscDeviceContextBackend T> class cupmContext;

template <PetscDeviceContextBackend T>
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
  using cupmType_t::cupmErrorNotReady;
  using cupmType_t::cupmStreamNonBlocking;

  // functions
  using cupmType_t::cupmName;
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
    cupmStream_t       stream;
    cupmEvent_t        event;
    cupmBlasHandle_t   blas;
    cupmSolverHandle_t solver;
  };

protected:
  // handle manipulation functions
  using cupmType_t::InitializeHandle;
  using cupmType_t::SetHandleStream;
  using cupmType_t::DestroyHandle;

  static cupmBlasHandle_t   _blashandle;
  static cupmSolverHandle_t _solverhandle;

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode FinalizeBLASHandle(void) PETSC_NOEXCEPT
  { return DestroyHandle(_blashandle);}

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode FinalizeSOLVERHandle(void) PETSC_NOEXCEPT
  { return DestroyHandle(_solverhandle);}

  PETSC_STATIC_INLINE PETSC_NODISCARD PetscErrorCode GetHandles(PetscDeviceContext_IMPLS *dci) PETSC_NOEXCEPT
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
    : ops{create,destroy,setUp,query,waitForContext,synchronize} {}

  static PETSC_NODISCARD PetscErrorCode destroy(PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode setUp(PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode query(PetscDeviceContext,PetscBool*) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode waitForContext(PetscDeviceContext,PetscDeviceContext) PETSC_NOEXCEPT;
  static PETSC_NODISCARD PetscErrorCode synchronize(PetscDeviceContext) PETSC_NOEXCEPT;
};

#define IMPLS_CAST(obj_) reinterpret_cast<PetscDeviceContext_IMPLS*>(obj_)
template <PetscDeviceContextBackend T>
PetscErrorCode cupmContext<T>::destroy(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = IMPLS_CAST(dctx->data);
  cupmError_t              cerr;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (dci->stream) {cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);}
  if (dci->event)  {cerr = cupmEventDestroy(dci->event);CHKERRCUPM(cerr);}
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <PetscDeviceContextBackend T>
PetscErrorCode cupmContext<T>::setUp(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = IMPLS_CAST(dctx->data);
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

template <PetscDeviceContextBackend T>
PetscErrorCode cupmContext<T>::query(PetscDeviceContext dctx, PetscBool *idle) PETSC_NOEXCEPT
{
  PetscFunctionBegin;
  *idle = cupmStreamQuery(IMPLS_CAST(dctx->data)->stream) == cupmErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

template <PetscDeviceContextBackend T>
PetscErrorCode cupmContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dcia = IMPLS_CAST(dctxa->data);
  PetscDeviceContext_IMPLS *dcib = IMPLS_CAST(dctxb->data);
  cupmError_t               cerr;

  PetscFunctionBegin;
  cerr = cupmEventRecord(dcib->event,dcib->stream);CHKERRCUPM(cerr);
  cerr = cupmStreamWaitEvent(dcia->stream,dcib->event,0);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <PetscDeviceContextBackend T>
PetscErrorCode cupmContext<T>::synchronize(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = IMPLS_CAST(dctx->data);
  cupmError_t               cerr;

  PetscFunctionBegin;
  /* in case anything was queued on the event */
  cerr = cupmStreamWaitEvent(dci->stream,dci->event,0);CHKERRCUPM(cerr);
  cerr = cupmStreamSynchronize(dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}
#undef IMPLS_CAST

// initialize the static member variables
template <PetscDeviceContextBackend T>
typename cupmContext<T>::cupmBlasHandle_t   cupmContext<T>::_blashandle   = NULL;

template <PetscDeviceContextBackend T>
typename cupmContext<T>::cupmSolverHandle_t cupmContext<T>::_solverhandle = NULL;

// shorten this one up a bit
typedef cupmContext<PetscDeviceContextBackend::CUDA> cupmContextCuda;
typedef cupmContext<PetscDeviceContextBackend::HIP>  cupmContextHip;

} // namespace Petsc

// make sure these doesn't leak out
#undef CHKERRCUPM
#undef ALIAS_FUNCTION

// shorthand for what is an EXTREMELY long name
#define PetscDeviceContext_(impls_) Petsc::cupmContext<Petsc::PetscDeviceContextBackend::impls_>::PetscDeviceContext_IMPLS

/* Silence undefined identifier errors for the op structs */
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext);
#endif /* PETSCDEVICECONTEXTCUDA_HPP */
