#ifndef PETSCCUPMINTERFACE_HPP
#define PETSCCUPMINTERFACE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/traithelpers.hpp>

#if defined(__cplusplus)
#if !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#error CUPMInterface requires c++11
#endif // PetscDefined(HAVE_CXX_DIALECT_CXX11)

namespace Petsc
{

// enum describing available cupm devices, this is used as the template parameter to any
// class subclassing the CUPMInterface or using it as a member variable
enum class CUPMDeviceKind : int {
  CUDA,
  HIP
};

static constexpr const char *const CUPMDeviceKinds[] = {"cuda","hip","CUPMDeviceKind","CUPMDeviceKind::",nullptr};

#if defined(CHKERRCUPM)
#error "Invalid redefinition of CHKERRCUPM, perhaps change order of header-file includes"
#endif // defined(CHKERRCUPM)

// A backend agnostic CHKERRCUPM() function, this will only work inside the member
// functions of a class inheriting from CUPMInterface
#define CHKERRCUPM(cerr) do {                                           \
    cupmError_t _cerr__ = cerr;                                         \
    if (PetscUnlikely(_cerr__)) {                                       \
      const char *name    = cupmGetErrorName(_cerr__);                  \
      const char *desc    = cupmGetErrorString(_cerr__);                \
      const char *backend = cupmName();                                 \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",   \
               backend,static_cast<PetscErrorCode>(_cerr__),name,desc); \
    }                                                                   \
  } while (0)

// A templated C++ struct that defines the entire CUPM interface. Use of templating vs
// preprocessor macros allows us to use both interfaces simultaneously as well as easily
// import them into classes.
template <CUPMDeviceKind T> struct CUPMInterface;

#define PETSC_ALIAS_CUPM_INTEGRAL_VALUE(prefix,stem) static const auto cupm##stem = prefix##stem
#define PETSC_ALIAS_CUPM_FUNCTION(prefix,stem)                          \
  PETSC_ALIAS_FUNCTION(static constexpr cupm##stem, prefix##stem)

#if PetscDefined(HAVE_CUDA)
template <>
struct CUPMInterface<CUPMDeviceKind::CUDA>
{
  static constexpr CUPMDeviceKind kind = CUPMDeviceKind::CUDA;

  PETSC_NODISCARD static constexpr const char* cupmName() noexcept
  { return CUPMDeviceKinds[static_cast<int>(kind)]; }

  // typedefs
  using cupmError_t        = cudaError_t;
  using cupmEvent_t        = cudaEvent_t;
  using cupmStream_t       = cudaStream_t;
  using cupmBlasHandle_t   = cublasHandle_t;
  using cupmBlasError_t    = cublasStatus_t;
  using cupmSolverHandle_t = cusolverDnHandle_t;
  using cupmSolverError_t  = cusolverStatus_t;
  using cupmDeviceProp_t   = cudaDeviceProp;

  // values
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(cuda,Success);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(cuda,ErrorNotReady);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(cuda,StreamNonBlocking);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(cuda,ErrorDeviceAlreadyInUse);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(cuda,ErrorSetOnActiveProcess);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(cuda,DeviceMapHost);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(cuda,MemcpyHostToDevice);

  // error functions
  PETSC_ALIAS_CUPM_FUNCTION(cuda,GetErrorName);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,GetErrorString);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,GetLastError);

  // device management
  PETSC_ALIAS_CUPM_FUNCTION(cuda,GetDeviceCount);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,GetDeviceProperties);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,GetDevice);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,SetDevice);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,GetDeviceFlags);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,SetDeviceFlags);

  // stream management
  PETSC_ALIAS_CUPM_FUNCTION(cuda,EventCreate);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,EventDestroy);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,EventRecord);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,EventSynchronize);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,EventElapsedTime);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,StreamCreate);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,StreamCreateWithFlags);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,StreamDestroy);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,StreamWaitEvent);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,StreamQuery);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,StreamSynchronize);

  // general purpose
  PETSC_ALIAS_CUPM_FUNCTION(cuda,Free);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,Malloc);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,Memcpy);
  PETSC_ALIAS_CUPM_FUNCTION(cuda,DeviceSynchronize);

  // There isn't a good way to auto-template this stuff between the cublas handle and
  // cusolver handle, not in the least because CHKERRCUBLAS and CHKERRCUSOLVER (not to
  // mention their HIP counterparts) do ~slightly~ different things. So we just overload
  // and accept the bloat
  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) noexcept
  {
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    if (handle) PetscFunctionReturn(0);
    for (int i = 0; i < 3; ++i) {
      PetscErrorCode ierr;

      cberr = cublasCreate(&handle);
      if (cberr == CUBLAS_STATUS_SUCCESS) break;
      if ((cberr != CUBLAS_STATUS_ALLOC_FAILED) && (cberr != CUBLAS_STATUS_NOT_INITIALIZED)) CHKERRCUBLAS(cberr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (PetscUnlikely(cberr != CUBLAS_STATUS_SUCCESS)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuBLAS");
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) noexcept
  {
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    if (handle) PetscFunctionReturn(0);
    for (int i = 0; i < 3; ++i) {
      PetscErrorCode ierr;

      cerr = cusolverDnCreate(&handle);
      if (cerr == CUSOLVER_STATUS_SUCCESS) break;
      if ((cerr != CUSOLVER_STATUS_NOT_INITIALIZED) && (cerr != CUSOLVER_STATUS_ALLOC_FAILED)) CHKERRCUSOLVER(cerr);
      if (i < 2) {ierr = PetscSleep(3);CHKERRQ(ierr);}
    }
    if (PetscUnlikely(cerr != CUSOLVER_STATUS_SUCCESS)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuSolverDn");
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmBlasHandle_t &handle, cupmStream_t &stream) noexcept
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

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) noexcept
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

  PETSC_NODISCARD static PetscErrorCode DestroyHandle(cupmBlasHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (handle) {
      cupmBlasError_t cberr;

      cberr  = cublasDestroy(handle);CHKERRCUBLAS(cberr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (handle) {
      cupmSolverError_t cerr;

      cerr   = cusolverDnDestroy(handle);CHKERRCUSOLVER(cerr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }
};
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
template <>
struct CUPMInterface<CUPMDeviceKind::HIP>
{
  static constexpr CUPMDeviceKind kind = CUPMDeviceKind::HIP;

  PETSC_NODISCARD static constexpr const char* cupmName() noexcept
  { return CUPMDeviceKinds[static_cast<int>(kind)];}

  // typedefs
  using cupmError_t        = hipError_t;
  using cupmEvent_t        = hipEvent_t;
  using cupmStream_t       = hipStream_t;
  using cupmBlasHandle_t   = hipblasHandle_t;
  using cupmBlasError_t    = hipblasStatus_t;
  using cupmSolverHandle_t = hipsolverHandle_t;
  using cupmSolverError_t  = hipsolverStatus_t;
  using cupmDeviceProp_t   = hipDeviceProp_t;

  // values
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(hip,Success);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(hip,ErrorNotReady);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(hip,StreamNonBlocking);
  // as of HIP v4.2 cudaErrorDeviceAlreadyInUse has no HIP equivalent
  static const auto cupmErrorDeviceAlreadyInUse = hipSuccess;
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(hip,ErrorSetOnActiveProcess);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(hip,DeviceMapHost);
  PETSC_ALIAS_CUPM_INTEGRAL_VALUE(hip,MemcpyHostToDevice);

  // error functions
  PETSC_ALIAS_CUPM_FUNCTION(hip,GetErrorName);
  PETSC_ALIAS_CUPM_FUNCTION(hip,GetErrorString);
  PETSC_ALIAS_CUPM_FUNCTION(hip,GetLastError);

  // device management
  PETSC_ALIAS_CUPM_FUNCTION(hip,GetDeviceCount);
  PETSC_ALIAS_CUPM_FUNCTION(hip,GetDeviceProperties);
  PETSC_ALIAS_CUPM_FUNCTION(hip,GetDevice);
  PETSC_ALIAS_CUPM_FUNCTION(hip,SetDevice);
  PETSC_ALIAS_CUPM_FUNCTION(hip,GetDeviceFlags);
  PETSC_ALIAS_CUPM_FUNCTION(hip,SetDeviceFlags);

  // stream management
  PETSC_ALIAS_CUPM_FUNCTION(hip,EventCreate);
  PETSC_ALIAS_CUPM_FUNCTION(hip,EventDestroy);
  PETSC_ALIAS_CUPM_FUNCTION(hip,EventRecord);
  PETSC_ALIAS_CUPM_FUNCTION(hip,EventSynchronize);
  PETSC_ALIAS_CUPM_FUNCTION(hip,EventElapsedTime);
  PETSC_ALIAS_CUPM_FUNCTION(hip,StreamCreate);
  PETSC_ALIAS_CUPM_FUNCTION(hip,StreamCreateWithFlags);
  PETSC_ALIAS_CUPM_FUNCTION(hip,StreamDestroy);
  PETSC_ALIAS_CUPM_FUNCTION(hip,StreamWaitEvent);
  PETSC_ALIAS_CUPM_FUNCTION(hip,StreamQuery);
  PETSC_ALIAS_CUPM_FUNCTION(hip,StreamSynchronize);

  // general purpose
  PETSC_ALIAS_CUPM_FUNCTION(hip,Free);
  PETSC_ALIAS_CUPM_FUNCTION(hip,Malloc);
  PETSC_ALIAS_CUPM_FUNCTION(hip,Memcpy);
  PETSC_ALIAS_CUPM_FUNCTION(hip,DeviceSynchronize);

  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmBlasHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (!handle) {cupmBlasError_t cberr = hipblasCreate(&handle);CHKERRHIPBLAS(cberr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (!handle) {cupmSolverError_t cerr = hipsolverCreate(&handle);CHKERRHIPSOLVER(cerr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmBlasHandle_t &handle, cupmStream_t &stream) noexcept
  {
    cupmStream_t    cupmStream;
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    cberr = hipblasGetStream(handle,&cupmStream);CHKERRHIPBLAS(cberr);
    if (cupmStream != stream) {cberr = hipblasSetStream(handle,stream);CHKERRHIPBLAS(cberr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) noexcept
  {
    cupmStream_t      cupmStream;
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    cerr = hipsolverGetStream(handle,&cupmStream);CHKERRHIPSOLVER(cerr);
    if (cupmStream != stream) {cerr = hipsolverSetStream(handle,stream);CHKERRHIPSOLVER(cerr);}
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode DestroyHandle(cupmBlasHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (handle) {
      cupmBlasError_t cberr;

      cberr  = hipblasDestroy(handle);CHKERRHIPBLAS(cberr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (handle) {
      cupmSolverError_t cerr;

      cerr   = hipsolverDestroy(handle);CHKERRHIPSOLVER(cerr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }
};
#endif // PetscDefined(HAVE_HIP)

} // namespace Petsc

// shorthand for bringing all of the typedefs from the base CUPMInterface class into your own,
// it's annoying that c++ doesn't have a way to do this automatically

#define PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(base_name_,Tp_)     \
  using base_name_ = CUPMInterface<Tp_>;                                \
  /* introspective typedefs */                                          \
  using base_name_::kind;                                               \
  /* types */                                                           \
  using typename base_name_::cupmError_t;                               \
  using typename base_name_::cupmEvent_t;                               \
  using typename base_name_::cupmStream_t;                              \
  using typename base_name_::cupmBlasError_t;                           \
  using typename base_name_::cupmSolverError_t;                         \
  using typename base_name_::cupmBlasHandle_t;                          \
  using typename base_name_::cupmSolverHandle_t;                        \
  using typename base_name_::cupmDeviceProp_t;                          \
  /* variables */                                                       \
  using base_name_::cupmSuccess;                                        \
  using base_name_::cupmErrorNotReady;                                  \
  using base_name_::cupmStreamNonBlocking;                              \
  using base_name_::cupmErrorDeviceAlreadyInUse;                        \
  using base_name_::cupmErrorSetOnActiveProcess;                        \
  using base_name_::cupmDeviceMapHost;                                  \
  using base_name_::cupmMemcpyHostToDevice;                             \
  /* functions */                                                       \
  using base_name_::cupmName;                                           \
  using base_name_::cupmGetErrorName;                                   \
  using base_name_::cupmGetErrorString;                                 \
  using base_name_::cupmGetLastError;                                   \
  using base_name_::cupmGetDeviceCount;                                 \
  using base_name_::cupmGetDeviceProperties;                            \
  using base_name_::cupmGetDevice;                                      \
  using base_name_::cupmSetDevice;                                      \
  using base_name_::cupmGetDeviceFlags;                                 \
  using base_name_::cupmSetDeviceFlags;                                 \
  using base_name_::cupmEventCreate;                                    \
  using base_name_::cupmEventDestroy;                                   \
  using base_name_::cupmEventRecord;                                    \
  using base_name_::cupmEventSynchronize;                               \
  using base_name_::cupmEventElapsedTime;                               \
  using base_name_::cupmStreamCreate;                                   \
  using base_name_::cupmStreamCreateWithFlags;                          \
  using base_name_::cupmStreamDestroy;                                  \
  using base_name_::cupmStreamWaitEvent;                                \
  using base_name_::cupmStreamQuery;                                    \
  using base_name_::cupmStreamSynchronize;                              \
  using base_name_::cupmFree;                                           \
  using base_name_::cupmMalloc;                                         \
  using base_name_::cupmMemcpy;                                         \
  using base_name_::cupmDeviceSynchronize;

#endif /* __cplusplus */

#endif /* PETSC_CUPMTRAITS_HPP */
