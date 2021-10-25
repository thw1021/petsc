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
enum class CUPMDeviceType : int {
  CUDA,
  HIP
};

static constexpr const char *const CUPMDeviceTypes[] = {
  "cuda",
  "hip",
  "CUPMDeviceType",
  "CUPMDeviceType::",
  nullptr
};

#if defined(CHKERRCUPM)
#  error "Invalid redefinition of CHKERRCUPM, perhaps change order of header-file includes"
#endif

// A backend agnostic CHKERRCUPM() function, this will only work inside the member
// functions of a class inheriting from CUPMInterface
#define CHKERRCUPM(cerr) do {                                           \
    cupmError_t _cerr__ = (cerr);                                       \
    if (PetscUnlikely(_cerr__ != cupmSuccess)) {                        \
      const auto name    = cupmGetErrorName(_cerr__);                   \
      const auto desc    = cupmGetErrorString(_cerr__);                 \
      const auto backend = cupmName();                                  \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",   \
               backend,static_cast<PetscErrorCode>(_cerr__),name,desc); \
    }                                                                   \
  } while (0)

// A templated C++ struct that defines the entire CUPM interface. Use of templating vs
// preprocessor macros allows us to use both interfaces simultaneously as well as easily
// import them into classes.
template <CUPMDeviceType T> struct CUPMInterface;

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT_(cupmprefix,mapped,prefix,original) \
  static const auto cupmprefix ## mapped = prefix ## original

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupmprefix,mapped,prefix,original) \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT_(cupmprefix,mapped,prefix,original)

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE(common)                         \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,common,PETSC_CUPM_STEM,common)

#define PETSC_CUPM_ALIAS_FUNCTION_EXACT_(cupmprefix,prefix,stem)        \
  PETSC_ALIAS_FUNCTION(static constexpr cupmprefix ## stem, prefix ## stem)

#define PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupmprefix,prefix,stem) \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT_(cupmprefix,prefix,stem)

#define PETSC_CUPM_ALIAS_FUNCTION(stem) PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupm,PETSC_CUPM_STEM,stem)

#if PetscDefined(HAVE_CUDA)
#define PETSC_CUPM_STEM cuda
template <>
struct CUPMInterface<CUPMDeviceType::CUDA>
{
  static constexpr CUPMDeviceType type = CUPMDeviceType::CUDA;

  PETSC_NODISCARD static constexpr PetscDeviceType cupmDeviceTypeToPetsc() noexcept
  { return PETSC_DEVICE_CUDA; }

  PETSC_NODISCARD static constexpr const char* cupmName() noexcept
  { return CUPMDeviceTypes[static_cast<int>(type)]; }

  // typedefs
  using cupmError_t        = cudaError_t;
  using cupmEvent_t        = cudaEvent_t;
  using cupmStream_t       = cudaStream_t;
  using cupmBlasHandle_t   = cublasHandle_t;
  using cupmBlasError_t    = cublasStatus_t;
  using cupmSolverHandle_t = cusolverDnHandle_t;
  using cupmSolverError_t  = cusolverStatus_t;
  using cupmDeviceProp_t   = cudaDeviceProp;
  using cupmMemcpy_t       = cudaMemcpyKind;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorDeviceAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
#if PETSC_PKG_CUDA_VERSION_GE(11,1,0)
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorStubLibrary);
#else
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,ErrorStubLibrary,cuda,ErrorInsufficientDriver);
#endif
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNoDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDefault);

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorName);
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorString);
  PETSC_CUPM_ALIAS_FUNCTION(GetLastError);

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceCount);
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceProperties);
  PETSC_CUPM_ALIAS_FUNCTION(GetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(SetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceFlags);
  PETSC_CUPM_ALIAS_FUNCTION(SetDeviceFlags);

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(EventCreate);
  PETSC_CUPM_ALIAS_FUNCTION(EventDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(EventRecord);
  PETSC_CUPM_ALIAS_FUNCTION(EventSynchronize);
  PETSC_CUPM_ALIAS_FUNCTION(EventElapsedTime);
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreate);
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreateWithFlags);
  PETSC_CUPM_ALIAS_FUNCTION(StreamDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(StreamWaitEvent);
  PETSC_CUPM_ALIAS_FUNCTION(StreamQuery);
  PETSC_CUPM_ALIAS_FUNCTION(StreamSynchronize);

  // general purpose
  PETSC_CUPM_ALIAS_FUNCTION(Free);
  PETSC_CUPM_ALIAS_FUNCTION(Malloc);
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy);
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize);
};
#undef PETSC_CUPM_STEM
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
#define PETSC_CUPM_STEM hip
template <>
struct CUPMInterface<CUPMDeviceType::HIP>
{
  static constexpr CUPMDeviceType type = CUPMDeviceType::HIP;

  PETSC_NODISCARD static constexpr PetscDeviceType cupmDeviceTypeToPetsc() noexcept
  { return PETSC_DEVICE_HIP; }

  PETSC_NODISCARD static constexpr const char* cupmName() noexcept
  { return CUPMDeviceTypes[static_cast<int>(type)]; }

  // typedefs
  using cupmError_t        = hipError_t;
  using cupmEvent_t        = hipEvent_t;
  using cupmStream_t       = hipStream_t;
  using cupmSolverHandle_t = hipsolverHandle_t;
  using cupmSolverError_t  = hipsolverStatus_t;
  using cupmDeviceProp_t   = hipDeviceProp_t;
  using cupmMemcpy_t       = hipMemcpyKind;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  // see https://github.com/ROCm-Developer-Tools/HIP/blob/develop/bin/hipify-perl
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,ErrorDeviceAlreadyInUse,hip,ErrorContextAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
  // as of HIP v4.2 cudaErrorStubLibrary has no HIP equivalent
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,ErrorStubLibrary,hip,ErrorInsufficientDriver);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNoDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(StreamNonBlocking);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(DeviceMapHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDeviceToDevice);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyHostToHost);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(MemcpyDefault);

  // error functions
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorName);
  PETSC_CUPM_ALIAS_FUNCTION(GetErrorString);
  PETSC_CUPM_ALIAS_FUNCTION(GetLastError);

  // device management
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceCount);
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceProperties);
  PETSC_CUPM_ALIAS_FUNCTION(GetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(SetDevice);
  PETSC_CUPM_ALIAS_FUNCTION(GetDeviceFlags);
  PETSC_CUPM_ALIAS_FUNCTION(SetDeviceFlags);

  // stream management
  PETSC_CUPM_ALIAS_FUNCTION(EventCreate);
  PETSC_CUPM_ALIAS_FUNCTION(EventDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(EventRecord);
  PETSC_CUPM_ALIAS_FUNCTION(EventSynchronize);
  PETSC_CUPM_ALIAS_FUNCTION(EventElapsedTime);
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreate);
  PETSC_CUPM_ALIAS_FUNCTION(StreamCreateWithFlags);
  PETSC_CUPM_ALIAS_FUNCTION(StreamDestroy);
  PETSC_CUPM_ALIAS_FUNCTION(StreamWaitEvent);
  PETSC_CUPM_ALIAS_FUNCTION(StreamQuery);
  PETSC_CUPM_ALIAS_FUNCTION(StreamSynchronize);

  // general purpose
  PETSC_CUPM_ALIAS_FUNCTION(Free);
  PETSC_CUPM_ALIAS_FUNCTION(Malloc);
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy);
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize);
};
#undef PETSC_CUPM_STEM
#endif // PetscDefined(HAVE_HIP)

} // namespace Petsc

// shorthand for bringing all of the typedefs from the base CUPMInterface class into your own,
// it's annoying that c++ doesn't have a way to do this automatically
#define PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING_(base_name_,Tp_)    \
  using base_name_ = CUPMInterface<Tp_>;                                \
  /* introspection */                                                   \
  using base_name_::type;                                               \
  using base_name_::cupmName;                                           \
  using base_name_::cupmDeviceTypeToPetsc;                              \
  /* types */                                                           \
  using typename base_name_::cupmError_t;                               \
  using typename base_name_::cupmEvent_t;                               \
  using typename base_name_::cupmStream_t;                              \
  using typename base_name_::cupmSolverError_t;                         \
  using typename base_name_::cupmSolverHandle_t;                        \
  using typename base_name_::cupmDeviceProp_t;                          \
  using typename base_name_::cupmMemcpy_t;                              \
  /* variables */                                                       \
  using base_name_::cupmSuccess;                                        \
  using base_name_::cupmErrorNotReady;                                  \
  using base_name_::cupmErrorDeviceAlreadyInUse;                        \
  using base_name_::cupmErrorSetOnActiveProcess;                        \
  using base_name_::cupmErrorStubLibrary;                               \
  using base_name_::cupmErrorNoDevice;                                  \
  using base_name_::cupmStreamNonBlocking;                              \
  using base_name_::cupmDeviceMapHost;                                  \
  using base_name_::cupmMemcpyHostToDevice;                             \
  using base_name_::cupmMemcpyDeviceToHost;                             \
  using base_name_::cupmMemcpyDeviceToDevice;                           \
  using base_name_::cupmMemcpyHostToHost;                               \
  using base_name_::cupmMemcpyDefault;                                  \
  /* functions */                                                       \
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

// allow any macros to expand in case someone needs it
#define PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(base_name_,Tp_)     \
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING_(base_name_,Tp_)

#endif /* __cplusplus */

#endif /* PETSCCUPMINTERFACE_HPP */
