#ifndef PETSCCUPMINTERFACE_HPP
#define PETSCCUPMINTERFACE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/petsctypetraits.hpp>

#if defined(__cplusplus)

#if !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#  error CUPMInterface requires c++11
#endif

#include <array>

namespace Petsc
{

// enum describing available cupm devices, this is used as the template parameter to any
// class subclassing the CUPMInterface or using it as a member variable
enum class CUPMDeviceType : int {
  CUDA,
  HIP
};

static constexpr std::array<const char*const,5> CUPMDeviceTypes = {
  "cuda",
  "hip",
  "CUPMDeviceType",
  "CUPMDeviceType::",
  nullptr
};

namespace Impl
{

namespace detail
{

static_assert(util::integral_value(CUPMDeviceType::CUDA) == 0,"");
static_assert(util::integral_value(CUPMDeviceType::HIP)  == 1,"");
static constexpr std::array<PetscDeviceType,2> CUPMDeviceTypeToPetscDeviceTypes = {
  PETSC_DEVICE_CUDA,
  PETSC_DEVICE_HIP
};

static_assert(util::integral_value(CUPMDeviceType::CUDA) == 0,"");
static_assert(util::integral_value(CUPMDeviceType::HIP)  == 1,"");
static constexpr std::array<PetscMemType,2> CUPMDeviceTypeToPetscMemTypes = {
  PETSC_MEMTYPE_CUDA,
  PETSC_MEMTYPE_HIP
};

// A backend agnostic CHKERRCUPM() function, this will only work inside the member
// functions of a class inheriting from CUPMInterface
#define CHKERRCUPM(expression) do {                                     \
    const cupmError_t _cerr__ = expression;                             \
    if (PetscUnlikely(_cerr__ != cupmSuccess)) {                        \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s) : %s",   \
               cupmName(),static_cast<PetscErrorCode>(_cerr__),         \
               cupmGetErrorName(_cerr__),cupmGetErrorString(_cerr__));  \
    }                                                                   \
  } while (0)

#undef CAT_
#undef CAT

#define CAT_(x,y) x ## y
#define CAT(x,y)  CAT_(x,y)

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(our_prefix,our_suffix,their_prefix,their_suffix) \
  static const auto CAT(our_prefix,our_suffix) = CAT(their_prefix,their_suffix)

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(our_suffix,their_suffix) \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(cupm,our_suffix,PETSC_CUPM_PREFIX,their_suffix)

#define PETSC_CUPM_ALIAS_INTEGRAL_VALUE(suffix)         \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(suffix,suffix)

#define PETSC_CUPM_ALIAS_FUNCTION_EXACT(our_prefix,our_suffix,their_prefix,their_suffix) \
  PETSC_ALIAS_FUNCTION(static constexpr CAT(our_prefix,our_suffix),CAT(their_prefix,their_suffix))

#define PETSC_CUPM_ALIAS_FUNCTION_COMMON(our_suffix,their_suffix)       \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupm,our_suffix,PETSC_CUPM_PREFIX,their_suffix)

#define PETSC_CUPM_ALIAS_FUNCTION(suffix) PETSC_CUPM_ALIAS_FUNCTION_COMMON(suffix,suffix)

#define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(our_prefix,our_suffix,their_prefix,their_suffix,N) \
  PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(static constexpr CAT(our_prefix,our_suffix),CAT(their_prefix,their_suffix),N)

#define PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(our_suffix,their_suffix,N) \
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_EXACT(cupm,our_suffix,PETSC_CUPM_PREFIX,their_suffix,N)

#define PETSC_CUPM_DEVICE_TYPE CUPMDeviceType::PETSC_CUPM_PREFIX_U

// Base class that holds stuff that can be directly determined with templates
template <CUPMDeviceType T>
struct CUPMInterfaceBase
{
  static constexpr const auto type = T;

  PETSC_CXX_COMPAT_DECL(constexpr const char*const cupmName())
  {
    return std::get<util::integral_value(type)>(CUPMDeviceTypes);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceType cupmDeviceTypeToPetscDeviceType())
  {
    return std::get<util::integral_value(type)>(CUPMDeviceTypeToPetscDeviceTypes);
  }

  PETSC_CXX_COMPAT_DECL(constexpr PetscMemType cupmDeviceTypeToPetscMemType())
  {
    return std::get<util::integral_value(type)>(CUPMDeviceTypeToPetscMemTypes);
  }

};

template <CUPMDeviceType T> constexpr const CUPMDeviceType CUPMInterfaceBase<T>::type;

#define PETSC_CUPM_BASE_CLASS_HEADER(DEVICE_TYPE)                       \
  using base_type = detail::CUPMInterfaceBase<DEVICE_TYPE>;             \
  using base_type::type;                                                \
  using base_type::cupmName;                                            \
  using base_type::cupmDeviceTypeToPetscDeviceType;                     \
  using base_type::cupmDeviceTypeToPetscMemType

} // namespace detail

// A templated C++ struct that defines the entire CUPM interface. Use of templating vs
// preprocessor macros allows us to use both interfaces simultaneously as well as easily
// import them into classes.
template <CUPMDeviceType T> struct CUPMInterface;

#if PetscDefined(HAVE_CUDA)
#define PETSC_CUPM_PREFIX      cuda
#define PETSC_CUPM_PREFIX_U    CUDA
template <>
struct CUPMInterface<PETSC_CUPM_DEVICE_TYPE> : detail::CUPMInterfaceBase<PETSC_CUPM_DEVICE_TYPE>
{
  PETSC_CUPM_BASE_CLASS_HEADER(PETSC_CUPM_DEVICE_TYPE);

  // typedefs
  using cupmError_t        = cudaError_t;
  using cupmEvent_t        = cudaEvent_t;
  using cupmStream_t       = cudaStream_t;
  using cupmBlasHandle_t   = cublasHandle_t;
  using cupmBlasError_t    = cublasStatus_t;
  using cupmSolverHandle_t = cusolverDnHandle_t;
  using cupmSolverError_t  = cusolverStatus_t;
  using cupmDeviceProp_t   = cudaDeviceProp;
  using cupmMemcpyKind_t   = cudaMemcpyKind;
  using cupmDim3           = dim3;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorDeviceAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
#if PETSC_PKG_CUDA_VERSION_GE(11,1,0)
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorStubLibrary);
#else
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorStubLibrary,ErrorInsufficientDriver);
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
  PETSC_CUPM_ALIAS_FUNCTION(LaunchKernel);

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
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize);

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(Free);
  PETSC_CUPM_ALIAS_FUNCTION(Malloc);
#if PETSC_PKG_CUDA_VERSION_GE(11,2,0)
  PETSC_CUPM_ALIAS_FUNCTION(FreeAsync);
  PETSC_CUPM_ALIAS_FUNCTION(MallocAsync);
#else
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(FreeAsync,Free,1);
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync,Malloc,1);
#endif
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy);
  PETSC_CUPM_ALIAS_FUNCTION(MemcpyAsync);
  PETSC_CUPM_ALIAS_FUNCTION(MallocHost);
  PETSC_CUPM_ALIAS_FUNCTION(FreeHost);
};
#undef PETSC_CUPM_PREFIX
#undef PETSC_CUPM_PREFIX_U
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
#define PETSC_CUPM_PREFIX   hip
#define PETSC_CUPM_PREFIX_U HIP
template <>
struct CUPMInterface<PETSC_CUPM_DEVICE_TYPE> : detail::CUPMInterfaceBase<PETSC_CUPM_DEVICE_TYPE>
{
  PETSC_CUPM_BASE_CLASS_HEADER(PETSC_CUPM_DEVICE_TYPE);

  // typedefs
  using cupmError_t        = hipError_t;
  using cupmEvent_t        = hipEvent_t;
  using cupmStream_t       = hipStream_t;
  using cupmSolverHandle_t = hipsolverHandle_t;
  using cupmSolverError_t  = hipsolverStatus_t;
  using cupmDeviceProp_t   = hipDeviceProp_t;
  using cupmMemcpyKind_t   = hipMemcpyKind;

  // values
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(Success);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorNotReady);
  // see https://github.com/ROCm-Developer-Tools/HIP/blob/develop/bin/hipify-perl
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorDeviceAlreadyInUse,ErrorContextAlreadyInUse);
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE(ErrorSetOnActiveProcess);
  // as of HIP v4.2 cudaErrorStubLibrary has no HIP equivalent
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_COMMON(ErrorStubLibrary,ErrorInsufficientDriver);
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
  PETSC_CUPM_ALIAS_FUNCTION(LaunchKernel);

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
  PETSC_CUPM_ALIAS_FUNCTION(DeviceSynchronize);

  // memory management
  PETSC_CUPM_ALIAS_FUNCTION(Free);
  PETSC_CUPM_ALIAS_FUNCTION(Malloc);
  // HIP has no hipFreeAsync
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(FreeAsync,Free,1);
  // HIP has no hipMallocAsync
  PETSC_CUPM_ALIAS_FUNCTION_GOBBLE_COMMON(MallocAsync,Malloc,1);
  PETSC_CUPM_ALIAS_FUNCTION(Memcpy);
  PETSC_CUPM_ALIAS_FUNCTION(MemcpyAsync);
  PETSC_CUPM_ALIAS_FUNCTION(MallocHost);
  PETSC_CUPM_ALIAS_FUNCTION(FreeHost);
};
#undef PETSC_CUPM_PREFIX
#undef PETSC_CUPM_PREFIX_U
#endif // PetscDefined(HAVE_HIP)

#undef PETSC_CUPM_BASE_CLASS_HEADER
#undef PETSC_CUPM_DEVICE_TYPE

} // namespace Impl

} // namespace Petsc

// shorthand for bringing all of the typedefs from the base CUPMInterface class into your own,
// it's annoying that c++ doesn't have a way to do this automatically
#define PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(base_name_,Tp_)     \
  using base_name_ = CUPMInterface<Tp_>;                                \
  /* introspection */                                                   \
  using base_name_::type;                                               \
  using base_name_::cupmName;                                           \
  using base_name_::cupmDeviceTypeToPetscDeviceType;                    \
  using base_name_::cupmDeviceTypeToPetscMemType;                       \
  /* types */                                                           \
  using typename base_name_::cupmError_t;                               \
  using typename base_name_::cupmEvent_t;                               \
  using typename base_name_::cupmStream_t;                              \
  using typename base_name_::cupmSolverError_t;                         \
  using typename base_name_::cupmSolverHandle_t;                        \
  using typename base_name_::cupmDeviceProp_t;                          \
  using typename base_name_::cupmMemcpyKind_t;                          \
  using typename base_name_::dim3;                                      \
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
  using base_name_::cupmLaunchKernel;                                   \
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
  using base_name_::cupmDeviceSynchronize;                              \
  using base_name_::cupmFree;                                           \
  using base_name_::cupmFreeAsync;                                      \
  using base_name_::cupmMalloc;                                         \
  using base_name_::cupmMallocAsync;                                    \
  using base_name_::cupmMemcpy;                                         \
  using base_name_::cupmMemcpyAsync;                                    \
  using base_name_::cupmMallocHost;                                     \
  using base_name_::cupmFreeHost

#endif /* __cplusplus */

#endif /* PETSCCUPMINTERFACE_HPP */
