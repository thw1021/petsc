#ifndef PETSCCUPMBLASINTERFACE_HPP
#define PETSCCUPMBLASINTERFACE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>

#if defined(__cplusplus)

#if !PetscDefined(HAVE_CXX_DIALECT_CXX11) || (__cplusplus < 201103L)
#  error "CUPMBlasInterface requires C++11"
#endif

namespace Petsc
{

namespace Impl
{

namespace detail
{

#define CHKERRCUPMBLAS(cberr) do {                                      \
    const cupmBlasError_t cberr__ = cberr;                              \
    if (PetscUnlikely(cberr__ != CUPMBLAS_STATUS_SUCCESS)) {            \
      if (((cberr__ == CUPMBLAS_STATUS_NOT_INITIALIZED) ||              \
           (cberr__ == CUPMBLAS_STATUS_ALLOC_FAILED))   &&              \
          PetscDeviceInitialized(cupmDeviceTypeToPetscDeviceType())) {  \
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                \
                 "%s error %d (%s). "                                   \
                 "Reports not initialized or alloc failed; "            \
                 "this indicates the GPU may have run out resources",   \
                 cupmBlasName(),static_cast<PetscErrorCode>(cberr__),   \
                 cupmBlasGetErrorName(cberr__));                        \
      } else {                                                          \
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s)",      \
                 cupmBlasName(),static_cast<PetscErrorCode>(cberr__),   \
                 cupmBlasGetErrorName(cberr__));                        \
      }                                                                 \
    }                                                                   \
  } while (0)

// given cupmBlas<T>axpy() then
// T = PETSC_CUPBLAS_FP_TYPE
// given cupmBlas<T><u>nrm2() then
// T = PETSC_CUPMBLAS_FP_INPUT_TYPE
// u = PETSC_CUPMBLAS_FP_RETURN_TYPE
#if PetscDefined(USE_COMPLEX)
#  if PetscDefined(USE_REAL_SINGLE)
#    define PETSC_CUPMBLAS_FP_TYPE_U        C
#    define PETSC_CUPMBLAS_FP_TYPE_L        c
#    define PETSC_CUPMBLAS_FP_INPUT_TYPE_U  S
#    define PETSC_CUPMBLAS_FP_INPUT_TYPE_L  s
#  elif PetscDefined(USE_REAL_DOUBLE)
#    define PETSC_CUPMBLAS_FP_TYPE_U        Z
#    define PETSC_CUPMBLAS_FP_TYPE_L        z
#    define PETSC_CUPMBLAS_FP_INPUT_TYPE_U  D
#    define PETSC_CUPMBLAS_FP_INPUT_TYPE_L  d
#  endif
#  define PETSC_CUPMBLAS_FP_RETURN_TYPE_U PETSC_CUPMBLAS_FP_TYPE_U
#  define PETSC_CUPMBLAS_FP_RETURN_TYPE_L PETSC_CUPMBLAS_FP_TYPE_L
#else
#  if PetscDefined(USE_REAL_SINGLE)
#    define PETSC_CUPMBLAS_FP_TYPE_U S
#    define PETSC_CUPMBLAS_FP_TYPE_L s
#  elif PetscDefined(USE_REAL_DOUBLE)
#    define PETSC_CUPMBLAS_FP_TYPE_U D
#    define PETSC_CUPMBLAS_FP_TYPE_L d
#  endif
#  define PETSC_CUPMBLAS_FP_INPUT_TYPE_U  PETSC_CUPMBLAS_FP_TYPE_U
#  define PETSC_CUPMBLAS_FP_INPUT_TYPE_L  PETSC_CUPMBLAS_FP_TYPE_L
#  define PETSC_CUPMBLAS_FP_RETURN_TYPE_U
#  define PETSC_CUPMBLAS_FP_RETURN_TYPE_L
#endif // USE_COMPLEX

#if !defined(PETSC_CUPMBLAS_FP_TYPE_U)
#  error "Unsupported CUPM Blas floating-point type"
#endif

#define PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE_EXACT(OUR_PREFIX,OUR_SUFFIX,THEIR_PREFIX,THEIR_SUFFIX) \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(OUR_PREFIX,OUR_SUFFIX,THEIR_PREFIX,THEIR_SUFFIX)

#define PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(COMMON)                     \
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE_EXACT(CUPMBLAS,COMMON,PETSC_CUPMBLAS_PREFIX_U,COMMON)

#define PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_MODIFIED(func)         \
  PETSC_CONCAT(PETSC_CONCAT(PETSC_CUPMBLAS_FP_INPUT_TYPE,PETSC_CUPMBLAS_FP_RETURN_TYPE),func)

#define PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_IFPTYPE(func)          \
  PETSC_CONCAT(I,PETSC_CONCAT(PETSC_CUPMBLAS_FP_TYPE_L,func))

#define PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_STANDARD(func) \
  PETSC_CONCAT(PETSC_CUPMBLAS_FP_TYPE,func)

#define PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS(MACRO_SUFFIX)          \
  PETSC_CONCAT(PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS_,MACRO_SUFFIX)

#define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(MACRO_SUFFIX,our_suffix,their_suffix) \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(                                      \
    cupmBlasX,                                                          \
    our_suffix,                                                         \
    PETSC_CUPMBLAS_PREFIX,                                              \
    PETSC_CUPMBLAS_BUILD_BLAS_FUNCTION_ALIAS(MACRO_SUFFIX)(their_suffix) \
  )

#define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MACRO_SUFFIX,suffix) \
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(MACRO_SUFFIX,suffix,suffix)

#define PETSC_CUPMBLAS_ALIAS_FUNCTION(suffix)                           \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupmBlas,suffix,PETSC_CUPMBLAS_PREFIX,suffix)

template <CUPMDeviceType T>
struct CUPMBlasInterfaceBase : CUPMInterface<T>
{
  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 const char* cupmBlasName())
  {
    switch (T) {
    case CUPMDeviceType::CUDA: return "cuBLAS";
    case CUPMDeviceType::HIP:  return "hipBLAS";
    }
  }
};

#define PETSC_CUPMBLAS_BASE_CLASS_HEADER(DEV_TYPE)                      \
  using base_type = detail::CUPMBlasInterfaceBase<DEV_TYPE>;            \
  using base_type::cupmBlasName;                                        \
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(interface_type,DEV_TYPE); \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupmBlas,GetErrorName,PETSC_CONCAT(Petsc,PETSC_CUPMBLAS_PREFIX_U),GetErrorName)

} // namespace detail

template <CUPMDeviceType T> struct CUPMBlasInterface;

#if PetscDefined(HAVE_CUDA)
#define PETSC_CUPMBLAS_PREFIX         cublas
#define PETSC_CUPMBLAS_PREFIX_U       CUBLAS
#define PETSC_CUPMBLAS_FP_TYPE        PETSC_CUPMBLAS_FP_TYPE_U
#define PETSC_CUPMBLAS_FP_INPUT_TYPE  PETSC_CUPMBLAS_FP_INPUT_TYPE_U
#define PETSC_CUPMBLAS_FP_RETURN_TYPE PETSC_CUPMBLAS_FP_RETURN_TYPE_L
template <>
struct CUPMBlasInterface<CUPMDeviceType::CUDA>
  : detail::CUPMBlasInterfaceBase<CUPMDeviceType::CUDA>
{
  PETSC_CUPMBLAS_BASE_CLASS_HEADER(CUPMDeviceType::CUDA);

  // typedefs
  using cupmBlasHandle_t   = cublasHandle_t;
  using cupmBlasError_t    = cublasStatus_t;
  using cupmBlasInt_t      = int;
  using cupmSolverHandle_t = cusolverDnHandle_t;
  using cupmSolverError_t  = cusolverStatus_t;

  // values
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(_STATUS_SUCCESS);
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(_STATUS_NOT_INITIALIZED);
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(_STATUS_ALLOC_FAILED);

  // utility functions
  PETSC_CUPMBLAS_ALIAS_FUNCTION(Create);
  PETSC_CUPMBLAS_ALIAS_FUNCTION(Destroy);
  PETSC_CUPMBLAS_ALIAS_FUNCTION(GetStream);
  PETSC_CUPMBLAS_ALIAS_FUNCTION(SetStream);

  // level 1 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,axpy);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,scal);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD,dot,PETSC_IF_PETSC_DEFINED(USE_COMPLEX,dotc,dot));
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD,dotu,PETSC_IF_PETSC_DEFINED(USE_COMPLEX,dotu,dot));
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,swap);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MODIFIED,nrm2);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(IFPTYPE,amax);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MODIFIED,asum);

  // level 2 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,gemv);

  // level 3 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,gemm);

  // BLAS extensions
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,geam);

  PETSC_CXX_COMPAT_DECL(PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle))
  {
    PetscFunctionBegin;
    if (handle) PetscFunctionReturn(0);
    for (auto i = 0; i < 3; ++i) {
      const auto cerr = cusolverDnCreate(&handle);
      if (PetscLikely(cerr == CUSOLVER_STATUS_SUCCESS)) break;
      if ((cerr != CUSOLVER_STATUS_NOT_INITIALIZED) && (cerr != CUSOLVER_STATUS_ALLOC_FAILED)) CHKERRCUSOLVER(cerr);
      if (i < 2) {
        auto ierr = PetscSleep(3);CHKERRQ(ierr);
        continue;
      }
      if (PetscUnlikely(cerr != CUSOLVER_STATUS_SUCCESS)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize cuSolverDn");
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream))
  {
    cupmStream_t      cupmStream;
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    cerr = cusolverDnGetStream(handle,&cupmStream);CHKERRCUSOLVER(cerr);
    if (cupmStream != stream) {cerr = cusolverDnSetStream(handle,stream);CHKERRCUSOLVER(cerr);}
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle))
  {
    PetscFunctionBegin;
    if (handle) {
      auto cerr = cusolverDnDestroy(handle);CHKERRCUSOLVER(cerr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }
};
#undef PETSC_CUPMBLAS_PREFIX
#undef PETSC_CUPMBLAS_PREFIX_U
#undef PETSC_CUPMBLAS_FP_TYPE
#undef PETSC_CUPMBLAS_FP_INPUT_TYPE
#undef PETSC_CUPMBLAS_FP_RETURN_TYPE
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
#define PETSC_CUPMBLAS_PREFIX         hipblas
#define PETSC_CUPMBLAS_PREFIX_U       HIPBLAS
#define PETSC_CUPMBLAS_FP_TYPE        PETSC_CUPMBLAS_FP_TYPE_U
#define PETSC_CUPMBLAS_FP_INPUT_TYPE  PETSC_CUPMBLAS_FP_INPUT_TYPE_U
#define PETSC_CUPMBLAS_FP_RETURN_TYPE PETSC_CUPMBLAS_FP_RETURN_TYPE_L
template <>
struct CUPMBlasInterface<CUPMDeviceType::HIP> : detail::CUPMBlasInterfaceBase<CUPMDeviceType::HIP>
{
  PETSC_CUPMBLAS_BASE_CLASS_HEADER(CUPMDeviceType::HIP);

  // typedefs
  using cupmBlasHandle_t   = hipblasHandle_t;
  using cupmBlasError_t    = hipblasStatus_t;
  using cupmBlasInt_t      = int; // rocblas will have its own
  using cupmSolverHandle_t = hipsolverHandle_t;
  using cupmSolverError_t  = hipsolverStatus_t;

  // values
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(_STATUS_SUCCESS);
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(_STATUS_NOT_INITIALIZED);
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(_STATUS_ALLOC_FAILED);

  // utility functions
  PETSC_CUPMBLAS_ALIAS_FUNCTION(Create);
  PETSC_CUPMBLAS_ALIAS_FUNCTION(Destroy);
  PETSC_CUPMBLAS_ALIAS_FUNCTION(GetStream);
  PETSC_CUPMBLAS_ALIAS_FUNCTION(SetStream);

  // level 1 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,axpy);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,scal);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD,dot,PETSC_IF_PETSC_DEFINED(USE_COMPLEX,dotc,dot));
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(STANDARD,dotu,PETSC_IF_PETSC_DEFINED(USE_COMPLEX,dotu,dot));
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,swap);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MODIFIED,nrm2);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(IFPTYPE,amax);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(MODIFIED,asum);

  // level 2 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,gemv);

  // level 3 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,gemm);

  // BLAS extensions
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(STANDARD,geam);

  PETSC_CXX_COMPAT_DECL(PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle))
  {
    PetscFunctionBegin;
    if (!handle) {auto cerr = hipsolverCreate(&handle);CHKERRHIPSOLVER(cerr);}
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream))
  {
    cupmStream_t      cupmStream;
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    cerr = hipsolverGetStream(handle,&cupmStream);CHKERRHIPSOLVER(cerr);
    if (cupmStream != stream) {cerr = hipsolverSetStream(handle,stream);CHKERRHIPSOLVER(cerr);}
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode DestroyHandle(cupmSolverHandle_t &handle))
  {
    PetscFunctionBegin;
    if (handle) {
      auto cerr = hipsolverDestroy(handle);CHKERRHIPSOLVER(cerr);
      handle = nullptr;
    }
    PetscFunctionReturn(0);
  }
};
#undef PETSC_CUPMBLAS_PREFIX
#undef PETSC_CUPMBLAS_PREFIX_U
#undef PETSC_CUPMBLAS_FP_TYPE
#undef PETSC_CUPMBLAS_FP_INPUT_TYPE
#undef PETSC_CUPMBLAS_FP_RETURN_TYPE
#endif // PetscDefined(HAVE_HIP)

#undef PETSC_CUPMBLAS_BASE_CLASS_HEADER

} // namespace Impl

} // namespace Petsc

#define PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(base_name,Tp)   \
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(cupmInterface_t,Tp);      \
  using base_name = Petsc::Impl::CUPMBlasInterface<Tp>;                 \
  /* introspection */                                                   \
  using base_name::cupmBlasName;                                        \
  using base_name::cupmBlasGetErrorName;                                \
  /* types */                                                           \
  using typename base_name::cupmBlasHandle_t;                           \
  using typename base_name::cupmBlasError_t;                            \
  using typename base_name::cupmBlasInt_t;                              \
  using typename base_name::cupmSolverHandle_t;                         \
  using typename base_name::cupmSolverError_t;                          \
  /* values */                                                          \
  using base_name::CUPMBLAS_STATUS_SUCCESS;                             \
  using base_name::CUPMBLAS_STATUS_NOT_INITIALIZED;                     \
  using base_name::CUPMBLAS_STATUS_ALLOC_FAILED;                        \
  /* utility functions */                                               \
  using base_name::cupmBlasCreate;                                      \
  using base_name::cupmBlasDestroy;                                     \
  using base_name::cupmBlasGetStream;                                   \
  using base_name::cupmBlasSetStream;                                   \
  /* level 1 BLAS */                                                    \
  using base_name::cupmBlasXaxpy;                                       \
  using base_name::cupmBlasXscal;                                       \
  using base_name::cupmBlasXdot;                                        \
  using base_name::cupmBlasXdotu;                                       \
  using base_name::cupmBlasXswap;                                       \
  using base_name::cupmBlasXnrm2;                                       \
  using base_name::cupmBlasXamax;                                       \
  using base_name::cupmBlasXasum;                                       \
  /* level 2 BLAS */                                                    \
  using base_name::cupmBlasXgemv;                                       \
  /* level 3 BLAS */                                                    \
  using base_name::cupmBlasXgemm;                                       \
  /* BLAS extensions */                                                 \
  using base_name::cupmBlasXgeam

#endif // defined(__cplusplus)

#endif // PETSCCUPMBLASINTERFACE_HPP
