#ifndef PETSCCUPMBLASINTERFACE_HPP
#define PETSCCUPMBLASINTERFACE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>

#if defined(__cplusplus)

#if !PetscDefined(HAVE_CXX_DIALECT_CXX11) || (__cplusplus < 201103L)
#  error "CUPMBlasInterface requires c++11"
#endif

#include <array>

namespace Petsc
{

static constexpr std::array<const char*const,5> CUPMBlasTypes = {
  "cuBLAS",
  "hipBLAS",
  "CUPMBlasType",
  "CUPMBlasType::",
  nullptr
};

namespace detail
{

#define CHKERRCUPMBLAS(cberr) do {                                      \
    cupmBlasError_t _cberr__ = cberr;                                   \
    if (PetscUnlikely(_cberr__ != CUPMBLAS_STATUS_SUCCESS)) {           \
      if (((_cberr__ == CUPMBLAS_STATUS_NOT_INITIALIZED) ||             \
           (_cberr__ == CUPMBLAS_STATUS_ALLOC_FAILED))   &&             \
          PetscDeviceInitialized(cupmDeviceTypeToPetsc())) {            \
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,                \
                 "%s error %d (%s). "                                   \
                 "Reports not initialized or alloc failed; "            \
                 "this indicates the GPU may have run out resources",   \
                 cupmBlasName(),static_cast<PetscErrorCode>(_cberr__),  \
                 cupmBlasGetErrorName(_cberr__));                       \
      } else {                                                          \
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"%s error %d (%s)",      \
                 cupmBlasName(),static_cast<PetscErrorCode>(_cberr__),  \
                 cupmBlasGetErrorName(_cberr__));                       \
      }                                                                 \
    }                                                                   \
  } while (0)

#if PetscDefined(USE_COMPLEX)
#  if PetscDefined(USE_REAL_SINGLE)
#    define PETSC_CUPMBLAS_FP_TYPE        C
#    define PETSC_CUPMBLAS_FP_RETURN_TYPE c
#  elif PetscDefined(USE_REAL_DOUBLE)
#    define PETSC_CUPMBLAS_FP_TYPE        Z
#    define PETSC_CUPMBLAS_FP_RETURN_TYPE z
#  endif
#else
#  if PetscDefined(USE_REAL_SINGLE)
#    define PETSC_CUPMBLAS_FP_TYPE S
#  elif PetscDefined(USE_REAL_DOUBLE)
#    define PETSC_CUPMBLAS_FP_TYPE D
#  endif
#  define PETSC_CUPMBLAS_FP_RETURN_TYPE
#endif // USE_COMPLEX

#if !defined(PETSC_CUPMBLAS_FP_TYPE)
#  error Unsupported CUPM Blas floating-point type
#endif

#define PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE_EXACT(OUR_PREFIX,OUR_SUFFIX,THEIR_PREFIX,THEIR_SUFFIX) \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(OUR_PREFIX,OUR_SUFFIX,THEIR_PREFIX,THEIR_SUFFIX)

#define PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(COMMON)                     \
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE_EXACT(CUPMBLAS,COMMON,PETSC_CUPMBLAS_PREFIX_U,COMMON)

#define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(our_prefix,their_prefix,FPTYPE,suffix) \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(CAT(our_prefix,X),suffix,CAT(their_prefix,FPTYPE),suffix)

#define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_WITH_RETURN_TYPE(suffix)     \
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(cupmBlas,PETSC_CUPMBLAS_PREFIX,CAT(PETSC_CUPMBLAS_FP_TYPE,PETSC_CUPMBLAS_FP_RETURN_TYPE),suffix)

#define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(suffix)                      \
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(cupmBlas,PETSC_CUPMBLAS_PREFIX,PETSC_CUPMBLAS_FP_TYPE,suffix)

#define PETSC_CUPMBLAS_ALIAS_FUNCTION(suffix)                           \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupmBlas,suffix,PETSC_CUPMBLAS_PREFIX,suffix)

template <CUPMDeviceType T>
struct CUPMBlasInterfaceBase : Impl::CUPMInterface<T>
{
  PETSC_NODISCARD static constexpr const char* cupmBlasName() noexcept
  {
    return std::get<util::integral_value(T)>(CUPMBlasTypes);
  }
};

#define PETSC_CUPMBLAS_BASE_CLASS_HEADER_(DEV_TYPE)                     \
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(interface_type,DEV_TYPE); \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupmBlas,GetErrorName,                \
                                  CAT(Petsc,PETSC_CUPMBLAS_PREFIX_U),   \
                                  GetErrorName);                        \
  using base_type = detail::CUPMBlasInterfaceBase<DEV_TYPE>;            \
  using base_type::cupmBlasName

#define PETSC_CUPMBLAS_BASE_CLASS_HEADER(DEV_TYPE) PETSC_CUPMBLAS_BASE_CLASS_HEADER_(DEV_TYPE)

} // namespace detail

namespace Impl
{

template <CUPMDeviceType T> struct CUPMBlasInterface;

#if PetscDefined(HAVE_CUDA)
#define PETSC_CUPMBLAS_PREFIX      cublas
#define PETSC_CUPMBLAS_PREFIX_U    CUBLAS
#define PETSC_CUPMBLAS_DEVICE_TYPE CUPMDeviceType::CUDA
template <>
struct CUPMBlasInterface<PETSC_CUPMBLAS_DEVICE_TYPE>
  : detail::CUPMBlasInterfaceBase<PETSC_CUPMBLAS_DEVICE_TYPE>
{
  PETSC_CUPMBLAS_BASE_CLASS_HEADER(PETSC_CUPMBLAS_DEVICE_TYPE);

  // typedefs
  using cupmBlasHandle_t = cublasHandle_t;
  using cupmBlasError_t  = cublasStatus_t;

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
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(axpy);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(scal);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(dot);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(swap);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_WITH_RETURN_TYPE(nrm2);

  // level 2 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(gemv);

  // level 3 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(gemm);

  // BLAS extensions
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(geam);

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

  PETSC_NODISCARD static PetscErrorCode SetHandleStream(cupmSolverHandle_t &handle, cupmStream_t &stream) noexcept
  {
    cupmStream_t      cupmStream;
    cupmSolverError_t cerr;

    PetscFunctionBegin;
    cerr = cusolverDnGetStream(handle,&cupmStream);CHKERRCUSOLVER(cerr);
    if (cupmStream != stream) {cerr = cusolverDnSetStream(handle,stream);CHKERRCUSOLVER(cerr);}
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
#undef PETSC_CUPMBLAS_PREFIX
#undef PETSC_CUPMBLAS_PREFIX_U
#undef PETSC_CUPMBLAS_DEVICE_TYPE
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
#define PETSC_CUPMBLAS_PREFIX      hipblas
#define PETSC_CUPMBLAS_PREFIX_U    HIPBLAS
#define PETSC_CUPMBLAS_DEVICE_TYPE CUPMDeviceType::HIP
template <>
struct CUPMBlasInterface<PETSC_CUPMBLAS_DEVICE_TYPE>
  : detail::CUPMBlasInterfaceBase<PETSC_CUPMBLAS_DEVICE_TYPE>
{
  PETSC_CUPMBLAS_BASE_CLASS_HEADER(PETSC_CUPMBLAS_DEVICE_TYPE);

  // typedefs
  using cupmBlasHandle_t = hipblasHandle_t;
  using cupmBlasError_t  = hipblasStatus_t;

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
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(axpy);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(scal);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(dot);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(swap);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_WITH_RETURN_TYPE(nrm2);

  // level 2 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(gemv);

  // level 3 BLAS
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(gemm);

  // BLAS extensions
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(geam);

  PETSC_NODISCARD static PetscErrorCode InitializeHandle(cupmSolverHandle_t &handle) noexcept
  {
    PetscFunctionBegin;
    if (!handle) {cupmSolverError_t cerr = hipsolverCreate(&handle);CHKERRHIPSOLVER(cerr);}
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
#undef PETSC_CUPMBLAS_PREFIX
#undef PETSC_CUPMBLAS_PREFIX_U
#undef PETSC_CUPMBLAS_DEVICE_TYPE
#endif // PetscDefined(HAVE_HIP)

#undef PETSC_CUPMBLAS_BASE_CLASS_HEADER_
#undef PETSC_CUPMBLAS_BASE_CLASS_HEADER

} // namespace Impl

} // namespace Petsc

#define PETSC_INHERIT_CUPMBLAS_INTERFACE_TYPEDEFS_USING_(base_name,Tp)  \
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(cupmInterface_t,Tp);      \
  using base_name = CUPMBlasInterface<Tp>;                              \
  /* introspection */                                                   \
  using base_name::cupmBlasName;                                        \
  using base_name::cupmBlasGetErrorName;                                \
  /* types */                                                           \
  using typename base_name::cupmBlasHandle_t;                           \
  using typename base_name::cupmBlasError_t;                            \
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
  using base_name::cupmBlasXswap;                                       \
  using base_name::cupmBlasXnrm2;                                       \
  /* level 2 BLAS */                                                    \
  using base_name::cupmBlasXgemv;                                       \
  /* level 3 BLAS */                                                    \
  using base_name::cupmBlasXgemm;                                       \
  /* BLAS extensions */                                                 \
  using base_name::cupmBlasXgeam

#define PETSC_INHERIT_CUPMBLAS_INTERFACE_TYPEDEFS_USING(base_name,Tp)   \
  PETSC_INHERIT_CUPMBLAS_INTERFACE_TYPEDEFS_USING_(base_name,Tp)

#endif // defined(__cplusplus)

#endif // PETSCCUPMBLASINTERFACE_HPP
