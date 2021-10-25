#ifndef PETSCCUPMBLASINTERFACE_HPP
#define PETSCCUPMBLASINTERFACE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>

#if defined(__cplusplus)

#if !PetscDefined(HAVE_CXX_DIALECT_CXX11) || (__cplusplus < 201103L)
#  error CUPMBlasInterface requires c++11
#endif

#include <array>

namespace Petsc
{

namespace detail
{

static constexpr std::array<const char*const,5> CUPMBlasTypes = {
  "cuBLAS",
  "hipBLAS",
  "CUPMBlasType",
  "CUPMBlasType::",
  nullptr
};

} // namespace detail

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
#    define PETSC_CUPMBLAS_FP_TYPE C
#  elif PetscDefined(USE_REAL_DOUBLE)
#    define PETSC_CUPMBLAS_FP_TYPE Z
#  endif
#else
#  if PetscDefined(USE_REAL_SINGLE)
#    define PETSC_CUPMBLAS_FP_TYPE S
#  elif PetscDefined(USE_REAL_DOUBLE)
#    define PETSC_CUPMBLAS_FP_TYPE D
#  endif
#endif // USE_COMPLEX

#if !defined(PETSC_CUPMBLAS_FP_TYPE)
#  error Unsupported CUPM Blas floating-point type
#endif

#define CAT_(x,y) x ## y
#define CAT(x,y)  CAT_(x,y)

#define PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE_EXACT(CUPMPREFIX,MAPPED,PREFIX,ORIGINAL) \
  PETSC_CUPM_ALIAS_INTEGRAL_VALUE_EXACT(CUPMPREFIX,MAPPED,PREFIX,ORIGINAL)

#define PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE(COMMON)                     \
  PETSC_CUPMBLAS_ALIAS_INTEGRAL_VALUE_EXACT(CUPMBLAS,COMMON,PETSC_CUPMBLAS_STEM_U,COMMON)

#define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(cupmprefix,prefix,blasfptype,suffix) \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(CAT(cupmprefix,X),CAT(prefix,blasfptype),suffix)

#define PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(suffix)                      \
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION_EXACT(cupmBlas,PETSC_CUPMBLAS_STEM,PETSC_CUPMBLAS_FP_TYPE,suffix)

#define PETSC_CUPMBLAS_ALIAS_FUNCTION(suffix)                           \
  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupmBlas,PETSC_CUPMBLAS_STEM,suffix)

template <CUPMDeviceType T> struct CUPMBlasInterface;

#if PetscDefined(HAVE_CUDA)
#define PETSC_CUPMBLAS_STEM   cublas
#define PETSC_CUPMBLAS_STEM_U CUBLAS
template <>
struct CUPMBlasInterface<CUPMDeviceType::CUDA> : CUPMInterface<CUPMDeviceType::CUDA>
{
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(cupmInterface_t,CUPMDeviceType::CUDA);

  PETSC_NODISCARD static constexpr const char* cupmBlasName() noexcept
  { return std::get<static_cast<int>(type)>(detail::CUPMBlasTypes); }

  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupmBlas,PetscCUBLAS,GetErrorName);

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

  // BLAS level 1
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(axpy);
  PETSC_CUPMBLAS_ALIAS_BLAS_FUNCTION(scal);

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
#undef PETSC_CUPMBLAS_STEM
#undef PETSC_CUPMBLAS_STEM_U
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
#define PETSC_CUPMBLAS_STEM   hipblas
#define PETSC_CUPMBLAS_STEM_U HIPBLAS
template <>
struct CUPMBlasInterface<CUPMDeviceType::HIP> : CUPMInterface<CUPMDeviceType::HIP>
{
  PETSC_INHERIT_CUPM_INTERFACE_TYPEDEFS_USING(cupmInterface_t,CUPMDeviceType::HIP);

  PETSC_NODISCARD static constexpr const char* cupmBlasName() noexcept
  { return std::get<static_cast<int>(type)>(detail::CUPMBlasTypes); }

  PETSC_CUPM_ALIAS_FUNCTION_EXACT(cupmBlas,PetscHIPBLAS,GetErrorName);

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

  // BLAS level 1
  PETSC_CUPMBLAS_ALIAS_FUNCTION(axpy);
  PETSC_CUPMBLAS_ALIAS_FUNCTION(scal);

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
#undef PETSC_CUPMBLAS_STEM
#undef PETSC_CUPMBLAS_STEM_U
#endif // PetscDefined(HAVE_HIP)

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
  /* Level 1 BLAS */                                                    \
  using base_name::cupmBlasXaxpy;                                       \
  using base_name::cupmBlasXscal

#define PETSC_INHERIT_CUPMBLAS_INTERFACE_TYPEDEFS_USING(base_name,Tp)   \
  PETSC_INHERIT_CUPMBLAS_INTERFACE_TYPEDEFS_USING_(base_name,Tp)

} // namespace Petsc

#endif /* defined(__cplusplus) */

#endif /* PETSCCUPMBLASINTERFACE_HPP */
