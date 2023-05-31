#include <petsc/private/cupmblasinterface.hpp>
#include <petsc/private/petscadvancedmacros.h>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

#define PETSC_CUPMBLAS_STATIC_VARIABLE_DEFN(THEIRS, DEVICE, OURS) const decltype(THEIRS) BlasInterfaceImpl<DeviceType::DEVICE>::OURS;

// in case either one or the other don't agree on a name, you can specify all three here:
//
// PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT(CUBLAS_STATUS_SUCCESS, rocblas_status_success,
// CUPMBLAS_STATUS_SUCCESS) ->
// const decltype(CUBLAS_STATUS_SUCCESS)  BlasInterface<DeviceType::CUDA>::CUPMBLAS_STATUS_SUCCESS;
// const decltype(rocblas_status_success) BlasInterface<DeviceType::HIP>::CUPMBLAS_STATUS_SUCCESS;
#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT(CUORIGINAL, HIPORIGINAL, OURS) \
  PetscIfPetscDefined(HAVE_CUDA, PETSC_CUPMBLAS_STATIC_VARIABLE_DEFN, PetscExpandToNothing)(CUORIGINAL, CUDA, OURS) PetscIfPetscDefined(HAVE_HIP, PETSC_CUPMBLAS_STATIC_VARIABLE_DEFN, PetscExpandToNothing)(HIPORIGINAL, HIP, OURS)

// if both cuda and hip agree on the same naming scheme i.e. CUBLAS_STATUS_SUCCESS and
// HIPBLAS_STATUS_SUCCESS:
//
// PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_PREFIX(STATUS_SUCCESS) ->
// const decltype(CUBLAS_STATUS_SUCCESS)  BlasInterface<DeviceType::CUDA>::CUPMBLAS_STATUS_SUCCESS;
// const decltype(HIPBLAS_STATUS_SUCCESS) BlasInterface<DeviceType::HIP>::CUPMBLAS_STATUS_SUCCESS;
#define PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(SUFFIX) PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_EXACT(PetscConcat(CUBLAS_, SUFFIX), PetscConcat(HIPBLAS_, SUFFIX), PetscConcat(CUPMBLAS_, SUFFIX))

PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(STATUS_SUCCESS)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(STATUS_NOT_INITIALIZED)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(STATUS_ALLOC_FAILED)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(POINTER_MODE_HOST)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(POINTER_MODE_DEVICE)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(OP_T)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(OP_N)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(OP_C)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(FILL_MODE_LOWER)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(FILL_MODE_UPPER)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(SIDE_LEFT)
PETSC_CUPMBLAS_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(DIAG_NON_UNIT)

// It would be nice to use PETSC_FUNCTION_ALIAS but I don't think I can do that for an extern "C" function
#define PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_SINGLE(DEV, SUBR, args, args_called) \
  PETSC_INTERN PetscErrorCode PetscDevice##SUBR##_Private_##DEV args \
  { \
    PetscFunctionBegin; \
    PetscCall(BlasInterface<DeviceType ::DEV>::SUBR##_Private args_called); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

#if PetscDefined(HAVE_CUDA)
  #define PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_CUDA(SUBR, args, args_called) PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_SINGLE(CUDA, SUBR, args, args_called)
#else
  #define PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_CUDA(SUBR, args, args_called)
#endif

#if PetscDefined(HAVE_HIP)
  #define PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_HIP(SUBR, args, args_called) PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_SINGLE(HIP, SUBR, args, args_called)
#else
  #define PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_HIP(SUBR, args, args_called)
#endif

#define PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC(SUBR, args, args_called) \
  PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_CUDA(SUBR, args, args_called) \
  PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC_HIP(SUBR, args, args_called)

#if PetscDefined(HAVE_CUDA)
template struct BlasInterface<DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
template struct BlasInterface<DeviceType::HIP>;
#endif

// Ugly passing the declaration and usage of the arguments separately, but again I don't think I can use the nice functional features
PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC(GEMM, (PetscDeviceContext dctx, PetscMemType memtype_scalar, char trans_A, char trans_B, PetscInt m, PetscInt n, PetscInt k, const PetscScalar *alpha, const PetscScalar A[], PetscInt lda, const PetscScalar B[], PetscInt ldb, const PetscScalar *beta, PetscScalar C[], PetscInt ldc), (dctx, memtype_scalar, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc))
PETSC_CUPMBLAS_DEFINE_DEVICE_BLAS_FUNC(GEMV, (PetscDeviceContext dctx, PetscMemType memtype_scalar, char trans, PetscInt m, PetscInt n, const PetscScalar *alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscInt inc_x, const PetscScalar *beta, PetscScalar y[], PetscInt inc_y), (dctx, memtype_scalar, trans, m, n, alpha, A, lda, x, inc_x, beta, y, inc_y))

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
