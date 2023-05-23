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

#if PetscDefined(HAVE_CUDA)
template struct BlasInterface<DeviceType::CUDA>;

PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private_Cuda(PetscDeviceContext dctx, PetscMemType memtype_scalar, char trans_A, char trans_B, PetscInt m, PetscInt n, PetscInt k, const PetscScalar *alpha, const PetscScalar A[], PetscInt lda, const PetscScalar B[], PetscInt ldb, const PetscScalar *beta, PetscScalar C[], PetscInt ldc)
{
  PetscFunctionBegin;
  PetscCall(BlasInterface<DeviceType::CUDA>::GEMM_Private(dctx, memtype_scalar, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#if PetscDefined(HAVE_HIP)
template struct BlasInterface<DeviceType::HIP>;

PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private_HIP(PetscDeviceContext dctx, PetscMemType memtype_scalar, char trans_A, char trans_B, PetscInt m, PetscInt n, PetscInt k, const PetscScalar *alpha, const PetscScalar A[], PetscInt lda, const PetscScalar B[], PetscInt ldb, const PetscScalar *beta, PetscScalar C[], PetscInt ldc)
{
  PetscFunctionBegin;
  PetscCall(BlasInterface<DeviceType::HIP>::GEMM_Private(dctx, memtype_scalar, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
