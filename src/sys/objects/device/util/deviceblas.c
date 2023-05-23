
#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/
#include <petscblaslapack.h>
#include <petsc/private/petsclegacycupmblas.h>
#include <petsc/private/deviceblas.h>

#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private_Cuda(PetscDeviceContext, PetscMemType, char, char, PetscInt, PetscInt, PetscInt, const PetscScalar *, const PetscScalar[], PetscInt, const PetscScalar[], PetscInt, const PetscScalar *, PetscScalar[], PetscInt);
#endif

#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private_HIP(PetscDeviceContext, PetscMemType, char, char, PetscInt, PetscInt, PetscInt, const PetscScalar *, const PetscScalar[], PetscInt, const PetscScalar[], PetscInt, const PetscScalar *, PetscScalar[], PetscInt);
#endif

PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private(PetscDeviceContext dctx, PetscMemType memtype_arrays, PetscMemType memtype_scalars, char trans_A, char trans_B, PetscInt m, PetscInt n, PetscInt k, const PetscScalar *alpha, const PetscScalar A[], PetscInt ld_A, const PetscScalar B[], PetscInt ld_B, const PetscScalar *beta, PetscScalar C[], PetscInt ld_C)
{
  PetscDeviceType device_type;
  PetscFunctionBegin;
  if (PetscMemTypeHost(memtype_arrays)) {
    PetscBLASInt   _m, _n, _k, _lda, _ldb, _ldc;
    PetscLogDouble flops;

    PetscCheck(PetscMemTypeHost(memtype_scalars), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Scalar references must be on the host for linear algebra on the host");
    flops = 2.0 * m * n * k + (*beta == 0.0 ? -1.0 : 1.0) * m * n + (*alpha == 1.0 ? 0.0 : 1.0) * PetscMin(m * n, PetscMin(m * k, n * k));
    PetscCall(PetscLogGpuFlops(flops));
    PetscCall(PetscBLASIntCast(m, &_m));
    PetscCall(PetscBLASIntCast(n, &_n));
    PetscCall(PetscBLASIntCast(k, &_k));
    PetscCall(PetscBLASIntCast(ld_A, &_lda));
    PetscCall(PetscBLASIntCast(ld_B, &_ldb));
    PetscCall(PetscBLASIntCast(ld_C, &_ldc));
    PetscCallBLAS("BLASgemm", BLASgemm_(&trans_A, &trans_B, &_m, &_n, &_k, alpha, A, &_lda, B, &_ldb, beta, C, &_ldc));
    PetscCall(PetscLogFlops(flops));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!dctx) PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
  if (device_type == PETSC_DEVICE_CUDA) {
    PetscCheck(PetscMemTypeCUDA(memtype_arrays), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incompatible array for device");
#if PetscDefined(HAVE_CUDA)
    PetscCall(PetscDeviceGEMM_Private_Cuda(dctx, memtype_scalars, trans_A, trans_B, m, n, k, alpha, A, ld_A, B, ld_B, beta, C, ld_C));
#endif
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (device_type == PETSC_DEVICE_HIP) {
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Could not dispatch GEMM for device");
  PetscFunctionReturn(PETSC_SUCCESS);
}
