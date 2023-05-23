
#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/
#include <petscblaslapack.h>
#include <petsc/private/petsclegacycupmblas.h>
#include <petsc/private/deviceblas.h>

PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private(PetscDeviceContext dctx, PetscMemType memtype_arrays, PetscMemType memtype_scalars, char trans_A, char trans_B, PetscInt m, PetscInt n, PetscInt k, const PetscScalar *alpha, const PetscScalar A[], PetscInt ld_A, const PetscScalar B[], PetscInt ld_B, const PetscScalar *beta, PetscScalar C[], PetscInt ld_C)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(memtype_arrays)) {
    PetscBLASInt _m, _n, _k, _lda, _ldb, _ldc;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}
