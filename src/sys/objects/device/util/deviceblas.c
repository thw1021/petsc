
#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/
#include <petscblaslapack.h>
#include <petsc/private/petsclegacycupmblas.h>
#include <petsc/private/deviceblas.h>

#define PETSC_DEVICE_BLAS_DISPATCH_SINGLE(DEV, SUBR, device_type, memtype_arrays, args) \
  if (device_type == PETSC_DEVICE_##DEV) { \
    PetscCheck(PetscMemType##DEV(memtype_arrays), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incompatible array for device"); \
    PetscCall(PetscDevice##SUBR##_Private_##DEV args); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

#define PETSC_DEVICE_BLAS_DECLARE_CUPM_PROTOTYPES(DEV) \
  PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private_##DEV(PetscDeviceContext, PetscMemType, char, char, PetscInt, PetscInt, PetscInt, const PetscScalar *, const PetscScalar[], PetscInt, const PetscScalar[], PetscInt, const PetscScalar *, PetscScalar[], PetscInt); \
  PETSC_INTERN PetscErrorCode PetscDeviceGEMV_Private_##DEV(PetscDeviceContext, PetscMemType, char, PetscInt, PetscInt, const PetscScalar *, const PetscScalar[], PetscInt, const PetscScalar[], PetscInt, const PetscScalar *, PetscScalar[], PetscInt);

#if PetscDefined(HAVE_CUDA)
PETSC_DEVICE_BLAS_DECLARE_CUPM_PROTOTYPES(CUDA)
  #define PETSC_DEVICE_BLAS_DISPATCH_CUDA(SUBR, device_type, memtype_arrays, args) PETSC_DEVICE_BLAS_DISPATCH_SINGLE(CUDA, SUBR, device_type, memtype_arrays, args)
#else
  #define PETSC_DEVICE_BLAS_DISPATCH_CUDA(SUBR, device_type, memtype_arrays, args)
#endif

#if PetscDefined(HAVE_HIP)
PETSC_DEVICE_BLAS_DECLARE_CUPM_PROTOTYPES(HIP)
  #define PETSC_DEVICE_BLAS_DISPATCH_HIP(SUBR, device_type, memtype_arrays, args) PETSC_DEVICE_BLAS_DISPATCH_SINGLE(HIP, SUBR, device_type, memtype_arrays, args)
#else
  #define PETSC_DEVICE_BLAS_DISPATCH_HIP(SUBR, device_type, memtype_arrays, args)
#endif

#define PETSC_DEVICE_BLAS_DISPATCH(SUBR, device_type, memtype_arrays, args) \
  PETSC_DEVICE_BLAS_DISPATCH_CUDA(SUBR, device_type, memtype_arrays, args) \
  PETSC_DEVICE_BLAS_DISPATCH_HIP(SUBR, device_type, memtype_arrays, args) \
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Could not dispatch " #SUBR " from device");

PETSC_INTERN PetscErrorCode PetscDeviceGEMM_Private(PetscDeviceContext dctx, PetscMemType memtype_arrays, PetscMemType memtype_scalars, char trans_A, char trans_B, PetscInt m, PetscInt n, PetscInt k, const PetscScalar *alpha, const PetscScalar A[], PetscInt ld_A, const PetscScalar B[], PetscInt ld_B, const PetscScalar *beta, PetscScalar C[], PetscInt ld_C)
{
  PetscDeviceType device_type;
  PetscFunctionBegin;
  if (PetscMemTypeHost(memtype_arrays)) {
    PetscBLASInt   _m, _n, _k, _lda, _ldb, _ldc;
    PetscLogDouble flops;

    PetscCheck(PetscMemTypeHost(memtype_scalars), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Scalar references must be on the host for linear algebra on the host");
    flops = 2.0 * m * n * k + (*beta == 0.0 ? -1.0 : 1.0) * m * n + (*alpha == 1.0 ? 0.0 : 1.0) * PetscMin(m * n, PetscMin(m * k, n * k));
    PetscCall(PetscBLASIntCast(m, &_m));
    PetscCall(PetscBLASIntCast(n, &_n));
    PetscCall(PetscBLASIntCast(k, &_k));
    PetscCall(PetscBLASIntCast(ld_A, &_lda));
    PetscCall(PetscBLASIntCast(ld_B, &_ldb));
    PetscCall(PetscBLASIntCast(ld_C, &_ldc));
    PetscCallBLAS("BLASgemm", BLASgemm_(&trans_A, &trans_B, &_m, &_n, &_k, alpha, A, &_lda, B, &_ldb, beta, C, &_ldc));
    PetscCall(PetscLogFlops(flops));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    if (!dctx) PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
    PETSC_DEVICE_BLAS_DISPATCH(GEMM, device_type, memtype_arrays, (dctx, memtype_scalars, trans_A, trans_B, m, n, k, alpha, A, ld_A, B, ld_B, beta, C, ld_C))
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscDeviceGEMV_Private(PetscDeviceContext dctx, PetscMemType memtype_arrays, PetscMemType memtype_scalars, char trans, PetscInt m, PetscInt n, const PetscScalar *alpha, const PetscScalar A[], PetscInt ld_A, const PetscScalar x[], PetscInt inc_x, const PetscScalar *beta, PetscScalar y[], PetscInt inc_y)
{
  PetscDeviceType device_type;
  PetscFunctionBegin;
  if (PetscMemTypeHost(memtype_arrays)) {
    PetscBLASInt   _m, _n, _lda, _inc_x, _inc_y;
    PetscLogDouble flops;

    PetscCheck(PetscMemTypeHost(memtype_scalars), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Scalar references must be on the host for linear algebra on the host");
    flops = 2.0 * m * n + (*beta == 0.0 ? -1.0 : 1.0) * ((trans == 'n' || trans == 'N') ? m : n) + (*alpha == 1.0 ? 0.0 : 1.0) * PetscMin(m, n);
    PetscCall(PetscBLASIntCast(m, &_m));
    PetscCall(PetscBLASIntCast(n, &_n));
    PetscCall(PetscBLASIntCast(ld_A, &_lda));
    PetscCall(PetscBLASIntCast(inc_x, &_inc_x));
    PetscCall(PetscBLASIntCast(inc_y, &_inc_y));
    PetscCallBLAS("BLASgemv", BLASgemv_(&trans, &_m, &_n, alpha, A, &_lda, x, &_inc_x, beta, y, &_inc_y));
    PetscCall(PetscLogFlops(flops));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    if (!dctx) PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextGetDeviceType(dctx, &device_type));
    PETSC_DEVICE_BLAS_DISPATCH(GEMV, device_type, memtype_arrays, (dctx, memtype_scalars, trans, m, n, alpha, A, ld_A, x, inc_x, beta, y, inc_y))
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
