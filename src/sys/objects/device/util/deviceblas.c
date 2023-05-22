
#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/
#include <petscblaslapack.h>
#include <petsc/private/petsclegacycupmblas.h>

#define TRANSOP_FROM_CHAR(PRE, t) (t == 'c' || t == 'C') ? PRE##BLAS_OP_C : (t == 't' || t == 'T') ? PRE##BLAS_OP_T : PRE##BLAS_OP_N

#if defined(PETSC_HAVE_CUDA)
static cublasOperation_t cublasOperationFromChar_Private(char t)
{
  return TRANSOP_FROM_CHAR(CU, t);
}
#endif

#if defined(PETSC_HAVE_HIP)
static hipblasOperation_t hipblasOperationFromChar_Private(char t)
{
  return TRANSOP_FROM_CHAR(HIP, t);
}
#endif

PETSC_INTERN PetscErrorCode PetscDeviceGEMM(PetscDeviceContext dctx, PetscMemType memtype, char trans_A, char trans_B, PetscInt m, PetscInt n, PetscInt k, PetscScalar alpha, const PetscScalar *A, PetscInt ld_A, const PetscScalar *B, PetscInt ld_B, PetscScalar beta, PetscScalar *C, PetscInt ld_C)
{
  PetscFunctionBegin;
  PetscDeviceContext dctx_default = NULL;
  if (dctx) {
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx_default));
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
  }
  PetscLogDouble flops = 2.0 * m * n * k + (beta == 0.0 ? -1.0 : 1.0) * m * n + (alpha == 1.0 ? 0.0 : 1.0) * PetscMin(m * n, PetscMin(m * k, n * k));
  switch (memtype) {
  case PETSC_MEMTYPE_HOST: {
    PetscBLASInt _m, _n, _k, _lda, _ldb, _ldc;

    PetscCall(PetscBLASIntCast(m, &_m));
    PetscCall(PetscBLASIntCast(n, &_n));
    PetscCall(PetscBLASIntCast(k, &_k));
    PetscCall(PetscBLASIntCast(ld_A, &_lda));
    PetscCall(PetscBLASIntCast(ld_B, &_ldb));
    PetscCall(PetscBLASIntCast(ld_C, &_ldc));
    PetscCallBLAS("BLASgemm", BLASgemm_(&trans_A, &trans_B, &_m, &_n, &_k, &alpha, A, &_lda, B, &_ldb, &beta, C, &_ldc));
    PetscCall(PetscLogFlops(flops));
  } break;
#if defined(PETSC_HAVE_CUDA)
  case PETSC_MEMTYPE_CUDA: {
    cublasOperation_t _transa = cublasOperationFromChar_Private(trans_A);
    cublasOperation_t _transb = cublasOperationFromChar_Private(trans_B);
    PetscCuBLASInt    _m, _n, _k, _lda, _ldb, _ldc;
    cublasHandle_t    _handle;
    PetscCall(PetscCuBLASIntCast(m, &_m));
    PetscCall(PetscCuBLASIntCast(n, &_n));
    PetscCall(PetscCuBLASIntCast(k, &_k));
    PetscCall(PetscCuBLASIntCast(ld_A, &_lda));
    PetscCall(PetscCuBLASIntCast(ld_B, &_ldb));
    PetscCall(PetscCuBLASIntCast(ld_C, &_ldc));
    PetscCall(PetscDeviceGetBLASHandle_Internal(dctx, (void *) &_handle));
    PetscCallCUBLAS(cublasXgemm(_handle, _transa, _transb, _m, _n, _k, &alpha, A, _lda, B, _ldb, &beta, C, _ldc));
    PetscCall(PetscLogGpuFlops(flops));
  } break;
#endif
#if defined(PETSC_HAVE_HIP)
  case PETSC_MEMTYPE_HIP: {
    hipblasOperation_t _transa = hipblasOperationFromChar_Private(trans_A);
    hipblasOperation_t _transb = hipblasOperationFromChar_Private(trans_B);
    PetscHipBLASInt    _m, _n, _k, _lda, _ldb, _ldc;
    hipblasHandle_t    _handle;
    PetscCall(PetscHipBLASIntCast(m, &_m));
    PetscCall(PetscHipBLASIntCast(n, &_n));
    PetscCall(PetscHipBLASIntCast(k, &_k));
    PetscCall(PetscHipBLASIntCast(ld_A, &_lda));
    PetscCall(PetscHipBLASIntCast(ld_B, &_ldb));
    PetscCall(PetscHipBLASIntCast(ld_C, &_ldf));
    PetscCall(PetscDeviceGetBLASHandle_Internal(dctx, (void *) &_handle));
    PetscCallHIPBLAS(hipblasXgemm(_handle, _transa, _transb, _m, _n, _k, &alpha, A, _lda, B, _ldb, &beta, C, _ldc));
    PetscCall(PetscLogGpuFlops(flops));
  } break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported device type");
    break;
  }
  if (dctx_default) {
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
