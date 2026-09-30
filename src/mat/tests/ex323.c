static const char help[] = "Test CUDA/HIP AIJ operations and array copies on the current device context.\n";

#include <petscmat.h>
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
#endif

static PetscErrorCode WriteValues(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[], PetscBool delay)
{
  PetscScalar *a;
  void        *stream;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCall(MatSeqAIJCUSPARSEGetArray(A, &a));
    if (delay) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    PetscCallCUDA(cudaMemcpyAsync(a, input, 4 * sizeof(*a), cudaMemcpyHostToDevice, *(cudaStream_t *)stream));
    PetscCall(MatSeqAIJCUSPARSERestoreArray(A, &a));
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCall(MatSeqAIJHIPSPARSEGetArray(A, &a));
    if (delay) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    PetscCallHIP(hipMemcpyAsync(a, input, 4 * sizeof(*a), hipMemcpyHostToDevice, *(hipStream_t *)stream));
    PetscCall(MatSeqAIJHIPSPARSERestoreArray(A, &a));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCopies(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  PetscErrorCode (*copy)(Mat, PetscInt, const PetscInt[], PetscScalar[]) = NULL;
  const PetscInt     idx[]                                               = {3, 1, 0, 2};
  const PetscScalar *a;
  PetscScalar       *host, *device;
  PetscStreamType    streamtype;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatSeqAIJCopySubArray_C", &copy));
  PetscCheck(copy, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing device subarray copy implementation");
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 4, &host));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, 4, &device));

  PetscCall(MatZeroEntries(A));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
  PetscCall(MatSeqAIJGetArrayRead(A, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Host readback overtook the device write at entry %" PetscInt_FMT, i);
  PetscCall(MatSeqAIJRestoreArrayRead(A, &a));

  for (PetscInt d = 0; d < 2; ++d) {
    for (PetscInt indexed = 0; indexed < 2; ++indexed) {
      PetscCall(MatZeroEntries(A));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
      PetscCall((*copy)(A, 4, indexed ? idx : NULL, d ? device : host));
      if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
        PetscBool idle;

        PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
        PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Subarray copy returned with work pending on a barrier context");
      }
      if (d) {
        PetscCall(PetscDeviceArrayCopy(dctx, host, device, 4));
        PetscCall(PetscDeviceContextSynchronize(dctx));
      }
      for (PetscInt i = 0; i < 4; ++i)
        PetscCheck(host[i] == input[indexed ? idx[i] : i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Subarray copy (device %" PetscInt_FMT ", indexed %" PetscInt_FMT ") overtook the device write at entry %" PetscInt_FMT, d, indexed, i);
      PetscCall((*copy)(A, 0, NULL, d ? device : host));
    }
  }
  PetscCall(PetscDeviceFree(dctx, device));
  PetscCall(PetscDeviceFree(dctx, host));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestGetDiagonal(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  Vec                diag;
  const PetscScalar *a;
  PetscStreamType    streamtype;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  PetscCall(MatCreateVecs(A, &diag, NULL));
  // Warm the allocation and kernel before delaying the matrix values.
  PetscCall(MatGetDiagonal(A, diag));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
  PetscCall(MatGetDiagonal(A, diag));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscBool idle;

    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatGetDiagonal() returned with work pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArrayRead(diag, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect diagonal entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(diag, &a));
  PetscCall(VecDestroy(&diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDiagonalScale(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  Vec                left, right;
  const PetscScalar *a;
  PetscStreamType    streamtype;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  PetscCall(MatCreateVecs(A, &right, &left));
  PetscCall(VecSet(left, 2));
  PetscCall(VecSet(right, 3));
  for (PetscInt mode = 0; mode < 3; ++mode) {
    PetscScalar factor = mode == 0 ? 2 : (mode == 1 ? 3 : 6);
    Vec         ll = mode == 1 ? NULL : left, rr = mode == 0 ? NULL : right;

    // Warm the scaling kernels before delaying the matrix values.
    PetscCall(MatDiagonalScale(A, ll, rr));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
    PetscCall(MatDiagonalScale(A, ll, rr));
    if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatDiagonalScale() mode %" PetscInt_FMT " returned with work pending on a barrier context", mode);
    }
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == factor * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatDiagonalScale() mode %" PetscInt_FMT ": incorrect entry %" PetscInt_FMT, mode, i);
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));
  }
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&right));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestGetIJ(PetscDeviceType type, PetscDeviceContext dctx)
{
  Mat             A;
  const PetscInt *i, *j;
  PetscInt       *rows, *cols;
  PetscBool       idle;
  void           *stream;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  PetscCall(MatSetValue(A, 1, 2, 7, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 5, &rows));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 1, &cols));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
  // Build the compressed device matrix before testing the full row-offset upload.
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCall(MatSeqAIJCUSPARSEGetIJ(A, PETSC_TRUE, &i, &j));
    PetscCall(MatSeqAIJCUSPARSERestoreIJ(A, PETSC_TRUE, &i, &j));
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_TRUE, &i, &j));
    PetscCall(MatSeqAIJHIPSPARSERestoreIJ(A, PETSC_TRUE, &i, &j));
  }
#endif
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscDeviceContextDelay(dctx, 0.2));
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) PetscCall(MatSeqAIJCUSPARSEGetIJ(A, PETSC_FALSE, &i, &j));
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_FALSE, &i, &j));
#endif
  PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
  PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSeqAIJGetIJ() returned before its current-context upload completed");
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCallCUDA(cudaMemcpyAsync(rows, i, 5 * sizeof(*rows), cudaMemcpyDeviceToHost, *(cudaStream_t *)stream));
    PetscCallCUDA(cudaMemcpyAsync(cols, j, sizeof(*cols), cudaMemcpyDeviceToHost, *(cudaStream_t *)stream));
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCallHIP(hipMemcpyAsync(rows, i, 5 * sizeof(*rows), hipMemcpyDeviceToHost, *(hipStream_t *)stream));
    PetscCallHIP(hipMemcpyAsync(cols, j, sizeof(*cols), hipMemcpyDeviceToHost, *(hipStream_t *)stream));
  }
#endif
  PetscCall(PetscDeviceContextSynchronize(dctx));
  for (PetscInt k = 0; k < 5; ++k) PetscCheck(rows[k] == (k > 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect row offset at entry %" PetscInt_FMT, k);
  PetscCheck(cols[0] == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect column index");
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) PetscCall(MatSeqAIJCUSPARSERestoreIJ(A, PETSC_FALSE, &i, &j));
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) PetscCall(MatSeqAIJHIPSPARSERestoreIJ(A, PETSC_FALSE, &i, &j));
#endif
  PetscCall(PetscDeviceFree(dctx, cols));
  PetscCall(PetscDeviceFree(dctx, rows));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SparsePointerMode(PetscDeviceType type, PetscDeviceContext dctx, PetscBool set, PetscBool device_mode)
{
  PetscFunctionBeginUser;
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    cusparseHandle_t      handle;
    cusparsePointerMode_t mode = device_mode ? CUSPARSE_POINTER_MODE_DEVICE : CUSPARSE_POINTER_MODE_HOST;

    PetscCall(PetscDeviceContextGetSPARSEHandle_Internal(dctx, &handle));
    if (set) PetscCallCUSPARSE(cusparseSetPointerMode(handle, mode));
    else {
      cusparsePointerMode_t actual;

      PetscCallCUSPARSE(cusparseGetPointerMode(handle, &actual));
      PetscCheck(actual == mode, PETSC_COMM_SELF, PETSC_ERR_PLIB, "cuSPARSE pointer mode was not restored");
    }
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    hipsparseHandle_t      handle;
    hipsparsePointerMode_t mode = device_mode ? HIPSPARSE_POINTER_MODE_DEVICE : HIPSPARSE_POINTER_MODE_HOST;

    PetscCall(PetscDeviceContextGetSPARSEHandle_Internal(dctx, &handle));
    if (set) PetscCallHIPSPARSE(hipsparseSetPointerMode(handle, mode));
    else {
      hipsparsePointerMode_t actual;

      PetscCallHIPSPARSE(hipsparseGetPointerMode(handle, &actual));
      PetscCheck(actual == mode, PETSC_COMM_SELF, PETSC_ERR_PLIB, "hipSPARSE pointer mode was not restored");
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSparseOperations(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  const PetscScalar  ones[] = {1, 1, 1, 1};
  const PetscScalar *a;
  PetscStreamType    streamtype;
  PetscBool          idle;
  Vec                x, y;
  Mat                B, C;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecSet(x, 1));
  PetscCall(SparsePointerMode(type, dctx, PETSC_TRUE, PETSC_FALSE));
  PetscCall(WriteValues(A, type, dctx, ones, PETSC_FALSE));
  PetscCall(MatMult(A, x, y));
  PetscCall(SparsePointerMode(type, dctx, PETSC_FALSE, PETSC_FALSE));
  PetscCall(PetscDeviceContextSynchronize(dctx));

  PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
  PetscCall(MatMult(A, x, y));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMult() returned with work pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArrayRead(y, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMult() returned an incorrect entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(y, &a));

  PetscCall(MatSetOption(A, MAT_FORM_EXPLICIT_TRANSPOSE, PETSC_TRUE));
  PetscCall(MatMultTranspose(A, x, y));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecSet(x, 0));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscDeviceContextDelay(dctx, 0.05));
  PetscCall(VecSet(x, 2));
  PetscCall(MatMultTranspose(A, x, y));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMultTranspose() returned with work pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArrayRead(y, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 2 * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMultTranspose() returned an incorrect entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(y, &a));
  PetscCall(MatSetOption(A, MAT_FORM_EXPLICIT_TRANSPOSE, PETSC_FALSE));
  PetscCall(VecSet(x, 1));

  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(WriteValues(B, type, dctx, ones, PETSC_FALSE));
  PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, 1.0, &C));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(WriteValues(B, type, dctx, input, PETSC_TRUE));
  PetscCall(MatMatMult(A, B, MAT_REUSE_MATRIX, 1.0, &C));
  PetscCall(SparsePointerMode(type, dctx, PETSC_FALSE, PETSC_FALSE));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMatMult() returned with work pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(MatSeqAIJGetArrayRead(C, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == input[i] * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMatMult() returned an incorrect entry %" PetscInt_FMT, i);
  PetscCall(MatSeqAIJRestoreArrayRead(C, &a));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSparseSolve(PetscDeviceType type, PetscDeviceContext saved, PetscDeviceContext dctx)
{
  Mat                A, F;
  IS                 perm;
  Vec                b, x;
  MatFactorInfo      info;
  PetscStreamType    streamtype;
  PetscBool          idle;
  const PetscScalar *a;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(MatSetValue(A, i, i, 2, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 4, 0, 1, &perm));
  PetscCall(MatCreateVecs(A, &b, &x));
  PetscCall(MatFactorInfoInitialize(&info));
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  for (PetscInt kind = 0; kind < 2; ++kind) {
    MatFactorType factortype = kind ? MAT_FACTOR_CHOLESKY : MAT_FACTOR_LU;

    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(MatGetFactor(A, type == PETSC_DEVICE_CUDA ? MATSOLVERCUSPARSE : MATSOLVERHIPSPARSE, factortype, &F));
    PetscCall(SparsePointerMode(type, saved, PETSC_TRUE, PETSC_TRUE));
    if (kind) {
      PetscCall(MatCholeskyFactorSymbolic(F, A, perm, &info));
      PetscCall(MatCholeskyFactorNumeric(F, A, &info));
    } else {
      PetscCall(MatLUFactorSymbolic(F, A, perm, perm, &info));
      PetscCall(MatLUFactorNumeric(F, A, &info));
    }
    PetscCall(SparsePointerMode(type, saved, PETSC_FALSE, PETSC_TRUE));
    PetscCall(SparsePointerMode(type, saved, PETSC_TRUE, PETSC_FALSE));
    PetscCall(PetscDeviceContextSynchronize(saved));
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
    for (PetscInt solve = 0; solve < 3; ++solve) {
      PetscCall(VecSet(b, 0));
      PetscCall(VecSet(x, 0));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(PetscDeviceContextDelay(dctx, 0.05));
      PetscCall(VecSet(b, 2));
      PetscCall(SparsePointerMode(type, dctx, PETSC_TRUE, PETSC_TRUE));
      if (solve == 1) PetscCall(MatSolveTranspose(F, b, x));
      else PetscCall(MatSolve(F, b, x));
      PetscCall(SparsePointerMode(type, dctx, PETSC_FALSE, PETSC_TRUE));
      PetscCall(SparsePointerMode(type, dctx, PETSC_TRUE, PETSC_FALSE));
      if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
        PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
        PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSolve() returned with work pending on a barrier context");
      }
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(VecGetArrayRead(x, &a));
      for (PetscInt i = 0; i < 4; ++i) PetscCheck(PetscAbsScalar(a[i] - 1) < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSolve() returned an incorrect entry %" PetscInt_FMT, i);
      PetscCall(VecRestoreArrayRead(x, &a));
    }
    PetscCall(MatDestroy(&F));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(ISDestroy(&perm));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat                   A, B;
  PetscDeviceContext    saved, current;
  PetscDevice           device;
  PetscDeviceType       type;
  PetscScalar          *input;
  const PetscScalar    *a;
  const PetscStreamType streams[] = {PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
  PetscCall(PetscDeviceMalloc(saved, PETSC_MEMTYPE_HOST, 4, &input));
  for (PetscInt i = 0; i < 4; ++i) input[i] = i + 1;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(MatSetValue(A, i, i, 1, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(WriteValues(A, type, saved, input, PETSC_FALSE));
  PetscCall(PetscDeviceContextSynchronize(saved));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(PetscDeviceContextSynchronize(saved));

  for (PetscInt k = 0; k < 2; ++k) {
    PetscCall(PetscDeviceContextDuplicate(saved, &current));
    PetscCall(PetscDeviceContextSetStreamType(current, streams[k]));
    PetscCall(PetscDeviceContextSetUp(current));
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    // Warm the zeroing operation before delaying the input write.
    PetscCall(MatZeroEntries(A));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_TRUE));
    PetscCall(MatZeroEntries(A));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: CSR entry %" PetscInt_FMT " is %g instead of zero", PetscStreamTypes[streams[k]], i, (double)PetscRealPart(a[i]));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));

    // Warm the BLAS handle and scaling kernel before checking completion on return.
    PetscCall(MatScale(A, 2));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_TRUE));
    PetscCall(MatScale(A, 2));
    if (streams[k] == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(current, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatScale() returned with work pending on a barrier context");
    }
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 2 * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: incorrectly scaled CSR entry %" PetscInt_FMT, PetscStreamTypes[streams[k]], i);
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));

    // Warm AXPY before delaying the update of its source matrix.
    PetscCall(MatAXPY(A, 2, B, SAME_NONZERO_PATTERN));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_FALSE));
    PetscCall(WriteValues(B, type, current, input, PETSC_TRUE));
    PetscCall(MatAXPY(A, 2, B, SAME_NONZERO_PATTERN));
    if (streams[k] == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(current, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatAXPY() returned with work pending on a barrier context");
    }
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 3 * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: incorrect CSR entry %" PetscInt_FMT " after MatAXPY()", PetscStreamTypes[streams[k]], i);
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));
    PetscCall(TestCopies(A, type, current, input));
    PetscCall(TestGetDiagonal(A, type, current, input));
    PetscCall(TestDiagonalScale(A, type, current, input));
    PetscCall(TestGetIJ(type, current));
    if (type == PETSC_DEVICE_CUDA) {
      PetscCall(TestSparseOperations(A, type, current, input));
      PetscCall(TestSparseSolve(type, saved, current));
    }
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&current));
  }

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscDeviceFree(saved, input));
  PetscCall(PetscDeviceContextSynchronize(saved));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out

    test:
      suffix: cuda
      requires: cuda

    test:
      suffix: hip
      requires: hip

TEST*/
