#pragma once
#include "blas_cyclic_cupm.h"
#include <petsc/private/cupminterface.hpp>
#include <petsc/private/cupmobject.hpp>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

template <DeviceType T>
struct BLASCyclic : CUPMObject<T> {
  PETSC_CUPMOBJECT_HEADER(T);

  static PetscErrorCode axpby(PetscDeviceContext, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscScalar, PetscScalar[], PetscInt) noexcept;
  static PetscErrorCode dmv(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], const PetscScalar[], PetscScalar, PetscScalar[]) noexcept;
  static PetscErrorCode dsv(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, const PetscScalar[], const PetscScalar[], PetscScalar[]) noexcept;
  static PetscErrorCode trsv(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[]) noexcept;
  static PetscErrorCode gemv(PetscDeviceContext, PetscBool, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar, PetscScalar[]) noexcept;
  static PetscErrorCode hemv(PetscDeviceContext, PetscInt, PetscInt, PetscInt, PetscScalar, const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar, PetscScalar[]) noexcept;
};

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::axpby(PetscDeviceContext dctx, PetscInt M, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar x[], PetscScalar beta, PetscScalar y[], PetscInt y_stride) noexcept
{
  PetscInt              N = next - oldest;
  cupmBlasInt_t         m, i_oldest, i_next;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  auto                  _x = cupmScalarPtrCast(x);
  auto                  _y = cupmScalarPtrCast(y);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(M, &m));
  PetscCall(PetscCUPMBlasIntCast(oldest % m, &i_oldest));
  PetscCall(PetscCUPMBlasIntCast(((next - 1) % m) + 1, &i_next));
  PetscCall(GetHandlesFrom_(dctx, &handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (N == m) {
    PetscCallCUPMBLAS(cupmBlasXscal(handle, m, &beta, _y, y_stride));
    PetscCallCUPMBLAS(cupmBlasXaxpy(handle, m, &alpha, _x, 1, _y, y_stride));
  } else if (i_next > i_oldest) {
    cupmBlasInt_t diff = i_next - i_oldest;

    PetscCallCUPMBLAS(cupmBlasXscal(handle, diff, &beta, &_y[i_oldest * y_stride], y_stride));
    PetscCallCUPMBLAS(cupmBlasXaxpy(handle, diff, &alpha, &_x[i_oldest], 1, &_y[i_oldest * y_stride], y_stride));
  } else {
    cupmBlasInt_t diff = m - i_oldest;

    if (i_next) {
      PetscCallCUPMBLAS(cupmBlasXscal(handle, i_next, &beta, _y, y_stride));
      PetscCallCUPMBLAS(cupmBlasXaxpy(handle, i_next, &alpha, _x, 1, _y, y_stride));
    }
    if (diff) {
      PetscCallCUPMBLAS(cupmBlasXscal(handle, diff, &beta, &_y[i_oldest * y_stride], y_stride));
      PetscCallCUPMBLAS(cupmBlasXaxpy(handle, diff, &alpha, &_x[i_oldest], 1, &_y[i_oldest * y_stride], y_stride));
    }
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(3.0 * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::dmv(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt M, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], const PetscScalar x[], PetscScalar beta, PetscScalar y[]) noexcept
{
  PetscInt              N = next - oldest;
  cupmBlasInt_t         m, i_oldest, i_next;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  const auto            _A = cupmScalarPtrCast(A);
  const auto            _x = cupmScalarPtrCast(x);
  const auto            _y = cupmScalarPtrCast(y);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(M, &m));
  PetscCall(PetscCUPMBlasIntCast(oldest % m, &i_oldest));
  PetscCall(PetscCUPMBlasIntCast(((next - 1) % m) + 1, &i_next));
  PetscCall(GetHandlesFrom_(dctx, &handle));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (i_next > i_oldest) {
    cupmBlasInt_t diff = i_next - i_oldest;

    PetscCallCUPMBLAS(cupmBlasXhbmv(handle, CUPMBLAS_FILL_MODE_UPPER, diff, 0, &alpha, &_A[i_oldest], 1, &_x[i_oldest], 1, &beta, &_y[i_oldest], 1));
  } else {
    cupmBlasInt_t diff = m - i_oldest;

    if (i_next) PetscCallCUPMBLAS(cupmBlasXhbmv(handle, CUPMBLAS_FILL_MODE_UPPER, i_next, 0, &alpha, _A, 1, _x, 1, &beta, _y, 1));
    if (diff) PetscCallCUPMBLAS(cupmBlasXhbmv(handle, CUPMBLAS_FILL_MODE_UPPER, diff, 0, &alpha, &_A[i_oldest], 1, &_x[i_oldest], 1, &beta, &_y[i_oldest], 1));
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(3.0 * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::dsv(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt M, PetscInt oldest, PetscInt next, const PetscScalar A[], const PetscScalar x[], PetscScalar y[]) noexcept
{
  PetscInt              N = next - oldest;
  cupmBlasInt_t         m, i_oldest, i_next;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  cupmStream_t          stream;
  const auto            _A    = cupmScalarPtrCast(A);
  const auto            _y    = cupmScalarPtrCast(y);
  auto                  trans = hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N;

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(M, &m));
  PetscCall(PetscCUPMBlasIntCast(oldest % m, &i_oldest));
  PetscCall(PetscCUPMBlasIntCast(((next - 1) % m) + 1, &i_next));
  PetscCall(GetHandlesFrom_(dctx, &handle, NULL, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (i_next > i_oldest) {
    cupmBlasInt_t diff = i_next - i_oldest;

    if (x != y) PetscCall(PetscCUPMMemcpyAsync(&y[i_oldest], &x[i_oldest], diff, cupmMemcpyDeviceToDevice, stream));
    PetscCallCUPMBLAS(cupmBlasXtbsv(handle, CUPMBLAS_FILL_MODE_UPPER, trans, CUPMBLAS_DIAG_NON_UNIT, diff, 0, &_A[i_oldest], 1, &_y[i_oldest], 1));
  } else {
    cupmBlasInt_t diff = m - i_oldest;

    if (i_next) {
      if (x != y) PetscCall(PetscCUPMMemcpyAsync(y, x, i_next, cupmMemcpyDeviceToDevice, stream));
      PetscCallCUPMBLAS(cupmBlasXtbsv(handle, CUPMBLAS_FILL_MODE_UPPER, trans, CUPMBLAS_DIAG_NON_UNIT, i_next, 0, _A, 1, _y, 1));
    }
    if (diff) {
      if (x != y) PetscCall(PetscCUPMMemcpyAsync(&y[i_oldest], &x[i_oldest], diff, cupmMemcpyDeviceToDevice, stream));
      PetscCallCUPMBLAS(cupmBlasXtbsv(handle, CUPMBLAS_FILL_MODE_UPPER, trans, CUPMBLAS_DIAG_NON_UNIT, diff, 0, &_A[i_oldest], 1, &_y[i_oldest], 1));
    }
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(3.0 * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::trsv(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar y[]) noexcept
{
  PetscInt              N        = next - oldest;
  PetscInt              i_oldest = oldest % m;
  PetscInt              i_next   = ((next - 1) % m) + 1;
  cupmBlasInt_t         n, n_old, n_new;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  cupmStream_t          stream;
  auto                  sone      = cupmScalarCast(1.0);
  auto                  minus_one = cupmScalarCast(-1.0);
  auto                  _A        = cupmScalarPtrCast(A);
  auto                  _y        = cupmScalarPtrCast(y);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(i_next - i_oldest, &n));
  PetscCall(PetscCUPMBlasIntCast(m - i_oldest, &n_old));
  PetscCall(PetscCUPMBlasIntCast(i_next, &n_new));
  PetscCall(GetHandlesFrom_(dctx, &handle, NULL, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (n > 0) {
    if (x != y) PetscCall(PetscCUPMMemcpyAsync(y, x, n, cupmMemcpyDeviceToDevice, stream));
    PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n, &_A[i_oldest * (lda + 1)], lda, _y, 1));
  } else if (!hermitian_transpose) {
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n_new, _A, lda, _y, 1));
    if (n_new > 0 && n_old > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_old, n_new, &minus_one, &_A[i_oldest], lda, _y, 1, &sone, &_y[i_oldest], 1));
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, n_old, &_A[i_oldest * (lda + 1)], lda, &_y[i_oldest], 1));
  } else {
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_C, CUPMBLAS_DIAG_NON_UNIT, n_old, &_A[i_oldest * (lda + 1)], lda, &_y[i_oldest], 1));
    if (n_new > 0 && n_old > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_old, n_new, &minus_one, &_A[i_oldest], lda, &_y[i_oldest], 1, &sone, _y, 1));
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXtrsv(handle, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_C, CUPMBLAS_DIAG_NON_UNIT, n_new, _A, lda, _y, 1));
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(1.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::hemv(PetscDeviceContext dctx, PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar beta, PetscScalar y[]) noexcept
{
  PetscInt              N        = next - oldest;
  PetscInt              i_oldest = oldest % m;
  PetscInt              i_next   = ((next - 1) % m) + 1;
  cupmBlasInt_t         n, n_old, n_new;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  cupmStream_t          stream;
  auto                  sone = cupmScalarCast(1.0);
  auto                  _A   = cupmScalarPtrCast(A);
  auto                  _x   = cupmScalarPtrCast(x);
  auto                  _y   = cupmScalarPtrCast(y);

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(i_next - i_oldest, &n));
  PetscCall(PetscCUPMBlasIntCast(m - i_oldest, &n_old));
  PetscCall(PetscCUPMBlasIntCast(i_next, &n_new));
  PetscCall(GetHandlesFrom_(dctx, &handle, NULL, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (n > 0) {
    PetscCallCUPMBLAS(cupmBlasXhemv(handle, CUPMBLAS_FILL_MODE_UPPER, n, &alpha, &_A[i_oldest * (lda + 1)], lda, &_x[i_oldest], 1, &beta, &_y[i_oldest], 1));
  } else {
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXhemv(handle, CUPMBLAS_FILL_MODE_UPPER, n_new, &alpha, _A, lda, _x, 1, &beta, _y, 1));
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXhemv(handle, CUPMBLAS_FILL_MODE_UPPER, n_old, &alpha, &_A[i_oldest * (lda + 1)], lda, &_x[i_oldest], 1, &beta, &_y[i_oldest], 1));
    if (n_new > 0 && n_old > 0) {
      PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_old, n_new, &alpha, &_A[i_oldest], lda, _x, 1, &sone, &_y[i_oldest], 1));
      PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_old, n_new, &alpha, &_A[i_oldest], lda, &_x[i_oldest], 1, &sone, _y, 1));
    }
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(2.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode BLASCyclic<T>::gemv(PetscDeviceContext dctx, PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar beta, PetscScalar y[]) noexcept
{
  PetscInt              N        = next - oldest;
  PetscInt              i_oldest = oldest % m;
  PetscInt              i_next   = ((next - 1) % m) + 1;
  cupmBlasInt_t         n, n_old, n_new;
  cupmBlasPointerMode_t pointer_mode;
  cupmBlasHandle_t      handle;
  cupmStream_t          stream;
  auto                  sone  = cupmScalarCast(1.0);
  auto                  _A    = cupmScalarPtrCast(A);
  auto                  _x    = cupmScalarPtrCast(x);
  auto                  _y    = cupmScalarPtrCast(y);
  auto                  trans = hermitian_transpose ? CUPMBLAS_OP_C : CUPMBLAS_OP_N;

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCUPMBlasIntCast(i_next - i_oldest, &n));
  PetscCall(PetscCUPMBlasIntCast(m - i_oldest, &n_old));
  PetscCall(PetscCUPMBlasIntCast(i_next, &n_new));
  PetscCall(GetHandlesFrom_(dctx, &handle, NULL, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUPMBLAS(cupmBlasGetPointerMode(handle, &pointer_mode));
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, CUPMBLAS_POINTER_MODE_HOST));
  if (N == m) {
    PetscCallCUPMBLAS(cupmBlasXgemv(handle, trans, N, N, &alpha, _A, lda, _x, 1, &beta, _y, 1));
  } else if (n > 0) {
    PetscCallCUPMBLAS(cupmBlasXgemv(handle, trans, n, n, &alpha, &_A[i_oldest * (lda + 1)], lda, &_x[i_oldest], 1, &beta, &_y[i_oldest], 1));
  } else {
    if (n_new > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, trans, n_new, n_new, &alpha, _A, lda, _x, 1, &beta, _y, 1));
    if (n_old > 0) PetscCallCUPMBLAS(cupmBlasXgemv(handle, trans, n_old, n_old, &alpha, &_A[i_oldest * (lda + 1)], lda, &_x[i_oldest], 1, &beta, &_y[i_oldest], 1));
    if (n_new > 0 && n_old > 0) {
      if (!hermitian_transpose) {
        PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_old, n_new, &alpha, &_A[i_oldest], lda, _x, 1, &sone, &_y[i_oldest], 1));
        PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_N, n_new, n_old, &alpha, &_A[i_oldest * lda], lda, &_x[i_oldest], 1, &sone, _y, 1));
      } else {
        PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_new, n_old, &alpha, &_A[i_oldest * lda], lda, _x, 1, &sone, &_y[i_oldest], 1));
        PetscCallCUPMBLAS(cupmBlasXgemv(handle, CUPMBLAS_OP_C, n_old, n_new, &alpha, &_A[i_oldest], lda, &_x[i_oldest], 1, &sone, _y, 1));
      }
    }
  }
  PetscCallCUPMBLAS(cupmBlasSetPointerMode(handle, pointer_mode));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(2.0 * N * N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc
