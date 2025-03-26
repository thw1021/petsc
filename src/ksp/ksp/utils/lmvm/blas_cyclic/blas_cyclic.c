#include "blas_cyclic.h"
#include "cupm/blas_cyclic_cupm.h"
#include <petsc/private/vecimpl.h>
#include <petsc/private/matimpl.h>
#include <petscblaslapack.h>

#define VecCheckAllEntriesFirstRank(a, arg) PetscCheck((a)->map->range[1] == (a)->map->N, PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_SIZ, "Vector argument # %" PetscInt_FMT " does not have all of its entries on the first rank", (arg));

#define MatCheckAllEntriesFirstRank(a, arg) \
  PetscCheck(((a)->rmap->range[1] == (a)->rmap->N) && ((a)->cmap->range[1] == (a)->cmap->N), PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_SIZ, "Matrix argument # %" PetscInt_FMT " does not have all of its entries on the first rank", (arg));

// Takes y_stride argument because this is also used for updating a row of a MatDense
static inline void AXPBY_Private(PetscInt m, PetscScalar alpha, const PetscScalar x[], PetscScalar beta, PetscScalar y[], PetscInt y_stride)
{
  for (PetscInt i = 0; i < m; i++) y[i * y_stride] = alpha * x[i] + beta * y[i * y_stride];
}

static PetscErrorCode AXPBYCylic_Private(PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar x[], PetscScalar beta, PetscScalar y[], PetscInt y_stride)
{
  PetscInt i_oldest = oldest % m;
  PetscInt i_next   = ((next - 1) % m) + 1;

  PetscFunctionBegin;
  if (next - oldest == m) {
    AXPBY_Private(m, alpha, x, beta, y, y_stride);
  } else if (i_next > i_oldest) {
    AXPBY_Private(i_next - i_oldest, alpha, &x[i_oldest], beta, &y[i_oldest * y_stride], y_stride);
  } else {
    AXPBY_Private(i_next, alpha, x, beta, y, y_stride);
    AXPBY_Private(m - i_oldest, alpha, &x[i_oldest], beta, &y[i_oldest * y_stride], y_stride);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode VecAXPBYCyclic(PetscInt oldest, PetscInt next, PetscScalar alpha, Vec x, PetscScalar beta, Vec y)
{
  const PetscScalar *_x;
  PetscScalar       *_y;
  PetscInt           m, m_local;
  PetscMemType       x_memtype, y_memtype;
  PetscBool          on_host = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 6);
  PetscCheckSameComm(x, 4, y, 6);
  VecCheckSameSize(x, 4, y, 6);
  VecCheckAllEntriesFirstRank(x, 4);
  VecCheckAllEntriesFirstRank(y, 6);
  PetscCall(VecGetSize(x, &m));
  PetscCall(VecGetLocalSize(x, &m_local));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayReadAndMemType(x, &_x, &x_memtype));
  PetscCall(VecGetArrayAndMemType(y, &_y, &y_memtype));
  if (PetscMemTypeDevice(x_memtype) && PetscMemTypeDevice(y_memtype)) {
    if (m_local == m) PetscCall(AXPBYCyclic_CUPM_Private(m, oldest, next, alpha, _x, beta, _y, 1));
  } else if (m_local == m) on_host = PETSC_TRUE;
  PetscCall(VecRestoreArrayReadAndMemType(x, &_x));
  PetscCall(VecRestoreArrayAndMemType(y, &_y));
  if (on_host) {
    PetscCall(VecGetArrayRead(x, &_x));
    PetscCall(VecGetArray(y, &_y));
    PetscCall(AXPBYCylic_Private(m, oldest, next, alpha, _x, beta, _y, 1));
    PetscCall(VecRestoreArray(y, &_y));
    PetscCall(VecRestoreArrayRead(x, &_x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void DMV_Private(PetscBool hermitian_transpose, PetscInt m, PetscScalar alpha, const PetscScalar A[], const PetscScalar x[], PetscScalar beta, PetscScalar y[])
{
  if (!hermitian_transpose) {
    for (PetscInt i = 0; i < m; i++) y[i] = alpha * A[i] * x[i] + beta * y[i];
  } else {
    for (PetscInt i = 0; i < m; i++) y[i] = alpha * PetscConj(A[i]) * x[i] + beta * y[i];
  }
}

static PetscErrorCode DMVCylic_Private(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], const PetscScalar x[], PetscScalar beta, PetscScalar y[])
{
  PetscInt i_oldest = oldest % m;
  PetscInt i_next   = ((next - 1) % m) + 1;

  PetscFunctionBegin;
  if (next - oldest == m) {
    DMV_Private(hermitian_transpose, m, alpha, A, x, beta, y);
  } else if (i_next > i_oldest) {
    DMV_Private(hermitian_transpose, i_next - i_oldest, alpha, &A[i_oldest], &x[i_oldest], beta, &y[i_oldest]);
  } else {
    DMV_Private(hermitian_transpose, i_next, alpha, A, x, beta, y);
    DMV_Private(hermitian_transpose, m - i_oldest, alpha, &A[i_oldest], &x[i_oldest], beta, &y[i_oldest]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode VecDMVCyclic(PetscBool hermitian_transpose, PetscInt oldest, PetscInt next, PetscScalar alpha, Vec A, Vec x, PetscScalar beta, Vec y)
{
  const PetscScalar *_A;
  const PetscScalar *_x;
  PetscScalar       *_y;
  PetscInt           m, m_local;
  PetscMemType       A_memtype, x_memtype, y_memtype;
  PetscBool          on_host = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 6);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 8);
  PetscCheckSameComm(A, 5, x, 6);
  PetscCheckSameComm(A, 5, y, 8);
  VecCheckSameSize(A, 5, x, 6);
  VecCheckSameSize(A, 5, y, 8);
  VecCheckAllEntriesFirstRank(A, 5);
  VecCheckAllEntriesFirstRank(x, 6);
  VecCheckAllEntriesFirstRank(y, 8);
  PetscCall(VecGetSize(A, &m));
  PetscCall(VecGetLocalSize(A, &m_local));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayReadAndMemType(A, &_A, &A_memtype));
  PetscCall(VecGetArrayReadAndMemType(x, &_x, &x_memtype));
  PetscCall(VecGetArrayAndMemType(y, &_y, &y_memtype));
  if (PetscMemTypeDevice(A_memtype) && PetscMemTypeDevice(x_memtype) && PetscMemTypeDevice(y_memtype)) {
    if (m_local == m) PetscCall(DMVCyclic_CUPM_Private(hermitian_transpose, m, oldest, next, alpha, _A, _x, beta, _y));
  } else if (m_local == m) on_host = PETSC_TRUE;
  PetscCall(VecRestoreArrayAndMemType(y, &_y));
  PetscCall(VecRestoreArrayReadAndMemType(x, &_x));
  PetscCall(VecRestoreArrayReadAndMemType(A, &_A));
  if (on_host) {
    PetscCall(VecGetArrayRead(A, &_A));
    PetscCall(VecGetArrayRead(x, &_x));
    PetscCall(VecGetArray(y, &_y));
    PetscCall(DMVCylic_Private(hermitian_transpose, m, oldest, next, alpha, _A, _x, beta, _y));
    PetscCall(VecRestoreArray(y, &_y));
    PetscCall(VecRestoreArrayRead(x, &_x));
    PetscCall(VecRestoreArrayRead(A, &_A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void DSV_Private(PetscBool hermitian_transpose, PetscInt m, const PetscScalar A[], const PetscScalar x[], PetscScalar y[])
{
  if (x != y) {
    if (!hermitian_transpose) {
      for (PetscInt i = 0; i < m; i++) y[i] = x[i] / A[i];
    } else {
      for (PetscInt i = 0; i < m; i++) y[i] = x[i] / PetscConj(A[i]);
    }
  } else {
    if (!hermitian_transpose) {
      for (PetscInt i = 0; i < m; i++) y[i] = y[i] / A[i];
    } else {
      for (PetscInt i = 0; i < m; i++) y[i] = y[i] / PetscConj(A[i]);
    }
  }
}

static PetscErrorCode DSVCyclic_Private(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], const PetscScalar x[], PetscScalar y[])
{
  PetscInt i_oldest = oldest % m;
  PetscInt i_next   = ((next - 1) % m) + 1;

  PetscFunctionBegin;
  if (next - oldest == m) {
    DSV_Private(hermitian_transpose, m, A, x, y);
  } else if (i_next > i_oldest) {
    DSV_Private(hermitian_transpose, i_next - i_oldest, &A[i_oldest], &x[i_oldest], &y[i_oldest]);
  } else {
    DSV_Private(hermitian_transpose, i_next, A, x, y);
    DSV_Private(hermitian_transpose, m - i_oldest, &A[i_oldest], &x[i_oldest], &y[i_oldest]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode VecDSVCyclic(PetscBool hermitian_transpose, PetscInt oldest, PetscInt next, Vec A, Vec x, Vec y)
{
  const PetscScalar *_A;
  const PetscScalar *_x = NULL;
  PetscScalar       *_y;
  PetscInt           m, m_local;
  PetscMemType       A_memtype, x_memtype, y_memtype;
  PetscBool          on_host = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 6);
  PetscCheckSameComm(A, 4, x, 5);
  PetscCheckSameComm(A, 4, y, 6);
  VecCheckSameSize(A, 4, x, 5);
  VecCheckSameSize(A, 4, y, 6);
  VecCheckAllEntriesFirstRank(A, 4);
  VecCheckAllEntriesFirstRank(x, 5);
  VecCheckAllEntriesFirstRank(y, 6);
  PetscCall(VecGetSize(A, &m));
  PetscCall(VecGetLocalSize(A, &m_local));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetArrayReadAndMemType(A, &_A, &A_memtype));
  PetscCall(VecGetArrayAndMemType(y, &_y, &y_memtype));
  if (x == y) {
    _x        = _y;
    x_memtype = y_memtype;
  } else {
    PetscCall(VecGetArrayReadAndMemType(x, &_x, &x_memtype));
  }
  if (PetscMemTypeDevice(A_memtype) && PetscMemTypeDevice(x_memtype) && PetscMemTypeDevice(y_memtype)) {
    if (m_local == m) PetscCall(DSVCyclic_CUPM_Private(hermitian_transpose, m, oldest, next, _A, _x, _y));
  } else if (m_local == m) on_host = PETSC_TRUE;
  if (x != y) PetscCall(VecRestoreArrayReadAndMemType(x, &_x));
  PetscCall(VecRestoreArrayAndMemType(y, &_y));
  PetscCall(VecRestoreArrayReadAndMemType(A, &_A));
  if (on_host) {
    PetscCall(VecGetArrayRead(A, &_A));
    PetscCall(VecGetArray(y, &_y));
    if (x == y) {
      _x = _y;
    } else {
      PetscCall(VecGetArrayRead(x, &_x));
    }
    PetscCall(DSVCyclic_Private(hermitian_transpose, m, oldest, next, _A, _x, _y));
    if (x != y) PetscCall(VecRestoreArrayRead(x, &_x));
    PetscCall(VecRestoreArray(y, &_y));
    PetscCall(VecRestoreArrayRead(A, &_A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TRSVCyclic_Private(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar y[])
{
  PetscBLASInt b_one = 1, blda, bm;
  PetscBLASInt i_oldest, i_next;
  PetscScalar  minus_one = -1.0, one = 1.0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(lda, &blda));
  PetscCall(PetscBLASIntCast(m, &bm));
  PetscCall(PetscBLASIntCast(oldest % m, &i_oldest));
  PetscCall(PetscBLASIntCast(((next - 1) % m) + 1, &i_next));
  if (i_next > i_oldest) {
    PetscBLASInt bn    = i_next - i_oldest;
    const char  *trans = hermitian_transpose ? "C" : "N";

    if (x != y) PetscCall(PetscArraycpy(&y[i_oldest], &x[i_oldest], bn));
    PetscCallBLAS("BLAStrsv", BLAStrsv_("U", trans, "N", &bn, &A[i_oldest * (lda + 1)], &blda, &y[i_oldest], &b_one));
  } else {
    PetscBLASInt bn = bm - i_oldest;
    if (x != y) {
      PetscCall(PetscArraycpy(y, x, i_next));
      PetscCall(PetscArraycpy(&y[i_oldest], &x[i_oldest], bn));
    }
    if (!hermitian_transpose) {
      if (i_next > 0) PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "N", "N", &i_next, A, &blda, y, &b_one));
      if (i_next > 0 && bn > 0) PetscCallBLAS("BLASgemv", BLASgemv_("N", &bn, &i_next, &minus_one, &A[i_oldest], &blda, y, &b_one, &one, &y[i_oldest], &b_one));
      if (bn > 0) PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "N", "N", &bn, &A[i_oldest * (lda + 1)], &blda, &y[i_oldest], &b_one));
    } else {
      if (bn > 0) PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "C", "N", &bn, &A[i_oldest * (lda + 1)], &blda, &y[i_oldest], &b_one));
      if (i_next > 0 && bn > 0) PetscCallBLAS("BLASgemv", BLASgemv_("C", &bn, &i_next, &minus_one, &A[i_oldest], &blda, &y[i_oldest], &b_one, &one, y, &b_one));
      if (i_next > 0) PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "C", "N", &i_next, A, &blda, y, &b_one));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqDenseTRSVCyclic(PetscBool hermitian_transpose, PetscInt oldest, PetscInt next, Mat A, Vec x, Vec y)
{
  const PetscScalar *_A;
  const PetscScalar *_x = NULL;
  PetscScalar       *_y;
  PetscInt           m, m_local, lda;
  PetscMemType       A_memtype, x_memtype, y_memtype;
  PetscBool          on_host = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 4);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 6);
  PetscCheckSameComm(A, 4, x, 5);
  PetscCheckSameComm(A, 4, y, 6);
  VecCheckMatCompatible(A, x, 5, y, 6);
  MatCheckAllEntriesFirstRank(A, 4);
  VecCheckAllEntriesFirstRank(x, 5);
  VecCheckAllEntriesFirstRank(y, 6);
  PetscCall(VecGetSize(x, &m));
  PetscCall(VecGetLocalSize(x, &m_local));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &_A, &A_memtype));
  PetscCall(VecGetArrayAndMemType(y, &_y, &y_memtype));
  if (x == y) {
    _x        = _y;
    x_memtype = y_memtype;
  } else {
    PetscCall(VecGetArrayReadAndMemType(x, &_x, &x_memtype));
  }
  if (PetscMemTypeDevice(A_memtype) && PetscMemTypeDevice(x_memtype) && PetscMemTypeDevice(y_memtype)) {
    if (m_local == m) PetscCall(TRSVCyclic_CUPM_Private(hermitian_transpose, m, oldest, next, _A, lda, _x, _y));
  } else if (m_local == m) on_host = PETSC_TRUE;
  if (x != y) PetscCall(VecRestoreArrayReadAndMemType(x, &_x));
  PetscCall(VecRestoreArrayAndMemType(y, &_y));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &_A));
  if (on_host) {
    PetscCall(MatDenseGetArrayRead(A, &_A));
    PetscCall(VecGetArray(y, &_y));
    if (x == y) {
      _x = _y;
    } else {
      PetscCall(VecGetArrayRead(x, &_x));
    }
    PetscCall(TRSVCyclic_Private(hermitian_transpose, m, oldest, next, _A, lda, _x, _y));
    if (x != y) PetscCall(VecRestoreArrayRead(x, &_x));
    PetscCall(VecRestoreArray(y, &_y));
    PetscCall(MatDenseRestoreArrayRead(A, &_A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HEMVCyclic_Private(PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar beta, PetscScalar y[])
{
  PetscBLASInt b_one = 1, blda, bm;
  PetscBLASInt i_oldest, i_next;
  PetscScalar  one = 1.0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(lda, &blda));
  PetscCall(PetscBLASIntCast(m, &bm));
  PetscCall(PetscBLASIntCast(oldest % m, &i_oldest));
  PetscCall(PetscBLASIntCast(((next - 1) % m) + 1, &i_next));
  if (i_next > i_oldest) {
    PetscBLASInt bn = i_next - i_oldest;

    PetscCallBLAS("BLAShemv", BLAShemv_("U", &bn, &alpha, &A[i_oldest * (lda + 1)], &blda, &x[i_oldest], &b_one, &beta, &y[i_oldest], &b_one));
  } else {
    PetscBLASInt bn = bm - i_oldest;
    if (i_next > 0) PetscCallBLAS("BLAShemv", BLAShemv_("U", &i_next, &alpha, A, &blda, x, &b_one, &beta, y, &b_one));
    if (bn > 0) PetscCallBLAS("BLAShemv", BLAShemv_("U", &bn, &alpha, &A[i_oldest * (lda + 1)], &blda, &x[i_oldest], &b_one, &beta, &y[i_oldest], &b_one));
    if (i_next > 0 && bn > 0) {
      PetscCallBLAS("BLASgemv", BLASgemv_("N", &bn, &i_next, &alpha, &A[i_oldest], &blda, x, &b_one, &one, &y[i_oldest], &b_one));
      PetscCallBLAS("BLASgemv", BLASgemv_("C", &bn, &i_next, &alpha, &A[i_oldest], &blda, &x[i_oldest], &b_one, &one, y, &b_one));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqDenseHEMVCyclic(PetscInt oldest, PetscInt next, PetscScalar alpha, Mat A, Vec x, PetscScalar beta, Vec y)
{
  const PetscScalar *_A;
  const PetscScalar *_x = NULL;
  PetscScalar       *_y;
  PetscInt           m, m_local, lda;
  PetscMemType       A_memtype, x_memtype, y_memtype;
  PetscBool          on_host = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 5);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 6);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 8);
  PetscCheckSameComm(A, 5, x, 6);
  PetscCheckSameComm(A, 5, y, 8);
  VecCheckMatCompatible(A, x, 6, y, 8);
  MatCheckAllEntriesFirstRank(A, 5);
  VecCheckAllEntriesFirstRank(x, 6);
  VecCheckAllEntriesFirstRank(y, 8);
  PetscCall(VecGetSize(x, &m));
  PetscCall(VecGetLocalSize(x, &m_local));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &_A, &A_memtype));
  PetscCall(VecGetArrayReadAndMemType(x, &_x, &x_memtype));
  PetscCall(VecGetArrayAndMemType(y, &_y, &y_memtype));
  if (PetscMemTypeDevice(A_memtype) && PetscMemTypeDevice(x_memtype) && PetscMemTypeDevice(y_memtype)) {
    if (m_local == m) PetscCall(HEMVCyclic_CUPM_Private(m, oldest, next, alpha, _A, lda, _x, beta, _y));
  } else if (m_local == m) on_host = PETSC_TRUE;
  PetscCall(VecRestoreArrayAndMemType(y, &_y));
  PetscCall(VecRestoreArrayReadAndMemType(x, &_x));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &_A));
  if (on_host) {
    PetscCall(MatDenseGetArrayRead(A, &_A));
    PetscCall(VecGetArrayRead(x, &_x));
    PetscCall(VecGetArray(y, &_y));
    PetscCall(HEMVCyclic_Private(m, oldest, next, alpha, _A, lda, _x, beta, _y));
    PetscCall(VecRestoreArray(y, &_y));
    PetscCall(VecRestoreArrayRead(x, &_x));
    PetscCall(MatDenseRestoreArrayRead(A, &_A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GEMVCyclic_Private(PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, PetscScalar alpha, const PetscScalar A[], PetscInt lda, const PetscScalar x[], PetscScalar beta, PetscScalar y[])
{
  PetscBLASInt b_one = 1, blda, bm;
  PetscBLASInt i_oldest, i_next;
  PetscScalar  one   = 1.0;
  const char  *trans = hermitian_transpose ? "C" : "N";

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(lda, &blda));
  PetscCall(PetscBLASIntCast(m, &bm));
  PetscCall(PetscBLASIntCast(oldest % m, &i_oldest));
  PetscCall(PetscBLASIntCast(((next - 1) % m) + 1, &i_next));
  if (next - oldest == m) {
    PetscCallBLAS("BLASgemv", BLASgemv_(trans, &bm, &bm, &alpha, A, &blda, x, &b_one, &beta, y, &b_one));
  } else if (i_next > i_oldest) {
    PetscBLASInt bn = i_next - i_oldest;

    PetscCallBLAS("BLASgemv", BLASgemv_(trans, &bn, &bn, &alpha, &A[i_oldest * (lda + 1)], &blda, &x[i_oldest], &b_one, &beta, &y[i_oldest], &b_one));
  } else {
    PetscBLASInt bn = bm - i_oldest;
    if (i_next > 0) PetscCallBLAS("BLASgemv", BLASgemv_(trans, &i_next, &i_next, &alpha, A, &blda, x, &b_one, &beta, y, &b_one));
    if (bn > 0) PetscCallBLAS("BLASgemv", BLASgemv_(trans, &bn, &bn, &alpha, &A[i_oldest * (lda + 1)], &blda, &x[i_oldest], &b_one, &beta, &y[i_oldest], &b_one));
    if (i_next > 0 && bn > 0) {
      if (!hermitian_transpose) {
        PetscCallBLAS("BLASgemv", BLASgemv_("N", &bn, &i_next, &alpha, &A[i_oldest], &blda, x, &b_one, &one, &y[i_oldest], &b_one));
        PetscCallBLAS("BLASgemv", BLASgemv_("N", &i_next, &bn, &alpha, &A[i_oldest * lda], &blda, &x[i_oldest], &b_one, &one, y, &b_one));
      } else {
        PetscCallBLAS("BLASgemv", BLASgemv_("C", &i_next, &bn, &alpha, &A[i_oldest * lda], &blda, x, &b_one, &one, &y[i_oldest], &b_one));
        PetscCallBLAS("BLASgemv", BLASgemv_("C", &bn, &i_next, &alpha, &A[i_oldest], &blda, &x[i_oldest], &b_one, &one, y, &b_one));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqDenseGEMVCyclic(PetscBool hermitian_transpose, PetscInt oldest, PetscInt next, PetscScalar alpha, Mat A, Vec x, PetscScalar beta, Vec y)
{
  const PetscScalar *_A;
  const PetscScalar *_x = NULL;
  PetscScalar       *_y;
  PetscInt           m, m_local, lda;
  PetscMemType       A_memtype, x_memtype, y_memtype;
  PetscBool          on_host = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 6);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 7);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 9);
  PetscCheckSameComm(A, 6, x, 7);
  PetscCheckSameComm(A, 6, y, 9);
  VecCheckMatCompatible(A, x, 7, y, 9);
  MatCheckAllEntriesFirstRank(A, 6);
  VecCheckAllEntriesFirstRank(x, 7);
  VecCheckAllEntriesFirstRank(y, 9);
  PetscCall(VecGetSize(x, &m));
  PetscCall(VecGetLocalSize(x, &m_local));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &_A, &A_memtype));
  PetscCall(VecGetArrayReadAndMemType(x, &_x, &x_memtype));
  PetscCall(VecGetArrayAndMemType(y, &_y, &y_memtype));
  if (PetscMemTypeDevice(A_memtype) && PetscMemTypeDevice(x_memtype) && PetscMemTypeDevice(y_memtype)) {
    if (m_local == m) PetscCall(GEMVCyclic_CUPM_Private(hermitian_transpose, m, oldest, next, alpha, _A, lda, _x, beta, _y));
  } else if (m_local == m) on_host = PETSC_TRUE;
  PetscCall(VecRestoreArrayAndMemType(y, &_y));
  PetscCall(VecRestoreArrayReadAndMemType(x, &_x));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &_A));
  if (on_host) {
    PetscCall(MatDenseGetArrayRead(A, &_A));
    PetscCall(VecGetArrayRead(x, &_x));
    PetscCall(VecGetArray(y, &_y));
    PetscCall(GEMVCyclic_Private(hermitian_transpose, m, oldest, next, alpha, _A, lda, _x, beta, _y));
    PetscCall(VecRestoreArray(y, &_y));
    PetscCall(VecRestoreArrayRead(x, &_x));
    PetscCall(MatDenseRestoreArrayRead(A, &_A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqDenseRowAXPBYCyclic(PetscInt oldest, PetscInt next, PetscScalar alpha, Vec x, PetscScalar beta, Mat Y, PetscInt row)
{
  const PetscScalar *_x = NULL;
  PetscScalar       *_y;
  PetscInt           m, m_local, ldy;
  PetscMemType       x_memtype, y_memtype;
  PetscBool          on_host = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(Y, MAT_CLASSID, 6);
  PetscCheckSameComm(x, 4, Y, 6);
  VecCheckMatCompatible(Y, x, 4, x, 4);
  VecCheckAllEntriesFirstRank(x, 4);
  MatCheckAllEntriesFirstRank(Y, 6);
  PetscCall(VecGetSize(x, &m));
  PetscCall(VecGetLocalSize(x, &m_local));
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDenseGetLDA(Y, &ldy));
  PetscCall(VecGetArrayReadAndMemType(x, &_x, &x_memtype));
  PetscCall(MatDenseGetArrayAndMemType(Y, &_y, &y_memtype));
  if (PetscMemTypeDevice(x_memtype) && PetscMemTypeDevice(y_memtype)) {
    if (m_local == m) PetscCall(AXPBYCyclic_CUPM_Private(m, oldest, next, alpha, _x, beta, &_y[row % m], ldy));
  } else if (m_local == m) on_host = PETSC_TRUE;
  PetscCall(MatDenseRestoreArrayAndMemType(Y, &_y));
  PetscCall(VecRestoreArrayReadAndMemType(x, &_x));
  if (on_host) {
    PetscCall(VecGetArrayRead(x, &_x));
    PetscCall(MatDenseGetArray(Y, &_y));
    PetscCall(AXPBYCylic_Private(m, oldest, next, alpha, _x, beta, &_y[row % m], ldy));
    PetscCall(MatDenseRestoreArray(Y, &_y));
    PetscCall(VecRestoreArrayRead(x, &_x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatMultColumnRange(Mat A, Vec xx, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_MultAdd, A, NULL, NULL, NULL));
  PetscUseMethod(A, "MatMultColumnRange_C", (Mat, Vec, Vec, PetscInt, PetscInt), (A, xx, yy, c_start, c_end));
  PetscCall(PetscLogEventEnd(MAT_MultAdd, A, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatMultAddColumnRange(Mat A, Vec xx, Vec zz, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_MultAdd, A, NULL, NULL, NULL));
  PetscUseMethod(A, "MatMultAddColumnRange_C", (Mat, Vec, Vec, Vec, PetscInt, PetscInt), (A, xx, zz, yy, c_start, c_end));
  PetscCall(PetscLogEventEnd(MAT_MultAdd, A, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatMultHermitianTransposeColumnRange(Mat A, Vec xx, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_MultTranspose, A, NULL, NULL, NULL));
  PetscUseMethod(A, "MatMultHermitianTransposeColumnRange_C", (Mat, Vec, Vec, PetscInt, PetscInt), (A, xx, yy, c_start, c_end));
  PetscCall(PetscLogEventEnd(MAT_MultTranspose, A, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatMultHermitianTransposeAddColumnRange(Mat A, Vec xx, Vec zz, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_MultTransposeAdd, A, NULL, NULL, NULL));
  PetscUseMethod(A, "MatMultHermitianTransposeAddColumnRange_C", (Mat, Vec, Vec, Vec, PetscInt, PetscInt), (A, xx, zz, yy, c_start, c_end));
  PetscCall(PetscLogEventEnd(MAT_MultTransposeAdd, A, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
