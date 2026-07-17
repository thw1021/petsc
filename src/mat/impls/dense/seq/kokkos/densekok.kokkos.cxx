#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petsc/private/petscimpl.h>
#include <petsc/private/kokkosimpl.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>

#include <../src/mat/impls/dense/seq/kokkos/densekok.hpp>

static PetscErrorCode MatSetOps_SeqDenseKokkos(Mat); /* Forward declaration */

/* Ensure A->spptr (Mat_SeqDenseKokkos) exists, wrapping the current host array Mat_SeqDense->v.
   The host array is assumed to hold the up-to-date values (this is the case after preallocation,
   MatSetValues(), or a conversion from a host matrix), so we mark the host side modified. */
static PetscErrorCode MatSeqDenseKokkosSetupSpptr(Mat A)
{
  Mat_SeqDense *aseq = static_cast<Mat_SeqDense *>(A->data);
  PetscInt      lda;

  PetscFunctionBegin;
  if (A->spptr) PetscFunctionReturn(PETSC_SUCCESS);
  /* Allocate the host array if it does not exist yet (e.g. a matrix created for a write accessor) */
  if (!aseq->v) PetscCall(MatSeqDenseSetPreallocation(A, NULL));
  lda = aseq->lda > 0 ? aseq->lda : A->rmap->n;
  PetscCheck(lda == A->rmap->n, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MATSEQDENSEKOKKOS requires a leading dimension (%" PetscInt_FMT ") equal to the number of rows (%" PetscInt_FMT ")", lda, A->rmap->n);
  PetscCallCXX(A->spptr = new Mat_SeqDenseKokkos(lda, A->cmap->n, aseq->v));
  PetscCallCXX(static_cast<Mat_SeqDenseKokkos *>(A->spptr)->a_dual.modify_host());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sync the matrix values to the device if the host side is newer */
PETSC_INTERN PetscErrorCode MatSeqDenseKokkosSyncDevice(Mat A)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Can't sync factorized matrix from host to device");
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCheck(densekok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing Mat_SeqDenseKokkos");
  PetscCall(KokkosDualViewSyncDevice(densekok->a_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sync the matrix values to the host if the device side is newer */
static PetscErrorCode MatSeqDenseKokkosSyncHost(Mat A)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Can't sync factorized matrix from device to host");
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCheck(densekok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing Mat_SeqDenseKokkos");
  PetscCall(KokkosDualViewSyncHost(densekok->a_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Mark the matrix values on device as modified */
PETSC_INTERN PetscErrorCode MatSeqDenseKokkosModifyDevice(Mat A)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Not supported for factorized matrices");
  PetscCheck(densekok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing Mat_SeqDenseKokkos");
  PetscCallCXX(densekok->a_dual.clear_sync_state());
  PetscCallCXX(densekok->a_dual.modify_device());
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Host array access (with sync) ------------------------- */

static PetscErrorCode MatSeqDenseGetArray_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  *array   = densekok->HostData();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArray_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->a_dual.modify_host());
  if (array) *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetArrayRead_SeqDenseKokkos(Mat A, const PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  *array   = densekok->HostData();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayRead_SeqDenseKokkos(Mat A, const PetscScalar *array[])
{
  PetscFunctionBegin;
  if (array) *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetArrayWrite_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  *array   = densekok->HostData(); /* write-only: no device->host sync */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayWrite_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->a_dual.clear_sync_state());
  PetscCallCXX(densekok->a_dual.modify_host());
  if (array) *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Device array access (memtype) ------------------------- */

static PetscErrorCode MatSeqDenseGetArrayAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a, PetscMemType *mtype)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  *a       = densekok->DeviceData();
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->a_dual.modify_device());
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  if (a) *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetArrayReadAndMemType_SeqDenseKokkos(Mat A, const PetscScalar **a, PetscMemType *mtype)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  *a       = densekok->DeviceData();
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayReadAndMemType_SeqDenseKokkos(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  if (a) *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetArrayWriteAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a, PetscMemType *mtype)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(densekok->a_dual.clear_sync_state()); /* write-only: no host->device sync */
  *a = densekok->DeviceData();
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayWriteAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->a_dual.modify_device());
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  if (a) *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Public Kokkos view accessors ------------------------- */

PetscErrorCode MatSeqDenseGetKokkosView(Mat A, ConstMatDenseKokkosView2D *kv)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(*kv = densekok->DeviceView2D(A->rmap->n, A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseRestoreKokkosView(Mat A, ConstMatDenseKokkosView2D *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseGetKokkosView(Mat A, MatDenseKokkosView2D *kv)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(*kv = densekok->DeviceView2D(A->rmap->n, A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseRestoreKokkosView(Mat A, MatDenseKokkosView2D *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseGetKokkosViewWrite(Mat A, MatDenseKokkosView2D *kv)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(densekok->a_dual.clear_sync_state());
  PetscCallCXX(*kv = densekok->DeviceView2D(A->rmap->n, A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseRestoreKokkosViewWrite(Mat A, MatDenseKokkosView2D *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Matrix-vector products (gemv) ------------------------- */

/* Generic y = op(A) x [+ z], op = N/T/C */
static PetscErrorCode MatMultAdd_SeqDenseKokkos_Private(Mat A, Vec xx, Vec yy, Vec zz, const char trans[])
{
  Mat_SeqDenseKokkos        *densekok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;
  PetscScalar                beta = 0.0;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  if (zz) {
    if (zz != yy) PetscCall(VecCopy(yy, zz)); /* z gets yy's latest data (possibly on host) */
    PetscCall(VecGetKokkosView(zz, &zv));
    beta = 1.0;
  } else {
    PetscCall(VecGetKokkosViewWrite(yy, &zv));
  }
  PetscCall(VecGetKokkosView(xx, &xv));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), trans, 1.0, densekok->DeviceView2D(A->rmap->n, A->cmap->n), xv, beta, zv));
  PetscCall(VecRestoreKokkosView(xx, &xv));
  if (zz) PetscCall(VecRestoreKokkosView(zz, &zv));
  else PetscCall(VecRestoreKokkosViewWrite(yy, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * A->rmap->n * A->cmap->n));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_SeqDenseKokkos(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseKokkos_Private(A, xx, yy, NULL, "N"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_SeqDenseKokkos(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseKokkos_Private(A, xx, yy, NULL, "T"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_SeqDenseKokkos(Mat A, Vec xx, Vec yy)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseKokkos_Private(A, xx, yy, NULL, "C"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_SeqDenseKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseKokkos_Private(A, xx, yy, zz, "N"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTransposeAdd_SeqDenseKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseKokkos_Private(A, xx, yy, zz, "T"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTransposeAdd_SeqDenseKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  PetscFunctionBegin;
  PetscCall(MatMultAdd_SeqDenseKokkos_Private(A, xx, yy, zz, "C"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Matrix-matrix products (gemm) ------------------------- */

/* C = op(A) op(B), on the device. transA/transB are "N"/"T"/"C". */
static PetscErrorCode MatMatMultNumeric_SeqDenseKokkos_Private(Mat A, Mat B, Mat C, const char transA[], const char transB[])
{
  Mat_SeqDenseKokkos *ckok;
  PetscInt            m = C->rmap->n, n = C->cmap->n, k = (transA[0] == 'N') ? A->cmap->n : A->rmap->n;
  PetscBool           Akok, Bkok;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSEKOKKOS, &Akok));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSEKOKKOS, &Bkok));
  /* If an operand is not a Kokkos matrix, fall back to the host BLAS implementation (auto-syncs via GetArray) */
  if (!Akok || !Bkok) {
    if (transA[0] == 'N' && transB[0] == 'N') PetscCall(MatMatMultNumeric_SeqDense_SeqDense(A, B, C));
    else if (transA[0] != 'N' && transB[0] == 'N') PetscCall(MatTransposeMatMultNumeric_SeqDense_SeqDense(A, B, C));
    else PetscCall(MatMatTransposeMultNumeric_SeqDense_SeqDense(A, B, C));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!m || !n) {
    PetscCall(MatZeroEntries(C));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  PetscCall(MatSeqDenseKokkosSyncDevice(B));
  PetscCall(MatSeqDenseKokkosSetupSpptr(C));
  ckok = static_cast<Mat_SeqDenseKokkos *>(C->spptr);
  {
    Mat_SeqDenseKokkos *akok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
    Mat_SeqDenseKokkos *bkok = static_cast<Mat_SeqDenseKokkos *>(B->spptr);

    PetscCallCXX(ckok->a_dual.clear_sync_state());
    PetscCallCXX(KokkosBlas::gemm(PetscGetKokkosExecutionSpace(), transA, transB, 1.0, akok->DeviceView2D(A->rmap->n, A->cmap->n), bkok->DeviceView2D(B->rmap->n, B->cmap->n), 0.0, ckok->DeviceView2D(m, n)));
  }
  PetscCall(MatSeqDenseKokkosModifyDevice(C));
  PetscCall(PetscLogGpuFlops(2.0 * m * n * k));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatMultNumeric_SeqDenseKokkos_SeqDenseKokkos(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseKokkos_Private(A, B, C, "N", "N"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTransposeMatMultNumeric_SeqDenseKokkos_SeqDenseKokkos(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseKokkos_Private(A, B, C, "T", "N"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatTransposeMultNumeric_SeqDenseKokkos_SeqDenseKokkos(Mat A, Mat B, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultNumeric_SeqDenseKokkos_Private(A, B, C, "N", "T"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Element-wise operations ------------------------- */

static PetscErrorCode MatScale_SeqDenseKokkos(Mat A, PetscScalar alpha)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(KokkosBlas::scal(PetscGetKokkosExecutionSpace(), densekok->a_dual.view_device(), alpha, densekok->a_dual.view_device()));
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscCall(PetscLogGpuFlops(1.0 * A->rmap->n * A->cmap->n));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* add alpha to the (main) diagonal */
static PetscErrorCode MatShift_SeqDenseKokkos(Mat A, PetscScalar alpha)
{
  Mat_SeqDenseKokkos *densekok;
  PetscInt            n = PetscMin(A->rmap->n, A->cmap->n);

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  {
    auto Aa = densekok->DeviceView2D(A->rmap->n, A->cmap->n);
    PetscCallCXX(Kokkos::parallel_for("MatShift", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt i) { Aa(i, i) += alpha; }));
  }
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscCall(PetscLogGpuFlops(1.0 * n));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_SeqDenseKokkos(Mat A)
{
  Mat_SeqDense *aseq = static_cast<Mat_SeqDense *>(A->data);

  PetscFunctionBegin;
  if (!aseq->v) { /* Not preallocated yet: let SeqDense allocate/zero on host */
    PetscCall(MatZeroEntries_SeqDense(A));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  {
    Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
    PetscCallCXX(densekok->a_dual.clear_sync_state());
    PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), densekok->a_dual.view_device(), 0.0));
  }
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_SeqDenseKokkos(Mat A, Vec x)
{
  Mat_SeqDenseKokkos   *densekok;
  PetscInt              n = PetscMin(A->rmap->n, A->cmap->n), len;
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x, &len));
  PetscCheck(len == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCall(VecGetKokkosViewWrite(x, &xv));
  {
    auto Aa = densekok->DeviceView2D(A->rmap->n, A->cmap->n);
    PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), xv, 0.0));
    PetscCallCXX(Kokkos::parallel_for("MatGetDiagonal", Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt i) { xv(i) = Aa(i, i); }));
  }
  PetscCall(VecRestoreKokkosViewWrite(x, &xv));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Y = alpha X + Y */
static PetscErrorCode MatAXPY_SeqDenseKokkos(Mat Y, PetscScalar alpha, Mat X, MatStructure str)
{
  Mat_SeqDenseKokkos *xkok, *ykok;
  PetscBool           Xkok;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)X, MATSEQDENSEKOKKOS, &Xkok));
  if (!Xkok) { /* X lives on host: fall back to the host implementation (Y auto-syncs via GetArray) */
    PetscCall(MatAXPY_SeqDense(Y, alpha, X, str));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(Y));
  PetscCall(MatSeqDenseKokkosSyncDevice(X));
  ykok = static_cast<Mat_SeqDenseKokkos *>(Y->spptr);
  xkok = static_cast<Mat_SeqDenseKokkos *>(X->spptr);
  PetscCallCXX(KokkosBlas::axpy(PetscGetKokkosExecutionSpace(), alpha, xkok->a_dual.view_device(), ykok->a_dual.view_device()));
  PetscCall(MatSeqDenseKokkosModifyDevice(Y));
  PetscCall(PetscLogGpuFlops(2.0 * Y->rmap->n * Y->cmap->n));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_SeqDenseKokkos(Mat A, Mat B, MatStructure str)
{
  Mat_SeqDenseKokkos *akok, *bkok;
  PetscBool           Bkok;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSEKOKKOS, &Bkok));
  if (A == B || !Bkok) {
    PetscCall(MatCopy_SeqDense(A, B, str));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(A->rmap->n == B->rmap->n && A->cmap->n == B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "size(B) != size(A)");
  PetscCall(MatSeqDenseKokkosSetupSpptr(B));
  bkok = static_cast<Mat_SeqDenseKokkos *>(B->spptr);
  akok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  if (!akok || !bkok) { /* one side has no device storage yet: copy on the host */
    PetscCall(MatCopy_SeqDense(A, B, str));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  PetscCallCXX(bkok->a_dual.clear_sync_state());
  PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), bkok->a_dual.view_device(), akok->a_dual.view_device()));
  PetscCall(MatSeqDenseKokkosModifyDevice(B));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Create / Convert / Duplicate / Destroy ------------------------- */

static PetscErrorCode MatConvert_SeqDenseKokkos_SeqDense(Mat, MatType, MatReuse, Mat *);

static PetscErrorCode MatDestroy_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    delete static_cast<Mat_SeqDenseKokkos *>(A->spptr);
    A->spptr = NULL;
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqdensekokkos_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayReadAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWriteAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayReadAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayWriteAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdensekokkos_seqdensekokkos_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdensekokkos_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdense_seqdensekokkos_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_seqdensekokkos_C", NULL));
  /* The host SeqDense destroy restores/frees the rest (data, host-composed functions) */
  PetscCall(MatDestroy_SeqDense(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_SeqDenseKokkos(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  PetscFunctionBegin;
  /* MatDuplicate_SeqDense() creates *B with A's type (MATSEQDENSEKOKKOS) and, for MAT_COPY_VALUES, copies
     values on the host through MatDenseGetArrayRead()/MatDenseGetArrayWrite(), which sync as needed. */
  PetscCall(MatDuplicate_SeqDense(A, cpvalues, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Convert a MATSEQDENSEKOKKOS back to a host MATSEQDENSE */
static PetscErrorCode MatConvert_SeqDenseKokkos_SeqDense(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat B;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatConvert_Basic(A, mtype, reuse, newmat));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* MAT_INPLACE_MATRIX */
  B = *newmat;
  PetscCall(MatSeqDenseKokkosSyncHost(B));
  delete static_cast<Mat_SeqDenseKokkos *>(B->spptr);
  B->spptr       = NULL;
  B->offloadmask = PETSC_OFFLOAD_CPU;

  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECSTANDARD, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQDENSE));

  /* Restore host ops that we overrode for the device type */
  B->ops->destroy                   = MatDestroy_SeqDense;
  B->ops->duplicate                 = MatDuplicate_SeqDense;
  B->ops->mult                      = MatMult_SeqDense;
  B->ops->multadd                   = MatMultAdd_SeqDense;
  B->ops->multtranspose             = MatMultTranspose_SeqDense;
  B->ops->multtransposeadd          = MatMultTransposeAdd_SeqDense;
  B->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqDense;
  B->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqDense;
  B->ops->matmultnumeric            = MatMatMultNumeric_SeqDense_SeqDense;
  B->ops->mattransposemultnumeric   = MatMatTransposeMultNumeric_SeqDense_SeqDense;
  B->ops->transposematmultnumeric   = MatTransposeMatMultNumeric_SeqDense_SeqDense;
  B->ops->scale                     = MatScale_SeqDense;
  B->ops->shift                     = MatShift_SeqDense;
  B->ops->axpy                      = MatAXPY_SeqDense;
  B->ops->copy                      = MatCopy_SeqDense;
  B->ops->zeroentries               = MatZeroEntries_SeqDense;
  B->ops->getdiagonal               = MatGetDiagonal_SeqDense;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqdensekokkos_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseGetArray_C", MatDenseGetArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseGetArrayRead_C", MatDenseGetArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseGetArrayWrite_C", MatDenseGetArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseRestoreArray_C", MatDenseRestoreArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseRestoreArrayRead_C", MatDenseRestoreArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseRestoreArrayWrite_C", MatDenseRestoreArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseGetArrayAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseGetArrayReadAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseGetArrayWriteAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseRestoreArrayAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseRestoreArrayReadAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDenseRestoreArrayWriteAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqdensekokkos_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqdensekokkos_seqdensekokkos_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqdensekokkos_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqdense_seqdensekokkos_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqDenseKokkos(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat           B;
  Mat_SeqDense *aseq;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatConvert_Basic(A, mtype, reuse, newmat)); /* MatSetType(newmat, ...) triggers the INPLACE path below */
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* MAT_INPLACE_MATRIX */
  B    = *newmat;
  aseq = static_cast<Mat_SeqDense *>(B->data);

  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECKOKKOS, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQDENSEKOKKOS));
  PetscCall(MatSetOps_SeqDenseKokkos(B));

  PetscCheck(!B->spptr, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Expected NULL (Mat_SeqDenseKokkos*)B->spptr");
  if (aseq->v) PetscCall(MatSeqDenseKokkosSetupSpptr(B)); /* host holds the current values */
  B->offloadmask = PETSC_OFFLOAD_KOKKOS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqdensekokkos_seqdense_C", MatConvert_SeqDenseKokkos_SeqDense));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATSEQDENSEKOKKOS - "seqdensekokkos" - A matrix type for sequential dense matrices whose data can live on a
   Kokkos backend (e.g. a GPU).

   Options Database Key:
.  -mat_type seqdensekokkos - sets the matrix type to `MATSEQDENSEKOKKOS` during a call to `MatSetFromOptions()`

   Level: beginner

   Note:
   The matrix values are stored in conventional column-major dense format and mirrored to the Kokkos default
   memory space. Compute-heavy operations (`MatMult()`, `MatMatMult()`, `MatScale()`, `MatAXPY()`, ...) run on
   the device; operations without a device implementation fall back to the host, syncing values as needed.
   Matrix factorization and triangular solves are performed on the host.

.seealso: [](ch_matrices), `Mat`, `MATDENSEKOKKOS`, `MATMPIDENSEKOKKOS`, `MATSEQDENSE`, `MatCreateSeqDenseKokkos()`, `MATSEQDENSECUDA`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate_SeqDense(A));
  PetscCall(MatConvert_SeqDense_SeqDenseKokkos(A, MATSEQDENSEKOKKOS, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateSeqDenseKokkos - Creates a sequential dense matrix of type `MATSEQDENSEKOKKOS`.

  Collective

  Input Parameters:
+ comm - MPI communicator, set to `PETSC_COMM_SELF`
. m    - number of rows
. n    - number of columns
- data - optional location of matrix data in column-major order. Use `NULL` to have PETSc control all
         matrix memory allocation.

  Output Parameter:
. A - the matrix

  Level: intermediate

  Note:
  The `data` argument is intended primarily for Fortran programmers who wish to allocate their own matrix
  memory. Most users should pass `NULL` for `data`.

.seealso: [](ch_matrices), `Mat`, `MATSEQDENSEKOKKOS`, `MatCreate()`, `MatCreateSeqDense()`, `MatCreateDenseKokkos()`
@*/
PetscErrorCode MatCreateSeqDenseKokkos(MPI_Comm comm, PetscInt m, PetscInt n, PetscScalar *data, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, m, n));
  PetscCall(MatSetType(*A, MATSEQDENSEKOKKOS));
  PetscCall(MatSeqDenseSetPreallocation(*A, data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Product setup ------------------------- */

static PetscErrorCode MatSetOps_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_KOKKOS; /* used only as a cheap Kokkos-type tag */
  A->boundtocpu  = PETSC_FALSE;

  A->ops->destroy                   = MatDestroy_SeqDenseKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqDenseKokkos;
  A->ops->mult                      = MatMult_SeqDenseKokkos;
  A->ops->multadd                   = MatMultAdd_SeqDenseKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqDenseKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqDenseKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqDenseKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqDenseKokkos;
  A->ops->matmultnumeric            = MatMatMultNumeric_SeqDenseKokkos_SeqDenseKokkos;
  A->ops->mattransposemultnumeric   = MatMatTransposeMultNumeric_SeqDenseKokkos_SeqDenseKokkos;
  A->ops->transposematmultnumeric   = MatTransposeMatMultNumeric_SeqDenseKokkos_SeqDenseKokkos;
  A->ops->scale                     = MatScale_SeqDenseKokkos;
  A->ops->shift                     = MatShift_SeqDenseKokkos;
  A->ops->axpy                      = MatAXPY_SeqDenseKokkos;
  A->ops->copy                      = MatCopy_SeqDenseKokkos;
  A->ops->zeroentries               = MatZeroEntries_SeqDenseKokkos;
  A->ops->getdiagonal               = MatGetDiagonal_SeqDenseKokkos;

  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArray_C", MatSeqDenseGetArray_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayRead_C", MatSeqDenseGetArrayRead_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWrite_C", MatSeqDenseGetArrayWrite_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArray_C", MatSeqDenseRestoreArray_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayRead_C", MatSeqDenseRestoreArrayRead_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayWrite_C", MatSeqDenseRestoreArrayWrite_SeqDenseKokkos));

  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayAndMemType_C", MatSeqDenseGetArrayAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayReadAndMemType_C", MatSeqDenseGetArrayReadAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWriteAndMemType_C", MatSeqDenseGetArrayWriteAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayAndMemType_C", MatSeqDenseRestoreArrayAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayReadAndMemType_C", MatSeqDenseRestoreArrayReadAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayWriteAndMemType_C", MatSeqDenseRestoreArrayWriteAndMemType_SeqDenseKokkos));

  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_seqdensekokkos_C", MatProductSetFromOptions_SeqAIJ_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdensekokkos_seqdensekokkos_C", MatProductSetFromOptions_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdensekokkos_seqdense_C", MatProductSetFromOptions_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdense_seqdensekokkos_C", MatProductSetFromOptions_SeqDense));
  PetscFunctionReturn(PETSC_SUCCESS);
}
