#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/kokkosimpl.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>

#include <../src/mat/impls/dense/seq/kokkos/densekok.hpp>

static PetscErrorCode MatSetOps_SeqDenseKokkos(Mat); /* Forward declaration */
static PetscErrorCode MatBindToCPU_SeqDenseKokkos(Mat, PetscBool);
static PetscErrorCode MatConvert_SeqDenseKokkos_SeqDense(Mat, MatType, MatReuse, Mat *);

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
  PetscCheck(lda == A->rmap->n || A->rmap->n == 0, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MATSEQDENSEKOKKOS requires a leading dimension (%" PetscInt_FMT ") equal to the number of rows (%" PetscInt_FMT ")", lda, A->rmap->n);
  PetscCallCXX(A->spptr = new Mat_SeqDenseKokkos(A->rmap->n, A->cmap->n, aseq->v));
  PetscCallCXX(static_cast<Mat_SeqDenseKokkos *>(A->spptr)->a_dual.modify_host());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Drop the device mirror; it is lazily rebuilt around the (possibly different) host array on the next device access */
static PetscErrorCode MatSeqDenseKokkosResetSpptr(Mat A)
{
  PetscFunctionBegin;
  delete static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  A->spptr = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sync the matrix values to the device if the host side is newer */
PETSC_INTERN PetscErrorCode MatSeqDenseKokkosSyncDevice(Mat A)
{
  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Can't sync factorized matrix from host to device");
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  PetscCall(KokkosDualViewSyncDevice(static_cast<Mat_SeqDenseKokkos *>(A->spptr)->a_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sync the matrix values to the host if the device side is newer */
static PetscErrorCode MatSeqDenseKokkosSyncHost(Mat A)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  if (A->factortype != MAT_FACTOR_NONE) {
    /* An in-place factorization (host LAPACK) made the host array the single fresh copy; there is nothing to sync */
    PetscCheck(!densekok->a_dual.need_sync_host(), PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Factorized matrix has newer values on the device");
    PetscFunctionReturn(PETSC_SUCCESS);
  }
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

/* Make the host array the single fresh copy before a host-side path reads and writes it outside the accessor protocol */
static PetscErrorCode MatSeqDenseKokkosPrepareHostWrite(Mat A)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(densekok->a_dual.clear_sync_state());
  PetscCallCXX(densekok->a_dual.modify_host());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Host array access (with sync) ------------------------- */

/* Also composed under MatDenseGetArrayRead_C and MatDenseGetArrayWrite_C: PETSc routes partial writers
   through the write accessor (e.g. MatDenseGetColumnVecWrite() obtains the whole array to write a single
   column), so the entries the caller does not touch must be current and a write access cannot skip the
   device-to-host sync. */
static PetscErrorCode MatSeqDenseGetArray_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  *array = static_cast<Mat_SeqDenseKokkos *>(A->spptr)->HostData();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Also composed under MatDenseRestoreArrayWrite_C, see MatSeqDenseGetArray_SeqDenseKokkos() */
static PetscErrorCode MatSeqDenseRestoreArray_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  /* A NULL spptr means the array was obtained through the host path (e.g. while bound to CPU); the
     host array is then already the single fresh copy and there is no device state to update */
  if (densekok) PetscCallCXX(densekok->a_dual.modify_host());
  if (array) *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayRead_SeqDenseKokkos(Mat A, const PetscScalar *array[])
{
  PetscFunctionBegin;
  if (array) *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Device array access (memtype) ------------------------- */

/* Also composed under the Read and Write variants; a device write access syncs for the same
   partial-writer reason as the host write accessor */
static PetscErrorCode MatSeqDenseGetArrayAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a, PetscMemType *mtype)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  *a = static_cast<Mat_SeqDenseKokkos *>(A->spptr)->DeviceData();
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Also composed under the Write variant. MatDenseRestoreArray*AndMemType() increases the object state */
static PetscErrorCode MatSeqDenseRestoreArrayAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->a_dual.modify_device());
  if (a) *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayReadAndMemType_SeqDenseKokkos(Mat A, const PetscScalar **a)
{
  PetscFunctionBegin;
  if (a) *a = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Matrix-vector products (gemv) ------------------------- */

/* VecGetKokkosView() on a non-Kokkos vector is an invalid cast that optimized builds do not catch;
   check explicitly so the user gets an error instead of a segfault */
static PetscErrorCode MatSeqDenseKokkosCheckVecKokkos(Vec v)
{
  PetscBool iskok;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)v, &iskok, VECSEQKOKKOS, VECMPIKOKKOS, ""));
  PetscCheck(iskok, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_WRONG, "Vector of type %s is not supported by MATSEQDENSEKOKKOS, which requires Kokkos vectors", ((PetscObject)v)->type_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Generic y = op(A) x [+ z], op = N/T/C */
static PetscErrorCode MatMultAdd_SeqDenseKokkos_Private(Mat A, Vec xx, Vec yy, Vec zz, const char trans[])
{
  Mat_SeqDenseKokkos        *densekok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;
  PetscScalar                beta = 0.0;

  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosCheckVecKokkos(xx));
  PetscCall(MatSeqDenseKokkosCheckVecKokkos(zz ? zz : yy));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  if (zz) {
    if (zz != yy) PetscCall(VecCopy(yy, zz)); /* z gets yy's latest data (possibly on host) */
    PetscCall(VecGetKokkosView(zz, &zv));
    beta = 1.0;
  } else PetscCall(VecGetKokkosViewWrite(yy, &zv));
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
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSetupSpptr(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), densekok->a_dual.view_device(), 0.0));
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
  PetscCall(MatSeqDenseKokkosCheckVecKokkos(x));
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
  Mat_SeqDenseKokkos *akok = static_cast<Mat_SeqDenseKokkos *>(A->spptr), *bkok;
  PetscBool           Bkok, Bdense;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSEKOKKOS, &Bkok));
  if (!Bkok && A != B) {
    PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSE, &Bdense));
    if (Bdense) {
      /* MatCopy_SeqDense() would take the per-row MatCopy_Basic() path because the copy ops differ;
         both matrices are Mat_SeqDense underneath, so copy through the array accessors directly */
      const PetscScalar *va;
      PetscScalar       *vb;
      PetscInt           m = A->rmap->n, n = A->cmap->n, lda, ldb;

      PetscCheck(m == B->rmap->n && n == B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "size(B) != size(A)");
      PetscCall(MatDenseGetLDA(A, &lda));
      PetscCall(MatDenseGetLDA(B, &ldb));
      PetscCall(MatDenseGetArrayRead(A, &va));
      PetscCall(MatDenseGetArrayWrite(B, &vb));
      if (lda > m || ldb > m) {
        for (PetscInt j = 0; j < n; j++) PetscCall(PetscArraycpy(vb + j * ldb, va + j * lda, m));
      } else PetscCall(PetscArraycpy(vb, va, m * n));
      PetscCall(MatDenseRestoreArrayWrite(B, &vb));
      PetscCall(MatDenseRestoreArrayRead(A, &va));
    } else PetscCall(MatCopy_Basic(A, B, str));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (A == B || !akok) { /* self copy, or A has no device storage yet: copy on the host */
    PetscCall(MatCopy_SeqDense(A, B, str));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(A->rmap->n == B->rmap->n && A->cmap->n == B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "size(B) != size(A)");
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  PetscCall(MatSeqDenseKokkosSetupSpptr(B));
  bkok = static_cast<Mat_SeqDenseKokkos *>(B->spptr);
  PetscCallCXX(Kokkos::deep_copy(PetscGetKokkosExecutionSpace(), bkok->a_dual.view_device(), akok->a_dual.view_device()));
  PetscCall(MatSeqDenseKokkosModifyDevice(B));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------- Host fallbacks that bypass the accessor protocol (sync first) ------------------- */

static PetscErrorCode MatSOR_SeqDenseKokkos(Mat A, Vec bb, PetscReal omega, MatSORType flag, PetscReal shift, PetscInt its, PetscInt lits, Vec xx)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatSOR_SeqDense(A, bb, omega, flag, shift, its, lits, xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrix_SeqDenseKokkos(Mat A, IS isrow, IS iscol, MatReuse scall, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatCreateSubMatrix_SeqDense(A, isrow, iscol, scall, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrices_SeqDenseKokkos(Mat A, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *B[])
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatCreateSubMatrices_SeqDense(A, n, irow, icol, scall, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* In-place factorizations run host LAPACK on Mat_SeqDense->v; afterwards the host array holds the factors
   and the device mirror is dead, which MatSeqDenseKokkosSyncHost() accounts for */
static PetscErrorCode MatLUFactor_SeqDenseKokkos(Mat A, IS row, IS col, const MatFactorInfo *minfo)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosPrepareHostWrite(A));
  PetscCall(MatLUFactor_SeqDense(A, row, col, minfo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactor_SeqDenseKokkos(Mat A, IS perm, const MatFactorInfo *minfo)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosPrepareHostWrite(A));
  PetscCall(MatCholeskyFactor_SeqDense(A, perm, minfo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatQRFactor_SeqDenseKokkos(Mat A, IS col, const MatFactorInfo *minfo)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosPrepareHostWrite(A));
  PetscCall(MatQRFactor_SeqDense(A, col, minfo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultColumnRange_SeqDenseKokkos(Mat A, Vec xx, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatMultColumnRange_SeqDense(A, xx, yy, c_start, c_end));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAddColumnRange_SeqDenseKokkos(Mat A, Vec xx, Vec yy, Vec zz, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatMultAddColumnRange_SeqDense(A, xx, yy, zz, c_start, c_end));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTransposeColumnRange_SeqDenseKokkos(Mat A, Vec xx, Vec yy, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatMultHermitianTransposeColumnRange_SeqDense(A, xx, yy, c_start, c_end));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTransposeAddColumnRange_SeqDenseKokkos(Mat A, Vec xx, Vec yy, Vec zz, PetscInt c_start, PetscInt c_end)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatMultHermitianTransposeAddColumnRange_SeqDense(A, xx, yy, zz, c_start, c_end));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------- Host-array bookkeeping (the host paths swap Mat_SeqDense->v) ------------------- */

static PetscErrorCode MatDensePlaceArray_SeqDenseKokkos(Mat A, const PetscScalar *array)
{
  PetscFunctionBegin;
  /* the current host array is stashed and later restored by MatDenseResetArray(); bring it up to date first */
  if (A->spptr) PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatSeqDenseKokkosResetSpptr(A));
  PetscCall(MatDensePlaceArray_SeqDense(A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseResetArray_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  /* the placed array must hold the latest values when it is handed back to its owner */
  if (A->spptr) PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatSeqDenseKokkosResetSpptr(A));
  PetscCall(MatDenseResetArray_SeqDense(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseReplaceArray_SeqDenseKokkos(Mat A, const PetscScalar *array)
{
  PetscFunctionBegin;
  /* the previous values are discarded by contract */
  PetscCall(MatSeqDenseKokkosResetSpptr(A));
  PetscCall(MatDenseReplaceArray_SeqDense(A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseSetPreallocation_SeqDenseKokkos(Mat A, PetscScalar *data)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosResetSpptr(A));
  PetscCall(MatSeqDenseSetPreallocation_SeqDense(A, data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseSetLDA_SeqDenseKokkos(Mat A, PetscInt lda)
{
  PetscFunctionBegin;
  /* a matrix without rows carries no data, so any lda is acceptable (LAPACK's max(1,m) convention) */
  PetscCheck(lda == A->rmap->n || A->rmap->n == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "MATSEQDENSEKOKKOS requires a leading dimension (%" PetscInt_FMT ") equal to the number of rows (%" PetscInt_FMT ")", lda, A->rmap->n);
  PetscCall(MatDenseSetLDA_SeqDense(A, lda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseGetSubMatrix_SeqDenseKokkos(Mat A, PetscInt rbegin, PetscInt rend, PetscInt cbegin, PetscInt cend, Mat *v)
{
  Mat_SeqDense *a    = static_cast<Mat_SeqDense *>(A->data);
  PetscBool     ckok = PETSC_FALSE;

  PetscFunctionBegin;
  /* the submatrix wraps the host array directly; the host side is the fresh one for both reads and writes
     until MatDenseRestoreSubMatrix() */
  PetscCall(MatSeqDenseKokkosPrepareHostWrite(A));
  if (a->cmat) PetscCall(PetscObjectTypeCompare((PetscObject)a->cmat, MATSEQDENSEKOKKOS, &ckok));
  if (rbegin == 0 && rend == A->rmap->n) {
    /* a full-row (column) window is packed like the parent (lda == rows), so it can be a device matrix */
    PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
    PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
    if (a->cmat && (!ckok || cend - cbegin != a->cmat->cmap->N || rend - rbegin != a->cmat->rmap->N)) PetscCall(MatDestroy(&a->cmat));
    if (!a->cmat) PetscCall(MatCreateSeqDenseKokkos(PetscObjectComm((PetscObject)A), rend - rbegin, cend - cbegin, PetscSafePointerPlusOffset(a->v, (size_t)cbegin * a->lda), &a->cmat));
    else PetscCall(MatDensePlaceArray(a->cmat, PetscSafePointerPlusOffset(a->v, (size_t)cbegin * a->lda)));
    a->matinuse = cbegin + 1;
    *v          = a->cmat;
  } else {
    /* a row subrange makes the window strided (lda > rows), which MATSEQDENSEKOKKOS does not support;
       hand out a host window instead (destroying a cached device window of the wrong kind first) */
    if (ckok) PetscCall(MatDestroy(&a->cmat));
    PetscCall(MatDenseGetSubMatrix_SeqDense(A, rbegin, rend, cbegin, cend, v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseRestoreSubMatrix_SeqDenseKokkos(Mat A, Mat *v)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(MatDenseRestoreSubMatrix_SeqDense(A, v));
  /* writes through the window landed in the host array (directly for a host window, via the sync in
     MatDenseResetArray() for a device window); a device op on A between Get and Restore may have cleared
     the host-modified mark set at Get time, so re-mark it */
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  if (densekok) {
    PetscCallCXX(densekok->a_dual.clear_sync_state());
    PetscCallCXX(densekok->a_dual.modify_host());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Column vector accessors (device) ------------------------- */

/* The cached column vector aliases the matrix's device column through VecKokkosPlaceArray() and keeps its
   own host buffer, so consumers move at most one column, not the whole matrix, between host and device.
   Also composed under the Write names: the write accessor must preserve the other columns, so it needs
   the same device sync as the read-write accessor. */
static PetscErrorCode MatDenseGetColumnVec_SeqDenseKokkos(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = static_cast<Mat_SeqDense *>(A->data);

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) PetscCall(MatDenseCreateColumnVec_Private(A, &a->cvec));
  a->vecinuse = col + 1;
  PetscCall(MatSeqDenseGetArrayAndMemType_SeqDenseKokkos(A, (PetscScalar **)&a->ptrinuse, NULL));
  PetscCall(VecKokkosPlaceArray(a->cvec, (PetscScalar *)PetscSafePointerPlusOffset(a->ptrinuse, (size_t)col * (size_t)a->lda)));
  *v = a->cvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseRestoreColumnVec_SeqDenseKokkos(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = static_cast<Mat_SeqDense *>(A->data);

  PetscFunctionBegin;
  PetscCheck(a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  VecCheckAssembled(a->cvec);
  a->vecinuse = 0;
  /* pushes host-side writes of the column back to the matrix's device column */
  PetscCall(VecKokkosResetArray(a->cvec));
  PetscCall(MatSeqDenseRestoreArrayAndMemType_SeqDenseKokkos(A, (PetscScalar **)&a->ptrinuse));
  PetscCall(PetscObjectStateIncrease((PetscObject)A)); /* the interface leaves the state increase to the implementation */
  if (v) *v = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseGetColumnVecRead_SeqDenseKokkos(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = static_cast<Mat_SeqDense *>(A->data);

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) PetscCall(MatDenseCreateColumnVec_Private(A, &a->cvec));
  a->vecinuse = col + 1;
  PetscCall(MatSeqDenseGetArrayAndMemType_SeqDenseKokkos(A, (PetscScalar **)&a->ptrinuse, NULL));
  PetscCall(VecKokkosPlaceArray(a->cvec, (PetscScalar *)PetscSafePointerPlusOffset(a->ptrinuse, (size_t)col * (size_t)a->lda)));
  PetscCall(VecLockReadPush(a->cvec));
  *v = a->cvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseRestoreColumnVecRead_SeqDenseKokkos(Mat A, PetscInt col, Vec *v)
{
  Mat_SeqDense *a = static_cast<Mat_SeqDense *>(A->data);

  PetscFunctionBegin;
  PetscCheck(a->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  VecCheckAssembled(a->cvec);
  a->vecinuse = 0;
  PetscCall(VecLockReadPop(a->cvec));
  PetscCall(VecKokkosResetArray(a->cvec));
  PetscCall(MatSeqDenseRestoreArrayReadAndMemType_SeqDenseKokkos(A, (const PetscScalar **)&a->ptrinuse));
  if (v) *v = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------- Create / Convert / Duplicate / Destroy ------------------------- */

static PetscErrorCode MatDestroy_SeqDenseKokkos(Mat A)
{
  Mat_SeqDense       *aseq     = static_cast<Mat_SeqDense *>(A->data);
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  /* If the user provided the host array, hand the freshest values back, like the CUDA/HIP dense types.
     Do not gate this on need_sync_host(): KokkosDualViewSyncHost() skips the copy on its own, but it
     always fences, which the user-owned array needs before it is handed back (an async device kernel
     that only reads the aliased array on unified memory never marks the device modified) */
  if (densekok && aseq->user_alloc) PetscCall(KokkosDualViewSyncHost(densekok->a_dual, PetscGetKokkosExecutionSpace()));
  PetscCall(MatSeqDenseKokkosResetSpptr(A));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqdensekokkos_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayReadAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWriteAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayReadAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayWriteAndMemType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_seqdensekokkos_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijkokkos_seqdensekokkos_C", NULL));
  /* The host SeqDense destroy restores/frees the rest (data, host-composed functions) */
  PetscCall(MatDestroy_SeqDense(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_SeqDenseKokkos(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  Mat_SeqDenseKokkos *akok      = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  MatDuplicateOption  hcpvalues = cpvalues;

  PetscFunctionBegin;
  /* When the device holds the freshest values, copy them device-to-device instead of bouncing the whole
     matrix through the host (MatDuplicate_SeqDense() copies on the host through the array accessors) */
  if (cpvalues == MAT_COPY_VALUES && akok && akok->a_dual.need_sync_host()) hcpvalues = MAT_DO_NOT_COPY_VALUES;
  PetscCall(MatDuplicate_SeqDense(A, hcpvalues, B));
  /* MatDuplicateNoCreate_SeqDense() preallocates only plain MATSEQDENSE duplicates (a MAT_COPY_VALUES
     duplicate is preallocated as a side effect of the array accessors); do it here so that the duplicate
     is usable with the other MatDuplicateOption values, like the CUDA/HIP dense types */
  if (hcpvalues != MAT_COPY_VALUES && !(*B)->preallocated) PetscCall(MatSeqDenseSetPreallocation(*B, NULL));
  if (hcpvalues != cpvalues) PetscCall(MatCopy_SeqDenseKokkos(A, *B, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Convert a MATSEQDENSEKOKKOS back to a host MATSEQDENSE */
static PetscErrorCode MatConvert_SeqDenseKokkos_SeqDense(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat           B;
  Mat_SeqDense *aseq;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatConvert_Basic(A, mtype, reuse, newmat));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* MAT_INPLACE_MATRIX */
  B    = *newmat;
  aseq = static_cast<Mat_SeqDense *>(B->data);
  if (B->spptr) PetscCall(MatSeqDenseKokkosSyncHost(B));
  PetscCall(MatSeqDenseKokkosResetSpptr(B));
  /* The cached column vector/matrix were created with the Kokkos vector type */
  PetscCall(VecDestroy(&aseq->cvec));
  PetscCall(MatDestroy(&aseq->cmat));
  B->offloadmask = PETSC_OFFLOAD_CPU;

  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECSTANDARD, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQDENSE));
  PetscCall(MatBindToCPU_SeqDenseKokkos(B, PETSC_TRUE)); /* installs the host ops and host composed functions */
  B->ops->destroy   = MatDestroy_SeqDense;
  B->ops->duplicate = MatDuplicate_SeqDense;
  B->ops->bindtocpu = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqdensekokkos_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqdensekokkos_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaijkokkos_seqdensekokkos_C", NULL));
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
  /* The cached column vector/matrix were created with the host vector type */
  PetscCall(VecDestroy(&aseq->cvec));
  PetscCall(MatDestroy(&aseq->cmat));

  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECKOKKOS, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQDENSEKOKKOS));
  PetscCall(MatSetOps_SeqDenseKokkos(B));

  PetscCheck(!B->spptr, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Expected NULL (Mat_SeqDenseKokkos*)B->spptr");
  if (aseq->v) PetscCall(MatSeqDenseKokkosSetupSpptr(B)); /* host holds the current values */
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

/* ------------------------- Ops installation ------------------------- */

/* One symmetric host/device table shared by MatBindToCPU(), the convert-to-device path (to_host =
   PETSC_FALSE) and the convert-to-host path (to_host = PETSC_TRUE), so that the two directions cannot
   drift apart */
static PetscErrorCode MatBindToCPU_SeqDenseKokkos(Mat A, PetscBool flg)
{
  Mat_SeqDense *a = static_cast<Mat_SeqDense *>(A->data);

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse, PetscObjectComm((PetscObject)A), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse, PetscObjectComm((PetscObject)A), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  /* Bound host paths write Mat_SeqDense->v directly; make the host array the single fresh copy */
  if (flg && A->spptr && A->factortype == MAT_FACTOR_NONE) PetscCall(MatSeqDenseKokkosPrepareHostWrite(A));
  A->boundtocpu = flg;

#define MatSetOp_SeqDenseKokkos(op, hostfn, devfn)       A->ops->op = flg ? (hostfn) : (devfn)
#define MatComposeOp_SeqDenseKokkos(name, hostfn, devfn) PetscObjectComposeFunction((PetscObject)A, name, flg ? (PetscVoidFn *)(hostfn) : (PetscVoidFn *)(devfn))
  MatSetOp_SeqDenseKokkos(mult, MatMult_SeqDense, MatMult_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(multadd, MatMultAdd_SeqDense, MatMultAdd_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(multtranspose, MatMultTranspose_SeqDense, MatMultTranspose_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(multtransposeadd, MatMultTransposeAdd_SeqDense, MatMultTransposeAdd_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(multhermitiantranspose, MatMultHermitianTranspose_SeqDense, MatMultHermitianTranspose_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(multhermitiantransposeadd, MatMultHermitianTransposeAdd_SeqDense, MatMultHermitianTransposeAdd_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(matmultnumeric, MatMatMultNumeric_SeqDense_SeqDense, MatMatMultNumeric_SeqDenseKokkos_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(mattransposemultnumeric, MatMatTransposeMultNumeric_SeqDense_SeqDense, MatMatTransposeMultNumeric_SeqDenseKokkos_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(transposematmultnumeric, MatTransposeMatMultNumeric_SeqDense_SeqDense, MatTransposeMatMultNumeric_SeqDenseKokkos_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(scale, MatScale_SeqDense, MatScale_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(shift, MatShift_SeqDense, MatShift_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(axpy, MatAXPY_SeqDense, MatAXPY_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(copy, MatCopy_SeqDense, MatCopy_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(zeroentries, MatZeroEntries_SeqDense, MatZeroEntries_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(getdiagonal, MatGetDiagonal_SeqDense, MatGetDiagonal_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(sor, MatSOR_SeqDense, MatSOR_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(createsubmatrix, MatCreateSubMatrix_SeqDense, MatCreateSubMatrix_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(createsubmatrices, MatCreateSubMatrices_SeqDense, MatCreateSubMatrices_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(lufactor, MatLUFactor_SeqDense, MatLUFactor_SeqDenseKokkos);
  MatSetOp_SeqDenseKokkos(choleskyfactor, MatCholeskyFactor_SeqDense, MatCholeskyFactor_SeqDenseKokkos);

  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetArray_C", MatDenseGetArray_SeqDense, MatSeqDenseGetArray_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetArrayRead_C", MatDenseGetArray_SeqDense, MatSeqDenseGetArray_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetArrayWrite_C", MatDenseGetArray_SeqDense, MatSeqDenseGetArray_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreArray_C", MatDenseRestoreArray_SeqDense, MatSeqDenseRestoreArray_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreArrayRead_C", MatDenseRestoreArray_SeqDense, MatSeqDenseRestoreArrayRead_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreArrayWrite_C", MatDenseRestoreArray_SeqDense, MatSeqDenseRestoreArray_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetArrayAndMemType_C", NULL, MatSeqDenseGetArrayAndMemType_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetArrayReadAndMemType_C", NULL, MatSeqDenseGetArrayAndMemType_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetArrayWriteAndMemType_C", NULL, MatSeqDenseGetArrayAndMemType_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreArrayAndMemType_C", NULL, MatSeqDenseRestoreArrayAndMemType_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreArrayReadAndMemType_C", NULL, MatSeqDenseRestoreArrayReadAndMemType_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreArrayWriteAndMemType_C", NULL, MatSeqDenseRestoreArrayAndMemType_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDensePlaceArray_C", MatDensePlaceArray_SeqDense, MatDensePlaceArray_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseResetArray_C", MatDenseResetArray_SeqDense, MatDenseResetArray_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseReplaceArray_C", MatDenseReplaceArray_SeqDense, MatDenseReplaceArray_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatSeqDenseSetPreallocation_C", MatSeqDenseSetPreallocation_SeqDense, MatSeqDenseSetPreallocation_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseSetLDA_C", MatDenseSetLDA_SeqDense, MatDenseSetLDA_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetSubMatrix_C", MatDenseGetSubMatrix_SeqDense, MatDenseGetSubMatrix_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreSubMatrix_C", MatDenseRestoreSubMatrix_SeqDense, MatDenseRestoreSubMatrix_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetColumnVec_C", MatDenseGetColumnVec_SeqDense, MatDenseGetColumnVec_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreColumnVec_C", MatDenseRestoreColumnVec_SeqDense, MatDenseRestoreColumnVec_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetColumnVecRead_C", MatDenseGetColumnVecRead_SeqDense, MatDenseGetColumnVecRead_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreColumnVecRead_C", MatDenseRestoreColumnVecRead_SeqDense, MatDenseRestoreColumnVecRead_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseGetColumnVecWrite_C", MatDenseGetColumnVecWrite_SeqDense, MatDenseGetColumnVec_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatDenseRestoreColumnVecWrite_C", MatDenseRestoreColumnVecWrite_SeqDense, MatDenseRestoreColumnVec_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatQRFactor_C", MatQRFactor_SeqDense, MatQRFactor_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatMultColumnRange_C", MatMultColumnRange_SeqDense, MatMultColumnRange_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatMultAddColumnRange_C", MatMultAddColumnRange_SeqDense, MatMultAddColumnRange_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatMultHermitianTransposeColumnRange_C", MatMultHermitianTransposeColumnRange_SeqDense, MatMultHermitianTransposeColumnRange_SeqDenseKokkos));
  PetscCall(MatComposeOp_SeqDenseKokkos("MatMultHermitianTransposeAddColumnRange_C", MatMultHermitianTransposeAddColumnRange_SeqDense, MatMultHermitianTransposeAddColumnRange_SeqDenseKokkos));
#undef MatSetOp_SeqDenseKokkos
#undef MatComposeOp_SeqDenseKokkos
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetOps_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_KOKKOS; /* the convention for Kokkos matrix types; freshness is tracked by the DualView flags */
  PetscCall(MatBindToCPU_SeqDenseKokkos(A, PETSC_FALSE));
  A->ops->destroy   = MatDestroy_SeqDenseKokkos;
  A->ops->duplicate = MatDuplicate_SeqDenseKokkos;
  A->ops->bindtocpu = MatBindToCPU_SeqDenseKokkos;
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqdensekokkos_seqdense_C", MatConvert_SeqDenseKokkos_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdensekokkos_seqdensekokkos_C", MatProductSetFromOptions_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdensekokkos_seqdense_C", MatProductSetFromOptions_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqdense_seqdensekokkos_C", MatProductSetFromOptions_SeqDense));
  /* Composed unconditionally (not in the bind-switched table): the host product implementation is the
     right handler in both bind states, and removing it on MatBindToCPU() would leave the AIJ products
     with no handler at all (ABt has no fallback in MatProductSetFromOptions()) */
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_seqdensekokkos_C", MatProductSetFromOptions_SeqAIJ_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaijkokkos_seqdensekokkos_C", MatProductSetFromOptions_SeqAIJ_SeqDense));
  PetscFunctionReturn(PETSC_SUCCESS);
}
