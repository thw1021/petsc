#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petscpkg_version.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/sfimpl.h>
#include <petsc/private/kokkosimpl.hpp>
#include <petscsystypes.h>
#include <petscerror.h>

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
//TODO cant tell which header has getrf..
#include <KokkosBlas_util.hpp>
#include <KokkosBatched_Getrf.hpp>
//Do I need this?
#include <KokkosBlas2_gemv.hpp>

#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_InverseLU_Decl.hpp>

#include <../src/mat/impls/dense/seq/kokkos/densekok.hpp>

//using KokkosKernels::Impl::transpose_matrix;

static PetscErrorCode MatSetOps_SeqDenseKokkos(Mat); /* Forward declaration */

static PetscErrorCode MatSeqDenseKokkosSyncHost(Mat A)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  auto                exec     = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  /* We do not expect one needs factors on host  */
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Can't sync factorized matrix from device to host");
  PetscCheck(densekok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing DenseKOK");
  PetscCall(KokkosDualViewSync<HostMirrorMemorySpace>(densekok->m_dual, exec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetArray_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  auto exec = PetscGetKokkosExecutionSpace();
  PetscCallCXX(densekok->m_dual.sync_host(exec));
  PetscCallCXX(exec.fence());
  *array = densekok->m_dual.view_host().data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArray_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  densekok->m_dual.modify_host();
  PetscFunctionReturn(PETSC_SUCCESS);
}

//For some reason, GetArray and GetArrayRead code are same?
static PetscErrorCode MatSeqDenseGetArrayRead_SeqDenseKokkos(Mat A, const PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  auto exec = PetscGetKokkosExecutionSpace();
  PetscCallCXX(densekok->m_dual.sync_host(exec));
  PetscCallCXX(exec.fence());
  *array = densekok->m_dual.view_host().data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayRead_SeqDenseKokkos(Mat A, const PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetArrayWrite_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  *array = densekok->m_dual.view_host().data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayWrite_SeqDenseKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  densekok->m_dual.clear_sync_state();
  densekok->m_dual.modify_host();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetArrayAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a, PetscMemType *mtype)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(densekok != NULL, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "densekok is NULL");

  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  *a = densekok->m_dual.view_device().data();
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  if (std::is_same<DefaultMemorySpace, HostMirrorMemorySpace>::value) {
    PetscCallCXX(densekok->m_dual.modify_host());
  } else {
    PetscCallCXX(densekok->m_dual.modify_device());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO ArrayRead is not there for VECKOKKOS...? does kokkos automatically take care of it
//as we are passing const?
static PetscErrorCode MatSeqDenseGetArrayReadAndMemType_SeqDenseKokkos(Mat A, const PetscScalar *array[], PetscMemType *mtype)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(densekok != NULL, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "densekok is NULL");
  if (array) {
    densekok->m_dual.sync_device();
    *array = densekok->m_device_data();
  }
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreArrayReadAndMemType_SeqDenseKokkos(Mat A, const PetscScalar *array[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(densekok != NULL, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "densekok is NULL");
  if (array) {
    densekok->m_dual.sync_device();
    *array = densekok->m_device_data();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetArrayWriteAndMemType_SeqDenseKokkos(Mat A, PetscScalar **a, PetscMemType *mtype)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  *a = densekok->m_dual.view_device().data();
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO VECKOKKOS doesnt have RestoreArrayWrite...

// CUPM dont have these..
#if 0
static PetscErrorCode MatSeqDenseGetColumn_SeqDenseKokkos(Mat A, PetscInt col, PetscScalar *vals[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreColumn_SeqDenseKokkos(Mat A, PetscScalar *vals[])
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode MatSeqDenseGetColumnVec_SeqDenseKokkos(Mat A, PetscInt col, Vec *x)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreColumnVec_SeqDenseKokkos(Mat A, PetscInt col, Vec *x)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetColumnVecRead_SeqDenseKokkos(Mat A, PetscInt col, Vec *x)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreColumnVecRead_SeqDenseKokkos(Mat A, PetscInt col, Vec *x)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseGetColumnVecWrite_SeqDenseKokkos(Mat A, PetscInt col, Vec *x)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqDenseRestoreColumnVecWrite_SeqDenseKokkos(Mat A, PetscInt col, Vec *x)
{
  Mat_SeqDenseKokkos *densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCallCXX(densekok->m_dual.clear_sync_state());
  PetscCall(KokkosDualViewSync<DefaultMemorySpace>(densekok->m_dual, PetscGetKokkosExecutionSpace()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseSetLDA_SeqDenseKokkos(Mat A, PetscInt lda)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseGetSubMatrix_SeqDenseKokkos(Mat A, PetscInt rbegin, PetscInt rend, PetscInt cbegin, PetscInt cend, Mat *v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseRestoreSubMatrix_SeqDenseKokkos(Mat A, Mat *v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDensePlaceArray_SeqDenseKokkos(Mat A, const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseReplaceArray_SeqDenseKokkos(Mat A, const PetscScalar *array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDenseResetArray_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = A x */
static PetscErrorCode MatMult_SeqDenseKokkos(Mat A, Vec xx, Vec yy)
{
  Mat_SeqDenseKokkos        *densekok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosViewWrite(yy, &yv));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  //is this corerct?
  PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), "N", 1.0 /*alpha*/, densekok->densemat, xv, 0.0 /*beta*/, yv)); /* y = alpha A x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosViewWrite(yy, &yv));
  //TODO should i do 2mn-m or just do 2mn?
  PetscCall(PetscLogGpuFlops(2.0 * densekok->nrows() * densekok->ncols()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = A^T x */
static PetscErrorCode MatMultTranspose_SeqDenseKokkos(Mat A, Vec xx, Vec yy)
{
  Mat_SeqDenseKokkos        *densekok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosViewWrite(yy, &yv));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), "T", 1.0 /*alpha*/, densekok->densemat, xv, 0.0 /*beta*/, yv)); /* y = alpha A x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosViewWrite(yy, &yv));
  PetscCall(PetscLogGpuFlops(2.0 * densekok->nrows() * densekok->ncols()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = A^H x */
static PetscErrorCode MatMultHermitianTranspose_SeqDenseKokkos(Mat A, Vec xx, Vec yy)
{
  Mat_SeqDenseKokkos        *densekok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosViewWrite(yy, &yv));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), "C", 1.0 /*alpha*/, densekok->densemat, xv, 0.0 /*beta*/, yv)); /* y = alpha A x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosViewWrite(yy, &yv));
  PetscCall(PetscLogGpuFlops(2.0 * densekok->nrows() * densekok->ncols()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = A x + y */
static PetscErrorCode MatMultAdd_SeqDenseKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqDenseKokkos        *densekok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  if (zz != yy) PetscCall(VecCopy(yy, zz)); // depending on yy's sync flags, zz might get its latest data on host
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosView(zz, &zv)); // do after VecCopy(yy, zz) to get the latest data on device
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), "N", 1.0 /*alpha*/, densekok->densemat, xv, 1.0 /*beta*/, zv)); /* y = alpha A x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosView(zz, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * densekok->nrows() * densekok->ncols()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = A^T x + y */
static PetscErrorCode MatMultTransposeAdd_SeqDenseKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqDenseKokkos        *densekok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  if (zz != yy) PetscCall(VecCopy(yy, zz));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosView(zz, &zv));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), "T", 1.0 /*alpha*/, densekok->densemat, xv, 1.0 /*beta*/, zv)); /* y = alpha A x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosView(zz, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * densekok->nrows() * densekok->ncols()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = A^H x + y */
static PetscErrorCode MatMultHermitianTransposeAdd_SeqDenseKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqDenseKokkos        *densekok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  if (zz != yy) PetscCall(VecCopy(yy, zz));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosView(zz, &zv));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), "C", 1.0 /*alpha*/, densekok->densemat, xv, 1.0 /*beta*/, zv)); /* y = alpha A x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosView(zz, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * densekok->nrows() * densekok->ncols()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Depending on reuse, either build a new mat, or use the existing mat */
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqDenseKokkos(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat_SeqDense *adense;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  if (reuse == MAT_INITIAL_MATRIX) {                      /* Build a brand new mat */
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, newmat));  /* the returned newmat is a SeqDenseKokkos */
  } else if (reuse == MAT_REUSE_MATRIX) {                 /* Reuse the mat created before */
    PetscCall(MatCopy(A, *newmat, SAME_NONZERO_PATTERN)); /* newmat is already a SeqDenseKokkos */
  } else if (reuse == MAT_INPLACE_MATRIX) {               /* newmat is A */
    PetscCheck(A == *newmat, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "A != *newmat with MAT_INPLACE_MATRIX");
    PetscCall(PetscFree(A->defaultvectype));
    PetscCall(PetscStrallocpy(VECKOKKOS, &A->defaultvectype)); /* Allocate and copy the string */
    PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATSEQDENSEKOKKOS));
    PetscCall(MatSetOps_SeqDenseKokkos(A));
    adense = static_cast<Mat_SeqDense *>(A->data);
    if (A->assembled) { /* Copy i, j (but not values) to device for an assembled matrix if not yet */
      PetscCheck(!A->spptr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expect NULL (Mat_SeqDenseKokkos*)A->spptr");
      A->spptr = new Mat_SeqDenseKokkos(A->rmap->n, A->cmap->n, adense, PETSC_FALSE);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatDuplicate always creates a new matrix. MatDuplicate can be called either on an assembled matrix or
   an unassembled matrix, even though MAT_COPY_VALUES is not allowed for unassembled matrix.
 */
static PetscErrorCode MatDuplicate_SeqDenseKokkos(Mat A, MatDuplicateOption dupOption, Mat *B)
{
  Mat_SeqDense       *bseq;
  Mat_SeqDenseKokkos *akok = static_cast<Mat_SeqDenseKokkos *>(A->spptr), *bkok;
  Mat                 mat;

  PetscFunctionBegin;
  /* Do not copy values on host as A's latest values might be on device. We don't want to do sync blindly */
  PetscCall(MatDuplicate_SeqDense(A, MAT_DO_NOT_COPY_VALUES, B));
  mat = *B;
  if (A->assembled) {
    bseq = static_cast<Mat_SeqDense *>(mat->data);
    bkok = new Mat_SeqDenseKokkos(mat->rmap->n, mat->cmap->n, bseq, PETSC_FALSE);
    bkok->m_dual.clear_sync_state(); /* Clear B's sync state as it will be decided below */
    /* Now copy values to B if needed */
    if (dupOption == MAT_COPY_VALUES) {
      if (akok->m_dual.need_sync_device()) {
        Kokkos::deep_copy(bkok->m_dual.view_host(), akok->m_dual.view_host());
        bkok->m_dual.modify_host();
      } else { /* If device has the latest data, we only copy data on device */
        Kokkos::deep_copy(bkok->m_dual.view_device(), akok->m_dual.view_device());
        bkok->m_dual.modify_device();
      }
    } else { /* MAT_DO_NOT_COPY_VALUES or MAT_SHARE_NONZERO_PATTERN. B's values should be zeroed */
      /* B's values on host should be already zeroed by MatDuplicate_SeqDense() */
      bkok->m_dual.modify_host();
    }
    mat->spptr = bkok;
  }

  PetscCall(PetscFree(mat->defaultvectype));
  PetscCall(PetscStrallocpy(VECKOKKOS, &mat->defaultvectype)); /* Allocate and copy the string */
  PetscCall(PetscObjectChangeTypeName((PetscObject)mat, MATSEQDENSEKOKKOS));
  PetscCall(MatSetOps_SeqDenseKokkos(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO MatTranspose. CUPM doesnt have it anyway..?

static PetscErrorCode MatDestroy_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  //TODO prevent copying back data if we own the data pointer A LA CUPM
  //Convert DenseKokkos to just Dense
  PetscCall(MatDestroy_SeqDense(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATSEQDENSEKOKKOS - MATDENSEKOKKOS = "(seq)densekokkos" - A matrix type to be used for dense matrices with Kokkos

   Options Database Key:
.  -mat_type densekokkos - sets the matrix type to `MATSEQDENSEKOKKOS` during a call to `MatSetFromOptions()`

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATMPIDENSEKOKKOS`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate_SeqDense(A));
  PetscCall(MatConvert_SeqDense_SeqDenseKokkos(A, MATSEQDENSEKOKKOS, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_SeqDenseKokkos(Mat A, PetscScalar a)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  KokkosBlas::scal(PetscGetKokkosExecutionSpace(), densekok->m_dual.view_device(), a, densekok->m_dual.view_device());
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscCall(PetscLogGpuFlops(densekok->m_dual.extent(0)));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// add a to A's diagonal (if A is square) or main diagonal (if A is rectangular)
static PetscErrorCode MatShift_SeqDenseKokkos(Mat A, PetscScalar a)
{
  PetscFunctionBegin;
  //TODO from aij code. it said no missing diagonal.
  //Should be true for dense?
  PetscInt n = PetscMin(A->rmap->n, A->cmap->n);

  //TODO should i do DualView for diag?
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  const auto  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  const auto &Aa       = densekok->m_dual.view_device();
  //TODO is this okay?
  PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt i) { Aa(i, i) += a; }));
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscCall(PetscLogGpuFlops(n));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_SeqDenseKokkos(Mat A)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  if (densekok) { /* Only zero the device if data is already there */
    KokkosBlas::fill(PetscGetKokkosExecutionSpace(), densekok->m_dual.view_device(), 0.0);
    PetscCall(MatSeqDenseKokkosModifyDevice(A));
  } else { /* Might be preallocated but not assembled */
    PetscCall(MatZeroEntries_SeqDense(A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_SeqDenseKokkos(Mat A, Vec x)
{
  Mat_SeqDenseKokkos   *densekok;
  PetscInt              n;
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCheck(n == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_SUP, "MatGetDiagonal_SeqDenseJKokkos not supported on factored matrices");

  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);

  const auto &Aa = densekok->m_dual.view_device();

  PetscCall(VecGetKokkosViewWrite(x, &xv));
  PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt i) { xv(i) = Aa(i,i); }));
  PetscCall(VecRestoreKokkosViewWrite(x, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Get a Kokkos View from a mat of type MatSeqDenseKokkos */
PetscErrorCode MatSeqDenseGetKokkosView(Mat A, ConstMatScalarKokkosDenseView *kv)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  *kv      = densekok->m_dual.view_device();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseRestoreKokkosView(Mat A, ConstMatScalarKokkosDenseView *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseGetKokkosView(Mat A, MatScalarKokkosDenseView *kv)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosSyncDevice(A));
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  *kv      = densekok->m_dual.view_device();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseRestoreKokkosView(Mat A, MatScalarKokkosDenseView *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseGetKokkosViewWrite(Mat A, MatScalarKokkosDenseView *kv)
{
  Mat_SeqDenseKokkos *densekok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  densekok = static_cast<Mat_SeqDenseKokkos *>(A->spptr);
  *kv      = densekok->m_dual.view_device();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqDenseRestoreKokkosViewWrite(Mat A, MatScalarKokkosDenseView *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosModifyDevice(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateSeqDenseKokkosWithKokkosViews(MPI_Comm comm, PetscInt m, PetscInt n, Kokkos::View<PetscScalar **> &a_d, Mat *A)
{
  Mat_SeqDenseKokkos *akok;

  PetscFunctionBegin;
  auto exec = PetscGetKokkosExecutionSpace();
  // Don't copy the vals to the host now
  auto a_h = Kokkos::create_mirror_view(HostMirrorMemorySpace(), a_d);

  MatScalarKokkosDenseDualView a_dual = MatScalarKokkosDenseDualView(a_d, a_h);
  // Note we have modified device data so it will copy lazily
  a_dual.modify_device();

  PetscCallCXX(akok = new Mat_SeqDenseKokkos(m, n, a_dual));
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSeqDenseKokkosWithDenseMatrix(*A, akok));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Computes Y += alpha X */
static PetscErrorCode MatAXPY_SeqDenseKokkos(Mat Y, PetscScalar alpha, Mat X, MatStructure pattern)
{
  Mat_SeqDenseKokkos           *xkok, *ykok;
  ConstMatScalarKokkosDenseView Xa;
  MatScalarKokkosDenseView      Ya;
  auto                          exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCheckTypeName(Y, MATSEQDENSEKOKKOS);
  PetscCheckTypeName(X, MATSEQDENSEKOKKOS);
  PetscCall(MatSeqDenseKokkosSyncDevice(Y));
  PetscCall(MatSeqDenseKokkosSyncDevice(X));
  PetscCall(PetscLogGpuTimeBegin());

  //TODO do i need to support AXPY against non-Kokkos matrix?
  ykok = static_cast<Mat_SeqDenseKokkos *>(Y->spptr);
  xkok = static_cast<Mat_SeqDenseKokkos *>(X->spptr);
  Xa   = xkok->m_dual.view_device();
  Ya   = ykok->m_dual.view_device();

  PetscCallCXX(KokkosBlas::axpy(exec, alpha, Xa, Ya));
  PetscCall(MatSeqDenseKokkosModifyDevice(Y));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(xkok->m_dual.extent(0) * xkok->m_dual.extent(1) * 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetOps_SeqDenseKokkos(Mat A)
{
  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_KOKKOS; /* We do not really use this flag */
  A->boundtocpu  = PETSC_FALSE;

  A->ops->destroy                   = MatDestroy_SeqDenseKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqDenseKokkos;
  A->ops->axpy                      = MatAXPY_SeqDenseKokkos;
  A->ops->scale                     = MatScale_SeqDenseKokkos;
  A->ops->zeroentries               = MatZeroEntries_SeqDenseKokkos;
  A->ops->mult                      = MatMult_SeqDenseKokkos;
  A->ops->multadd                   = MatMultAdd_SeqDenseKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqDenseKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqDenseKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqDenseKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqDenseKokkos;
  A->ops->getdiagonal               = MatGetDiagonal_SeqDenseKokkos;
  A->ops->shift                     = MatShift_SeqDenseKokkos;
  //TODO CUPM has MatComposeOp_CUPM to double dispatch for host and gpu.
  //Does kokkos need to care?
  //TODO PlaceArray
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArray_C", MatSeqDenseGetArray_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayRead_C", MatSeqDenseGetArrayRead_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWrite_C", MatSeqDenseGetArrayWrite_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayAndMemType_C", MatSeqDenseGetArrayAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayReadAndMemType_C", MatSeqDenseGetArrayReadAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetArrayWriteAndMemType_C", MatSeqDenseGetArrayWriteAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArray_C", MatSeqDenseRestoreArray_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayRead_C", MatSeqDenseRestoreArrayRead_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayWrite_C", MatSeqDenseRestoreArrayWrite_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayAndMemType_C", MatSeqDenseRestoreArrayAndMemType_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayReadAndMemType_C", MatSeqDenseRestoreArrayReadAndMemType_SeqDenseKokkos));
  //PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreArrayWriteAndMemType_C", MatSeqDenseRestoreArrayWriteAndMemType_SeqDenseKokkos));

  //CUPM dont have them...
  //PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumn_C", MatDenseGetColumn_SeqDenseKokkos));
  //PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumn_C", MatDenseRestoreColumn_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVec_C", MatSeqDenseGetColumnVec_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVecRead_C", MatSeqDenseGetColumnVecRead_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetColumnVecWrite_C", MatSeqDenseGetColumnVecWrite_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVec_C", MatSeqDenseRestoreColumnVec_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVecRead_C", MatSeqDenseRestoreColumnVecRead_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreColumnVecWrite_C", MatSeqDenseRestoreColumnVecWrite_SeqDenseKokkos));

  //somehow this is in ops, but the above are not?
  A->ops->getcolumnvector = NULL;
  //TODO lda?
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseSetLDA_C", MatDenseSetLDA_SeqDenseKokkos));

  //TODO submatrix!
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseGetSubMatrix_C", MatDenseGetSubMatrix_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseRestoreSubMatrix_C", MatDenseRestoreSubMatrix_SeqDenseKokkos));

  //Array stuff TODO
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDensePlaceArray_C", MatDensePlaceArray_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseReplaceArray_C", MatDenseReplaceArray_SeqDenseKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatDenseResetArray_C", MatDenseResetArray_SeqDenseKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO is this right? instead of Mat_SeqDenseKokkos, something View? dense doesnt have createfromarray...
PETSC_INTERN PetscErrorCode MatCreateSeqDenseKokkosWithView2D(MPI_Comm comm, Mat_SeqDenseKokkos *akok, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  //TODO ??
//  PetscCall(MatSetSeqDenseKokkosWithView2D(*A, akok));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateSeqDenseKokkos - Creates a dense matrix in `MATSEQDENSEKOKKOS` (view 2D) format.
  This matrix will ultimately be handled by Kokkos for calculations.

  Collective

  Input Parameters:
+ comm - MPI communicator, set to `PETSC_COMM_SELF`
. m    - number of rows
. n    - number of columns
- data - optional location of matrix data in column major order.  Use `NULL` for PETSc
         to control all matrix memory allocation.

  Output Parameter:
. A - the matrix

  Level: intermediate

  Notes:
  The data input variable is intended primarily for Fortran programmers
  who wish to allocate their own matrix memory space.  Most users should
  set `data` = `NULL`.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateDense()`, `MatSetValues()`
@*/
PetscErrorCode MatCreateSeqDenseKokkos(MPI_Comm comm, PetscInt m, PetscInt n, PetscScalar data[], Mat *A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, m, n));
  PetscCall(MatSetType(*A, MATSEQDENSEKOKKOS));
  //TODO ??
//  PetscCall(MatSeqDenseSetPreallocation_SeqDense(*A, nz, (PetscInt *)nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}


#if 0
// After matrix numeric factorization, there are still steps to do before triangular solve can be called.
// For example, for transpose solve, we might need to compute the transpose matrices if the solver does not support it (such as KK, while cusparse does).
// In cusparse, one has to call cusparseSpSV_analysis() with updated triangular matrix values before calling cusparseSpSV_solve().
// Simiarily, in KK sptrsv_symbolic() has to be called before sptrsv_solve(). We put these steps in MatSeqAIJKokkos{Transpose}SolveCheck.
static PetscErrorCode MatSeqAIJKokkosSolveCheck(Mat A)
{
  Mat_SeqAIJKokkosTriFactors *factors   = (Mat_SeqAIJKokkosTriFactors *)A->spptr;
  const PetscBool             has_lower = factors->iL_d.extent(0) ? PETSC_TRUE : PETSC_FALSE; // false with Choleksy
  const PetscBool             has_upper = factors->iU_d.extent(0) ? PETSC_TRUE : PETSC_FALSE; // true with LU and Choleksy

  PetscFunctionBegin;
  if (!factors->sptrsv_symbolic_completed) { // If sptrsv_symbolic was not called yet
    if (has_upper) PetscCallCXX(sptrsv_symbolic(&factors->khU, factors->iU_d, factors->jU_d, factors->aU_d));
    if (has_lower) PetscCallCXX(sptrsv_symbolic(&factors->khL, factors->iL_d, factors->jL_d, factors->aL_d));
    factors->sptrsv_symbolic_completed = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJKokkosTransposeSolveCheck(Mat A)
{
  const PetscInt              n         = A->rmap->n;
  Mat_SeqAIJKokkosTriFactors *factors   = (Mat_SeqAIJKokkosTriFactors *)A->spptr;
  const PetscBool             has_lower = factors->iL_d.extent(0) ? PETSC_TRUE : PETSC_FALSE; // false with Choleksy
  const PetscBool             has_upper = factors->iU_d.extent(0) ? PETSC_TRUE : PETSC_FALSE; // true with LU or Choleksy

  PetscFunctionBegin;
  if (!factors->transpose_updated) {
    if (has_upper) {
      if (!factors->iUt_d.extent(0)) {                                 // Allocate Ut on device if not yet
        factors->iUt_d = MatRowMapKokkosView("factors->iUt_d", n + 1); // KK requires this view to be initialized to 0 to call transpose_matrix
        factors->jUt_d = MatColIdxKokkosView(NoInit("factors->jUt_d"), factors->jU_d.extent(0));
        factors->aUt_d = MatScalarKokkosView(NoInit("factors->aUt_d"), factors->aU_d.extent(0));
      }

      if (factors->iU_h.extent(0)) { // If U is on host (factorization was done on host), we also compute the transpose on host
        if (!factors->U) {
          Mat_SeqAIJ *seq;

          PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, factors->iU_h.data(), factors->jU_h.data(), factors->aU_h.data(), &factors->U));
          PetscCall(MatTranspose(factors->U, MAT_INITIAL_MATRIX, &factors->Ut));

          seq            = static_cast<Mat_SeqAIJ *>(factors->Ut->data);
          factors->iUt_h = MatRowMapKokkosViewHost(seq->i, n + 1);
          factors->jUt_h = MatColIdxKokkosViewHost(seq->j, seq->nz);
          factors->aUt_h = MatScalarKokkosViewHost(seq->a, seq->nz);
        } else {
          PetscCall(MatTranspose(factors->U, MAT_REUSE_MATRIX, &factors->Ut)); // Matrix Ut' data is aliased with {i, j, a}Ut_h
        }
        // Copy Ut from host to device
        PetscCallCXX(Kokkos::deep_copy(factors->iUt_d, factors->iUt_h));
        PetscCallCXX(Kokkos::deep_copy(factors->jUt_d, factors->jUt_h));
        PetscCallCXX(Kokkos::deep_copy(factors->aUt_d, factors->aUt_h));
      } else { // If U was computed on device, we also compute the transpose there
        // TODO: KK transpose_matrix() does not sort column indices, however cusparse requires sorted indices. We have to sort the indices, until KK provides finer control options.
        PetscCallCXX(transpose_matrix<ConstMatRowMapKokkosView, ConstMatColIdxKokkosView, ConstMatScalarKokkosView, MatRowMapKokkosView, MatColIdxKokkosView, MatScalarKokkosView, MatRowMapKokkosView, DefaultExecutionSpace>(n, n, factors->iU_d,
                                                                                                                                                                                                                               factors->jU_d, factors->aU_d,
                                                                                                                                                                                                                               factors->iUt_d, factors->jUt_d,
                                                                                                                                                                                                                               factors->aUt_d));
        PetscCallCXX(sort_crs_matrix<DefaultExecutionSpace, MatRowMapKokkosView, MatColIdxKokkosView, MatScalarKokkosView>(factors->iUt_d, factors->jUt_d, factors->aUt_d));
      }
      PetscCallCXX(sptrsv_symbolic(&factors->khUt, factors->iUt_d, factors->jUt_d, factors->aUt_d));
    }

    // do the same for L with LU
    if (has_lower) {
      if (!factors->iLt_d.extent(0)) {                                 // Allocate Lt on device if not yet
        factors->iLt_d = MatRowMapKokkosView("factors->iLt_d", n + 1); // KK requires this view to be initialized to 0 to call transpose_matrix
        factors->jLt_d = MatColIdxKokkosView(NoInit("factors->jLt_d"), factors->jL_d.extent(0));
        factors->aLt_d = MatScalarKokkosView(NoInit("factors->aLt_d"), factors->aL_d.extent(0));
      }

      if (factors->iL_h.extent(0)) { // If L is on host, we also compute the transpose on host
        if (!factors->L) {
          Mat_SeqAIJ *seq;

          PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, factors->iL_h.data(), factors->jL_h.data(), factors->aL_h.data(), &factors->L));
          PetscCall(MatTranspose(factors->L, MAT_INITIAL_MATRIX, &factors->Lt));

          seq            = static_cast<Mat_SeqAIJ *>(factors->Lt->data);
          factors->iLt_h = MatRowMapKokkosViewHost(seq->i, n + 1);
          factors->jLt_h = MatColIdxKokkosViewHost(seq->j, seq->nz);
          factors->aLt_h = MatScalarKokkosViewHost(seq->a, seq->nz);
        } else {
          PetscCall(MatTranspose(factors->L, MAT_REUSE_MATRIX, &factors->Lt)); // Matrix Lt' data is aliased with {i, j, a}Lt_h
        }
        // Copy Lt from host to device
        PetscCallCXX(Kokkos::deep_copy(factors->iLt_d, factors->iLt_h));
        PetscCallCXX(Kokkos::deep_copy(factors->jLt_d, factors->jLt_h));
        PetscCallCXX(Kokkos::deep_copy(factors->aLt_d, factors->aLt_h));
      } else { // If L was computed on device, we also compute the transpose there
        // TODO: KK transpose_matrix() does not sort column indices, however cusparse requires sorted indices. We have to sort the indices, until KK provides finer control options.
        PetscCallCXX(transpose_matrix<ConstMatRowMapKokkosView, ConstMatColIdxKokkosView, ConstMatScalarKokkosView, MatRowMapKokkosView, MatColIdxKokkosView, MatScalarKokkosView, MatRowMapKokkosView, DefaultExecutionSpace>(n, n, factors->iL_d,
                                                                                                                                                                                                                               factors->jL_d, factors->aL_d,
                                                                                                                                                                                                                               factors->iLt_d, factors->jLt_d,
                                                                                                                                                                                                                               factors->aLt_d));
        PetscCallCXX(sort_crs_matrix<DefaultExecutionSpace, MatRowMapKokkosView, MatColIdxKokkosView, MatScalarKokkosView>(factors->iLt_d, factors->jLt_d, factors->aLt_d));
      }
      PetscCallCXX(sptrsv_symbolic(&factors->khLt, factors->iLt_d, factors->jLt_d, factors->aLt_d));
    }
    PetscCallCXX(KokkosBlas::);

    factors->transpose_updated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
//TODO above maybe i need the above? maybe not?

#if 0
// Solve Ax = b, with RAR = U^T D U, where R is the row (and col) permutation matrix on A.
// R is represented by rowperm in factors. If R is identity (i.e, no reordering), then rowperm is empty.
static PetscErrorCode MatSolve_SeqDenseKokkos_Cholesky(Mat A, Vec bb, Vec xx)
{
  auto                        exec    = PetscGetKokkosExecutionSpace();
  PetscInt                    m       = A->rmap->n;
  PetscScalarKokkosView       X, Y, B; // alias
  ConstPetscScalarKokkosView  b;
  PetscScalarKokkosView       x;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSolveCheck(A));          // for UX = T
  PetscCall(MatSeqDenseKokkosTransposeSolveCheck(A)); // for U^T Y = B
  PetscCall(VecGetKokkosView(bb, &b));
  PetscCall(VecGetKokkosViewWrite(xx, &x));

  // Solve U^T Y = B
  if (identity) { // Reorder b with the row permutation
    B = PetscScalarKokkosView(const_cast<PetscScalar *>(b.data()), b.extent(0));
    Y = factors->workVector;
  } else {
    B = factors->workVector;
    PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { B(i) = b(rowperm(i)); }));
    Y = x;
  }
  PetscCallCXX(sptrsv_solve(exec, &factors->khUt, factors->iUt_d, factors->jUt_d, factors->aUt_d, B, Y));

  //PetscCallCXX(KokkosBlas::gemv(PetscGetKokkosExecutionSpace(), "N", 1.0 /*alpha*/, densekok->densemat, xv, 0.0 /*beta*/, yv)); /* y = alpha A x + beta y */
  //TODO no potrf on Kokkos... just getrf...
  PetscCallCXX(KokkosBlas::Impl:potrf(PetscGetKokkosExecutionSpace(), "L", ));
  PetscCall(VecRestoreKokkosView(bb, &b));
  PetscCall(VecRestoreKokkosViewWrite(xx, &x));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#if 0
// Solve Ax = b, with RAC = LU, where R and C are row and col permutation matrices on A respectively.
// R and C are represented by rowperm and colperm in factors.
// If R or C is identity (i.e, no reordering), then rowperm or colperm is empty.
static PetscErrorCode MatSolve_SeqDenseKokkos_LU(Mat A, Vec bb, Vec xx)
{
  auto                        exec    = PetscGetKokkosExecutionSpace();
  PetscInt                    m       = A->rmap->n;
  PetscScalarKokkosView       X, Y, B; // alias
  ConstPetscScalarKokkosView  b;
  PetscScalarKokkosView       x;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosSolveCheck(A));
  PetscCall(VecGetKokkosView(bb, &b));
  PetscCall(VecGetKokkosViewWrite(xx, &x));

  // Solve L Y = B (i.e., L (U C^- x) = R b).  R b indicates applying the row permutation on b.
  //PetscCallCXX(sptrsv_solve(exec, &factors->khL, factors->iL_d, factors->jL_d, factors->aL_d, B, Y));
  //something?

  // Solve U C^- x = Y
  //PetscCallCXX(sptrsv_solve(exec, &factors->khU, factors->iU_d, factors->jU_d, factors->aU_d, Y, X));
  //something??

  // x = C X; Reorder X with the inverse col permutation
  //PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { x(colperm(i)) = X(i); }));
  // ???

  PetscCall(VecRestoreKokkosView(bb, &b));
  PetscCall(VecRestoreKokkosViewWrite(xx, &x));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

#if 0
// Solve A^T x = b, with RAC = LU, where R and C are row and col permutation matrices on A respectively.
// R and C are represented by rowperm and colperm in factors.
// If R or C is identity (i.e, no reordering), then rowperm or colperm is empty.
// A = R^-1 L U C^-1, so A^T = C^-T U^T L^T R^-T. But since C^- = C^T, R^- = R^T, we have A^T = C U^T L^T R.
static PetscErrorCode MatSolveTranspose_SeqDenseKokkos_LU(Mat A, Vec bb, Vec xx)
{
  auto                        exec    = PetscGetKokkosExecutionSpace();
  PetscInt                    m       = A->rmap->n;
  PetscScalarKokkosView       X, Y, B; // alias
  ConstPetscScalarKokkosView  b;
  PetscScalarKokkosView       x;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqDenseKokkosTransposeSolveCheck(A)); // Update L^T, U^T if needed, and do sptrsv symbolic for L^T, U^T
  PetscCall(VecGetKokkosView(bb, &b));
  PetscCall(VecGetKokkosViewWrite(xx, &x));

  // Solve U^T Y = B (i.e., U^T (L^T R x) = C^- b).  Note C^- b = C^T b, which means applying the column permutation on b.
  PetscCallCXX(sptrsv_solve(exec, &factors->khUt, factors->iUt_d, factors->jUt_d, factors->aUt_d, B, Y));

  // Solve L^T X = Y
  PetscCallCXX(sptrsv_solve(exec, &factors->khLt, factors->iLt_d, factors->jLt_d, factors->aLt_d, Y, X));

  // x = R^- X = R^T X; Reorder X with the inverse row permutation

  PetscCall(VecRestoreKokkosView(bb, &b));
  PetscCall(VecRestoreKokkosViewWrite(xx, &x));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode MatLUFactorNumeric_SeqDenseKokkos(Mat B, Mat A, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  //PetscCall(MatLUFactorNumeric_SeqDense(B, A, info));

#if 0
  if (!info->solveonhost) { // if solve on host, then we don't need to copy L, U to device
    Mat_SeqAIJKokkosTriFactors *factors = (Mat_SeqAIJKokkosTriFactors *)B->spptr;
    Mat_SeqAIJ                 *b       = static_cast<Mat_SeqAIJ *>(B->data);
    const PetscInt             *Bi = b->i, *Bj = b->j, *Bdiag = b->diag;
    const MatScalar            *Ba = b->a;
    PetscInt                    m = B->rmap->n, n = B->cmap->n;

    if (factors->iL_h.extent(0) == 0) { // Allocate memory and copy the L, U structure for the first time
      // Allocate memory and copy the structure
      factors->iL_h = MatRowMapKokkosViewHost(NoInit("iL_h"), m + 1);
      factors->jL_h = MatColIdxKokkosViewHost(NoInit("jL_h"), (Bi[m] - Bi[0]) + m); // + the diagonal entries
      factors->aL_h = MatScalarKokkosViewHost(NoInit("aL_h"), (Bi[m] - Bi[0]) + m);
      factors->iU_h = MatRowMapKokkosViewHost(NoInit("iU_h"), m + 1);
      factors->jU_h = MatColIdxKokkosViewHost(NoInit("jU_h"), (Bdiag[0] - Bdiag[m]));
      factors->aU_h = MatScalarKokkosViewHost(NoInit("aU_h"), (Bdiag[0] - Bdiag[m]));

      PetscInt *Li = factors->iL_h.data();
      PetscInt *Lj = factors->jL_h.data();
      PetscInt *Ui = factors->iU_h.data();
      PetscInt *Uj = factors->jU_h.data();

      Li[0] = Ui[0] = 0;
      for (PetscInt i = 0; i < m; i++) {
        PetscInt llen = Bi[i + 1] - Bi[i];       // exclusive of the diagonal entry
        PetscInt ulen = Bdiag[i] - Bdiag[i + 1]; // inclusive of the diagonal entry

        PetscArraycpy(Lj + Li[i], Bj + Bi[i], llen); // entries of L on the left of the diagonal
        Lj[Li[i] + llen] = i;                        // diagonal entry of L

        Uj[Ui[i]] = i;                                                  // diagonal entry of U
        PetscArraycpy(Uj + Ui[i] + 1, Bj + Bdiag[i + 1] + 1, ulen - 1); // entries of U on  the right of the diagonal

        Li[i + 1] = Li[i] + llen + 1;
        Ui[i + 1] = Ui[i] + ulen;
      }

      factors->iL_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->iL_h);
      factors->jL_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->jL_h);
      factors->iU_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->iU_h);
      factors->jU_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->jU_h);
      factors->aL_d = Kokkos::create_mirror_view(DefaultMemorySpace(), factors->aL_h);
      factors->aU_d = Kokkos::create_mirror_view(DefaultMemorySpace(), factors->aU_h);

      // Copy row/col permutation to device
      IS        rowperm = ((Mat_SeqAIJ *)B->data)->row;
      PetscBool row_identity;
      PetscCall(ISIdentity(rowperm, &row_identity));
      if (!row_identity) {
        const PetscInt *ip;

        PetscCall(ISGetIndices(rowperm, &ip));
        factors->rowperm = PetscIntKokkosView(NoInit("rowperm"), m);
        PetscCallCXX(Kokkos::deep_copy(factors->rowperm, PetscIntKokkosViewHost(const_cast<PetscInt *>(ip), m)));
        PetscCall(ISRestoreIndices(rowperm, &ip));
        PetscCall(PetscLogCpuToGpu(m * sizeof(PetscInt)));
      }

      IS        colperm = ((Mat_SeqAIJ *)B->data)->col;
      PetscBool col_identity;
      PetscCall(ISIdentity(colperm, &col_identity));
      if (!col_identity) {
        const PetscInt *ip;

        PetscCall(ISGetIndices(colperm, &ip));
        factors->colperm = PetscIntKokkosView(NoInit("colperm"), n);
        PetscCallCXX(Kokkos::deep_copy(factors->colperm, PetscIntKokkosViewHost(const_cast<PetscInt *>(ip), n)));
        PetscCall(ISRestoreIndices(colperm, &ip));
        PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));
      }

      /* Create sptrsv handles for L, U and their transpose */
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
      auto sptrsv_alg = SPTRSVAlgorithm::SPTRSV_CUSPARSE;
#else
      auto sptrsv_alg = SPTRSVAlgorithm::SEQLVLSCHD_TP1;
#endif
      factors->khL.create_sptrsv_handle(sptrsv_alg, m, true /* L is lower tri */);
      factors->khU.create_sptrsv_handle(sptrsv_alg, m, false /* U is not lower tri */);
      factors->khLt.create_sptrsv_handle(sptrsv_alg, m, false /* L^T is not lower tri */);
      factors->khUt.create_sptrsv_handle(sptrsv_alg, m, true /* U^T is lower tri */);
    }

    // Copy the value
    for (PetscInt i = 0; i < m; i++) {
      PetscInt        llen = Bi[i + 1] - Bi[i];
      PetscInt        ulen = Bdiag[i] - Bdiag[i + 1];
      const PetscInt *Li   = factors->iL_h.data();
      const PetscInt *Ui   = factors->iU_h.data();

      PetscScalar *La = factors->aL_h.data();
      PetscScalar *Ua = factors->aU_h.data();

      PetscArraycpy(La + Li[i], Ba + Bi[i], llen); // entries of L
      La[Li[i] + llen] = 1.0;                      // diagonal entry

      Ua[Ui[i]] = 1.0 / Ba[Bdiag[i]];                                 // diagonal entry
      PetscArraycpy(Ua + Ui[i] + 1, Ba + Bdiag[i + 1] + 1, ulen - 1); // entries of U
    }

    PetscCallCXX(Kokkos::deep_copy(factors->aL_d, factors->aL_h));
    PetscCallCXX(Kokkos::deep_copy(factors->aU_d, factors->aU_h));
    // Once the factors' values have changed, we need to update their transpose and redo sptrsv symbolic
    factors->transpose_updated         = PETSC_FALSE;
    factors->sptrsv_symbolic_completed = PETSC_FALSE;

    B->ops->solve          = MatSolve_SeqAIJKokkos_LU;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJKokkos_LU;
  }
#endif

  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
static PetscErrorCode MatILUFactorNumeric_SeqDenseKokkos_ILU0(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat_SeqDenseKokkos *densekok = (Mat_SeqDenseKokkos *)A->spptr;
  PetscInt           fill_lev  = info->levels;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCheck(!info->factoronhost, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "MatFactorInfo.factoronhost should be false");
  PetscCall(MatSeqDenseKokkosSyncDevice(A));

  auto a_d = densekok->m_dual.view_device();

  //PetscCallCXX(spiluk_numeric(&factors->kh, fill_lev, i_d, j_d, a_d, factors->iL_d, factors->jL_d, factors->aL_d, factors->iU_d, factors->jU_d, factors->aU_d));

  B->assembled              = PETSC_TRUE;
  B->preallocated           = PETSC_TRUE;
  B->ops->solve             = MatSolve_SeqDenseKokkos_LU;
  B->ops->solvetranspose    = MatSolveTranspose_SeqDenseKokkos_LU;
  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;

  /* Once the factors' value changed, we need to update their transpose and sptrsv handle */
  //factors->transpose_updated         = PETSC_FALSE;
  //factors->sptrsv_symbolic_completed = PETSC_FALSE;
  /* TODO: log flops, but how to know that? */
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
// Use KK's spiluk_symbolic() to do ILU0 symbolic factorization, with no row/col reordering
static PetscErrorCode MatILUFactorSymbolic_SeqDenseKokkos_ILU0(Mat B, Mat A, IS, IS, const MatFactorInfo *info)
{
  //Mat_SeqDenseKokkos *densekok;
  //Mat_SeqDense       *b;
  //PetscInt           fill_lev = info->levels;
  //PetscInt           nnzA     = ((Mat_SeqDense*)A->data)->nz, nnzL, nnzU;
  //PetscInt           n        = A->rmap->n;

  PetscFunctionBegin;
#if 0
  PetscCheck(!info->factoronhost, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "MatFactorInfo's factoronhost should be false as we are doing it on device right now");
  PetscCall(MatSeqDenseKokkosSyncDevice(A));

  /* Create a spiluk handle and then do symbolic factorization */
  nnzL = nnzU = PetscRealIntMultTruncate(info->fill, nnzA);

  /* TODO: add options to select sptrsv algorithms */
  /* Create sptrsv handles for L, U and their transpose */
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
  auto sptrsv_alg = SPTRSVAlgorithm::SPTRSV_CUSPARSE;
#else
  auto sptrsv_alg = SPTRSVAlgorithm::SEQLVLSCHD_TP1;
#endif

  /* Fill fields of the factor matrix B */
  PetscCall(MatSeqDenseSetPreallocation_SeqDense(B, MAT_SKIP_ALLOCATION, NULL));
  b     = (Mat_SeqDense *)B->data;
  b->nz = b->maxnz          = spiluk_handle->get_nnzL() + spiluk_handle->get_nnzU();
  B->info.fill_ratio_given  = info->fill;
  B->info.fill_ratio_needed = nnzA > 0 ? ((PetscReal)b->nz) / ((PetscReal)nnzA) : 1.0;

  B->ops->lufactornumeric = MatILUFactorNumeric_SeqDenseKokkos_ILU0;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_SeqDenseKokkos(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatLUFactorSymbolic_SeqDense(B, A, isrow, iscol, info));
  PetscCheck(!B->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqDenseKokkos;
  //TODO ???
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatILUFactorSymbolic_SeqDenseKokkos(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  PetscBool row_identity = PETSC_FALSE, col_identity = PETSC_FALSE;

  PetscFunctionBegin;
  if (!info->factoronhost) {
    PetscCall(ISIdentity(isrow, &row_identity));
    PetscCall(ISIdentity(iscol, &col_identity));
  }

  PetscCheck(!B->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");
  //PetscCallCXX(B->spptr = new Mat_SeqDenseKokkosTriFactors(B->rmap->n));

  if (!info->factoronhost && !info->levels && row_identity && col_identity) { // if level 0 and no reordering
    PetscCall(MatILUFactorSymbolic_SeqDenseKokkos_ILU0(B, A, isrow, iscol, info));
  } else {
  //  PetscCall(MatILUFactorSymbolic_SeqDense(B, A, isrow, iscol, info)); // otherwise, use PETSc's ILU on host
    B->ops->lufactornumeric = MatLUFactorNumeric_SeqDenseKokkos;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
static PetscErrorCode MatCholeskyFactorNumeric_SeqDenseKokkos(Mat B, Mat A, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatSeqDenseKokkosSyncHost(A));
  PetscCall(MatCholeskyFactorNumeric_SeqDense(B, A, info));

  if (!info->solveonhost) { // if solve on host, then we don't need to copy L, U to device
    Mat_SeqAIJKokkosTriFactors *factors = (Mat_SeqAIJKokkosTriFactors *)B->spptr;
    Mat_SeqAIJ                 *b       = static_cast<Mat_SeqAIJ *>(B->data);
    const PetscInt             *Bi = b->i, *Bj = b->j, *Bdiag = b->diag;
    const MatScalar            *Ba = b->a;
    PetscInt                    m  = B->rmap->n;

    if (factors->iU_h.extent(0) == 0) { // First time of numeric factorization
      // Allocate memory and copy the structure
      factors->iU_h = PetscIntKokkosViewHost(const_cast<PetscInt *>(Bi), m + 1); // wrap Bi as iU_h
      factors->jU_h = MatColIdxKokkosViewHost(NoInit("jU_h"), Bi[m]);
      factors->aU_h = MatScalarKokkosViewHost(NoInit("aU_h"), Bi[m]);
      factors->D_h  = MatScalarKokkosViewHost(NoInit("D_h"), m);
      factors->aU_d = Kokkos::create_mirror_view(DefaultMemorySpace(), factors->aU_h);
      factors->D_d  = Kokkos::create_mirror_view(DefaultMemorySpace(), factors->D_h);

      // Build jU_h from the skewed Aj
      PetscInt *Uj = factors->jU_h.data();
      for (PetscInt i = 0; i < m; i++) {
        PetscInt ulen = Bi[i + 1] - Bi[i];
        Uj[Bi[i]]     = i;                                              // diagonal entry
        PetscCall(PetscArraycpy(Uj + Bi[i] + 1, Bj + Bi[i], ulen - 1)); // entries of U on the right of the diagonal
      }

      // Copy iU, jU to device
      PetscCallCXX(factors->iU_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->iU_h));
      PetscCallCXX(factors->jU_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->jU_h));

      // Copy row/col permutation to device
      IS        rowperm = ((Mat_SeqAIJ *)B->data)->row;
      PetscBool row_identity;
      PetscCall(ISIdentity(rowperm, &row_identity));
      if (!row_identity) {
        const PetscInt *ip;

        PetscCall(ISGetIndices(rowperm, &ip));
        PetscCallCXX(factors->rowperm = PetscIntKokkosView(NoInit("rowperm"), m));
        PetscCallCXX(Kokkos::deep_copy(factors->rowperm, PetscIntKokkosViewHost(const_cast<PetscInt *>(ip), m)));
        PetscCall(ISRestoreIndices(rowperm, &ip));
        PetscCall(PetscLogCpuToGpu(m * sizeof(PetscInt)));
      }

      // Create sptrsv handles for U and U^T
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
      auto sptrsv_alg = SPTRSVAlgorithm::SPTRSV_CUSPARSE;
#else
      auto sptrsv_alg = SPTRSVAlgorithm::SEQLVLSCHD_TP1;
#endif
      factors->khU.create_sptrsv_handle(sptrsv_alg, m, false /* U is not lower tri */);
      factors->khUt.create_sptrsv_handle(sptrsv_alg, m, true /* U^T is lower tri */);
    }
    // These pointers were set MatCholeskyFactorNumeric_SeqAIJ(), so we always need to update them
    B->ops->solve          = MatSolve_SeqAIJKokkos_Cholesky;
    B->ops->solvetranspose = MatSolve_SeqAIJKokkos_Cholesky;

    // Copy the value
    PetscScalar *Ua = factors->aU_h.data();
    PetscScalar *D  = factors->D_h.data();
    for (PetscInt i = 0; i < m; i++) {
      D[i]      = Ba[Bdiag[i]];     // actually Aa[Adiag[i]] is the inverse of the diagonal
      Ua[Bi[i]] = (PetscScalar)1.0; // set the unit diagonal for U
      for (PetscInt k = 0; k < Bi[i + 1] - Bi[i] - 1; k++) Ua[Bi[i] + 1 + k] = -Ba[Bi[i] + k];
    }
    PetscCallCXX(Kokkos::deep_copy(factors->aU_d, factors->aU_h));
    PetscCallCXX(Kokkos::deep_copy(factors->D_d, factors->D_h));

    factors->sptrsv_symbolic_completed = PETSC_FALSE; // When numeric value changed, we must do these again
    factors->transpose_updated         = PETSC_FALSE;
  }

  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode MatICCFactorSymbolic_SeqDenseKokkos(Mat B, Mat A, IS perm, const MatFactorInfo *info)
{
  PetscFunctionBegin;
#if 0
  if (info->solveonhost) {
    // If solve on host, we have to change the type, as eventually we need to call MatSolve_SeqSBAIJ_1_NaturalOrdering() etc.
    PetscCall(MatSetType(B, MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(B, 1, MAT_SKIP_ALLOCATION, NULL));
  }

  PetscCall(MatICCFactorSymbolic_SeqAIJ(B, A, perm, info));

  if (!info->solveonhost) {
    // If solve on device, B is still a MATSEQAIJKOKKOS, so we are good to allocate B->spptr
    PetscCheck(!B->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");
    PetscCallCXX(B->spptr = new Mat_SeqAIJKokkosTriFactors(B->rmap->n));
    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJKokkos;
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqDenseKokkos(Mat B, Mat A, IS perm, const MatFactorInfo *info)
{
  PetscFunctionBegin;
#if 0
  if (info->solveonhost) {
    // If solve on host, we have to change the type, as eventually we need to call MatSolve_SeqSBAIJ_1_NaturalOrdering() etc.
    PetscCall(MatSetType(B, MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(B, 1, MAT_SKIP_ALLOCATION, NULL));
  }

  PetscCall(MatCholeskyFactorSymbolic_SeqAIJ(B, A, perm, info)); // it sets B's two ISes ((Mat_SeqAIJ*)B->data)->{row, col} to perm

  if (!info->solveonhost) {
    // If solve on device, B is still a MATSEQAIJKOKKOS, so we are good to allocate B->spptr
    PetscCheck(!B->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");
    PetscCallCXX(B->spptr = new Mat_SeqAIJKokkosTriFactors(B->rmap->n));
    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJKokkos;
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The _Kokkos suffix means we will use Kokkos as a solver for the SeqAIJKokkos matrix
static PetscErrorCode MatFactorGetSolverType_SeqDenseKokkos_Kokkos(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERKOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSOLVERKOKKOS = "Kokkos" - A matrix solver type providing solvers for sequential matrices
  on a single GPU of type, `MATSEQDENSEKOKKOS`, `MATDENSEKOKKOS`.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `PCFactorSetMatSolverType()`, `MatSolverType`, `MatCreateSeqDenseKokkos()`, `MATDenseKOKKOS`, `MatKokkosSetFormat()`, `MatKokkosStorageFormat`, `MatKokkosFormatOperation`
M*/
PETSC_EXTERN PetscErrorCode MatGetFactor_SeqDenseKokkos_Kokkos(Mat A, MatFactorType ftype, Mat *B) /* MatGetFactor_<MatType>_<MatSolverType> */
{
  PetscInt n = A->rmap->n;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, n, n));
  PetscCall(MatSetBlockSizesFromMats(*B, A, A));
  (*B)->factortype = ftype;
  PetscCall(MatSetType(*B, MATSEQDENSEKOKKOS));
//  PetscCall(MatSeqDenseSetPreallocation(*B, MAT_SKIP_ALLOCATION, NULL));
  PetscCheck(!(*B)->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqDenseKokkos;
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqDenseKokkos;
    PetscCall(PetscStrallocpy(MATORDERINGND, (char **)&(*B)->preferredordering[MAT_FACTOR_LU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ILU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ILUDT]));
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqDenseKokkos;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDenseKokkos;
    PetscCall(PetscStrallocpy(MATORDERINGND, (char **)&(*B)->preferredordering[MAT_FACTOR_CHOLESKY]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ICC]));
  } else SETERRQ(comm, PETSC_ERR_SUP, "MatFactorType %s is not supported by MatType SeqDenseKokkos", MatFactorTypes[ftype]);

  // The factorization can use the ordering provided in MatLUFactorSymbolic(), MatCholeskyFactorSymbolic() etc, though we do it on host
  (*B)->canuseordering = PETSC_TRUE;
  PetscCall(PetscObjectComposeFunction((PetscObject)*B, "MatFactorGetSolverType_C", MatFactorGetSolverType_SeqDenseKokkos_Kokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_Kokkos(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS, MATSEQDENSEKOKKOS, MAT_FACTOR_LU, MatGetFactor_SeqDenseKokkos_Kokkos));
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS, MATSEQDENSEKOKKOS, MAT_FACTOR_CHOLESKY, MatGetFactor_SeqDenseKokkos_Kokkos));
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS, MATSEQDENSEKOKKOS, MAT_FACTOR_ILU, MatGetFactor_SeqDenseKokkos_Kokkos));
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS, MATSEQDENSEKOKKOS, MAT_FACTOR_ICC, MatGetFactor_SeqDenseKokkos_Kokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}
