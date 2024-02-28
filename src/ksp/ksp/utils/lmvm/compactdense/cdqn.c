#include <../src/ksp/ksp/utils/lmvm/compactdense/cdqn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>
#include <petscblaslapack.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscis.h>
#include <petscoptions.h>
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>
#if defined(PETSC_HAVE_CUDA)
  #include <petscdevice_cuda.h>
  #include <petsc/private/vecimpl.h>
  #include <cuda_profiler_api.h>
#endif

PetscLogEvent CDDFP_MatMult;
PetscLogEvent CDDFP_MatSolve;
PetscLogEvent CDQN_MatMult;
PetscLogEvent CDQN_MatSolve;
PetscLogEvent CDBFGS_MatMult;
PetscLogEvent CDBFGS_MatSolve;
PetscLogEvent CDQN_J0Inv;
PetscLogEvent CDQN_J0Fwd;

static PetscErrorCode MatMult_LMVMCDQN(Mat, Vec, Vec);
static PetscErrorCode MatMult_LMVMCDBFGS(Mat, Vec, Vec);
static PetscErrorCode MatMult_LMVMCDDFP(Mat, Vec, Vec);
static PetscErrorCode MatSolve_LMVMCDQN(Mat, Vec, Vec);
static PetscErrorCode MatSolve_LMVMCDBFGS(Mat, Vec, Vec);
static PetscErrorCode MatSolve_LMVMCDDFP(Mat, Vec, Vec);

static inline PetscInt recycle_index(PetscInt m, PetscInt idx)
{
  return idx % m;
}

static inline PetscInt history_index(PetscInt m, PetscInt num_updates, PetscInt idx)
{
  return (idx - num_updates) + PetscMin(m, num_updates);
}

static inline PetscInt oldest_update(PetscInt m, PetscInt idx)
{
  return PetscMax(0, idx - m);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatView_LMVMCDQN(Mat B, PetscViewer pv)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *ldfp = (Mat_CDQN *)lmvm->ctx;

  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  PetscCall(MatView_LMVM(B, pv));
  if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) { PetscCall(MatView(ldfp->diag_qn, pv)); }
  if (isascii) { PetscCall(PetscViewerASCIIPrintf(pv, "Counts: S x : %" PetscInt_FMT ", S^T x : %" PetscInt_FMT ", Y x : %" PetscInt_FMT ",  Y^T x: %" PetscInt_FMT "\n", ldfp->S_count, ldfp->St_count, ldfp->Y_count, ldfp->Yt_count)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMCDQN(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *lqn  = (Mat_CDQN *)lmvm->ctx;
  PetscBool is_cdbfgs, is_cddfp, is_cdqn, same, allocate = PETSC_FALSE;
  VecType   vec_type;
  PetscInt  m, n, M, N;
  MPI_Comm  comm = PetscObjectComm((PetscObject)B);

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDBFGS, &is_cdbfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDDFP, &is_cddfp));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDQN, &is_cdqn));

  if (lmvm->allocated) {
    PetscCall(VecGetType(X, &vec_type));
    PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->Xprev, vec_type, &same));
    if (!same) {
      /* Given X vector has a different type than allocated X-type data structures.
         We need to destroy all of this and duplicate again out of the given vector. */
      allocate = PETSC_TRUE;
      PetscCall(MatLMVMReset(B, PETSC_TRUE));
    } else {
      VecCheckMatCompatible(B, X, 2, F, 3);
    }
  } else {
    allocate = PETSC_TRUE;
  }
  if (allocate) {
    PetscCall(VecGetLocalSize(X, &n));
    PetscCall(VecGetSize(X, &N));
    PetscCall(VecGetLocalSize(F, &m));
    PetscCall(VecGetSize(F, &M));
    PetscCheck(N == M, comm, PETSC_ERR_ARG_SIZ, "Incorrect problem sizes! dim(X) not equal to dim(F)");
    PetscCall(MatSetSizes(B, m, n, M, N));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));
    PetscCall(VecDuplicate(X, &lmvm->Xprev));
    PetscCall(VecDuplicate(F, &lmvm->Fprev));
    if (lmvm->m > 0) {
      PetscMPIInt rank;
      PetscInt    m, M;

      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      M = lmvm->m;
      m = (rank == 0) ? M : 0;

      /* For CDBFGS: Create data needed for MatSolve() eagerly; data needed for MatMult() will be created on demand
       * For CDDFP : Create data needed for MatMult() eagerly; data needed for MatSolve() will be created on demand
       * For CDQN  : Create all data eagerly */
      PetscCall(VecGetType(X, &vec_type));
      PetscCall(MatCreateDenseFromVecType(comm, vec_type, n, m, N, M, -1, NULL, &lqn->Sfull));
      PetscCall(MatDuplicate(lqn->Sfull, MAT_SHARE_NONZERO_PATTERN, &lqn->Yfull));
      PetscCall(MatZeroEntries(lqn->Sfull));
      PetscCall(MatZeroEntries(lqn->Yfull));
      if (is_cdqn) {
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lqn->StY_triu));
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lqn->YtS_triu));
        PetscCall(MatCreateVecs(lqn->StY_triu, &lqn->diag_vec, &lqn->rwork1));
        PetscCall(MatCreateVecs(lqn->StY_triu, &lqn->rwork2, &lqn->rwork3));
      } else if (is_cddfp) {
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lqn->YtS_triu));
        PetscCall(MatDuplicate(lqn->Sfull, MAT_SHARE_NONZERO_PATTERN, &lqn->HY));
        PetscCall(MatCreateVecs(lqn->YtS_triu, &lqn->diag_vec, &lqn->rwork1));
        PetscCall(MatCreateVecs(lqn->YtS_triu, &lqn->rwork2, &lqn->rwork3));
      } else if (is_cdbfgs) {
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lqn->StY_triu));
        PetscCall(MatDuplicate(lqn->Sfull, MAT_SHARE_NONZERO_PATTERN, &lqn->BS));
        PetscCall(MatCreateVecs(lqn->StY_triu, &lqn->diag_vec, &lqn->rwork1));
        PetscCall(MatCreateVecs(lqn->StY_triu, &lqn->rwork2, &lqn->rwork3));
      } else {
        SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "MatAllocate_LMVMCDQN is only available for compact-dense derived types. (CDBFGS, CDDFP, CDQN");
      }
      /* initialize StY_triu and YtS_triu to identity, if they exist, so it is invertible */
      if (lqn->StY_triu) {
        PetscCall(MatZeroEntries(lqn->StY_triu));
        PetscCall(MatShift(lqn->StY_triu, 1.0));
      }
      if (lqn->YtS_triu) {
        PetscCall(MatZeroEntries(lqn->YtS_triu));
        PetscCall(MatShift(lqn->YtS_triu, 1.0));
      }
      PetscCall(VecDuplicate(lqn->rwork2, &lqn->cyclic_work_vec));
      PetscCall(VecZeroEntries(lqn->rwork1));
      PetscCall(VecZeroEntries(lqn->rwork2));
      PetscCall(VecZeroEntries(lqn->rwork3));
      PetscCall(VecZeroEntries(lqn->diag_vec));
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lqn->column_work));
    if (!(lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale)) { PetscCall(MatLMVMAllocate(lqn->diag_qn, X, F)); }
    lmvm->allocated = PETSC_TRUE;
    B->preallocated = PETSC_TRUE;
    B->assembled    = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMCDQN(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscInt    m, n, M, N;
  PetscMPIInt size;
  MPI_Comm    comm = PetscObjectComm((PetscObject)B);
  Vec         Xtmp, Ftmp;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  PetscCall(MatGetSize(B, &M, &N));
  PetscCheck(M != 0 && N != 0, comm, PETSC_ERR_ORDER, "MatSetSizes() must be called before MatSetUp()");
  if (!lmvm->allocated) {
    PetscCallMPI(MPI_Comm_size(comm, &size));
    if (size == 1) {
      PetscCall(VecCreateSeq(comm, N, &Xtmp));
      PetscCall(VecCreateSeq(comm, M, &Ftmp));
    } else {
      PetscCall(MatGetLocalSize(B, &m, &n));
      PetscCall(VecCreateMPI(comm, n, N, &Xtmp));
      PetscCall(VecCreateMPI(comm, m, M, &Ftmp));
    }
    PetscCall(MatAllocate_LMVMCDQN(B, Xtmp, Ftmp));
    PetscCall(VecDestroy(&Xtmp));
    PetscCall(VecDestroy(&Ftmp));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/
static PetscErrorCode MatSetFromOptions_LMVMCDQN_Private(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *lqn  = (Mat_CDQN *)lmvm->ctx;
  PetscBool is_cdbfgs, is_cddfp, is_cdqn;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDBFGS, &is_cdbfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDDFP, &is_cddfp));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDQN, &is_cdqn));
  if (is_cdqn) {
    PetscCall(PetscOptionsEnum("-mat_lqn_type", "Implementation options for L-QN", "MatLMVMCompactDenseType", MatLMVMCompactDenseTypes, (PetscEnum)lqn->strategy, (PetscEnum *)&lqn->strategy, NULL));
    PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "MatLMVMSymBrdnScaleType", MatLMVMSymBroydenScaleTypes, (PetscEnum)lqn->scale_type, (PetscEnum *)&lqn->scale_type, NULL));
  } else if (is_cdbfgs) {
    PetscCall(PetscOptionsEnum("-mat_lbfgs_type", "Implementation options for L-BFGS", "MatLMVMCompactDenseType", MatLMVMCompactDenseTypes, (PetscEnum)lqn->strategy, (PetscEnum *)&lqn->strategy, NULL));
    PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "MatLMVMSymBrdnScaleType", MatLMVMSymBroydenScaleTypes, (PetscEnum)lqn->scale_type, (PetscEnum *)&lqn->scale_type, NULL));
  } else if (is_cddfp) {
    PetscCall(PetscOptionsEnum("-mat_ldfp_type", "Implementation options for L-DFP", "MatLMVMCompactDenseType", MatLMVMCompactDenseTypes, (PetscEnum)lqn->strategy, (PetscEnum *)&lqn->strategy, NULL));
    PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "MatLMVMSymBrdnScaleType", MatLMVMSymBroydenScaleTypes, (PetscEnum)lqn->scale_type, (PetscEnum *)&lqn->scale_type, NULL));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "MatSetFromOptions_LMVMCDQN is only available for compact-dense derived types. (CDBFGS, CDDFP, CDQN");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMCDQN(Mat B, PetscOptionItems *PetscOptionsObject)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *lqn  = (Mat_CDQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsBegin(PetscObjectComm((PetscObject)B), ((PetscObject)B)->prefix, "Compact dense QN method (MATLMVMCDQN,MATLMVMCDBFGS,MATLMVMCDDFP)", NULL);
  PetscCall(MatSetFromOptions_LMVMCDQN_Private(B, PetscOptionsObject));
  lqn->allocated = PETSC_FALSE;
  PetscOptionsEnd();
  if (lqn->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    const char *prefix;

    PetscCall(MatGetOptionsPrefix(B, &prefix));
    PetscCall(MatSetOptionsPrefix(lqn->diag_qn, prefix));
    PetscCall(MatAppendOptionsPrefix(lqn->diag_qn, "J0_"));
    PetscCall(MatSetFromOptions(lqn->diag_qn));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatLMVMCDQNResetDestructive(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *lqn  = (Mat_CDQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&lqn->Sfull));
  PetscCall(MatDestroy(&lqn->Yfull));
  PetscCall(MatDestroy(&lqn->HY));
  PetscCall(MatDestroy(&lqn->BS));
  PetscCall(MatDestroy(&lqn->StY_triu));
  PetscCall(MatDestroy(&lqn->YtS_triu));
  PetscCall(VecDestroy(&lqn->StFprev));
  PetscCall(VecDestroy(&lqn->YtXprev));
  PetscCall(VecDestroy(&lqn->Fprev_ref));
  PetscCall(VecDestroy(&lqn->Xprev_ref));
  lqn->Fprev_state = 0;
  lqn->Xprev_state = 0;
  PetscCall(MatDestroy(&lqn->YtS_triu_strict));
  PetscCall(MatDestroy(&lqn->StY_triu_strict));
  PetscCall(MatDestroy(&lqn->StBS));
  PetscCall(MatDestroy(&lqn->YtHY));
  PetscCall(MatDestroy(&lqn->J));
  PetscCall(MatDestroy(&lqn->temp_mat));
  PetscCall(VecDestroy(&lqn->diag_vec));
  PetscCall(VecDestroy(&lqn->diag_vec_recycle_order));
  PetscCall(VecDestroy(&lqn->inv_diag_vec));
  PetscCall(VecDestroy(&lqn->column_work));
  PetscCall(VecDestroy(&lqn->rwork1));
  PetscCall(VecDestroy(&lqn->rwork2));
  PetscCall(VecDestroy(&lqn->rwork3));
  PetscCall(VecDestroy(&lqn->rwork2_local));
  PetscCall(VecDestroy(&lqn->rwork3_local));
  PetscCall(VecDestroy(&lqn->cyclic_work_vec));
  lqn->allocated = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMCDQN(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *lqn  = (Mat_CDQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatLMVMCDQNResetDestructive(B));
  PetscCall(MatDestroy(&lqn->diag_qn));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMCDQN(Mat B, PetscBool destructive)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *lqn  = (Mat_CDQN *)lmvm->ctx;

  PetscFunctionBegin;
  lqn->watchdog = 0;
  if (lqn->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(MatLMVMReset(lqn->diag_qn, destructive));
  if (lqn->Sfull) PetscCall(MatZeroEntries(lqn->Sfull));
  if (lqn->Yfull) PetscCall(MatZeroEntries(lqn->Yfull));
  if (lqn->BS) PetscCall(MatZeroEntries(lqn->BS));
  if (lqn->HY) PetscCall(MatZeroEntries(lqn->HY));
  if (lqn->StY_triu) { /* Set to identity by default so it is invertible */
    PetscCall(MatZeroEntries(lqn->StY_triu));
    PetscCall(MatShift(lqn->StY_triu, 1.0));
  }
  if (lqn->YtS_triu) {
    PetscCall(MatZeroEntries(lqn->YtS_triu));
    PetscCall(MatShift(lqn->YtS_triu, 1.0));
  }
  if (lqn->YtS_triu_strict) PetscCall(MatZeroEntries(lqn->YtS_triu_strict));
  if (lqn->StY_triu_strict) PetscCall(MatZeroEntries(lqn->StY_triu_strict));
  if (lqn->StBS) {
    PetscCall(MatZeroEntries(lqn->StBS));
    PetscCall(MatShift(lqn->StBS, 1.0));
  }
  if (lqn->YtHY) {
    PetscCall(MatZeroEntries(lqn->YtHY));
    PetscCall(MatShift(lqn->YtHY, 1.0));
  }
  if (lqn->Fprev_ref) PetscCall(VecDestroy(&lqn->Fprev_ref));
  if (lqn->Xprev_ref) PetscCall(VecDestroy(&lqn->Xprev_ref));
  lqn->Fprev_state = 0;
  lqn->Xprev_state = 0;
  if (lqn->StFprev) PetscCall(VecZeroEntries(lqn->StFprev));
  if (lqn->YtXprev) PetscCall(VecZeroEntries(lqn->YtXprev));
  if (destructive) { PetscCall(MatLMVMCDQNResetDestructive(B)); }
  lqn->num_updates      = 0;
  lqn->num_mult_updates = 0;
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMCDQN(Mat B, Vec X, Vec F)
{
  Mat_LMVM     *lmvm    = (Mat_LMVM *)B->data;
  Mat_CDQN     *lqn     = (Mat_CDQN *)lmvm->ctx;
  Mat_LMVM     *dbase   = (Mat_LMVM *)lqn->diag_qn->data;
  Mat_DiagBrdn *diagctx = (Mat_DiagBrdn *)dbase->ctx;

  PetscBool          is_cddfp, is_cdbfgs, is_cdqn;
  PetscScalar        curvature, yTy;
  PetscReal          curvtol;
  Vec                workvec1;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDBFGS, &is_cdbfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDDFP, &is_cddfp));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMCDQN, &is_cdqn));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  if (lmvm->prev_set) {
    Vec         FX[2];
    Vec         XF[2];
    PetscScalar dotFX[2];
    PetscScalar dotXF[2];
    PetscScalar stFprev;
    PetscScalar ytXprev;

    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPXAsync_Private(lmvm->Xprev, -1.0, X, dctx));
    /* Test if the updates can be accepted */
    FX[0] = lmvm->Fprev; /* dotFX[0] = s^T Fprev */
    FX[1] = F;           /* dotFX[1] = s^T F     */
    XF[0] = lmvm->Xprev; /* dotXF[0] = y^T Xprev */
    XF[1] = X;           /* dotXF[1] = y^T X     */
    PetscCall(VecMDot(lmvm->Xprev, 2, FX, dotFX));
    PetscCall(VecMDot(lmvm->Fprev, 2, XF, dotXF));
    PetscCall(VecAYPXAsync_Private(lmvm->Fprev, -1.0, F, dctx));
    PetscCall(VecDot(lmvm->Fprev, lmvm->Fprev, &yTy));
    stFprev   = dotFX[0];
    ytXprev   = dotXF[0];
    curvature = (dotFX[1] - dotFX[0]); /* s^T y */
    if (PetscRealPart(yTy) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(yTy);
    }
    if (PetscRealPart(curvature) > curvtol) {
      PetscInt m     = lmvm->m;
      PetscInt k     = lqn->num_updates;
      PetscInt h_new = k + 1 - oldest_update(m, k + 1);
      PetscInt idx   = recycle_index(m, k);
      PetscInt StYidx;

      /* Update is good, accept it */
      lmvm->nupdates++;
      lqn->num_updates++;
      lqn->watchdog = 0;

      if (lmvm->k != m - 1) {
        lmvm->k++;
      } else if (lqn->strategy == MAT_LMVM_CD_REORDER) {
        if (is_cdqn) {
          PetscCall(MatMove_LR3(B, lqn->StY_triu, m - 1));
          PetscCall(MatMove_LR3(B, lqn->YtS_triu, m - 1));
        } else if (is_cdbfgs) {
          PetscCall(MatMove_LR3(B, lqn->StY_triu, m - 1));
        } else if (is_cddfp) {
          PetscCall(MatMove_LR3(B, lqn->YtS_triu, m - 1));
        } else {
          SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "MatUpdate_LMVMCDQN is only available for compact-dense derived types. (CDBFGS, CDDFP, CDQN");
        }
      }

      /* First update the S^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lqn->Sfull, idx, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Xprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(lqn->Sfull, idx, &workvec1));

      /* Now repeat update for the Y^T matrix */
      PetscCall(MatDenseGetColumnVecWrite(lqn->Yfull, idx, &workvec1));
      PetscCall(VecCopyAsync_Private(lmvm->Fprev, workvec1, dctx));
      PetscCall(MatDenseRestoreColumnVecWrite(lqn->Yfull, idx, &workvec1));

      StYidx = (lqn->strategy == MAT_LMVM_CD_REORDER) ? history_index(m, lqn->num_updates, k) : idx;

      { /* implement the scheme of Byrd, Nocedal, and Schnabel to save a MatMultTranspose call in the common case the       *
         * H_k is immediately applied to F after begin updated.   The S^T y computation can be split up as S^T (F - F_prev) */
        PetscInt     local_n;
        PetscScalar *StFprev;
        PetscMemType memtype;

        if (!lqn->StFprev) {
          PetscCall(VecDuplicate(lqn->rwork1, &lqn->StFprev));
          PetscCall(VecZeroEntries(lqn->StFprev));
        }
        PetscCall(VecGetLocalSize(lqn->StFprev, &local_n));
        PetscCall(VecGetArrayAndMemType(lqn->StFprev, &StFprev, &memtype));
        if (local_n) {
          if (PetscMemTypeHost(memtype)) {
            StFprev[idx] = stFprev;
          } else {
            PetscCall(PetscDeviceRegisterMemory(&stFprev, PETSC_MEMTYPE_HOST, 1 * sizeof(stFprev)));
            PetscCall(PetscDeviceRegisterMemory(StFprev, memtype, local_n * sizeof(*StFprev)));
            PetscCall(PetscDeviceArrayCopy(dctx, &StFprev[idx], &stFprev, 1));
          }
        }
        PetscCall(VecRestoreArrayAndMemType(lqn->StFprev, &StFprev));

        if (is_cdqn || is_cdbfgs) {
          Vec this_sy_col;
          /* Now StFprev is updated for the new S vector.  Write -StFprev into the appropriate row */
          PetscCall(MatDenseGetColumnVecWrite(lqn->StY_triu, StYidx, &this_sy_col));
          PetscCall(VecAXPBYAsync_Private(this_sy_col, -1.0, 0.0, lqn->StFprev, dctx));

          /* Now compute the new StFprev */
          PetscCall(MatMultTransposeColumnRange(lqn->Sfull, F, lqn->StFprev, 0, h_new));
          lqn->St_count++;

          /* Now add StFprev: this_sy_col == S^T (F - Fprev) == S^T y */
          PetscCall(VecAXPYAsync_Private(this_sy_col, 1.0, lqn->StFprev, dctx));

          if (lqn->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, this_sy_col, lqn->num_updates, lqn->cyclic_work_vec));
          PetscCall(MatDenseRestoreColumnVecWrite(lqn->StY_triu, StYidx, &this_sy_col));
        } else {
          PetscCall(MatMultTransposeColumnRange(lqn->Sfull, F, lqn->StFprev, 0, h_new));
          lqn->St_count++;
        }
      }

      { /* implement the scheme of Byrd, Nocedal, and Schnabel to save a MatMultTranspose call in the common case the       *
         * B_k is immediately applied to X after begin updated.   The Y^T x computation can be split up as Y^T (X - X_prev) */
        PetscInt     local_n;
        PetscScalar *YtXprev;
        PetscMemType memtype;

        if (!lqn->YtXprev) {
          PetscCall(VecDuplicate(lqn->rwork1, &lqn->YtXprev));
          PetscCall(VecZeroEntries(lqn->YtXprev));
        }

        if (is_cdqn || is_cddfp) {
          Vec this_sy_col;
          PetscCall(MatDenseGetColumnVecWrite(lqn->YtS_triu, StYidx, &this_sy_col));
          /* Unlike cdbfgs, due to initial gradient, additional MatMult is introduced per update */
          PetscCall(MatMultTransposeColumnRange(lqn->Yfull, lmvm->Xprev, this_sy_col, 0, h_new));

          /* Now compute the new YtXprev */
          PetscCall(MatMultTransposeColumnRange(lqn->Yfull, X, lqn->YtXprev, 0, h_new));
          lqn->Yt_count++;

          if (lqn->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, this_sy_col, lqn->num_updates, lqn->cyclic_work_vec));
          PetscCall(MatDenseRestoreColumnVecWrite(lqn->YtS_triu, StYidx, &this_sy_col));
        } else {
          PetscCall(VecGetLocalSize(lqn->YtXprev, &local_n));
          PetscCall(VecGetArrayAndMemType(lqn->YtXprev, &YtXprev, &memtype));
          if (local_n) {
            if (PetscMemTypeHost(memtype)) {
              YtXprev[idx] = ytXprev;
            } else {
              PetscCall(PetscDeviceRegisterMemory(&ytXprev, PETSC_MEMTYPE_HOST, 1 * sizeof(ytXprev)));
              PetscCall(PetscDeviceRegisterMemory(YtXprev, memtype, local_n * sizeof(*YtXprev)));
              PetscCall(PetscDeviceArrayCopy(dctx, &YtXprev[idx], &ytXprev, 1));
            }
          }
          PetscCall(VecRestoreArrayAndMemType(lqn->YtXprev, &YtXprev));

          /* Now compute the new YtXprev */
          PetscCall(MatMultTransposeColumnRange(lqn->Yfull, X, lqn->YtXprev, 0, h_new));
          lqn->Yt_count++;
        }
      }

      if (is_cdbfgs || is_cdqn) {
        PetscCall(MatGetDiagonal(lqn->StY_triu, lqn->diag_vec));
      } else if (is_cddfp) {
        PetscCall(MatGetDiagonal(lqn->YtS_triu, lqn->diag_vec));
      } else {
        SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "MatUpdate_LMVMCDQN is only available for compact-dense derived types. (CDBFGS, CDDFP, CDQN");
      }

      if (lqn->strategy == MAT_LMVM_CD_REORDER) {
        if (!lqn->diag_vec_recycle_order) PetscCall(VecDuplicate(lqn->diag_vec, &lqn->diag_vec_recycle_order));
        PetscCall(VecCopyAsync_Private(lqn->diag_vec, lqn->diag_vec_recycle_order, dctx));
        PetscCall(VecHistoryOrderToRecycleOrder(B, lqn->diag_vec_recycle_order, lqn->num_updates, lqn->cyclic_work_vec));
      } else {
        if (!lqn->diag_vec_recycle_order) {
          PetscCall(PetscObjectReference((PetscObject)lqn->diag_vec));
          lqn->diag_vec_recycle_order = lqn->diag_vec;
        }
      }

      if (lqn->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) {
        PetscScalar sTy = curvature;

        PetscCall(VecDot(lmvm->Fprev, lmvm->Fprev, &yTy));
        diagctx->sigma = PetscRealPart(sTy) / PetscRealPart(yTy);
      } else if (lqn->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
        PetscBool   forward = PETSC_TRUE;
        PetscScalar sTy     = curvature;
        PetscReal   mu      = 0.25;

        if (!diagctx->invD) {
          PetscCall(VecDuplicate(lmvm->Fprev, &diagctx->invD));
          PetscCall(VecSetAsync_Private(diagctx->invD, PetscRealPart(sTy) / PetscRealPart(yTy), dctx));
        }
        if (!diagctx->U) PetscCall(VecDuplicate(lmvm->Fprev, &diagctx->U));
        if (!diagctx->V) PetscCall(VecDuplicate(lmvm->Fprev, &diagctx->V));
        if (!diagctx->W) PetscCall(VecDuplicate(lmvm->Fprev, &diagctx->W));

        if (forward) {
          PetscScalar sTDs, yTDy;

          /* diagonal Broyden */
          PetscCall(VecReciprocalAsync_Private(diagctx->invD, dctx));
          PetscCall(VecPointwiseMultAsync_Private(diagctx->V, diagctx->invD, lmvm->Xprev, dctx));
          PetscCall(VecPointwiseMultAsync_Private(diagctx->U, lmvm->Fprev, lmvm->Fprev, dctx));
          PetscCall(VecAXPYAsync_Private(diagctx->invD, 1.0 / sTy, diagctx->U, dctx));
          PetscCall(VecDot(diagctx->V, lmvm->Xprev, &sTDs));
          PetscCall(VecPointwiseMultAsync_Private(diagctx->V, diagctx->V, diagctx->V, dctx));
          PetscCall(VecAXPYAsync_Private(diagctx->invD, -1.0 / PetscMax(PetscRealPart(sTDs), diagctx->tol), diagctx->V, dctx));
          PetscCall(VecReciprocalAsync_Private(diagctx->invD, dctx));
          PetscCall(VecAbsAsync_Private(diagctx->invD, dctx));
          PetscCall(VecDot(diagctx->U, diagctx->invD, &yTDy));
          PetscCall(VecScaleAsync_Private(diagctx->invD, PetscRealPart(sTy) / PetscRealPart(yTDy), dctx));
        } else {
          PetscScalar sTs;
          PetscCall(VecDot(lmvm->Xprev, lmvm->Xprev, &sTs));

          PetscCall(VecSetAsync_Private(diagctx->V, PetscRealPart(sTy) / PetscRealPart(yTy), dctx));
          PetscCall(VecPointwiseMinAsync_Private(diagctx->V, diagctx->V, diagctx->invD, dctx)); /* lower bound = min(invD, sTy / yTy) */
          PetscCall(VecSetAsync_Private(diagctx->W, PetscRealPart(sTs) / PetscRealPart(sTy), dctx));
          PetscCall(VecPointwiseMaxAsync_Private(diagctx->W, diagctx->W, diagctx->invD, dctx)); /* lower bound = max(invD, sTs / sTy) */

          PetscCall(VecPointwiseMultAsync_Private(diagctx->U, lmvm->Fprev, lmvm->Xprev, dctx));
          PetscCall(VecAYPXAsync_Private(diagctx->invD, mu, diagctx->U, dctx));
          PetscCall(VecPointwiseMultAsync_Private(diagctx->U, lmvm->Fprev, lmvm->Fprev, dctx));
          PetscCall(VecShiftAsync_Private(diagctx->U, mu, dctx));
          PetscCall(VecPointwiseDivideAsync_Private(diagctx->invD, diagctx->invD, diagctx->U, dctx)); /* (s o y + mu invD) / (y o y + mu) */
          PetscCall(VecPointwiseMaxAsync_Private(diagctx->invD, diagctx->invD, diagctx->V, dctx));    /* enforce lower bound              */
          PetscCall(VecPointwiseMinAsync_Private(diagctx->invD, diagctx->invD, diagctx->W, dctx));    /* enforce upper bound              */
        }
      }
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lqn->watchdog;
      lmvm->k    = lmvm->k - 1;
      PetscInt m = lmvm->m;
      PetscInt k = lqn->num_updates;
      PetscInt h = k - oldest_update(m, k);

      /* we still have to maintain StFprev */
      if (!lqn->StFprev) {
        PetscCall(VecDuplicate(lqn->rwork1, &lqn->StFprev));
        PetscCall(VecZeroEntries(lqn->StFprev));
      }
      PetscCall(MatMultTransposeColumnRange(lqn->Sfull, F, lqn->StFprev, 0, h));
      lqn->St_count++;
      /* we still have to maintain YtXprev */
      if (!lqn->YtXprev) {
        PetscCall(VecDuplicate(lqn->rwork1, &lqn->YtXprev));
        PetscCall(VecZeroEntries(lqn->YtXprev));
      }
      PetscCall(MatMultTransposeColumnRange(lqn->Yfull, X, lqn->YtXprev, 0, h));
      lqn->Yt_count++;
    }
  } else {
    switch (lqn->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(VecSetAsync_Private(diagctx->invD, diagctx->delta, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      diagctx->sigma = diagctx->delta;
      break;
    default:
      diagctx->sigma = 1.0;
      break;
    }
  }

  if (lqn->watchdog > lqn->max_seq_rejects) PetscCall(MatLMVMReset(B, PETSC_FALSE));

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopyAsync_Private(X, lmvm->Xprev, dctx));
  PetscCall(VecCopyAsync_Private(F, lmvm->Fprev, dctx));
  PetscCall(PetscObjectReference((PetscObject)F));
  PetscCall(VecDestroy(&lqn->Fprev_ref));
  lqn->Fprev_ref = F;
  PetscCall(PetscObjectStateGet((PetscObject)F, &lqn->Fprev_state));

  PetscCall(PetscObjectReference((PetscObject)X));
  PetscCall(VecDestroy(&lqn->Xprev_ref));
  lqn->Xprev_ref = X;
  PetscCall(PetscObjectStateGet((PetscObject)X, &lqn->Xprev_state));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMCDQN(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM *bdata = (Mat_LMVM *)B->data;
  Mat_CDQN *blqn  = (Mat_CDQN *)bdata->ctx;
  Mat_LMVM *mdata = (Mat_LMVM *)M->data;
  Mat_CDQN *mlqn  = (Mat_CDQN *)mdata->ctx;

  PetscFunctionBegin;
  mlqn->watchdog        = blqn->watchdog;
  mlqn->max_seq_rejects = blqn->max_seq_rejects;
  if (!(bdata->J0 || bdata->user_pc || bdata->user_ksp || bdata->user_scale)) { PetscCall(MatCopy(blqn->diag_qn, mlqn->diag_qn, SAME_NONZERO_PATTERN)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMCDQN(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(MatMult_LMVMCDDFP(B, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMCDQN(Mat H, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_LMVMCDBFGS(H, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMCDQN(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_CDQN *ldfp;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMCDQN));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view           = MatView_LMVMCDQN;
  B->ops->setup          = MatSetUp_LMVMCDQN;
  B->ops->setfromoptions = MatSetFromOptions_LMVMCDQN;
  B->ops->destroy        = MatDestroy_LMVMCDQN;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->square        = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMCDQN;
  lmvm->ops->reset    = MatReset_LMVMCDQN;
  lmvm->ops->update   = MatUpdate_LMVMCDQN;
  lmvm->ops->mult     = MatMult_LMVMCDQN;
  lmvm->ops->solve    = MatSolve_LMVMCDQN;
  lmvm->ops->copy     = MatCopy_LMVMCDQN;

  PetscCall(PetscNew(&ldfp));
  lmvm->ctx             = (void *)ldfp;
  ldfp->allocated       = PETSC_FALSE;
  ldfp->watchdog        = 0;
  ldfp->max_seq_rejects = lmvm->m / 2;
  ldfp->strategy        = MAT_LMVM_CD_INPLACE;
  ldfp->scale_type      = MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &ldfp->diag_qn));
  PetscCall(MatSetType(ldfp->diag_qn, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetOptionsPrefix(ldfp->diag_qn, "J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMCDQN - Creates a compact dense representation of the limited-memory
  Quasi-Newton approximation to a Hessian. This compact dense representation uses
  Davidon-Fletcher-Powell (DFP) for MatMult, and Broyden-Fletcher-Goldfarb-Shanno (BFGS)
  for MatSolve. This implementation results in avoiding costly Cholesky factorization,
  at the cost of duality cap. Please refer to MatLMVMCDDFP and MatLMVMCDBFGS for more
  information.

  Collective

  Input Parameters:
+ comm - MPI communicator, set to PETSC_COMM_SELF
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
  paradigm instead of this routine directly.

  Level: advanced

.seealso: `MatCreate()`, `MATLMVM`, `MATLMVMCDBFGS`, `MATLMVMCDDFP`, `MatCreateLMVMCDDFP()`, `MatCreateLMVMCDBFGS()`
@*/
PetscErrorCode MatCreateLMVMCDQN(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMCDQN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCDQNApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *lqn  = (Mat_CDQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDQN_J0Fwd, B, X, Z, 0));
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    lqn->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  } else {
    PetscDeviceContext dctx;
    Mat_LMVM          *dbase   = (Mat_LMVM *)lqn->diag_qn->data;
    Mat_DiagBrdn      *diagctx = (Mat_DiagBrdn *)dbase->ctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    switch (lqn->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      PetscCall(VecAXPBYAsync_Private(Z, 1.0 / diagctx->sigma, 0.0, X, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(VecPointwiseDivideAsync_Private(Z, diagctx->invD, X, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    default:
      PetscCall(VecCopyAsync_Private(X, Z, dctx));
      break;
    }
  }
  PetscCall(PetscLogEventEnd(CDQN_J0Fwd, B, X, Z, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCDQNApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *lqn  = (Mat_CDQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDQN_J0Inv, B, F, dX, 0));
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    lqn->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  } else {
    PetscDeviceContext dctx;
    Mat_LMVM          *dbase   = (Mat_LMVM *)lqn->diag_qn->data;
    Mat_DiagBrdn      *diagctx = (Mat_DiagBrdn *)dbase->ctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    switch (lqn->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      PetscCall(VecAXPBYAsync_Private(dX, diagctx->sigma, 0.0, F, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(VecPointwiseMultAsync_Private(dX, diagctx->invD, F, dctx));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    default:
      PetscCall(VecCopyAsync_Private(F, dX, dctx));
      break;
    }
  }
  PetscCall(PetscLogEventEnd(CDQN_J0Inv, B, F, dX, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This is not Bunch-Kaufman LDLT: here L is strictly lower triangular part of STY */
static PetscErrorCode MatGetLDLT(Mat B, Mat result)
{
  Mat_LMVM *lmvm  = (Mat_LMVM *)B->data;
  Mat_CDQN *lbfgs = (Mat_CDQN *)lmvm->ctx;
  PetscInt  m_local;

  PetscFunctionBegin;
  if (!lbfgs->temp_mat) PetscCall(MatDuplicate(lbfgs->YtS_triu_strict, MAT_SHARE_NONZERO_PATTERN, &lbfgs->temp_mat));
  PetscCall(MatCopy(lbfgs->YtS_triu_strict, lbfgs->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(lbfgs->temp_mat, lbfgs->inv_diag_vec, NULL));
  PetscCall(MatGetLocalSize(result, &m_local, NULL));
  if (m_local) {
    Mat temp_local, YtS_local, result_local;
    PetscCall(MatDenseGetLocalMatrix(lbfgs->YtS_triu_strict, &YtS_local));
    PetscCall(MatDenseGetLocalMatrix(lbfgs->temp_mat, &temp_local));
    PetscCall(MatDenseGetLocalMatrix(result, &result_local));
    PetscCall(MatTransposeMatMult(YtS_local, temp_local, MAT_REUSE_MATRIX, PETSC_DEFAULT, &result_local));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCDBFGSUpdateMultData(Mat B)
{
  Mat_LMVM *lmvm  = (Mat_LMVM *)B->data;
  Mat_CDQN *lbfgs = (Mat_CDQN *)lmvm->ctx;
  PetscInt  m     = lmvm->m, m_local;
  PetscInt  k     = lbfgs->num_updates;
  PetscInt  h     = k - oldest_update(m, k);
  PetscInt  j_0;
  PetscInt  prev_oldest;
  Mat       J_local;

  PetscFunctionBegin;
  if (!lbfgs->YtS_triu_strict) {
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->YtS_triu_strict));
    PetscCall(MatDestroy(&lbfgs->StBS));
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->StBS));
    PetscCall(MatDestroy(&lbfgs->J));
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->J));
    PetscCall(MatDestroy(&lbfgs->BS));
    PetscCall(MatDuplicate(lbfgs->Yfull, MAT_SHARE_NONZERO_PATTERN, &lbfgs->BS));
    PetscCall(MatZeroEntries(lbfgs->YtS_triu_strict));
    PetscCall(MatZeroEntries(lbfgs->BS));
    PetscCall(MatZeroEntries(lbfgs->StBS));
    PetscCall(MatZeroEntries(lbfgs->J));
    PetscCall(MatShift(lbfgs->StBS, 1.0));
    lbfgs->num_mult_updates = oldest_update(m, k);
  }
  if (lbfgs->num_mult_updates == k) PetscFunctionReturn(PETSC_SUCCESS);

  /* B_0 may have been updated, we must recompute B_0 S and S^T B_0 S */
  for (PetscInt j = oldest_update(m, k); j < k; j++) {
    Vec      s_j;
    Vec      Bs_j;
    Vec      StBs_j;
    PetscInt S_idx    = recycle_index(m, j);
    PetscInt StBS_idx = lbfgs->strategy == MAT_LMVM_CD_INPLACE ? S_idx : history_index(m, k, j);

    PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, S_idx, &Bs_j));
    PetscCall(MatDenseGetColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatCDQNApplyJ0Fwd(B, s_j, Bs_j));
    PetscCall(MatDenseRestoreColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(lbfgs->StBS, StBS_idx, &StBs_j));
    PetscCall(MatMultTransposeColumnRange(lbfgs->Sfull, Bs_j, StBs_j, 0, h));
    lbfgs->St_count++;
    if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, StBs_j, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StBS, StBS_idx, &StBs_j));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, S_idx, &Bs_j));
  }
  prev_oldest = oldest_update(m, lbfgs->num_mult_updates);
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER && prev_oldest < oldest_update(m, k)) {
    /* move the YtS entries that have been computed and need to be kept back up */
    PetscInt m_keep = m - (oldest_update(m, k) - prev_oldest);

    PetscCall(MatMove_LR3(B, lbfgs->YtS_triu_strict, m_keep));
  }
  PetscCall(MatGetLocalSize(lbfgs->YtS_triu_strict, &m_local, NULL));
  j_0 = PetscMax(lbfgs->num_mult_updates, oldest_update(m, k));
  for (PetscInt j = j_0; j < k; j++) {
    PetscInt S_idx   = recycle_index(m, j);
    PetscInt YtS_idx = lbfgs->strategy == MAT_LMVM_CD_INPLACE ? S_idx : history_index(m, k, j);
    Vec      s_j, Yts_j;

    PetscCall(MatDenseGetColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(lbfgs->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatMultTransposeColumnRange(lbfgs->Yfull, s_j, Yts_j, 0, h));
    lbfgs->Yt_count++;
    if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, Yts_j, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatDenseRestoreColumnVecRead(lbfgs->Sfull, S_idx, &s_j));
    /* zero the corresponding row */
    if (m_local > 0) {
      Mat YtS_local, YtS_row;

      PetscCall(MatDenseGetLocalMatrix(lbfgs->YtS_triu_strict, &YtS_local));
      PetscCall(MatDenseGetSubMatrix(YtS_local, YtS_idx, YtS_idx + 1, PETSC_DECIDE, PETSC_DECIDE, &YtS_row));
      PetscCall(MatZeroEntries(YtS_row));
      PetscCall(MatDenseRestoreSubMatrix(YtS_local, &YtS_row));
    }
  }
  {
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

    if (!lbfgs->inv_diag_vec) PetscCall(VecDuplicate(lbfgs->diag_vec, &lbfgs->inv_diag_vec));
    PetscCall(VecCopyAsync_Private(lbfgs->diag_vec, lbfgs->inv_diag_vec, dctx));
    PetscCall(VecReciprocalAsync_Private(lbfgs->inv_diag_vec, dctx));
  }
  PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
  PetscCall(MatSetFactorType(J_local, MAT_FACTOR_NONE));
  PetscCall(MatGetLDLT(B, lbfgs->J));
  PetscCall(MatAXPY(lbfgs->J, 1.0, lbfgs->StBS, SAME_NONZERO_PATTERN));
  if (m_local) {
    PetscCall(MatSetOption(J_local, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(J_local, NULL, NULL));
  }
  lbfgs->num_mult_updates = lbfgs->num_updates;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
 * [ I | -S R^{-T} ] [  I  | 0 ] [ H_0 | 0 ] [ I | Y ] [      I      ]
 *                   [-----+---] [-----+---] [---+---] [-------------]
 *                   [ Y^T | I ] [  0  | D ] [ 0 | I ] [ -R^{-1} S^T ]  */

static PetscErrorCode MatSolve_LMVMCDBFGS(Mat H, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM *)H->data;
  Mat_CDQN          *lbfgs = (Mat_CDQN *)lmvm->ctx;
  PetscDeviceContext dctx;
  Vec                rwork1 = lbfgs->rwork1;
  PetscInt           m      = lmvm->m;
  PetscInt           k      = lbfgs->num_updates;
  PetscInt           h      = k - oldest_update(m, k);
  PetscObjectState   Fstate;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_MatSolve, H, F, dX, 0));
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Block Version */
  if (!lbfgs->num_updates) {
    PetscCall(MatCDQNApplyJ0Inv(H, F, dX));
    PetscCall(PetscLogEventEnd(CDBFGS_MatSolve, H, F, dX, 0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(PetscObjectStateGet((PetscObject)F, &Fstate));
  if (F == lbfgs->Fprev_ref && Fstate == lbfgs->Fprev_state) {
    PetscCall(VecCopyAsync_Private(lbfgs->StFprev, rwork1, dctx));
  } else {
    PetscCall(MatMultTransposeColumnRange(lbfgs->Sfull, F, rwork1, 0, h));
    lbfgs->St_count++;
  }

  /* Reordering rwork1, as STY is in history order, while S is in recycled order */
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(H, lbfgs->StY_triu, rwork1, PETSC_FALSE, lbfgs->num_updates, lbfgs->strategy));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));

  PetscCall(VecCopyAsync_Private(F, lbfgs->column_work, dctx));
  PetscCall(MatMultAddColumnRange(lbfgs->Yfull, rwork1, lbfgs->column_work, lbfgs->column_work, 0, h));
  lbfgs->Y_count++;

  PetscCall(VecPointwiseMultAsync_Private(rwork1, lbfgs->diag_vec_recycle_order, rwork1, dctx));
  PetscCall(MatCDQNApplyJ0Inv(H, lbfgs->column_work, dX));

  PetscCall(MatMultTransposeAddColumnRange(lbfgs->Yfull, dX, rwork1, rwork1, 0, h));
  lbfgs->Yt_count++;

  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(H, lbfgs->StY_triu, rwork1, PETSC_TRUE, lbfgs->num_updates, lbfgs->strategy));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));

  PetscCall(MatMultAddColumnRange(lbfgs->Sfull, rwork1, dX, dX, 0, h));
  lbfgs->S_count++;
  PetscCall(PetscLogEventEnd(CDBFGS_MatSolve, H, F, dX, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
   B_0 - [ Y | B_0 S] [ -D  |    L^T    ]^-1 [   Y^T   ]
                      [-----+-----------]    [---------]
                      [  L  | S^T B_0 S ]    [ S^T B_0 ]

   Above is equivalent to

   B_0 - [ Y | B_0 S] [[     I     | 0 ][ -D  | 0 ][ I | -D^{-1} L^T ]]^-1 [   Y^T   ]
                      [[-----------+---][-----+---][---+-------------]]    [---------]
                      [[ -L D^{-1} | I ][  0  | J ][ 0 |       I     ]]    [ S^T B_0 ]

   where J = S^T B_0 S + L D^{-1} L^T

   becomes

   B_0 - [ Y | B_0 S] [ I | D^{-1} L^T ][ -D^{-1}  |   0    ][    I     | 0 ] [   Y^T   ]
                      [---+------------][----------+--------][----------+---] [---------]
                      [ 0 |     I      ][     0    | J^{-1} ][ L D^{-1} | I ] [ S^T B_0 ]

                      =

   B_0 + [ Y | B_0 S] [ D^{-1} | 0 ][ I | L^T ][ I |    0    ][     I    | 0 ] [   Y^T   ]
                      [--------+---][---+-----][---+---------][----------+---] [---------]
                      [ 0      | I ][ 0 |  I  ][ 0 | -J^{-1} ][ L D^{-1} | I ] [ S^T B_0 ]

                      (Note that YtS_triu_strict is L^T)
   Byrd, Nocedal, Schnabel 1994

   Alternative approach: considering the fact that DFP is dual to BFGS, use MatMult of DPF:
   (See cddfp.c's MatMult_LMVMCDDFP)

*/

static PetscErrorCode MatMult_LMVMCDBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm  = (Mat_LMVM *)B->data;
  Mat_CDQN          *lbfgs = (Mat_CDQN *)lmvm->ctx;
  Mat                J_local;
  PetscInt           m_local;
  PetscInt           m = lmvm->m;
  PetscInt           k = lbfgs->num_updates;
  PetscInt           h = k - oldest_update(m, k);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDBFGS_MatMult, B, X, Z, 0));
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Cholesky Version */
  /* Start with the B0 term */
  PetscCall(MatCDQNApplyJ0Fwd(B, X, Z));
  if (!lbfgs->num_updates) {
    PetscCall(PetscLogEventEnd(CDBFGS_MatMult, B, X, Z, 0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(MatLMVMCDBFGSUpdateMultData(B));
  PetscCall(MatMultTransposeColumnRange(lbfgs->Yfull, X, lbfgs->rwork1, 0, h));
  lbfgs->Yt_count++;
  PetscCall(MatMultTransposeColumnRange(lbfgs->Sfull, Z, lbfgs->rwork2, 0, h));
  lbfgs->St_count++;
  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) {
    PetscCall(VecRecycleOrderToHistoryOrder(B, lbfgs->rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(VecRecycleOrderToHistoryOrder(B, lbfgs->rwork2, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  }

  PetscCall(VecPointwiseMultAsync_Private(lbfgs->rwork3, lbfgs->rwork1, lbfgs->inv_diag_vec, dctx));

  PetscCall(MatMultTransposeAdd(lbfgs->YtS_triu_strict, lbfgs->rwork3, lbfgs->rwork2, lbfgs->rwork2));

  if (!lbfgs->rwork2_local) PetscCall(VecCreateLocalVector(lbfgs->rwork2, &lbfgs->rwork2_local));
  if (!lbfgs->rwork3_local) PetscCall(VecCreateLocalVector(lbfgs->rwork3, &lbfgs->rwork3_local));
  PetscCall(VecGetLocalVectorRead(lbfgs->rwork2, lbfgs->rwork2_local));
  PetscCall(VecGetLocalVector(lbfgs->rwork3, lbfgs->rwork3_local));
  PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
  PetscCall(VecGetSize(lbfgs->rwork2_local, &m_local));
  if (m_local) {
    Mat J_local;

    PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
    PetscCall(MatSolve(J_local, lbfgs->rwork2_local, lbfgs->rwork3_local));
  }
  PetscCall(VecRestoreLocalVector(lbfgs->rwork3, lbfgs->rwork3_local));
  PetscCall(VecRestoreLocalVectorRead(lbfgs->rwork2, lbfgs->rwork2_local));
  PetscCall(VecScale(lbfgs->rwork3, -1.0));

  PetscCall(MatMultAdd(lbfgs->YtS_triu_strict, lbfgs->rwork3, lbfgs->rwork1, lbfgs->rwork1));

  PetscCall(VecPointwiseMultAsync_Private(lbfgs->rwork1, lbfgs->rwork1, lbfgs->inv_diag_vec, dctx));

  if (lbfgs->strategy == MAT_LMVM_CD_REORDER) {
    PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->rwork3, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  }

  PetscCall(MatMultAddColumnRange(lbfgs->Yfull, lbfgs->rwork1, Z, Z, 0, h));
  lbfgs->Y_count++;
  PetscCall(MatMultAddColumnRange(lbfgs->BS, lbfgs->rwork3, Z, Z, 0, h));
  lbfgs->S_count++;
  PetscCall(PetscLogEventEnd(CDBFGS_MatMult, B, X, Z, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMCDBFGS(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_CDQN *lbfgs;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMCDBFGS));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view           = MatView_LMVMCDQN;
  B->ops->setup          = MatSetUp_LMVMCDQN;
  B->ops->setfromoptions = MatSetFromOptions_LMVMCDQN;
  B->ops->destroy        = MatDestroy_LMVMCDQN;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->square        = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMCDQN;
  lmvm->ops->reset    = MatReset_LMVMCDQN;
  lmvm->ops->update   = MatUpdate_LMVMCDQN;
  lmvm->ops->mult     = MatMult_LMVMCDBFGS;
  lmvm->ops->solve    = MatSolve_LMVMCDBFGS;
  lmvm->ops->copy     = MatCopy_LMVMCDQN;

  PetscCall(PetscNew(&lbfgs));
  lmvm->ctx              = (void *)lbfgs;
  lbfgs->allocated       = PETSC_FALSE;
  lbfgs->watchdog        = 0;
  lbfgs->max_seq_rejects = lmvm->m / 2;
  lbfgs->strategy        = MAT_LMVM_CD_INPLACE;
  lbfgs->scale_type      = MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &lbfgs->diag_qn));
  PetscCall(MatSetType(lbfgs->diag_qn, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetOptionsPrefix(lbfgs->diag_qn, "J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMCDBFGS - Creates a compact dense (CD) representation of the limited-memory
  Broyden-Fletcher-Goldfarb-Shanno (BFGS) approximation to a Hessian. This CD representation
  reduces the L-BFGS update to a series of matrix-vector products with CD matrices
  in lieu of the conventional matrix-free two-loop algorithm. For most problems on CPUs,
  this representation is not as fast as the regular implementation provided via MATLMVMBFGS.
  However, it may be faster on GPUs for large enough problems (note: requires CUDA/HIP/KOKKOS).

  Collective

  Input Parameters:
+ comm - MPI communicator, set to PETSC_COMM_SELF
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
  paradigm instead of this routine directly.

  Level: advanced

.seealso: `MatCreate()`, `MATLMVM`, `MATLMVMCDBFGS`, `MatCreateLMVMBFGS()`
@*/
PetscErrorCode MatCreateLMVMCDBFGS(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMCDBFGS));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* here R is strictly upper triangular part of STY */
static PetscErrorCode MatGetRTDR(Mat B, Mat result)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *ldfp = (Mat_CDQN *)lmvm->ctx;
  PetscInt  m_local;

  PetscFunctionBegin;
  if (!ldfp->temp_mat) PetscCall(MatDuplicate(ldfp->StY_triu_strict, MAT_SHARE_NONZERO_PATTERN, &ldfp->temp_mat));
  PetscCall(MatCopy(ldfp->StY_triu_strict, ldfp->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(ldfp->temp_mat, ldfp->inv_diag_vec, NULL));
  PetscCall(MatGetLocalSize(result, &m_local, NULL));
  if (m_local) {
    Mat temp_local, StY_local, result_local;
    PetscCall(MatDenseGetLocalMatrix(ldfp->StY_triu_strict, &StY_local));
    PetscCall(MatDenseGetLocalMatrix(ldfp->temp_mat, &temp_local));
    PetscCall(MatDenseGetLocalMatrix(result, &result_local));
    PetscCall(MatTransposeMatMult(StY_local, temp_local, MAT_REUSE_MATRIX, PETSC_DEFAULT, &result_local));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCDDFPUpdateSolveData(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_CDQN *ldfp = (Mat_CDQN *)lmvm->ctx;
  PetscInt  m    = lmvm->m, m_local;
  PetscInt  k    = ldfp->num_updates;
  PetscInt  h    = k - oldest_update(m, k);
  PetscInt  j_0;
  PetscInt  prev_oldest;
  Mat       J_local;

  PetscFunctionBegin;
  if (!ldfp->StY_triu_strict) {
    PetscCall(MatDuplicate(ldfp->YtS_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->StY_triu_strict));
    PetscCall(MatDestroy(&ldfp->YtHY));
    PetscCall(MatDuplicate(ldfp->YtS_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->YtHY));
    PetscCall(MatDestroy(&ldfp->J));
    PetscCall(MatDuplicate(ldfp->YtS_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->J));
    PetscCall(MatDestroy(&ldfp->HY));
    PetscCall(MatDuplicate(ldfp->Yfull, MAT_SHARE_NONZERO_PATTERN, &ldfp->HY));
    PetscCall(MatZeroEntries(ldfp->StY_triu_strict));
    PetscCall(MatZeroEntries(ldfp->HY));
    PetscCall(MatZeroEntries(ldfp->YtHY));
    PetscCall(MatZeroEntries(ldfp->J));
    PetscCall(MatShift(ldfp->YtHY, 1.0));
    ldfp->num_mult_updates = oldest_update(m, k);
  }
  if (ldfp->num_mult_updates == k) PetscFunctionReturn(PETSC_SUCCESS);

  /* H_0 may have been updated, we must recompute H_0 Y and Y^T H_0 Y */
  for (PetscInt j = oldest_update(m, k); j < k; j++) {
    Vec      y_j;
    Vec      Hy_j;
    Vec      YtHy_j;
    PetscInt Y_idx    = recycle_index(m, j);
    PetscInt YtHY_idx = ldfp->strategy == MAT_LMVM_CD_INPLACE ? Y_idx : history_index(m, k, j);

    PetscCall(MatDenseGetColumnVecWrite(ldfp->HY, Y_idx, &Hy_j));
    PetscCall(MatDenseGetColumnVecRead(ldfp->Yfull, Y_idx, &y_j));
    PetscCall(MatCDQNApplyJ0Inv(B, y_j, Hy_j));
    PetscCall(MatDenseRestoreColumnVecRead(ldfp->Yfull, Y_idx, &y_j));
    PetscCall(MatDenseGetColumnVecWrite(ldfp->YtHY, YtHY_idx, &YtHy_j));
    PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, Hy_j, YtHy_j, 0, h));
    ldfp->Yt_count++;
    if (ldfp->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, YtHy_j, ldfp->num_updates, ldfp->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->YtHY, YtHY_idx, &YtHy_j));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->HY, Y_idx, &Hy_j));
  }
  prev_oldest = oldest_update(m, ldfp->num_mult_updates);
  if (ldfp->strategy == MAT_LMVM_CD_REORDER && prev_oldest < oldest_update(m, k)) {
    /* move the YtS entries that have been computed and need to be kept back up */
    PetscInt m_keep = m - (oldest_update(m, k) - prev_oldest);

    PetscCall(MatMove_LR3(B, ldfp->StY_triu_strict, m_keep));
  }
  PetscCall(MatGetLocalSize(ldfp->StY_triu_strict, &m_local, NULL));
  j_0 = PetscMax(ldfp->num_mult_updates, oldest_update(m, k));
  for (PetscInt j = j_0; j < k; j++) {
    PetscInt Y_idx   = recycle_index(m, j);
    PetscInt StY_idx = ldfp->strategy == MAT_LMVM_CD_INPLACE ? Y_idx : history_index(m, k, j);
    Vec      y_j, Sty_j;

    PetscCall(MatDenseGetColumnVecRead(ldfp->Yfull, Y_idx, &y_j));
    PetscCall(MatDenseGetColumnVecWrite(ldfp->StY_triu_strict, StY_idx, &Sty_j));
    PetscCall(MatMultTransposeColumnRange(ldfp->Sfull, y_j, Sty_j, 0, h));
    ldfp->St_count++;
    if (ldfp->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, Sty_j, ldfp->num_updates, ldfp->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->StY_triu_strict, StY_idx, &Sty_j));
    PetscCall(MatDenseRestoreColumnVecRead(ldfp->Yfull, Y_idx, &y_j));
    /* zero the corresponding row */
    if (m_local > 0) {
      Mat StY_local, StY_row;

      PetscCall(MatDenseGetLocalMatrix(ldfp->StY_triu_strict, &StY_local));
      PetscCall(MatDenseGetSubMatrix(StY_local, StY_idx, StY_idx + 1, PETSC_DECIDE, PETSC_DECIDE, &StY_row));
      PetscCall(MatZeroEntries(StY_row));
      PetscCall(MatDenseRestoreSubMatrix(StY_local, &StY_row));
    }
  }
  {
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

    if (!ldfp->inv_diag_vec) PetscCall(VecDuplicate(ldfp->diag_vec, &ldfp->inv_diag_vec));
    PetscCall(VecCopyAsync_Private(ldfp->diag_vec, ldfp->inv_diag_vec, dctx));
    PetscCall(VecReciprocalAsync_Private(ldfp->inv_diag_vec, dctx));
  }
  PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
  PetscCall(MatSetFactorType(J_local, MAT_FACTOR_NONE));
  PetscCall(MatGetRTDR(B, ldfp->J));
  PetscCall(MatAXPY(ldfp->J, 1.0, ldfp->YtHY, SAME_NONZERO_PATTERN));
  if (m_local) {
    PetscCall(MatSetOption(J_local, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(J_local, NULL, NULL));
  }
  ldfp->num_mult_updates = ldfp->num_updates;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for

   H_0 - [ S | H_0 Y] [ -D  |    R.T    ]^-1 [   S^T   ]
                      [-----+-----------]    [---------]
                      [  R  | Y^T H_0 Y ]    [ Y^T H_0 ]

   Above is equivalent to

   H_0 - [ S | H_0 Y] [[     I     | 0 ][ -D | 0 ][ I | -D^{-1} R^T ]]^-1 [   S^T   ]
                      [[-----------+---][----+---][---+-------------]]    [---------]
                      [[ -R D^{-1} | I ][  0 | J ][ 0 |      I      ]]    [ Y^T H_0 ]

   where J = Y^T H_0 Y + R D^{-1} R.T

   becomes

   H_0 - [ S | H_0 Y] [ I | D^{-1} R^T ][ -D^{-1}  |   0    ][     I    | 0 ] [   S^T   ]
                      [---+------------][----------+--------][----------+---] [---------]
                      [ 0 |      I     ][     0    | J^{-1} ][ R D^{-1} | I ] [ Y^T H_0 ]

                      =

   H_0 + [ S | H_0 Y] [ D^{-1} | 0 ][ I | R^T ][ I |    0    ][     I    | 0 ] [   S^T   ]
                      [--------+---][---+-----][---+---------][----------+---] [---------]
                      [ 0      | I ][ 0 |  I  ][ 0 | -J^{-1} ][ R D^{-1} | I ] [ Y^T H_0 ]

                      (Note that StY_triu_strict is R)
   Byrd, Nocedal, Schnabel 1994

*/
static PetscErrorCode MatSolve_LMVMCDDFP(Mat H, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM *)H->data;
  Mat_CDQN          *ldfp = (Mat_CDQN *)lmvm->ctx;
  PetscInt           m    = lmvm->m;
  PetscInt           k    = ldfp->num_updates;
  PetscInt           h    = k - oldest_update(m, k);
  PetscInt           m_local;
  Mat                J_local;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDDFP_MatSolve, H, F, dX, 0));
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Cholesky Version */
  /* Start with the B0 term */
  PetscCall(MatCDQNApplyJ0Inv(H, F, dX));
  if (!ldfp->num_updates) {
    PetscCall(PetscLogEventEnd(CDDFP_MatSolve, H, F, dX, 0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(MatLMVMCDDFPUpdateSolveData(H));

  PetscCall(MatMultTransposeColumnRange(ldfp->Sfull, F, ldfp->rwork1, 0, h));
  ldfp->St_count++;
  PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, dX, ldfp->rwork2, 0, h));
  ldfp->Yt_count++;
  if (ldfp->strategy == MAT_LMVM_CD_REORDER) {
    PetscCall(VecRecycleOrderToHistoryOrder(H, ldfp->rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));
    PetscCall(VecRecycleOrderToHistoryOrder(H, ldfp->rwork2, ldfp->num_updates, ldfp->cyclic_work_vec));
  }

  PetscCall(VecPointwiseMultAsync_Private(ldfp->rwork3, ldfp->rwork1, ldfp->inv_diag_vec, dctx));
  PetscCall(MatMultTransposeAdd(ldfp->StY_triu_strict, ldfp->rwork3, ldfp->rwork2, ldfp->rwork2));

  if (!ldfp->rwork2_local) PetscCall(VecCreateLocalVector(ldfp->rwork2, &ldfp->rwork2_local));
  if (!ldfp->rwork3_local) PetscCall(VecCreateLocalVector(ldfp->rwork3, &ldfp->rwork3_local));
  PetscCall(VecGetLocalVectorRead(ldfp->rwork2, ldfp->rwork2_local));
  PetscCall(VecGetLocalVector(ldfp->rwork3, ldfp->rwork3_local));
  PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
  PetscCall(VecGetSize(ldfp->rwork2_local, &m_local));
  if (m_local) {
    Mat J_local;

    PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
    PetscCall(MatSolve(J_local, ldfp->rwork2_local, ldfp->rwork3_local));
  }
  PetscCall(VecRestoreLocalVector(ldfp->rwork3, ldfp->rwork3_local));
  PetscCall(VecRestoreLocalVectorRead(ldfp->rwork2, ldfp->rwork2_local));
  PetscCall(VecScale(ldfp->rwork3, -1.0));

  PetscCall(MatMultAdd(ldfp->StY_triu_strict, ldfp->rwork3, ldfp->rwork1, ldfp->rwork1));

  PetscCall(VecPointwiseMultAsync_Private(ldfp->rwork1, ldfp->rwork1, ldfp->inv_diag_vec, dctx));

  if (ldfp->strategy == MAT_LMVM_CD_REORDER) {
    PetscCall(VecHistoryOrderToRecycleOrder(H, ldfp->rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));
    PetscCall(VecHistoryOrderToRecycleOrder(H, ldfp->rwork3, ldfp->num_updates, ldfp->cyclic_work_vec));
  }

  PetscCall(MatMultAddColumnRange(ldfp->Sfull, ldfp->rwork1, dX, dX, 0, h));
  ldfp->S_count++;
  PetscCall(MatMultAddColumnRange(ldfp->HY, ldfp->rwork3, dX, dX, 0, h));
  ldfp->Y_count++;
  PetscCall(PetscLogEventEnd(CDDFP_MatSolve, H, F, dX, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
   (Theorem 1, Erway, Jain, and Marcia, 2013)

   B_0 - [ Y | B_0 S] [ -R^{-T} (D + S^T B_0 S) R^{-1} | R^{-T} ] [   Y^T   ]
                      ---------------------------------+--------] [---------]
                      [             R^{-1}             |   0    ] [ S^T B_0 ]

   (Note: R above is right triangular part of YTS)
   which becomes,

   [ I | -Y L^{-T} ] [  I  | 0 ] [ B_0 | 0 ] [ I | S ] [      I      ]
                     [-----+---] [-----+---] [---+---] [-------------]
                     [ S^T | I ] [  0  | D ] [ 0 | I ] [ -L^{-1} Y^T ]

   (Note: L above is right triangular part of STY)

*/
static PetscErrorCode MatMult_LMVMCDDFP(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm   = (Mat_LMVM *)B->data;
  Mat_CDQN          *ldfp   = (Mat_CDQN *)lmvm->ctx;
  Vec                rwork1 = ldfp->rwork1;
  PetscInt           m      = lmvm->m;
  PetscInt           k      = ldfp->num_updates;
  PetscInt           h      = k - oldest_update(m, k);
  PetscDeviceContext dctx;
  PetscObjectState   Xstate;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(CDDFP_MatMult, B, X, Z, 0));
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* DFP Version. Erway, Jain, Marcia, 2013, Theorem 1 */
  /* Block Version */
  if (!ldfp->num_updates) {
    PetscCall(MatCDQNApplyJ0Fwd(B, X, Z));
    PetscCall(PetscLogEventEnd(CDDFP_MatMult, B, X, Z, 0));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  PetscCall(PetscObjectStateGet((PetscObject)X, &Xstate));
  if (X == ldfp->Xprev_ref && Xstate == ldfp->Xprev_state) {
    PetscCall(VecCopyAsync_Private(ldfp->YtXprev, rwork1, dctx));
  } else {
    PetscCall(MatMultTransposeColumnRange(ldfp->Yfull, X, rwork1, 0, h));
    ldfp->Yt_count++;
  }

  /* Reordering rwork1, as STY is in history order, while Y is in recycled order */
  if (ldfp->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(B, ldfp->YtS_triu, rwork1, PETSC_FALSE, ldfp->num_updates, ldfp->strategy));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (ldfp->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(B, rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));

  PetscCall(VecCopyAsync_Private(X, ldfp->column_work, dctx));
  PetscCall(MatMultAddColumnRange(ldfp->Sfull, rwork1, ldfp->column_work, ldfp->column_work, 0, h));
  ldfp->S_count++;

  PetscCall(VecPointwiseMultAsync_Private(rwork1, ldfp->diag_vec_recycle_order, rwork1, dctx));
  PetscCall(MatCDQNApplyJ0Fwd(B, ldfp->column_work, Z));

  PetscCall(MatMultTransposeAddColumnRange(ldfp->Sfull, Z, rwork1, rwork1, 0, h));
  ldfp->St_count++;

  if (ldfp->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(B, ldfp->YtS_triu, rwork1, PETSC_TRUE, ldfp->num_updates, ldfp->strategy));
  PetscCall(VecScaleAsync_Private(rwork1, -1.0, dctx));
  if (ldfp->strategy == MAT_LMVM_CD_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(B, rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));

  PetscCall(MatMultAddColumnRange(ldfp->Yfull, rwork1, Z, Z, 0, h));
  ldfp->Y_count++;
  PetscCall(PetscLogEventEnd(CDDFP_MatMult, B, X, Z, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMCDDFP(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_CDQN *ldfp;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMCDDFP));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view           = MatView_LMVMCDQN;
  B->ops->setup          = MatSetUp_LMVMCDQN;
  B->ops->setfromoptions = MatSetFromOptions_LMVMCDQN;
  B->ops->destroy        = MatDestroy_LMVMCDQN;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->square        = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMCDQN;
  lmvm->ops->reset    = MatReset_LMVMCDQN;
  lmvm->ops->update   = MatUpdate_LMVMCDQN;
  lmvm->ops->mult     = MatMult_LMVMCDDFP;
  lmvm->ops->solve    = MatSolve_LMVMCDDFP;
  lmvm->ops->copy     = MatCopy_LMVMCDQN;

  PetscCall(PetscNew(&ldfp));
  lmvm->ctx             = (void *)ldfp;
  ldfp->allocated       = PETSC_FALSE;
  ldfp->watchdog        = 0;
  ldfp->max_seq_rejects = lmvm->m / 2;
  ldfp->strategy        = MAT_LMVM_CD_INPLACE;
  ldfp->scale_type      = MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &ldfp->diag_qn));
  PetscCall(MatSetType(ldfp->diag_qn, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetOptionsPrefix(ldfp->diag_qn, "J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

/*@
  MatCreateLMVMCDDFP - Creates a compact dense (CD) representation of the limited-memory
  Davidon-Fletcher-Powell (DFP) approximation to a Hessian. This CD representation
  reduces the L-DFP update to a series of matrix-vector products with CD matrices
  in lieu of the conventional matrix-free two-loop algorithm. For most problems on CPUs,
  this representation is not as fast as the regular implementation provided via MATLMVMDFP.
  However, it may be faster on GPUs for large enough problems (note: requires CUDA/HIP/KOKKOS).

  Collective

  Input Parameters:
+ comm - MPI communicator, set to PETSC_COMM_SELF
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
  paradigm instead of this routine directly.

  Level: advanced

.seealso: `MatCreate()`, `MATLMVM`, `MATLMVMCDDFP`, `MatCreateLMVMDFP()`
@*/
PetscErrorCode MatCreateLMVMCDDFP(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMCDDFP));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
