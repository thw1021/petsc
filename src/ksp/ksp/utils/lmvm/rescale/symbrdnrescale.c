#include <petscdevice.h>
#include "symbrdnrescale.h"

PetscLogEvent SBRDN_Rescale;

const char *const MatLMVMSymBroydenScaleTypes[] = {"none", "scalar", "diagonal", "user", "decide", "MatLMVMSymBroydenScaleType", "MAT_LMVM_SYMBROYDEN_SCALING_", NULL};

static PetscErrorCode SymBroydenRescaleUpdateScalar(Mat B, SymBroydenRescale ldb)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscReal a, b, c, signew;
  PetscReal sigma_inv, sigma;
  PetscInt  oldest, next;

  PetscFunctionBegin;
  next   = ldb->k;
  oldest = PetscMax(0, ldb->k - ldb->sigma_hist);
  PetscCall(MatNorm(lmvm->J0, NORM_INFINITY, &sigma_inv));
  sigma = 1.0 / sigma_inv;
  if (ldb->sigma_hist == 0) {
    signew = 1.0;
  } else {
    signew = 0.0;
    if (ldb->alpha == 1.0) {
      for (PetscInt i = 0; i < next - oldest; ++i) signew += ldb->yts[i] / ldb->yty[i];
    } else if (ldb->alpha == 0.5) {
      for (PetscInt i = 0; i < next - oldest; ++i) signew += ldb->sts[i] / ldb->yty[i];
      signew = PetscSqrtReal(signew);
    } else if (ldb->alpha == 0.0) {
      for (PetscInt i = 0; i < next - oldest; ++i) signew += ldb->sts[i] / ldb->yts[i];
    } else {
      /* compute coefficients of the quadratic */
      a = b = c = 0.0;
      for (PetscInt i = 0; i < next - oldest; ++i) {
        a += ldb->yty[i];
        b += ldb->yts[i];
        c += ldb->sts[i];
      }
      a *= ldb->alpha;
      b *= -(2.0 * ldb->alpha - 1.0);
      c *= ldb->alpha - 1.0;
      /* use quadratic formula to find roots */
      PetscReal sqrtdisc = PetscSqrtReal(b * b - 4 * a * c);
      if (b >= 0.0) {
        if (a >= 0.0) {
          signew = (2 * c) / (-b - sqrtdisc);
        } else {
          signew = (-b - sqrtdisc) / (2 * a);
        }
      } else {
        if (a >= 0.0) {
          signew = (-b + sqrtdisc) / (2 * a);
        } else {
          signew = (2 * c) / (-b + sqrtdisc);
        }
      }
      PetscCheck(signew > 0.0, PetscObjectComm((PetscObject)B), PETSC_ERR_CONV_FAILED, "Cannot find positive scalar");
    }
  }
  sigma = ldb->rho * signew + (1.0 - ldb->rho) * sigma;
  PetscCall(MatLMVMSetJ0Scale(B, 1.0 / sigma));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode DiagonalUpdate(SymBroydenRescale ldb, Vec D, Vec s, Vec y, Vec V, Vec W, Vec BFGS, Vec DFP, PetscReal theta, PetscReal yts)
{
  PetscFunctionBegin;
  /*  V = |y o y| */
  PetscCall(VecPointwiseMult(V, y, y));
  if (PetscDefined(USE_COMPLEX)) PetscCall(VecAbs(V));

  /*  W = D o s */
  PetscReal stDs;
  PetscCall(VecPointwiseMult(W, D, s));
  PetscCall(VecDotRealPart(W, s, &stDs));

  PetscCall(VecAXPY(D, 1.0 / yts, ldb->V));

  /*  Safeguard stDs */
  stDs = PetscMax(stDs, ldb->tol);

  if (theta != 1.0) {
    /*  BFGS portion of the update */

    /*  U = |(D o s) o (D o s)| */
    PetscCall(VecPointwiseMult(BFGS, W, W));
    if (PetscDefined(USE_COMPLEX)) PetscCall(VecAbs(BFGS));

    /*  Assemble */
    PetscCall(VecScale(BFGS, -1.0 / stDs));
  }

  if (theta != 0.0) {
    /*  DFP portion of the update */
    /*  U = Real(conj(y) o D o s) */
    PetscCall(VecCopy(y, DFP));
    PetscCall(VecConjugate(DFP));
    PetscCall(VecPointwiseMult(DFP, DFP, W));
    if (PetscDefined(USE_COMPLEX)) {
      PetscCall(VecCopy(DFP, W));
      PetscCall(VecConjugate(W));
      PetscCall(VecAXPY(DFP, 1.0, W));
    } else {
      PetscCall(VecScale(DFP, 2.0));
    }

    /*  Assemble */
    PetscCall(VecAXPBY(DFP, stDs / yts, -1.0, V));
  }

  if (theta == 0.0) {
    PetscCall(VecAXPY(D, 1.0, BFGS));
  } else if (theta == 1.0) {
    PetscCall(VecAXPY(D, 1.0 / yts, DFP));
  } else {
    /*  Broyden update Dkp1 = Dk + (1-theta)*P + theta*Q + y_i^2/yts*/
    PetscCall(VecAXPBYPCZ(D, 1.0 - theta, theta / yts, 1.0, BFGS, DFP));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymBroydenRescaleUpdateDiagonal(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleUpdate(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetDelta(SymBroydenRescale ldb, PetscReal delta)
{
  PetscFunctionBegin;
  ldb->delta = delta;
  ldb->delta = PetscMin(ldb->delta, ldb->delta_max);
  ldb->delta = PetscMax(ldb->delta, ldb->delta_min);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleCopy(SymBroydenRescale bctx, SymBroydenRescale mctx)
{
  PetscInt k = bctx->sigma_hist;

  PetscFunctionBegin;
  mctx->scale_type = bctx->scale_type;
  mctx->theta      = bctx->theta;
  mctx->alpha      = bctx->alpha;
  mctx->beta       = bctx->beta;
  mctx->rho        = bctx->rho;
  mctx->delta      = bctx->delta;
  mctx->delta_min  = bctx->delta_min;
  mctx->delta_max  = bctx->delta_max;
  mctx->tol        = bctx->tol;
  mctx->sigma_hist = bctx->sigma_hist;
  mctx->forward    = bctx->forward;
  for (PetscInt i = 0; i < k; ++i) {
    mctx->yty[i] = bctx->yty[i];
    mctx->yts[i] = bctx->yts[i];
    mctx->sts[i] = bctx->sts[i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetDiagonalMode(SymBroydenRescale ldb, PetscBool forward)
{
  PetscFunctionBegin;
  ldb->forward = forward;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleGetType(SymBroydenRescale ldb, MatLMVMSymBroydenScaleType *stype)
{
  PetscFunctionBegin;
  *stype = ldb->scale_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetType(SymBroydenRescale ldb, MatLMVMSymBroydenScaleType stype)
{
  PetscFunctionBegin;
  ldb->scale_type = stype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetFromOptions(Mat B, SymBroydenRescale ldb, PetscOptionItems PetscOptionsObject)
{
  MatLMVMSymBroydenScaleType stype = ldb->scale_type;
  PetscBool                  flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Restricted Broyden method for updating diagonal Jacobian approximation (MATLMVMDIAGBRDN)");
  PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "MatLMVMSymBroydenScaleType", MatLMVMSymBroydenScaleTypes, (PetscEnum)stype, (PetscEnum *)&stype, &flg));
  PetscCall(PetscOptionsReal("-mat_lmvm_theta", "(developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling", "", ldb->theta, &ldb->theta, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_rho", "(developer) update limiter in the J0 scaling", "", ldb->rho, &ldb->rho, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_tol", "(developer) tolerance for bounding rescaling denominator", "", ldb->tol, &ldb->tol, NULL));
  PetscCall(PetscOptionsRangeReal("-mat_lmvm_alpha", "(developer) convex ratio in the J0 scaling", "", ldb->alpha, &ldb->alpha, NULL, 0.0, 1.0));
  PetscCall(PetscOptionsBool("-mat_lmvm_forward", "Forward -> Update diagonal scaling for B. Else -> diagonal scaling for H.", "", ldb->forward, &ldb->forward, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_beta", "(developer) exponential factor in the diagonal J0 scaling", "", ldb->beta, &ldb->beta, NULL));
  PetscCall(PetscOptionsBoundedInt("-mat_lmvm_sigma_hist", "(developer) number of past updates to use in the default J0 scalar", "", ldb->sigma_hist, &ldb->sigma_hist, NULL, 0));
  PetscOptionsHeadEnd();
  PetscCheck(!(ldb->theta < 0.0) && !(ldb->theta > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the diagonal J0 scale cannot be outside the range of [0, 1]");
  PetscCheck(!(ldb->alpha < 0.0) && !(ldb->alpha > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCheck(!(ldb->rho < 0.0) && !(ldb->rho > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex update limiter in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCheck(ldb->sigma_hist >= 0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "J0 scaling history length cannot be negative");
  if (flg) PetscCall(SymBroydenRescaleSetType(ldb, stype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetUp(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DECIDE) {
    Mat       J0;
    PetscBool is_constant_or_diagonal;

    PetscCall(MatLMVMGetJ0(B, &J0));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)J0, &is_constant_or_diagonal, MATCONSTANTDIAGONAL, MATDIAGONAL, ""));
    ldb->scale_type = is_constant_or_diagonal ? MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL : MAT_LMVM_SYMBROYDEN_SCALE_NONE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleInitializeJ0(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleUpdateJ0(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleSetUp(B, ldb));
  if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) PetscCall(SymBroydenRescaleUpdateScalar(B, ldb));
  else if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(SymBroydenRescaleUpdateDiagonal(B, ldb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleView(SymBroydenRescale ldb, PetscViewer pv)
{
  PetscFunctionBegin;
  PetscBool isascii;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "Rescale type: %s\n", MatLMVMSymBroydenScaleTypes[ldb->scale_type]));
    if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR || ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
      PetscCall(PetscViewerASCIIPrintf(pv, "Rescale history: %" PetscInt_FMT "\n", ldb->sigma_hist));
      PetscCall(PetscViewerASCIIPrintf(pv, "Rescale params: alpha=%g, beta=%g, rho=%g\n", (double)ldb->alpha, (double)ldb->beta, (double)ldb->rho));
    }
    if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(PetscViewerASCIIPrintf(pv, "Rescale convex factor: theta=%g\n", (double)ldb->theta));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleReset(Mat B, SymBroydenRescale ldb, PetscBool destructive)
{
  PetscFunctionBegin;
  if (B && !destructive) PetscCall(SymBroydenRescaleInitializeJ0(B, ldb));
  if (destructive && ldb->allocated) {
    PetscCall(PetscFree3(ldb->yty, ldb->yts, ldb->sts));
    PetscCall(VecDestroy(&ldb->invDnew));
    PetscCall(VecDestroy(&ldb->BFGS));
    PetscCall(VecDestroy(&ldb->DFP));
    PetscCall(VecDestroy(&ldb->U));
    PetscCall(VecDestroy(&ldb->V));
    PetscCall(VecDestroy(&ldb->W));
    ldb->allocated = PETSC_FALSE;
  }
  ldb->k = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleAllocate(Mat B, SymBroydenRescale ldb)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!ldb->allocated) {
    PetscCall(PetscMalloc3(ldb->sigma_hist, &ldb->yty, ldb->sigma_hist, &ldb->yts, ldb->sigma_hist, &ldb->sts));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->invDnew));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->BFGS));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->DFP));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->U));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->V));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->W));
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleDestroy(SymBroydenRescale *ldb)
{
  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleReset(NULL, *ldb, PETSC_TRUE));
  PetscCall(PetscFree(*ldb));
  *ldb = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleCreate(SymBroydenRescale *ldb)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(ldb));
  (*ldb)->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_DECIDE;
  (*ldb)->theta      = 0.0;
  (*ldb)->alpha      = 1.0;
  (*ldb)->rho        = 1.0;
  (*ldb)->forward    = PETSC_TRUE;
  (*ldb)->beta       = 0.5;
  (*ldb)->delta      = 1.0;
  (*ldb)->delta_min  = 1e-7;
  (*ldb)->delta_max  = 100.0;
  (*ldb)->tol        = 1e-8;
  (*ldb)->sigma_hist = 1;
  (*ldb)->allocated  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
