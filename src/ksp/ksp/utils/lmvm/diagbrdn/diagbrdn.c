#include <petscdevice.h>
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h> /*I "petscksp.h" I*/

const char *const MatLMVMSymBroydenScaleTypes[] = {"none", "scalar", "diagonal", "user", "decide", "MatLMVMSymBroydenScaleType", "MAT_LMVM_SYMBROYDEN_SCALING_", NULL};

static PetscErrorCode MatSolve_DiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatSolve(lmvm->J0, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_DiagBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatMult(lmvm->J0, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

static PetscErrorCode DiagonalUpdate(SymBroydenRescale ldb, Vec D, Vec s, Vec y, Vec V, Vec W, Vec BFGS, Vec DFP, PetscReal theta, PetscReal yts)
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
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  PetscInt    oldest, next;
  Vec         invD, s_last, y_last;
  LMBasis     S, Y;
  PetscScalar yts;
  PetscReal   sigma;

  PetscFunctionBegin;
  next   = ldb->k;
  oldest = PetscMax(0, ldb->k - ldb->sigma_hist);
  PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, next - 1, &yts));
  PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_S, &S, NULL, NULL));
  PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_Y, &Y, NULL, NULL));
  PetscCall(LMBasisGetVecRead(S, next - 1, &s_last));
  PetscCall(LMBasisGetVecRead(Y, next - 1, &y_last));
  PetscCall(MatLMVMGetJ0InvDiag(B, &invD));
  if (ldb->forward) {
    /* We are doing diagonal scaling of the forward Hessian B */
    /*  BFGS = DFP = inv(D); */
    PetscCall(VecCopy(invD, ldb->invDnew));
    PetscCall(VecReciprocal(ldb->invDnew));
    PetscCall(DiagonalUpdate(ldb, ldb->invDnew, s_last, y_last, ldb->V, ldb->W, ldb->BFGS, ldb->DFP, ldb->theta, PetscRealPart(yts)));
    /*  Obtain inverse and ensure positive definite */
    PetscCall(VecReciprocal(ldb->invDnew));
  } else {
    /* Inverse Hessian update instead. */
    PetscCall(VecCopy(invD, ldb->invDnew));
    PetscCall(DiagonalUpdate(ldb, ldb->invDnew, y_last, s_last, ldb->V, ldb->W, ldb->DFP, ldb->BFGS, 1.0 - ldb->theta, PetscRealPart(yts)));
  }
  PetscCall(VecAbs(ldb->invDnew));
  PetscCall(LMBasisRestoreVecRead(Y, next - 1, &y_last));
  PetscCall(LMBasisRestoreVecRead(S, next - 1, &s_last));

  if (ldb->sigma_hist > 0) {
    // We are computing the scaling factor sigma that minimizes
    //
    // Sum_i || sigma^(alpha) (D^(-beta) o y_i) - sigma^(alpha-1) (D^(1-beta) o s_i) ||_2^2
    //                        `-------.-------'                   `--------.-------'
    //                               v_i                                  w_i
    //
    // To do this we first have to compute the sums of the dot product terms
    //
    // yy_sum = Sum_i v_i^T v_i,
    // ys_sum = Sum_i v_i^T w_i, and
    // ss_sum = Sum_i w_i^T w_i.
    //
    // These appear in the quadratic equation for the optimality condition for sigma,
    //
    // [alpha yy_sum] sigma^2 - [(2 alpha - 1) ys_sum] * sigma + [(alpha - 1) * ss_sum] = 0
    //
    // which we solve for sigma.

    PetscReal yy_sum = 0; /*  No safeguard required */
    PetscReal ys_sum = 0; /*  No safeguard required */
    PetscReal ss_sum = 0; /*  No safeguard required */
    PetscInt  start  = PetscMax(oldest, lmvm->k - ldb->sigma_hist);

    Vec D_minus_beta             = NULL;
    Vec D_minus_beta_squared     = NULL;
    Vec D_one_minus_beta         = NULL;
    Vec D_one_minus_beta_squared = NULL;
    if (ldb->beta == 0.5) {
      D_minus_beta_squared = ldb->invDnew; // (D^(-0.5))^2 = D^-1

      PetscCall(VecCopy(ldb->invDnew, ldb->U));
      PetscCall(VecReciprocal(ldb->U));
      D_one_minus_beta_squared = ldb->U; // (D^(1-0.5))^2 = D
    } else if (ldb->beta == 0.0) {
      PetscCall(VecCopy(ldb->invDnew, ldb->U));
      PetscCall(VecReciprocal(ldb->U));
      D_one_minus_beta = ldb->U; // D^1
    } else if (ldb->beta == 1.0) {
      D_minus_beta = ldb->invDnew; // D^-1
    } else {
      PetscCall(VecCopy(ldb->invDnew, ldb->DFP));
      PetscCall(VecPow(ldb->DFP, ldb->beta));
      D_minus_beta = ldb->DFP;

      PetscCall(VecCopy(ldb->invDnew, ldb->BFGS));
      PetscCall(VecPow(ldb->BFGS, ldb->beta - 1));
      D_one_minus_beta = ldb->BFGS;
    }
    for (PetscInt i = start - oldest; i < next - oldest; ++i) {
      Vec s_i, y_i;

      PetscCall(LMBasisGetVecRead(S, oldest + i, &s_i));
      PetscCall(LMBasisGetVecRead(Y, oldest + i, &y_i));
      if (ldb->beta == 0.5) {
        PetscReal ytDinvy, stDs;

        PetscCall(VecPointwiseMult(ldb->V, y_i, D_minus_beta_squared));
        PetscCall(VecPointwiseMult(ldb->W, s_i, D_one_minus_beta_squared));
        PetscCall(VecDotRealPart(ldb->W, s_i, &stDs));
        PetscCall(VecDotRealPart(ldb->V, y_i, &ytDinvy));
        ss_sum += stDs;        // ||s||_{D^(2*(1-beta))}^2
        ys_sum += ldb->yts[i]; // s^T D^(1 - 2*beta) y
        yy_sum += ytDinvy;     // ||y||_{D^(-2*beta)}^2
      } else if (ldb->beta == 0.0) {
        PetscScalar ytDs_scalar;
        PetscReal   stDsr;

        PetscCall(VecPointwiseMult(ldb->W, s_i, D_one_minus_beta));
        PetscCall(VecDotNorm2(y_i, ldb->W, &ytDs_scalar, &stDsr));
        ss_sum += stDsr;                      // ||s||_{D^(2*(1-beta))}^2
        ys_sum += PetscRealPart(ytDs_scalar); // s^T D^(1 - 2*beta) y
        yy_sum += ldb->yty[i];                // ||y||_{D^(-2*beta)}^2
      } else if (ldb->beta == 1.0) {
        PetscScalar ytDs_scalar;
        PetscReal   ytDyr;

        PetscCall(VecPointwiseMult(ldb->V, y_i, D_minus_beta));
        PetscCall(VecDotNorm2(s_i, ldb->V, &ytDs_scalar, &ytDyr));
        ss_sum += ldb->sts[i];                // ||s||_{D^(2*(1-beta))}^2
        ys_sum += PetscRealPart(ytDs_scalar); // s^T D^(1 - 2*beta) y
        yy_sum += ytDyr;                      // ||y||_{D^(-2*beta)}^2
      } else {
        PetscScalar ytDs_scalar;
        PetscReal   ytDyr, stDs;

        PetscCall(VecPointwiseMult(ldb->V, y_i, D_minus_beta));
        PetscCall(VecPointwiseMult(ldb->W, s_i, D_one_minus_beta));
        PetscCall(VecDotNorm2(ldb->W, ldb->V, &ytDs_scalar, &ytDyr));
        PetscCall(VecDotRealPart(ldb->W, ldb->W, &stDs));
        ss_sum += stDs;                       // ||s||_{D^(2*(1-beta))}^2
        ys_sum += PetscRealPart(ytDs_scalar); // s^T D^(1 - 2*beta) y
        yy_sum += ytDyr;                      // ||y||_{D^(-2*beta)}^2
      }
      PetscCall(LMBasisRestoreVecRead(Y, oldest + i, &y_i));
      PetscCall(LMBasisRestoreVecRead(S, oldest + i, &s_i));
    }

    if (ldb->alpha == 0.0) {
      /*  Safeguard ys_sum  */
      ys_sum = PetscMax(ldb->tol, ys_sum);

      sigma = ss_sum / ys_sum;
    } else if (1.0 == ldb->alpha) {
      /* yy_sum is never 0; if it were, we'd be at the minimum */
      sigma = ys_sum / yy_sum;
    } else {
      PetscReal a         = ldb->alpha * yy_sum;
      PetscReal b         = -(2.0 * ldb->alpha - 1.0) * ys_sum;
      PetscReal c         = (ldb->alpha - 1.0) * ss_sum;
      PetscReal sqrt_disc = PetscSqrtReal(b * b - 4 * a * c);

      // numerically stable computation of positive root
      if (b >= 0.0) {
        if (a >= 0) {
          PetscReal denom = PetscMax(-b - sqrt_disc, ldb->tol);

          sigma = (2 * c) / denom;
        } else {
          PetscReal denom = PetscMax(2 * a, ldb->tol);

          sigma = (-b - sqrt_disc) / denom;
        }
      } else {
        if (a >= 0) {
          PetscReal denom = PetscMax(2 * a, ldb->tol);

          sigma = (-b + sqrt_disc) / denom;
        } else {
          PetscReal denom = PetscMax(-b + sqrt_disc, ldb->tol);

          sigma = (2 * c) / denom;
        }
      }
    }
  } else {
    sigma = 1.0;
  }
  /*  If Q has small values, then Q^(r_beta - 1)
      can have very large values.  Hence, ys_sum
      and ss_sum can be infinity.  In this case,
      sigma can either be not-a-number or infinity. */
  if (PetscIsNormalReal(sigma)) { PetscCall(VecScale(ldb->invDnew, sigma)); }
  /* Combine the old diagonal and the new diagonal using a convex limiter */
  if (ldb->rho == 1.0) {
    PetscCall(VecCopy(ldb->invDnew, invD));
  } else if (ldb->rho) PetscCall(VecAXPBY(invD, 1.0 - ldb->rho, ldb->rho, ldb->invDnew));
  PetscCall(MatLMVMRestoreJ0InvDiag(B, &invD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpdate_DiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    SymBroydenRescale ldb = (SymBroydenRescale)lmvm->ctx;
    PetscScalar       curvature;
    PetscReal         curvtol, ststmp;
    PetscInt          oldest, next;

    PetscCall(MatLMVMGetRange(B, &oldest, &next));
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));

    /* Test if the updates can be accepted */
    PetscCall(VecDotNorm2(lmvm->Fprev, lmvm->Xprev, &curvature, &ststmp));
    if (ststmp < lmvm->eps) curvtol = 0.0;
    else curvtol = lmvm->eps * ststmp;

    /* Test the curvature for the update */
    if (PetscRealPart(curvature) > curvtol) {
      /* Update is good so we accept it */
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      PetscCall(MatLMVMProductsInsertDiagonalValue(B, LMBASIS_Y, LMBASIS_S, next, PetscRealPart(curvature)));
      PetscCall(MatLMVMProductsInsertDiagonalValue(B, LMBASIS_S, LMBASIS_S, next, ststmp));
      PetscCall(SymBroydenRescaleUpdate(B, ldb));
    } else {
      /* reset */
      PetscCall(SymBroydenRescaleInitializeJ0(B, ldb));
    }
    /* End DiagBrdn update */
  }
  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleUpdate(Mat B, SymBroydenRescale ldb)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleSetUp(B, ldb));
  if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_NONE || ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_USER) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > ldb->k) {
    PetscInt new_oldest = PetscMax(0, next - ldb->sigma_hist);
    PetscInt ldb_oldest = PetscMax(0, ldb->k - ldb->sigma_hist);

    if (new_oldest > ldb_oldest) {
      for (PetscInt i = new_oldest; i < ldb->k; i++) {
        ldb->yty[i - new_oldest] = ldb->yty[i - ldb_oldest];
        ldb->yts[i - new_oldest] = ldb->yts[i - ldb_oldest];
        ldb->sts[i - new_oldest] = ldb->sts[i - ldb_oldest];
      }
    }
    for (PetscInt i = PetscMax(new_oldest, ldb->k); i < next; i++) {
      PetscScalar yty, sts, yts;

      PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_Y, LMBASIS_Y, i, &yty));
      PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, i, &yts));
      PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_S, LMBASIS_S, i, &sts));
      ldb->yty[i - new_oldest] = PetscRealPart(yty);
      ldb->yts[i - new_oldest] = PetscRealPart(yts);
      ldb->sts[i - new_oldest] = PetscRealPart(sts);
    }
    ldb->k = next;
  }
  PetscCall(SymBroydenRescaleUpdateJ0(B, ldb));
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

static PetscErrorCode MatCopy_DiagBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM         *bdata = (Mat_LMVM *)B->data;
  SymBroydenRescale bctx  = (SymBroydenRescale)bdata->ctx;
  Mat_LMVM         *mdata = (Mat_LMVM *)M->data;
  SymBroydenRescale mctx  = (SymBroydenRescale)mdata->ctx;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleCopy(bctx, mctx));
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

static PetscErrorCode MatView_DiagBrdn(Mat B, PetscViewer pv)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatView_LMVM(B, pv));
  PetscCall(SymBroydenRescaleView(ldb, pv));
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
  PetscCall(SymBroydenRescaleSetUp(B, ldb));
  switch (ldb->scale_type) {
  case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL: {
    Vec invD;

    PetscCall(MatLMVMGetJ0InvDiag(B, &invD));
    PetscCall(VecSet(invD, ldb->delta));
    PetscCall(MatLMVMRestoreJ0InvDiag(B, &invD));
    break;
  }
  case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
    PetscCall(MatLMVMSetJ0Scale(B, 1.0 / ldb->delta));
    break;
  default:
    break;
  }
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

static PetscErrorCode MatSetFromOptions_DiagBrdn(Mat B, PetscOptionItems PetscOptionsObject)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscCall(SymBroydenRescaleSetFromOptions(B, ldb, PetscOptionsObject));
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

static PetscErrorCode MatReset_DiagBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleReset(B, ldb, destructive));
  PetscCall(MatReset_LMVM(B, destructive));
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

static PetscErrorCode MatAllocate_DiagBrdn_Internal(Mat B)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;
  Vec               invD;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleAllocate(B, ldb));

  PetscCall(MatLMVMGetJ0InvDiag(B, &invD));
  PetscCall(VecSet(invD, ldb->delta));
  PetscCall(MatLMVMRestoreJ0InvDiag(B, &invD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAllocate_DiagBrdn(Mat B, Vec X, Vec F)
{
  PetscFunctionBegin;
  PetscCall(MatAllocate_LMVM(B, X, F));
  PetscCall(MatAllocate_DiagBrdn_Internal(B));
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

static PetscErrorCode MatDestroy_DiagBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleDestroy((SymBroydenRescale *)&lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_DiagBrdn(Mat B)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  PetscCall(MatAllocate_DiagBrdn_Internal(B));
  PetscCall(SymBroydenRescaleSetUp(B, ldb));
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

PetscErrorCode MatCreate_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM         *lmvm;
  SymBroydenRescale ldb;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDIAGBROYDEN));
  B->ops->setup          = MatSetUp_DiagBrdn;
  B->ops->setfromoptions = MatSetFromOptions_DiagBrdn;
  B->ops->destroy        = MatDestroy_DiagBrdn;
  B->ops->view           = MatView_DiagBrdn;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->square        = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_DiagBrdn;
  lmvm->ops->reset    = MatReset_DiagBrdn;
  lmvm->ops->mult     = MatMult_DiagBrdn;
  lmvm->ops->solve    = MatSolve_DiagBrdn;
  lmvm->ops->update   = MatUpdate_DiagBrdn;
  lmvm->ops->copy     = MatCopy_DiagBrdn;

  PetscCall(MatLMVMSetHistorySize(B, 1));

  PetscCall(SymBroydenRescaleCreate(&ldb));
  lmvm->ctx = (void *)ldb;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMDiagBroyden - DiagBrdn creates a symmetric Broyden-type diagonal matrix used
  for approximating Hessians.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Options Database Keys:
+ -mat_lmvm_theta      - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
. -mat_lmvm_rho        - (developer) update limiter for the J0 scaling
. -mat_lmvm_alpha      - (developer) coefficient factor for the quadratic subproblem in J0 scaling
. -mat_lmvm_beta       - (developer) exponential factor for the diagonal J0 scaling
. -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling.
. -mat_lmvm_tol        - (developer) tolerance for bounding the denominator of the rescaling away from 0.
- -mat_lmvm_forward    - (developer) whether or not to use the forward or backward Broyden update to the diagonal

  Level: intermediate

  Notes:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

  It consists of a convex combination of DFP and BFGS
  diagonal approximation schemes, such that DiagBrdn = (1-theta)*BFGS + theta*DFP.
  To preserve symmetric positive-definiteness, we restrict theta to be in [0, 1].
  We also ensure positive definiteness by taking the `VecAbs()` of the final vector.

  There are two ways of approximating the diagonal: using the forward (B) update
  schemes for BFGS and DFP and then taking the inverse, or directly working with
  the inverse (H) update schemes for the BFGS and DFP updates, derived using the
  Sherman-Morrison-Woodbury formula. We have implemented both, controlled by a
  parameter below.

  In order to use the DiagBrdn matrix with other vector types, i.e. doing matrix-vector products
  and matrix solves, the matrix must first be created using `MatCreate()` and `MatSetType()`,
  followed by `MatLMVMAllocate()`. Then it will be available for updating
  (via `MatLMVMUpdate()`) in one's favored solver implementation.

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMDIAGBRDN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBroyden()`, `MatCreateLMVMSymBroyden()`
@*/
PetscErrorCode MatCreateLMVMDiagBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
