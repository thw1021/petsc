/*
   Implementation of IDR(s) (Induced Dimension Reduction), an iterative method for
   nonsymmetric linear systems Ax = b. Follows Algorithm 2 of {cite}`vangijzensonneveld2011`.
*/
#include <../src/ksp/ksp/impls/idr/idrimpl.h> /*I "petscksp.h" I*/

static PetscErrorCode KSPIDRInitShadowSpace_IDR(KSP ksp)
{
  KSP_IDR    *idr = (KSP_IDR *)ksp->data;
  PetscRandom rnd;
  PetscInt    k, j;
  PetscScalar alpha;

  PetscFunctionBegin;
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)ksp), &rnd));
  PetscCall(PetscRandomSetSeed(rnd, 0x12345678ULL));
  PetscCall(PetscRandomSeed(rnd));
  for (k = 0; k < idr->s; k++) PetscCall(VecSetRandom(idr->PP[k], rnd));
  PetscCall(PetscRandomDestroy(&rnd));
  /* Modified Gram-Schmidt orthonormalisation of the shadow space */
  for (k = 0; k < idr->s; k++) {
    PetscCall(VecNormalize(idr->PP[k], NULL));
    for (j = k + 1; j < idr->s; j++) {
      PetscCall(VecDot(idr->PP[j], idr->PP[k], &alpha));
      PetscCall(VecAXPY(idr->PP[j], -alpha, idr->PP[k]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPSetUp_IDR(KSP ksp)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;
  Mat      A;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatCreateVecs(A, &idr->r, NULL));
  PetscCall(VecDuplicateVecs(idr->r, idr->s, &idr->GG));
  PetscCall(VecDuplicateVecs(idr->r, idr->s, &idr->UU));
  PetscCall(VecDuplicateVecs(idr->r, idr->s, &idr->PP));
  PetscCall(VecDuplicate(idr->r, &idr->v));
  PetscCall(VecDuplicate(idr->r, &idr->t));
  PetscCall(PetscMalloc3(idr->s * idr->s, &idr->M, idr->s, &idr->f, idr->s, &idr->c));
  PetscCall(KSPIDRInitShadowSpace_IDR(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPSolve_IDR(KSP ksp)
{
  KSP_IDR     *idr = (KSP_IDR *)ksp->data;
  Mat          A;
  Vec          X, B, R, V, T;
  Vec         *GG, *UU, *PP;
  PetscScalar *M, *f, *c;
  PetscScalar  alpha, beta, omg, tdr, pdot;
  PetscReal    tdt, rho_thresh, rnrm, tnrm, dp = 0.0;
  PetscInt     s, k, j, i, it, nck;
  PetscBool    pc_is_left;

  PetscFunctionBegin;
  s          = idr->s;
  X          = ksp->vec_sol;
  B          = ksp->vec_rhs;
  R          = idr->r;
  V          = idr->v;
  T          = idr->t;
  GG         = idr->GG;
  UU         = idr->UU;
  PP         = idr->PP;
  M          = idr->M;
  f          = idr->f;
  c          = idr->c;
  pc_is_left = (PetscBool)(ksp->pc_side == PC_LEFT);
  PetscCall(KSPGetOperators(ksp, &A, NULL));

  /* KSPInitialResidual returns: K^{-1}(B - A X) for PC_LEFT,  B - A X otherwise. */
  PetscCall(KSPInitialResidual(ksp, X, V, T, R, B));

  if (ksp->normtype != KSP_NORM_NONE) {
    PetscCall(VecNorm(R, NORM_2, &dp));
    KSPCheckNorm(ksp, dp);
  }
  ksp->its   = 0;
  ksp->rnorm = dp;
  PetscCall(KSPLogResidualHistory(ksp, dp));
  PetscCall(KSPMonitor(ksp, 0, dp));
  PetscCall((*ksp->converged)(ksp, 0, dp, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* Initialise U = G = 0; M = I; omega = 1. */
  for (k = 0; k < s; k++) {
    PetscCall(VecSet(UU[k], 0.0));
    PetscCall(VecSet(GG[k], 0.0));
  }
  PetscCall(PetscArrayzero(M, s * s));
  for (k = 0; k < s; k++) M[k * s + k] = 1.0;
  omg        = 1.0;
  rho_thresh = PetscSqrtReal(2.0) / 2.0;
  it         = 0;

  do {
    /* f = P^T R   (one allreduce, s values) */
    PetscCall(VecMDot(R, s, PP, f));

    for (k = 0; k < s; k++) {
      nck = s - k;

      /* Solve trailing lower-triangular system M[k:s, k:s] c[0:nck-1] = f[k:s].
         M is column-major: M[row=i, col=j] = M[j*s + i]; the (i,j) entry of the trailing block
         is M[(k+j)*s + (k+i)]. */
      for (j = 0; j < nck; j++) {
        PetscScalar sum = f[k + j];
        for (i = 0; i < j; i++) sum -= M[(k + i) * s + (k + j)] * c[i];
        c[j] = sum / M[(k + j) * s + (k + j)];
      }

      /* V = sum_{i=0..nck-1} c[i] * GG[k+i]; then V = R - V. */
      PetscCall(VecSet(V, 0.0));
      PetscCall(VecMAXPY(V, nck, c, &GG[k]));
      PetscCall(VecAYPX(V, -1.0, R)); /* V = R - V */

      /* Q = sum_{i=0..nck-1} c[i] * UU[k+i]; store in GG[k] (will be overwritten with A*UU[k]). */
      PetscCall(VecSet(GG[k], 0.0));
      PetscCall(VecMAXPY(GG[k], nck, c, &UU[k]));

      if (pc_is_left) {
        /* PC_LEFT: R is the preconditioned residual; V is already an x-space direction. */
        PetscCall(VecCopy(GG[k], UU[k]));            /* UU[k] = Q */
        PetscCall(VecAXPY(UU[k], omg, V));           /* UU[k] = Q + omega * V */
        PetscCall(KSP_PCApplyBAorAB(ksp, UU[k], GG[k], V)); /* GG[k] = K^{-1} A UU[k]; V is scratch */
      } else {
        /* PC_RIGHT or PC_NONE: turn V into x-space via K^{-1}, then A. */
        PetscCall(KSP_PCApply(ksp, V, T));           /* T = K^{-1} V */
        PetscCall(VecCopy(GG[k], UU[k]));            /* UU[k] = Q */
        PetscCall(VecAXPY(UU[k], omg, T));           /* UU[k] = Q + omega * (K^{-1} V) */
        PetscCall(KSP_MatMult(ksp, A, UU[k], GG[k]));/* GG[k] = A UU[k] */
      }

      /* Bi-orthogonalise GG[k] and UU[k] against PP[0..k-1] (modified Gram-Schmidt). */
      for (i = 0; i < k; i++) {
        PetscCall(VecDot(GG[k], PP[i], &pdot)); /* pdot = PP[i]^H GG[k] = <P[i], G[k]> */
        alpha = pdot / M[i * s + i];
        PetscCall(VecAXPY(GG[k], -alpha, GG[i]));
        PetscCall(VecAXPY(UU[k], -alpha, UU[i]));
      }

      /* New column-k entries of M for rows k..s-1: M[i, k] = <P[i], G[k]>.
         Column-major: stored at &M[k*s + k], &M[k*s + k+1], ..., &M[k*s + s-1]. */
      PetscCall(VecMDot(GG[k], nck, &PP[k], &M[k * s + k]));

      /* Breakdown check: M[k, k] should be nonzero. */
      if (PetscAbsScalar(M[k * s + k]) == 0.0) {
        PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve breakdown: zero diagonal of M in IDR(s)");
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        PetscCall(PetscInfo(ksp, "Breakdown: M[%" PetscInt_FMT ",%" PetscInt_FMT "] = 0 in IDR(s)\n", k, k));
        PetscFunctionReturn(PETSC_SUCCESS);
      }

      beta = f[k] / M[k * s + k];

      /* R -= beta * G[k];  X += beta * U[k] */
      PetscCall(VecAXPY(R, -beta, GG[k]));
      PetscCall(VecAXPY(X, beta, UU[k]));

      /* f[k+1..s-1] -= beta * M[k+1..s-1, k]   (column-major: &M[k*s + k+1]) */
      for (j = k + 1; j < s; j++) f[j] -= beta * M[k * s + j];

      it++;
      if (ksp->normtype != KSP_NORM_NONE) {
        PetscCall(VecNorm(R, NORM_2, &dp));
        KSPCheckNorm(ksp, dp);
      }
      ksp->its   = it;
      ksp->rnorm = dp;
      PetscCall(KSPLogResidualHistory(ksp, dp));
      PetscCall(KSPMonitor(ksp, it, dp));
      PetscCall((*ksp->converged)(ksp, it, dp, &ksp->reason, ksp->cnvP));
      if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);
      if (it >= ksp->max_it) {
        ksp->reason = KSP_DIVERGED_ITS;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }

    /* Step s+1: omega update. R is orthogonal to PP[0..s-1] by construction. */
    if (pc_is_left) {
      PetscCall(KSP_PCApplyBAorAB(ksp, R, T, V)); /* T = K^{-1} A R; V is scratch */
    } else {
      PetscCall(KSP_PCApply(ksp, R, V));     /* V = K^{-1} R */
      PetscCall(KSP_MatMult(ksp, A, V, T));  /* T = A V */
    }
    PetscCall(VecDotNorm2(R, T, &tdr, &tdt)); /* tdr = T^H R; tdt = ||T||^2 */
    if (tdt == 0.0) {
      ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* Anti-stagnation omega: boost when T and R are nearly orthogonal. */
    omg = tdr / tdt;
    PetscCall(VecNorm(R, NORM_2, &rnrm));
    KSPCheckNorm(ksp, rnrm);
    tnrm = PetscSqrtReal(tdt);
    if (rnrm > 0.0 && tnrm > 0.0) {
      PetscReal rho = PetscAbsScalar(tdr) / (tnrm * rnrm);
      if (rho < rho_thresh) omg *= rho_thresh / rho;
    }
    idr->omega = PetscRealPart(omg);
    if (pc_is_left) {
      PetscCall(VecAXPY(X, omg, R));   /* X += omega * R (R is the preconditioned residual = x-space dir) */
    } else {
      PetscCall(VecAXPY(X, omg, V));   /* X += omega * V (V = K^{-1} R) */
    }
    PetscCall(VecAXPY(R, -omg, T));    /* R -= omega * T */

    it++;
    if (ksp->normtype != KSP_NORM_NONE) {
      PetscCall(VecNorm(R, NORM_2, &dp));
      KSPCheckNorm(ksp, dp);
    }
    ksp->its   = it;
    ksp->rnorm = dp;
    PetscCall(KSPLogResidualHistory(ksp, dp));
    PetscCall(KSPMonitor(ksp, it, dp));
    PetscCall((*ksp->converged)(ksp, it, dp, &ksp->reason, ksp->cnvP));
    if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);
  } while (it < ksp->max_it);

  if (it >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPReset_IDR(KSP ksp)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&idr->r));
  PetscCall(VecDestroy(&idr->v));
  PetscCall(VecDestroy(&idr->t));
  PetscCall(VecDestroyVecs(idr->s, &idr->GG));
  PetscCall(VecDestroyVecs(idr->s, &idr->UU));
  PetscCall(VecDestroyVecs(idr->s, &idr->PP));
  PetscCall(PetscFree3(idr->M, idr->f, idr->c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPDestroy_IDR(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPReset_IDR(ksp));
  PetscCall(KSPDestroyDefault(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRSetS_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRGetS_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPView_IDR(KSP ksp, PetscViewer viewer)
{
  KSP_IDR  *idr = (KSP_IDR *)ksp->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  s (shadow space dimension) = %" PetscInt_FMT "\n", idr->s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPSetFromOptions_IDR(KSP ksp, PetscOptionItems PetscOptionsObject)
{
  KSP_IDR  *idr = (KSP_IDR *)ksp->data;
  PetscInt  s;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP IDR options");
  PetscCall(PetscOptionsInt("-ksp_idr_s", "Shadow space dimension", "KSPIDRSetS", idr->s, &s, &flg));
  if (flg) PetscCall(KSPIDRSetS(ksp, s));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPIDRSetS_IDR(KSP ksp, PetscInt s)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(s >= 1, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Shadow space dimension s must be >= 1, got %" PetscInt_FMT, s);
  if (!ksp->setupstage) {
    idr->s = s;
  } else if (idr->s != s) {
    PetscCall(KSPReset_IDR(ksp));
    idr->s          = s;
    ksp->setupstage = KSP_SETUP_NEW;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPIDRGetS_IDR(KSP ksp, PetscInt *s)
{
  KSP_IDR *idr = (KSP_IDR *)ksp->data;

  PetscFunctionBegin;
  *s = idr->s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPIDRSetS - Sets the shadow space dimension s of the `KSPIDR` (IDR(s)) Krylov solver.

  Logically Collective

  Input Parameters:
+ ksp - iterative context of type `KSPIDR`
- s   - shadow space dimension; must be `>= 1`

  Options Database Key:
. -ksp_idr_s s - shadow space dimension (default 4)

  Level: intermediate

  Notes:
  Larger s improves convergence at the cost of s additional vectors and s extra inner
  products per cycle. `s = 1` is mathematically equivalent to `KSPBCGS` (BiCGSTAB);
  `s = 4` typically converges as fast as `KSPGMRES`(50). The default is 4.

.seealso: [](ch_ksp), `KSP`, `KSPIDR`, `KSPIDRGetS()`
@*/
PetscErrorCode KSPIDRSetS(KSP ksp, PetscInt s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, s, 2);
  PetscTryMethod(ksp, "KSPIDRSetS_C", (KSP, PetscInt), (ksp, s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPIDRGetS - Gets the shadow space dimension s of the `KSPIDR` (IDR(s)) Krylov solver.

  Not Collective

  Input Parameter:
. ksp - iterative context of type `KSPIDR`

  Output Parameter:
. s - shadow space dimension

  Level: intermediate

.seealso: [](ch_ksp), `KSP`, `KSPIDR`, `KSPIDRSetS()`
@*/
PetscErrorCode KSPIDRGetS(KSP ksp, PetscInt *s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(s, 2);
  PetscUseMethod(ksp, "KSPIDRGetS_C", (KSP, PetscInt *), (ksp, s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   KSPIDR - Implements the IDR(s) (Induced Dimension Reduction) Krylov method for general
   nonsymmetric linear systems {cite}`vangijzensonneveld2011`.

   Options Database Key:
.  -ksp_idr_s s - shadow space dimension (default 4); larger s improves convergence at the
   cost of s additional vectors and s extra inner products per cycle, see `KSPIDRSetS()`

   Level: intermediate

   Notes:
   IDR(s) is a short-recurrence, non-restarting Krylov method for general nonsymmetric
   linear systems. It requires no growing subspace and avoids the restart stagnation of
   `KSPGMRES`. The parameter s controls the trade-off between memory and convergence
   speed: `s = 1` is mathematically equivalent to `KSPBCGS` (BiCGSTAB); `s = 4` typically
   converges as fast as `KSPGMRES`(50); `s = 8` often outperforms `KSPGMRES`(100).

   Memory usage is (3s + 3) vectors plus an s-by-s dense matrix.

.seealso: [](ch_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPBCGS`,
          `KSPBCGSL`, `KSPGMRES`, `KSPIDRSetS()`, `KSPIDRGetS()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_IDR(KSP ksp)
{
  KSP_IDR *idr;

  PetscFunctionBegin;
  PetscCall(PetscNew(&idr));
  idr->s     = 4;
  idr->omega = 1.0;
  ksp->data  = (void *)idr;

  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));

  ksp->ops->setup          = KSPSetUp_IDR;
  ksp->ops->solve          = KSPSolve_IDR;
  ksp->ops->reset          = KSPReset_IDR;
  ksp->ops->destroy        = KSPDestroy_IDR;
  ksp->ops->view           = KSPView_IDR;
  ksp->ops->setfromoptions = KSPSetFromOptions_IDR;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRSetS_C", KSPIDRSetS_IDR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPIDRGetS_C", KSPIDRGetS_IDR));
  PetscFunctionReturn(PETSC_SUCCESS);
}
