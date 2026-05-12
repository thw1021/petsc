/*
Context for Bounded Regularized Gauss-Newton algorithm.

TAOBRGN is a thin wrapper around TAOBNLS that composes two TaoTerms on the
subsolver via TaoAddTerm():

  subterm 0: TAOTERMGAUSSNEWTON over the parent Tao  (scale 1.0)
             -- contributes (1/2) ||R(x)||^2 with Gauss-Newton Hessian J^T J
  subterm 1: regularizer  (scale = lambda)
             -- one of L2PURE, L2PROX, L1DICT (with optional dictionary D),
                LM (Levenberg-Marquardt diag(J^T J) damping), or a user term

Backward-compat callback setters (TaoBRGNSetRegularizerObjectiveAndGradientRoutine,
TaoBRGNSetRegularizerHessianRoutine) are adapted by building a TAOTERMSHELL
that wraps the user's callbacks.
*/

#pragma once

#include <petsc/private/taoimpl.h>
#include <petsctaoterm.h>

typedef struct {
  Mat       D;       /* L1DICT dictionary (optional, may be NULL) */
  Vec       damping; /* stable Vec returned by TaoBRGNGetDampingVector() */
  Tao       subsolver, parent;
  PetscReal lambda;  /* current regularizer weight (cached so we can rebuild subterms) */
  PetscReal epsilon; /* L1 smoothing parameter (cached so it survives type swaps) */
  PetscReal fc_old;  /* for LM uphill/downhill detection */
  PetscReal downhill_lambda_change, uphill_lambda_change;

  TaoBRGNRegularizationType reg_type;

  /* USER regularizer state (set by legacy callback APIs); used to build a TAOTERMSHELL */
  PetscErrorCode (*user_objgrad)(Tao, Vec, PetscReal *, Vec, PetscCtx);
  PetscCtx user_objgrad_ctx;
  PetscErrorCode (*user_hessian)(Tao, Vec, Mat, PetscCtx);
  PetscCtx user_hessian_ctx;
  Mat      user_hessian_mat; /* matrix the user passed to TaoBRGNSetRegularizerHessianRoutine() */
} TAO_BRGN;

PETSC_INTERN PetscErrorCode TaoTermCreateBRGNLMDamping(TaoTerm gn, TaoTerm *lm);
PETSC_INTERN PetscErrorCode TaoTermBRGNLMDampingCopyDiagonal(TaoTerm lm, Vec dst);
