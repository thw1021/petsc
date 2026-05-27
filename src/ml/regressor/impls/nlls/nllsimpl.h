#pragma once
#include <petsc/private/regressorimpl.h>
#include <petsctao.h>

typedef struct {
  /* User-provided model and (optional) Jacobian callbacks */
  PetscRegressorNLLSFunctionFn *modelfn;
  void                         *modelctx;
  Vec                           f_template; /* template for model output f(X, p); layout matches y; passed to Tao as its residual vector */

  PetscRegressorNLLSJacobianFn *jacfn;
  void                         *jacctx;
  Mat                           J, Jpre; /* Jacobian (M x N) */

  /* Solver state */
  Vec parameters;  /* fitted parameters (length N); Tao's solution vector */
  Vec parameters0; /* user-provided initial guess */

  /* Data matrix in play for the current Fit() or Predict() call; the Tao
     residual/Jacobian wrappers hand it to the user callback. */
  Mat current_X;
} PetscRegressor_NLLS;
