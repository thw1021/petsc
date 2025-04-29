#include <petsc/private/linesearchimpl.h>
#include <petscsnes.h>
#include <petsc/private/viewerimpl.h>
#include <petsc/private/snesimpl.h>

static PetscErrorCode SNESLineSearchApply_CP(SNESLineSearch linesearch)
{
  PetscBool   changed_y, changed_w;
  Vec         X, Y, F, W;
  SNES        snes;
  PetscReal   xnorm, ynorm, gnorm, steptol, atol, rtol, ltol, maxstep;
  PetscReal   lambda, lambda_old, lambda_update, delLambda;
  PetscScalar fty, fty_init, fty_old, fty_mid1, fty_mid2, s;
  PetscInt    it, max_its;
  PetscViewer converged_monitor;
  const char **cdata = NULL;
  const char **csuccess = NULL;
  const char **cwarning = NULL;
  PetscBool    keep_going;
  SNESLineSearchReason reason;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its));
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));
  converged_monitor = linesearch->converged_monitor;
  if (converged_monitor) {
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_SUCCESS, &csuccess));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
  }

  /* precheck */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));

  it = 0;
  PetscCall(SNESLineSearchComputeObjectiveDefault_Internal(linesearch, it, PETSC_TRUE, ynorm, 0.0, &fty_init, &keep_going));
  if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);
  fty = fty_old = fty_init;

  // at it == 0, this should just check atol convergence
  PetscCall(SNESLineSearchCheckConvergenceDefault_Internal(linesearch, it, ynorm, fty, 0.0, 0.0, fty_init, &keep_going));
  if (!keep_going) {
    PetscCall(SNESLineSearchGetReason(linesearch, &reason));
    if (reason == SNES_LINESEARCH_SUCCEEDED) goto postcheck;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  lambda_old = 0.0;
  for (it = 1; it <= max_its; it++) {
    PetscCall(SNESLineSearchComputeObjectiveDefault_Internal(linesearch, it, PETSC_TRUE, ynorm, lambda, &fty, &keep_going));
    if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(SNESLineSearchCheckConvergenceDefault_Internal(linesearch, it, ynorm, fty, lambda, lambda_old, fty_init, &keep_going));
    if (!keep_going) {

      PetscCall(SNESLineSearchGetReason(linesearch, &reason));
      if (reason == SNES_LINESEARCH_SUCCEEDED) goto postcheck;
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    /* compute the search direction */
    delLambda = lambda - lambda_old;
    if (linesearch->order == SNES_LINESEARCH_ORDER_LINEAR) {
      s = (fty - fty_old) / delLambda;
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      PetscCall(SNESLineSearchComputeObjectiveDefault_Internal(linesearch, it, PETSC_FALSE, ynorm, 0.5 * (lambda + lambda_old), &fty_mid1, &keep_going));
      if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);
      s = (3. * fty - 4. * fty_mid1 + fty_old) / delLambda;
    } else {
      PetscCall(SNESLineSearchComputeObjectiveDefault_Internal(linesearch, it, PETSC_FALSE, ynorm, 0.5 * (lambda + lambda_old), &fty_mid1, &keep_going));
      if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);
      PetscCall(SNESLineSearchComputeObjectiveDefault_Internal(linesearch, it, PETSC_FALSE, ynorm, (lambda + 0.5 * (lambda - lambda_old)), &fty_mid2, &keep_going));
      if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);
      s = (2. * fty_mid2 + 3. * fty - 6. * fty_mid1 + fty_old) / (3. * delLambda);
    }
    /* if the solve is going in the wrong direction, fix it */
    if (PetscRealPart(s) > 0.) s = -s;
    if (s == 0.0) {
      PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it, lambda, SNES_LINESEARCH_SUCCEEDED, "critical point breakdown"));
      goto postcheck;
    }
    lambda_update = lambda - PetscRealPart(fty / s);

    /* switch directions if we stepped out of bounds */
    if (lambda_update < steptol) lambda_update = lambda + PetscRealPart(fty / s);

    if (PetscIsInfOrNanReal(lambda_update)) {
      PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it, lambda, SNES_LINESEARCH_SUCCEEDED, "critical point breakdown"));
      goto postcheck;
    }
    if (lambda_update > maxstep) {
      PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it, lambda, SNES_LINESEARCH_SUCCEEDED, "maximum step length"));
      goto postcheck;
    }

    /* compute the new state of the line search */
    lambda_old = lambda;
    lambda     = lambda_update;
    fty_old    = fty;
  }

  if (lambda <= steptol) {
    PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it - 1, lambda, SNES_LINESEARCH_FAILED_REDUCT, "minimum step length"));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it - 1, lambda, SNES_LINESEARCH_SUCCEEDED, "maximum iterations"));
  }

postcheck:
  /* construct the solution */
  PetscCall(VecWAXPY(W, -lambda, Y, X));
  if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  /* postcheck */
  PetscCall(SNESLineSearchSetLambda(linesearch, lambda));
  PetscCall(SNESLineSearchPostCheck(linesearch, X, Y, W, &changed_y, &changed_w));
  if (changed_y) {
    if (!changed_w) PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  }
  PetscCall(VecCopy(W, X));
  PetscCall((*linesearch->ops->snesfunc)(snes, X, F));

  PetscCall(SNESLineSearchComputeNorms(linesearch));
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHCP - Critical point line search. This line search assumes that there exists some
   artificial $G(x)$ for which the `SNESFunctionFn` $ F(x) = grad G(x)$.  Therefore, this line search seeks
   to find roots of $ F^T Y$ via a secant method.

   Options Database Keys:
+  -snes_linesearch_minlambda <minlambda> - the minimum acceptable lambda
.  -snes_linesearch_maxstep <length>      - the algorithm insures that a step length is never longer than this value
.  -snes_linesearch_damping <damping>     - initial trial step length is scaled by this factor on entry to the line search, default is 1.0
-  -snes_linesearch_max_it <max_it>       - the maximum number of secant steps performed.

   Level: advanced

   Notes:
   This method does NOT use the objective function if it is provided with `SNESSetObjective()`.

   This method is the preferred line search for `SNESQN` and `SNESNCG`.

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`, `SNESLINESEARCHBISECTION`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_CP(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_CP;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;
  linesearch->order               = SNES_LINESEARCH_ORDER_LINEAR;

  linesearch->max_its = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
