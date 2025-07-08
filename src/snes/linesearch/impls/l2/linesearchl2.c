#include <petsc/private/linesearchimpl.h>
#include <petscsnes.h>
#include <petsc/private/snesimpl.h>
#include <petsc/private/viewerimpl.h>

static PetscErrorCode SNESLineSearchMonitor_L2(SNESLineSearch linesearch, PetscInt it, PetscReal fnorm, PetscReal lambda)
{
  SNES snes;
  SNESObjectiveFn *objective;
  PetscViewer monitor = linesearch->monitor;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  if (monitor) {
    const char **data = NULL;
    char         f_fmt[PETSC_MONITOR_REAL_LENGTH];
    char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

    PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
    PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
    if (!objective) {
      PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, fnorm, PETSC_REAL_FMT_SHORT, f_fmt));
      PetscCall(PetscViewerASCIIPrintf(monitor, "SNESLineSearch " PetscColorFmt("%3" PetscInt_FMT) " function norm " PetscColorFmt("%s") " lambda " PetscColorFmt("%s") "\n", PetscColorArg(data, it), PetscColorArg(data, f_fmt), PetscColorArg(data, lambda_fmt)));
    } else {
      PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, fnorm, PETSC_REAL_FMT_SHORT | PETSC_REAL_FMT_SIGNED, f_fmt));
      PetscCall(PetscViewerASCIIPrintf(monitor, "SNESLineSearch " PetscColorFmt("%3" PetscInt_FMT) " objective value " PetscColorFmt("%s") " lambda " PetscColorFmt("%s") "\n", PetscColorArg(data, it), PetscColorArg(data, f_fmt), PetscColorArg(data, lambda_fmt)));
    }
    PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode SNESLineSearchComputeObjective_L2(SNESLineSearch linesearch, PetscInt it, PetscBool monitor, PetscReal lambda, PetscReal gnorm, PetscReal *fnorm, PetscBool *keep_going)
{
  SNES snes;
  Vec         X, F, Y, W, G;
  PetscViewer converged_monitor = linesearch->converged_monitor;
  SNESObjectiveFn *objective;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    if (converged_monitor) {
      const char **cdata;
      const char **cwarning;

      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "function count"), PetscColorArg(cdata, it)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (lambda == 0.0) {
    if (!objective) {
      *fnorm = gnorm * gnorm;
    } else {
      PetscCall(SNESComputeObjective(snes, X, fnorm));
    }
  } else {
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (!objective) {
      PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
      if (linesearch->ops->vinorm) {
        *fnorm = gnorm;
        PetscCall((*linesearch->ops->vinorm)(snes, F, W, fnorm));
      } else {
        PetscCall(VecNorm(F, NORM_2, fnorm));
      }
      *fnorm = PetscSqr(*fnorm);
    } else {
      PetscCall(SNESComputeObjective(snes, W, fnorm));
    }
  }
  if (monitor) PetscCall(SNESLineSearchMonitor_L2(linesearch, it, *fnorm, lambda));
  if (PetscIsInfOrNanReal(*fnorm)) {
    if (converged_monitor) {
      const char **cdata;
      const char **cwarning;

      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      if (PetscIsNanScalar(*fnorm)) PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because dot(F,Y) is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "NaN"), PetscColorArg(cdata, it)));
      else PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because dot(F,Y) is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "Inf"), PetscColorArg(cdata, it)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *keep_going = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchApply_L2(SNESLineSearch linesearch)
{
  PetscBool        changed_y, changed_w;
  Vec              X;
  Vec              F;
  Vec              Y;
  Vec              W;
  SNES             snes;
  PetscReal        gnorm;
  PetscReal        ynorm;
  PetscReal        xnorm;
  PetscReal        steptol, maxstep, rtol, atol, ltol;
  PetscViewer      monitor;
  PetscReal        lambda, lambda_old, lambda_mid, lambda_update, delLambda;
  PetscReal        fnrm, fnrm_old, fnrm_mid;
  PetscReal        delFnrm, delFnrm_old, del2Fnrm;
  PetscInt         it, max_its;
  SNESObjectiveFn *objective;
  PetscBool        keep_going;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL));
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its));
  PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));

  PetscCall(SNESGetObjective(snes, &objective, NULL));

  /* precheck */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));
  it = 0;
  PetscCall(SNESLineSearchComputeObjective_L2(linesearch, it, PETSC_TRUE, 0.0, gnorm, &fnrm_old, &keep_going));
  if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);

  lambda_old = 0.0;
  lambda_mid = 0.5 * (lambda + lambda_old);
  for (it = 1; it <= max_its; it++) {
    while (PETSC_TRUE) {
      PetscCall(SNESLineSearchComputeObjective_L2(linesearch, it, PETSC_FALSE, lambda_mid, gnorm, &fnrm_mid, &keep_going));
      PetscCall(SNESLineSearchComputeObjective_L2(linesearch, it, PETSC_FALSE, lambda, gnorm, &fnrm, &keep_going));
      if (!PetscIsInfOrNanReal(fnrm)) break;
      PetscCall(PetscInfo(linesearch, "objective function at lambdas = %g is Inf or Nan, cutting lambda\n", (double)lambda));
      if (lambda <= steptol) {
        PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it, lambda, SNES_LINESEARCH_FAILED_REDUCT, "minimum step length"));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      maxstep    = .95 * lambda; /* forbid the search from ever going back to the "failed" length that generates Nan or Inf */
      lambda     = .5 * (lambda + lambda_old);
      lambda_mid = .5 * (lambda + lambda_old);
    }
    PetscCall(SNESLineSearchMonitor_L2(linesearch, it, fnrm, lambda));

    delLambda = lambda - lambda_old;
    /* compute f'() at the end points using second order one sided differencing */
    delFnrm     = (3. * fnrm - 4. * fnrm_mid + 1. * fnrm_old) / delLambda;
    delFnrm_old = (-3. * fnrm_old + 4. * fnrm_mid - 1. * fnrm) / delLambda;
    /* compute f''() at the midpoint using centered differencing */
    del2Fnrm = (delFnrm - delFnrm_old) / delLambda;

    if (!objective) {
      PetscCall(PetscInfo(linesearch, "lambdas = [%g, %g, %g], fnorms = [%g, %g, %g]\n", (double)lambda, (double)lambda_mid, (double)lambda_old, (double)PetscSqrtReal(fnrm), (double)PetscSqrtReal(fnrm_mid), (double)PetscSqrtReal(fnrm_old)));
    } else {
      PetscCall(PetscInfo(linesearch, "lambdas = [%g, %g, %g], obj = [%g, %g, %g]\n", (double)lambda, (double)lambda_mid, (double)lambda_old, (double)fnrm, (double)fnrm_mid, (double)fnrm_old));
    }

    /* compute the secant (Newton) update -- always go downhill */
    if (del2Fnrm > 0.) lambda_update = lambda - delFnrm / del2Fnrm;
    else if (del2Fnrm < 0.) lambda_update = lambda + delFnrm / del2Fnrm;
    else {
      PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it, lambda, SNES_LINESEARCH_SUCCEEDED, "critical point breakdown"));
      break;
    }

    if (lambda_update < steptol) lambda_update = 0.5 * (lambda + lambda_old);

    if (PetscIsInfOrNanReal(lambda_update)) {
      PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it, lambda, SNES_LINESEARCH_SUCCEEDED, "critical point breakdown"));
      goto postcheck;
    }

    if (lambda_update > maxstep) {
      PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it, lambda, SNES_LINESEARCH_SUCCEEDED, "maximum step length"));
      goto postcheck;
    }

    /* update the endpoints and the midpoint of the bracketed secant region */
    lambda_old = lambda;
    lambda     = lambda_update;
    fnrm_old   = fnrm;
    lambda_mid = 0.5 * (lambda + lambda_old);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHL2 - Secant search in the L2 norm of the function or the objective function, if it is provided with `SNESSetObjective()`.

   Attempts to solve $ \min_{\lambda} f(x + \lambda y) $ using the secant method with the initial bracketing of $ \lambda $ between [0,damping].
   Differences of $f()$ are used to approximate the first and second derivative of $f()$ with respect to
   $\lambda$, $f'()$ and $f''()$. The secant method is run for `maxit` iterations.

   When an objective function is provided $f(w)$ is the objective function otherwise $f(w) = ||F(w)||^2$.
   $x$ is the current step and $y$ is the search direction.

   This has no checks on whether the secant method is actually converging.

   Options Database Keys:
+  -snes_linesearch_max_it <maxit>        - maximum number of iterations, default is 1
.  -snes_linesearch_maxstep <length>      - the algorithm insures that a step length is never longer than this value
.  -snes_linesearch_damping <damping>     - initial step is scaled back by this factor, default is 1.0
-  -snes_linesearch_minlambda <minlambda> - minimum allowable lambda

   Level: advanced

   Developer Note:
   A better name for this method might be `SNESLINESEARCHSECANT`, L2 is not descriptive

.seealso: [](ch_snes), `SNESLINESEARCHBT`, `SNESLINESEARCHCP`, `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_L2(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_L2;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;

  linesearch->max_its = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
