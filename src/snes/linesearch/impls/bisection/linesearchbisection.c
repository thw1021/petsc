#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>
#include <petsc/private/viewerimpl.h>

static PetscErrorCode SNESLineSearchMonitor_Bisection(SNESLineSearch linesearch, PetscInt it, PetscScalar fty, PetscReal ynorm, PetscReal lambda)
{
  PetscViewer monitor = linesearch->monitor;
  PetscFunctionBegin;
  if (monitor) {
    const char **data = NULL;
    char         ratio_fmt[PETSC_MONITOR_REAL_LENGTH];
    char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

    PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
    PetscCall(PetscViewerASCIIGetColor(monitor, PETSC_COLOR_DATA, &data));
    PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, PetscRealPart(fty)/ ynorm, PETSC_REAL_FMT_SHORT | PETSC_REAL_FMT_SIGNED, ratio_fmt));
    PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
    PetscCall(PetscViewerASCIIPrintf(monitor, "SNESLineSearch " PetscColorFmt("%3" PetscInt_FMT) " dot(f,y)/||y|| " PetscColorFmt("%s") ", lambda " PetscColorFmt("%s") "\n", PetscColorArg(data, it), PetscColorArg(data, ratio_fmt), PetscColorArg(data, lambda_fmt)));
    PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESLineSearchCheckConvergenceDefault_Internal(SNESLineSearch linesearch, PetscInt it, PetscScalar ynorm, PetscScalar fty, PetscScalar lambda, PetscScalar fty_old, PetscScalar lambda_old, PetscScalar fty_initial, PetscBool *keep_going)
{
  const char **cdata = NULL;
  const char **csuccess = NULL;
  const char **cwarning = NULL;
  PetscReal rtol, atol, ltol;
  PetscInt  max_its;
  char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];
  PetscViewer converged_monitor = linesearch->converged_monitor;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetTolerances(linesearch, NULL, NULL, &rtol, &atol, &ltol, &max_its));
  if (converged_monitor) {
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_SUCCESS, &csuccess));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
  }
  if (PetscIsInfOrNanScalar(fty)) {
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
      if (PetscIsNanScalar(fty)) PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because dot(f,y) is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(cwarning, "NaN"), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
      else PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because dot(f,y) is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "Inf"), PetscColorArg(cdata, it)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* check absolute tolerance */
  if (PetscAbsScalar(fty) <= atol * ynorm) {
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(csuccess, "atol"), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* check relative tolerance */
  if (it > 0 && PetscAbsScalar(fty) / PetscAbsScalar(fty_initial) <= rtol) {
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(csuccess, "rtol"), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* check maximum number of iterations */
  if (it > max_its) {
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed due to " PetscColorFmt("%s") " " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(cwarning, "maximum iterations"), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* check change of lambda tolerance */
  if (it > 0 && PetscAbsReal(lambda - lambda_old) < ltol) {
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(csuccess, "ltol"), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *keep_going = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchApply_Bisection(SNESLineSearch linesearch)
{
  PetscBool   changed_y, changed_w;
  Vec         X, F, Y, W, G;
  SNES        snes;
  PetscReal   ynorm;
  PetscReal   lambda_left, lambda, lambda_right, lambda_old;
  PetscScalar fty_left, fty, fty_initial;
  PetscViewer monitor, converged_monitor;
  PetscReal   rtol, atol, ltol, minlambda;
  PetscInt    it, max_its;
  const char **cdata = NULL;
  const char **csuccess = NULL;
  const char **cwarning = NULL;
  char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &minlambda, NULL, &rtol, &atol, &ltol, &max_its));
  PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));
  converged_monitor = linesearch->converged_monitor;
  if (converged_monitor) {
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_SUCCESS, &csuccess));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
  }

  /* pre-check */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));

  /* compute ynorm to normalize search direction */
  PetscCall(VecNorm(Y, NORM_2, &ynorm));

  /* initialize interval for bisection */
  lambda_left  = 0.0;
  lambda_right = lambda;
  it           = 0;

  /* compute fty at left end of interval */
  if (linesearch->ops->vidirderiv) {
    PetscCall((*linesearch->ops->vidirderiv)(snes, F, X, Y, &fty_left));
  } else {
    PetscCall(VecDot(F, Y, &fty_left));
  }
  PetscCall(SNESLineSearchMonitor_Bisection(linesearch, it, fty_left, ynorm, lambda_left));
  fty_initial = fty_left;

  it++;
  /* compute fty at right end of interval (initial lambda) */
  PetscCall(VecWAXPY(W, -lambda, Y, X));
  if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
  if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    PetscCall(PetscInfo(snes, "Exceeded maximum function evaluations during line search!\n"));
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (linesearch->ops->vidirderiv) {
    PetscCall((*linesearch->ops->vidirderiv)(snes, G, W, Y, &fty));
  } else {
    PetscCall(VecDot(G, Y, &fty));
  }
  PetscCall(SNESLineSearchMonitor_Bisection(linesearch, it, fty, ynorm, lambda));

  /* check whether sign changes in interval */
  if (!PetscIsInfOrNanScalar(fty) && (PetscRealPart(fty_left * fty) > 0.0)) {
    /* no change of sign: accept full step */
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, 1.0, PETSC_REAL_FMT_SHORT, lambda_fmt));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iteration " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(csuccess, "no sign change"), PetscColorArg(cdata, 1), PetscColorArg(cdata, lambda_fmt)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
    }
    goto postcheck;
  }
  /* change of sign: iteratively bisect interval */
  lambda_old = 0.0;

  while (PETSC_TRUE) {
    PetscBool keep_going;
    PetscCall(SNESLineSearchCheckConvergenceDefault_Internal(linesearch, it, ynorm, fty, lambda, PETSC_DEFAULT, lambda_old, fty_initial, &keep_going));
    if (!keep_going) {
      SNESLineSearchReason reason;

      PetscCall(SNESLineSearchGetReason(linesearch, &reason));
      if (reason == SNES_LINESEARCH_SUCCEEDED || (reason == SNES_LINESEARCH_FAILED_REDUCT && lambda > minlambda)) goto postcheck;
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    /* determine direction of bisection (not necessary for 0th iteration) */
    if (it > 1) {
      if (PetscRealPart(fty * fty_left) <= 0.0) {
        lambda_right = lambda;
      } else {
        lambda_left = lambda;
        /* also update fty_left for direction check in next iteration */
        fty_left = fty;
      }
    }

    /* bisect interval */
    lambda_old = lambda;
    lambda     = 0.5 * (lambda_left + lambda_right);

    it++;
    /* compute fty at new lambda */
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
    PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
    if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
      PetscCall(PetscInfo(snes, "Exceeded maximum function evaluations during line search!\n"));
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (linesearch->ops->vidirderiv) {
      PetscCall((*linesearch->ops->vidirderiv)(snes, G, W, Y, &fty));
    } else {
      PetscCall(VecDot(G, Y, &fty));
    }
    PetscCall(SNESLineSearchMonitor_Bisection(linesearch, it, fty, ynorm, lambda));
  }

postcheck:

  /* post-check */
  PetscCall(SNESLineSearchSetLambda(linesearch, lambda));
  PetscCall(SNESLineSearchPostCheck(linesearch, X, Y, W, &changed_y, &changed_w));
  if (changed_y) {
    if (!changed_w) PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  }

  /* update solution*/
  PetscCall(VecCopy(W, X));
  PetscCall((*linesearch->ops->snesfunc)(snes, X, F));
  PetscCall(SNESLineSearchComputeNorms(linesearch));

  /* finalization */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHBISECTION - Bisection line search.
   Similar to the critical point line search, `SNESLINESEARCHCP`, the bisection line search assumes that there exists some $G(x)$ for which the `SNESFunctionFn` $F(x) = grad G(x)$.
   This line search seeks to find the root of the directional derivative along the search direction $F^T Y$ through bisection.

   Options Database Keys:
+  -snes_linesearch_max_it <50> - maximum number of iterations for the line search
.  -snes_linesearch_damping <1.0> - initial trial step length on entry to the line search
.  -snes_linesearch_rtol <1e\-8> - relative tolerance for the directional derivative
.  -snes_linesearch_atol <1e\-6> - absolute tolerance for the directional derivative
-  -snes_linesearch_ltol <1e\-6> - minimum absolute change in lambda allowed

   Level: intermediate

   Note:
   This method does NOT use the objective function if it is provided with `SNESSetObjective()`.
   This line search will always give a step size in the interval [0, damping].

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`, `SNESLINESEARCHCP`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Bisection(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_Bisection;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;

  /* set default option values */
  linesearch->max_its = 50;
  linesearch->rtol    = 1e-8;
  linesearch->atol    = 1e-6;
  linesearch->ltol    = 1e-6;
  PetscFunctionReturn(PETSC_SUCCESS);
}
