#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>
#include <petsc/private/viewerimpl.h>

PetscErrorCode SNESLineSearchSetConvergenceReasonDefault_Internal(SNESLineSearch linesearch, PetscInt it, PetscReal lambda, SNESLineSearchReason reason, const char explanation[])
{
  PetscViewer converged_monitor;
  const char **cdata = NULL;
  const char **cwarning = NULL;
  char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

  PetscFunctionBegin;
  converged_monitor = linesearch->converged_monitor;
  if (converged_monitor) {
    const char **code;

    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
    code = (reason == SNES_LINESEARCH_SUCCEEDED) ? cdata : cwarning;
    PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
    PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch %s due to " PetscColorFmt("%s") " iteration " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", reason == SNES_LINESEARCH_SUCCEEDED ? "terminated" : "failed", PetscColorArg(code, explanation), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
    PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscCall(SNESLineSearchSetReason(linesearch, reason));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESLineSearchComputeObjectiveDefault_Internal(SNESLineSearch linesearch, PetscInt count, PetscBool monitor, PetscReal ynorm, PetscReal lambda, PetscReal *fty, PetscBool *keep_going)
{
  SNES snes;
  Vec         X, F, Y, W, G;
  PetscViewer converged_monitor = linesearch->converged_monitor;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    if (converged_monitor) {
      const char **cdata;
      const char **cwarning;

      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "function count"), PetscColorArg(cdata, count)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* compute fty at new lambda */
  if (lambda == 0.0) {
    if (linesearch->ops->vidirderiv) {
      PetscCall((*linesearch->ops->vidirderiv)(snes, F, X, Y, fty));
    } else {
      PetscCall(VecDot(F, Y, fty));
    }
  } else {
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
    PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
    if (linesearch->ops->vidirderiv) {
      PetscCall((*linesearch->ops->vidirderiv)(snes, G, W, Y, fty));
    } else {
      PetscCall(VecDot(G, Y, fty));
    }
  }
  if (monitor) PetscCall(SNESLineSearchMonitorDefault_Internal(linesearch, count, *fty, ynorm, lambda));
  if (PetscIsInfOrNanReal(*fty)) {
    if (converged_monitor) {
      const char **cdata;
      const char **cwarning;

      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      if (PetscIsNanScalar(*fty)) PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because dot(F,Y) is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "NaN"), PetscColorArg(cdata, count)));
      else PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because dot(F,Y) is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "Inf"), PetscColorArg(cdata, count)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *keep_going = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESLineSearchMonitorDefault_Internal(SNESLineSearch linesearch, PetscInt it, PetscScalar fty, PetscReal ynorm, PetscReal lambda)
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
    PetscCall(PetscViewerASCIIPrintf(monitor, "SNESLineSearch " PetscColorFmt("%3" PetscInt_FMT) " dot(F,y)/||y|| " PetscColorFmt("%s") ", lambda " PetscColorFmt("%s") "\n", PetscColorArg(data, it), PetscColorArg(data, ratio_fmt), PetscColorArg(data, lambda_fmt)));
    PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SNESLineSearchCheckConvergenceDefault_Internal(SNESLineSearch linesearch, PetscInt it, PetscScalar ynorm, PetscScalar fty, PetscScalar lambda, PetscScalar lambda_old, PetscScalar fty_initial, PetscBool *keep_going)
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
      if (PetscIsNanScalar(fty)) PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because dot(F,y) is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(cwarning, "NaN"), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
      else PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because dot(F,y) is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(cwarning, "Inf"), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
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
  PetscScalar fty_left, fty, fty_init;
  PetscViewer monitor, converged_monitor;
  PetscReal   rtol, atol, ltol, minlambda;
  PetscInt    it, max_its;
  const char **cdata = NULL;
  const char **csuccess = NULL;
  const char **cwarning = NULL;
  char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];
    PetscBool keep_going;

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

  /* compute fty at left end of interval */
  it = 0;
  PetscCall(SNESLineSearchComputeObjectiveDefault_Internal(linesearch, it, PETSC_TRUE, ynorm, lambda_left, &fty_left, &keep_going));
  if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);
  fty_init = fty_left;

  lambda_old = 0.0;
  for (it = 1; it <= max_its; it++) {
    PetscCall(SNESLineSearchComputeObjectiveDefault_Internal(linesearch, it, PETSC_TRUE, ynorm, lambda, &fty, &keep_going));
    if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);

    if (it == 1 && PetscRealPart(fty_left * fty) > 0.0) {
      /* no change of sign: accept full step */
      if (converged_monitor) {
        PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, 1.0, PETSC_REAL_FMT_SHORT, lambda_fmt));
        PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iteration " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(csuccess, "no sign change"), PetscColorArg(cdata, it), PetscColorArg(cdata, lambda_fmt)));
        PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
      }
      goto postcheck;
    }

    PetscCall(SNESLineSearchCheckConvergenceDefault_Internal(linesearch, it, ynorm, fty, lambda, lambda_old, fty_init, &keep_going));
    if (!keep_going) {
      SNESLineSearchReason reason;

      PetscCall(SNESLineSearchGetReason(linesearch, &reason));
      if (reason == SNES_LINESEARCH_SUCCEEDED) goto postcheck;
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

    /* compute fty at new lambda */
    it++;
    PetscCall(SNESLineSearchComputeObjectiveDefault_Internal(linesearch, PETSC_TRUE, it, ynorm, lambda, &fty, &keep_going));
  }

  if (lambda <= minlambda) {
    PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it - 1, lambda, SNES_LINESEARCH_FAILED_REDUCT, "minimum step length"));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it - 1, lambda, SNES_LINESEARCH_SUCCEEDED, "maximum iterations"));
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
