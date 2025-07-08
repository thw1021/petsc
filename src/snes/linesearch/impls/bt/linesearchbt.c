#include <petsc/private/linesearchimpl.h> /*I  "petscsnes.h"  I*/
#include <petsc/private/snesimpl.h>
#include <petsc/private/viewerimpl.h>

typedef struct {
  PetscReal alpha; /* sufficient decrease parameter */
} SNESLineSearch_BT;

/*@
  SNESLineSearchBTSetAlpha - Sets the descent parameter, `alpha`, in the `SNESLINESEARCHBT` `SNESLineSearch` variant.

  Input Parameters:
+ linesearch - linesearch context
- alpha      - The descent parameter

  Level: intermediate

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchSetLambda()`, `SNESLineSearchGetTolerances()`, `SNESLINESEARCHBT`, `SNESLineSearchBTGetAlpha()`
@*/
PetscErrorCode SNESLineSearchBTSetAlpha(SNESLineSearch linesearch, PetscReal alpha)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  bt->alpha = alpha;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchBTGetAlpha - Gets the descent parameter, `alpha`, in the `SNESLINESEARCHBT` variant that was set with `SNESLineSearchBTSetAlpha()`

  Input Parameter:
. linesearch - linesearch context

  Output Parameter:
. alpha - The descent parameter

  Level: intermediate

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchGetLambda()`, `SNESLineSearchGetTolerances()`, `SNESLINESEARCHBT`, `SNESLineSearchBTSetAlpha()`
@*/
PetscErrorCode SNESLineSearchBTGetAlpha(SNESLineSearch linesearch, PetscReal *alpha)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  *alpha = bt->alpha;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchMonitor_BT(SNESLineSearch linesearch, PetscInt it, PetscScalar f, PetscReal fnorm, PetscReal ynorm, PetscReal lambda)
{
  SNES snes;
  SNESObjectiveFn *objective;
  PetscViewer monitor = linesearch->monitor;
  const char *const  ordStr[] = {"Linear", "Quadratic", "Cubic"};
  const char *order = ordStr[linesearch->order - 1];

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
      PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, f, PETSC_REAL_FMT_SHORT, f_fmt));
      PetscCall(PetscViewerASCIIPrintf(monitor, "SNESLineSearch " PetscColorFmt("%3" PetscInt_FMT) " %s step, function norm " PetscColorFmt("%s") " lambda " PetscColorFmt("%s") "\n", PetscColorArg(data, it), order, PetscColorArg(data, f_fmt), PetscColorArg(data, lambda_fmt)));
    } else {
      PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, fnorm, PETSC_REAL_FMT_SHORT | PETSC_REAL_FMT_SIGNED, f_fmt));
      PetscCall(PetscViewerASCIIPrintf(monitor, "SNESLineSearch " PetscColorFmt("%3" PetscInt_FMT) " %s step, objective value " PetscColorFmt("%s") " lambda " PetscColorFmt("%s") "\n", PetscColorArg(data, it), order, PetscColorArg(data, f_fmt), PetscColorArg(data, lambda_fmt)));
    }
    PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchConvergedSufficientReduction_BT(SNESLineSearch linesearch, PetscInt it, PetscReal lambda)
{
  PetscViewer converged_monitor = linesearch->converged_monitor;

  PetscFunctionBegin;
  if (converged_monitor) {
    const char **cdata = NULL;
    const char **csuccess = NULL;
    char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

    PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_SUCCESS, &csuccess));
    PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
    PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(csuccess, "sufficient reduction"), PetscColorArg(cdata, it)));
    PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchComputeObjective_BT(SNESLineSearch linesearch, PetscInt count, PetscReal fnorm, Vec W, Vec G, PetscReal *g, PetscReal *gnorm, PetscBool fail_nan, PetscBool *keep_going)
{
  SNES snes;
  PetscViewer converged_monitor = linesearch->converged_monitor;
  SNESObjectiveFn   *objective;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    PetscCall(PetscInfo(snes, "Exceeded maximum function evaluations, while checking full step length!\n"));
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

  if (objective) {
    PetscCall(SNESComputeObjective(snes, W, g));
  } else {
    PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
    if (linesearch->ops->vinorm) {
      *gnorm = fnorm;
      PetscCall((*linesearch->ops->vinorm)(snes, G, W, gnorm));
    } else {
      PetscCall(VecNorm(G, NORM_2, gnorm));
    }
    *g = 0.5 * PetscSqr(*gnorm);
  }
  if (fail_nan && PetscIsInfOrNanReal(*g)) {
    if (converged_monitor) {
      const char **cdata;
      const char **cwarning;

      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
      PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      if (PetscIsNanScalar(*g)) PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because g is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "NaN"), PetscColorArg(cdata, count)));
      else PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because g is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "Inf"), PetscColorArg(cdata, count)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
    PetscCall(PetscInfo(snes, "Aborted due to Nan or Inf in function evaluation\n"));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *keep_going = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchCheckConvergence_BT(SNESLineSearch linesearch, PetscInt count, PetscReal lambda, PetscReal g, PetscReal f, PetscReal initslope, PetscReal fnorm, PetscReal gnorm, PetscReal ynorm, PetscBool *keep_going)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;
  PetscReal alpha = bt->alpha;
  PetscReal minlambda;
  PetscInt max_its;
  SNES snes;
  SNESObjectiveFn   *objective;
  PetscViewer converged_monitor = linesearch->converged_monitor;
  const char **cdata = NULL;
  const char **cwarning = NULL;
  char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

  PetscFunctionBegin;
  if (converged_monitor) {
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
  }
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &minlambda, NULL, NULL, NULL, NULL, &max_its));
  if (g <= f + lambda * alpha * initslope) { /* Sufficient reduction or step tolerance convergence */
    PetscCall(SNESLineSearchConvergedSufficientReduction_BT(linesearch, count, lambda));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (lambda <= minlambda) {
    PetscCall(PetscInfo(linesearch, "unable to find good step length! After %" PetscInt_FMT " tries \n", count));
    if (!objective) PetscCall(PetscInfo(linesearch, "fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n", (double)fnorm, (double)gnorm, (double)ynorm, (double)minlambda, (double)lambda, (double)initslope));
    else PetscCall(PetscInfo(linesearch, "    Line search: obj(0)=%18.16e, obj=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n", (double)f, (double)g, (double)ynorm, (double)minlambda, (double)lambda, (double)initslope));
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "min lambda"), PetscColorArg(cdata, count)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (count >= max_its) {
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed due to " PetscColorFmt("%s") " " PetscColorFmt("%" PetscInt_FMT) " lambda " PetscColorFmt("%s") "\n", PetscColorArg(cwarning, "maximum iterations"), PetscColorArg(cdata, count), PetscColorArg(cdata, lambda_fmt)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
    *keep_going = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *keep_going = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchApply_BT(SNESLineSearch linesearch)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;
  PetscBool          changed_y, changed_w;
  Vec                X, F, Y, W, G;
  SNES               snes;
  PetscReal          fnorm, xnorm, ynorm, gnorm;
  PetscReal          lambda, lambdatemp, lambdaprev, minlambda, maxstep, initslope, alpha, stol;
  PetscReal          t1, t2, a, b, d;
  PetscReal          f;
  PetscReal          g, gprev;
  PetscViewer        monitor, converged_monitor;
  PetscInt           max_its, count;
  Mat                jac;
  SNESObjectiveFn   *objective;
  const char **data = NULL;
  const char **cdata = NULL;
  const char **csuccess = NULL;
  const char **cwarning = NULL;
  PetscBool          keep_going;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G));
  PetscCall(SNESLineSearchGetNorms(linesearch, NULL, &fnorm, NULL));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));
  if (monitor) {
    PetscCall(PetscViewerASCIIGetColor(monitor, PETSC_COLOR_DATA, &data));
  }
  converged_monitor = linesearch->converged_monitor;
  if (converged_monitor) {
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_DATA, &cdata));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_SUCCESS, &csuccess));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
  }
  PetscCall(SNESLineSearchGetTolerances(linesearch, &minlambda, &maxstep, NULL, NULL, NULL, &max_its));
  PetscCall(SNESGetTolerances(snes, NULL, NULL, &stol, NULL, NULL));
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  alpha = bt->alpha;

  PetscCall(SNESGetJacobian(snes, &jac, NULL, NULL, NULL));
  PetscCheck(jac || objective, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "SNESLineSearchBT requires a Jacobian matrix");

  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));

  PetscCall(VecNormBegin(Y, NORM_2, &ynorm));
  PetscCall(VecNormBegin(X, NORM_2, &xnorm));
  PetscCall(VecNormEnd(Y, NORM_2, &ynorm));
  PetscCall(VecNormEnd(X, NORM_2, &xnorm));

  if (ynorm == 0.0) {
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch did not run because initial direction and size is 0\n"));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(VecCopy(X, W));
    PetscCall(VecCopy(F, G));
    PetscCall(SNESLineSearchSetNorms(linesearch, xnorm, fnorm, ynorm));
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (ynorm > maxstep) { /* Step too big, so scale back */
    PetscCall(PetscInfo(linesearch, "Scaling step by %14.12e old ynorm %14.12e\n", (double)(maxstep / ynorm), (double)ynorm));
    PetscCall(VecScale(Y, maxstep / ynorm));
    ynorm = maxstep;
  }

  /* if the SNES has an objective set, use that instead of the function value */
  if (objective) {
    PetscCall(SNESComputeObjective(snes, X, &f));
  } else {
    f = 0.5 * PetscSqr(fnorm);
  }

  count = 0;
  PetscCall(SNESLineSearchMonitor_BT(linesearch, count, f, fnorm, ynorm, 0.0));

  /* compute the initial slope */
  if (objective) {
    /* slope comes from the function (assumed to be the gradient of the objective) */
    PetscCall(VecDotRealPart(Y, F, &initslope));
  } else {
    /* slope comes from the normal equations */
    PetscCall(MatMult(jac, Y, W));
    PetscCall(VecDotRealPart(F, W, &initslope));
    if (initslope > 0.0) initslope = -initslope;
    if (initslope == 0.0) initslope = -1.0;
  }

  while (PETSC_TRUE) {
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    PetscCall(SNESLineSearchComputeObjective_BT(linesearch, count, fnorm, W, G, &g, &gnorm, /* fail NaN */ PETSC_FALSE, &keep_going));
    if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);
    if (!PetscIsInfOrNanReal(g)) break;
    PetscCall(PetscInfo(linesearch, "objective function at lambdas = %g is Inf or Nan, cutting lambda\n", (double)lambda));
    if (lambda <= minlambda) SNESCheckFunctionNorm(snes, g);
    lambda *= .5;
  }

  count++;
  PetscCall(SNESLineSearchMonitor_BT(linesearch, count, g, gnorm, ynorm, lambda));

  if (!objective) PetscCall(PetscInfo(snes, "Initial fnorm %14.12e gnorm %14.12e\n", (double)fnorm, (double)gnorm));
  if (g <= f + lambda * alpha * initslope) { /* Sufficient reduction or step tolerance convergence */
    if (!objective) PetscCall(PetscInfo(linesearch, "Using full step: fnorm %14.12e gnorm %14.12e\n", (double)fnorm, (double)gnorm));
    else PetscCall(PetscInfo(linesearch, "Using full step: old obj %14.12e new obj %14.12e\n", (double)f, (double)g));
    PetscCall(SNESLineSearchConvergedSufficientReduction_BT(linesearch, count, lambda));
    goto postcheck;
  }
  if (stol * xnorm > ynorm) {
    /* Since the full step didn't give sufficient decrease and the step is tiny, exit */
    PetscCall(SNESLineSearchSetNorms(linesearch, xnorm, fnorm, ynorm));
    if (converged_monitor) {
      PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(csuccess, "stol"), PetscColorArg(cdata, count)));
      PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Here to avoid -Wmaybe-uninitiliazed warnings */
  lambdaprev = lambda;
  gprev      = g;
  if (linesearch->order != SNES_LINESEARCH_ORDER_LINEAR) {
    /* Fit points with quadratic */
    lambdatemp = -initslope * PetscSqr(lambda) / (2.0 * (g - f - lambda * initslope));
    lambda     = PetscClipInterval(lambdatemp, .1 * lambda, .5 * lambda);

    PetscCall(VecWAXPY(W, -lambda, Y, X));
    PetscCall(SNESLineSearchComputeObjective_BT(linesearch, count, fnorm, W, G, &g, &gnorm, /* fail NaN */ PETSC_TRUE, &keep_going));
    if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);
    if (!objective) PetscCall(PetscInfo(linesearch, "gnorm after quadratic fit %14.12e\n", (double)gnorm));
    else PetscCall(PetscInfo(linesearch, "obj after quadratic fit %14.12e\n", (double)g));
    count++;
    PetscCall(SNESLineSearchMonitor_BT(linesearch, count, g, gnorm, ynorm, lambda));
  }
  while (PETSC_TRUE) {
    PetscCall(SNESLineSearchCheckConvergence_BT(linesearch, count, lambda, g, f, initslope, fnorm, gnorm, ynorm, &keep_going));
    if (!keep_going) {
      SNESLineSearchReason reason;

      PetscCall(SNESLineSearchGetReason(linesearch, &reason));
      if (reason == SNES_LINESEARCH_SUCCEEDED) goto postcheck;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
      /* Fit points with cubic */
      t1 = g - f - lambda * initslope;
      t2 = gprev - f - lambdaprev * initslope;
      a  = (t1 / (lambda * lambda) - t2 / (lambdaprev * lambdaprev)) / (lambda - lambdaprev);
      b  = (-lambdaprev * t1 / (lambda * lambda) + lambda * t2 / (lambdaprev * lambdaprev)) / (lambda - lambdaprev);
      d  = b * b - 3 * a * initslope;
      if (d < 0.0) d = 0.0;
      if (a == 0.0) lambdatemp = -initslope / (2.0 * b);
      else lambdatemp = (-b + PetscSqrtReal(d)) / (3.0 * a);
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      lambdatemp = -initslope * PetscSqr(lambda) / (2.0 * (g - f - lambda * initslope));
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_LINEAR) { /* Just backtrack */
      lambdatemp = .5 * lambda;
    } else SETERRQ(PetscObjectComm((PetscObject)linesearch), PETSC_ERR_SUP, "Line search order %" PetscInt_FMT " for type bt", linesearch->order);
    lambdaprev = lambda;
    gprev      = g;

    lambda = PetscClipInterval(lambdatemp, .1 * lambda, .5 * lambda);
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    PetscCall(SNESLineSearchComputeObjective_BT(linesearch, count, fnorm, W, G, &g, &gnorm, /* fail NaN */ PETSC_TRUE, &keep_going));
    if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);
    count++;
    PetscCall(SNESLineSearchMonitor_BT(linesearch, count, g, gnorm, ynorm, lambda));
  }

postcheck:
  /* postcheck */
  PetscCall(SNESLineSearchSetLambda(linesearch, lambda));
  PetscCall(SNESLineSearchPostCheck(linesearch, X, Y, W, &changed_y, &changed_w));
  if (changed_y) {
    if (!changed_w) PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  }
  if (changed_y || changed_w || objective) { /* recompute the function norm if the step has changed or the objective isn't the norm */
    PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
    if (linesearch->ops->vinorm) {
      gnorm = fnorm;
      PetscCall((*linesearch->ops->vinorm)(snes, G, W, &gnorm));
    } else {
      PetscCall(VecNorm(G, NORM_2, &gnorm));
    }
    PetscCall(VecNorm(Y, NORM_2, &ynorm));
    if (PetscIsInfOrNanReal(gnorm)) {
      if (converged_monitor) {
        PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
        if (PetscIsNanScalar(gnorm)) PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because g is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "NaN"), PetscColorArg(cdata, count)));
        else PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because g is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "Inf"), PetscColorArg(cdata, count)));
        PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      }
      PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
      PetscCall(PetscInfo(snes, "Aborted due to Nan or Inf in function evaluation\n"));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  /* copy the solution over */
  PetscCall(VecCopy(W, X));
  PetscCall(VecCopy(G, F));
  PetscCall(VecNorm(X, NORM_2, &xnorm));
  PetscCall(SNESLineSearchSetNorms(linesearch, xnorm, gnorm, ynorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchView_BT(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscBool          iascii;
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  interpolation: cubic\n"));
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  interpolation: quadratic\n"));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "  alpha=%e\n", (double)bt->alpha));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchDestroy_BT(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(linesearch->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchSetFromOptions_BT(SNESLineSearch linesearch, PetscOptionItems PetscOptionsObject)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SNESLineSearch BT options");
  PetscCall(PetscOptionsReal("-snes_linesearch_alpha", "Descent tolerance", "SNESLineSearchBT", bt->alpha, &bt->alpha, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHBT - Backtracking line search {cite}`dennis:83`.

   This line search finds the minimum of a polynomial fitting of the L2 norm of the
   function or the objective function if it is provided with `SNESSetObjective()`.
   If this fit does not satisfy the conditions for progress, the interval shrinks
   and the fit is reattempted at most `max_it` times or until $\lambda$ is below `minlambda`.

   Options Database Keys:
+  -snes_linesearch_alpha <1e\-4>      - slope descent parameter
.  -snes_linesearch_damping <1.0>      - scaling of initial step length on entry to the line search
.  -snes_linesearch_maxstep <length>   - if the length the initial step is larger than this then the
                                         step is scaled back to be of this length at the beginning of the line search
.  -snes_linesearch_max_it <40>        - maximum number of shrinking steps
.  -snes_linesearch_minlambda <1e\-12> - minimum step length allowed
-  -snes_linesearch_order <1,2,3>      - order of the approximation. With order 1, it performs a simple backtracking without any curve fitting

   Level: advanced

   Note:
   This line search will always produce a step that is less than or equal to, in length, the full step size.

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_BT(SNESLineSearch linesearch)
{
  SNESLineSearch_BT *bt;

  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_BT;
  linesearch->ops->destroy        = SNESLineSearchDestroy_BT;
  linesearch->ops->setfromoptions = SNESLineSearchSetFromOptions_BT;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = SNESLineSearchView_BT;
  linesearch->ops->setup          = NULL;

  PetscCall(PetscNew(&bt));

  linesearch->data    = (void *)bt;
  linesearch->max_its = 40;
  linesearch->order   = SNES_LINESEARCH_ORDER_CUBIC;
  bt->alpha           = 1e-4;
  PetscFunctionReturn(PETSC_SUCCESS);
}
