#include <petsc/private/linesearchimpl.h>
#include <petscsnes.h>
#include <petsc/private/viewerimpl.h>

static PetscErrorCode SNESLineSearchApply_CP(SNESLineSearch linesearch)
{
  PetscBool   changed_y, changed_w;
  Vec         X, Y, F, W;
  SNES        snes;
  PetscReal   xnorm, ynorm, gnorm, steptol, atol, rtol, ltol, maxstep;
  PetscReal   lambda, lambda_old, lambda_update, delLambda;
  PetscScalar fty, fty_init, fty_old, fty_mid1, fty_mid2, s;
  PetscInt    count, max_its;
  PetscViewer monitor, converged_monitor;
  const char **data = NULL;
  const char **cdata = NULL;
  const char **csuccess = NULL;
  const char **cwarning = NULL;
  char         ratio_fmt[PETSC_MONITOR_REAL_LENGTH];
  char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL));
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its));
  PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
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

  /* precheck */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));
  lambda_old = 0.0;

  if (linesearch->ops->vidirderiv) {
    PetscCall((*linesearch->ops->vidirderiv)(snes, F, X, Y, &fty));
  } else {
    PetscCall(VecDot(F, Y, &fty));
  }
  fty_init = fty_old = fty;

  for (count = 0; count < max_its; count++) {
      /* check for NaN or Inf */
      if (PetscIsInfOrNanScalar(fty)) {
        if (converged_monitor) {
          PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
          if (PetscIsNanScalar(fty)) PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because fty is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "NaN"), PetscColorArg(cdata, count)));
          else PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch failed because fty is " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(cwarning, "Inf"), PetscColorArg(cdata, count)));
          PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
        }
        PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
        PetscFunctionReturn(PETSC_SUCCESS);
        break;
      }
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, PetscRealPart(fty)/ ynorm, PETSC_REAL_FMT_SHORT | PETSC_REAL_FMT_SIGNED, ratio_fmt));
      PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
      PetscCall(PetscViewerASCIIPrintf(monitor, "SNESLineSearch " PetscColorFmt("%3" PetscInt_FMT) " fty/||y|| " PetscColorFmt("%s") ", lambda " PetscColorFmt("%s") "\n", PetscColorArg(data, 0), PetscColorArg(data, ratio_fmt), PetscColorArg(data, lambda_fmt)));
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }

    if (PetscAbsScalar(fty) < atol * ynorm) {
      if (converged_monitor) {
        PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(csuccess, "atol"), PetscColorArg(cdata, count)));
        PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      }
      PetscCall(SNESSetConvergedReason(linesearch->snes, SNES_CONVERGED_FNORM_ABS));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* compute the norm at lambda */
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
    PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
    if (linesearch->ops->vidirderiv) {
      PetscCall((*linesearch->ops->vidirderiv)(snes, F, W, Y, &fty));
    } else {
      PetscCall(VecDot(F, Y, &fty));
    }

    delLambda = lambda - lambda_old;

    /* check for convergence */
    if (PetscAbsReal(delLambda) < steptol * lambda) break;
    if (PetscAbsScalar(fty) / PetscAbsScalar(fty_init) < rtol) break;
    if (PetscAbsScalar(fty) < atol * ynorm && count > 0) break;
    PetscCall(PetscInfo(linesearch, "lambdas = [%g, %g], ftys = [%g, %g]\n", (double)lambda, (double)lambda_old, (double)PetscRealPart(fty), (double)PetscRealPart(fty_old)));

    /* compute the search direction */
    if (linesearch->order == SNES_LINESEARCH_ORDER_LINEAR) {
      s = (fty - fty_old) / delLambda;
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      PetscCall(VecWAXPY(W, -0.5 * (lambda + lambda_old), Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
      if (linesearch->ops->vidirderiv) {
        PetscCall((*linesearch->ops->vidirderiv)(snes, F, W, Y, &fty_mid1));
      } else {
        PetscCall(VecDot(F, Y, &fty_mid1));
      }
      s = (3. * fty - 4. * fty_mid1 + fty_old) / delLambda;
    } else {
      PetscCall(VecWAXPY(W, -0.5 * (lambda + lambda_old), Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
      if (linesearch->ops->vidirderiv) {
        PetscCall((*linesearch->ops->vidirderiv)(snes, F, W, Y, &fty_mid1));
      } else {
        PetscCall(VecDot(F, Y, &fty_mid1));
      }
      PetscCall(VecWAXPY(W, -(lambda + 0.5 * (lambda - lambda_old)), Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
      if (linesearch->ops->vidirderiv) {
        PetscCall((*linesearch->ops->vidirderiv)(snes, F, W, Y, &fty_mid2));
      } else {
        PetscCall(VecDot(F, Y, &fty_mid2));
      }
      s = (2. * fty_mid2 + 3. * fty - 6. * fty_mid1 + fty_old) / (3. * delLambda);
    }
    /* if the solve is going in the wrong direction, fix it */
    if (PetscRealPart(s) > 0.) s = -s;
    if (s == 0.0) break;
    lambda_update = lambda - PetscRealPart(fty / s);

    /* switch directions if we stepped out of bounds */
    if (lambda_update < steptol) lambda_update = lambda + PetscRealPart(fty / s);

    if (PetscIsInfOrNanReal(lambda_update)) break;
    if (lambda_update > maxstep) break;

    /* compute the new state of the line search */
    lambda_old = lambda;
    lambda     = lambda_update;
    fty_old    = fty;
  }
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

  if (monitor) {
    PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search terminated: lambda = %g, fnorms = %g\n", (double)lambda, (double)gnorm));
    PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
  }
  if (lambda <= steptol) PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
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
