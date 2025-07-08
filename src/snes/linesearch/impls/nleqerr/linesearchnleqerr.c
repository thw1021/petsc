#include <petsc/private/linesearchimpl.h> /*I  "petscsnes.h"  I*/
#include <petsc/private/snesimpl.h>
#include <petsc/private/viewerimpl.h>

typedef struct {
  PetscReal norm_delta_x_prev;     /* norm of previous update */
  PetscReal norm_bar_delta_x_prev; /* norm of previous bar update */
  PetscReal mu_curr;               /* current local Lipschitz estimate */
  PetscReal lambda_prev;           /* previous step length: for some reason SNESLineSearchGetLambda returns 1 instead of the previous step length */
} SNESLineSearch_NLEQERR;

static PetscBool  NLEQERR_cited      = PETSC_FALSE;
static const char NLEQERR_citation[] = "@book{deuflhard2011,\n"
                                       "  title = {Newton Methods for Nonlinear Problems},\n"
                                       "  author = {Peter Deuflhard},\n"
                                       "  volume = 35,\n"
                                       "  year = 2011,\n"
                                       "  isbn = {978-3-642-23898-7},\n"
                                       "  doi  = {10.1007/978-3-642-23899-4},\n"
                                       "  publisher = {Springer-Verlag},\n"
                                       "  address = {Berlin, Heidelberg}\n}\n";

static PetscErrorCode SNESLineSearchReset_NLEQERR(SNESLineSearch linesearch)
{
  SNESLineSearch_NLEQERR *nleqerr = (SNESLineSearch_NLEQERR *)linesearch->data;

  PetscFunctionBegin;
  nleqerr->mu_curr               = 0.0;
  nleqerr->norm_delta_x_prev     = -1.0;
  nleqerr->norm_bar_delta_x_prev = -1.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchMonitor_NLEQERR(SNESLineSearch linesearch, PetscInt it, PetscScalar wnorm, PetscReal lambda)
{
  SNES snes;
  SNESObjectiveFn *objective;
  PetscViewer monitor = linesearch->monitor;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  if (monitor) {
    const char **data = NULL;
    char         wnorm_fmt[PETSC_MONITOR_REAL_LENGTH];
    char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

    PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
    PetscCall(PetscViewerASCIIGetColor(monitor, PETSC_COLOR_DATA, &data));
    PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
    PetscCall(PetscViewerASCIIFormatMonitorReal(monitor, wnorm, PETSC_REAL_FMT_SHORT, wnorm_fmt));
    PetscCall(PetscViewerASCIIPrintf(monitor, "SNESLineSearch " PetscColorFmt("%3" PetscInt_FMT) " Jacobian-preconditioned function norm " PetscColorFmt("%s") " lambda " PetscColorFmt("%s") "\n", PetscColorArg(data, it), PetscColorArg(data, wnorm_fmt), PetscColorArg(data, lambda_fmt)));
    PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchComputeFunction_NLEQERR(SNESLineSearch linesearch, PetscInt count, PetscReal lambda, Vec W, Vec G, PetscBool *keep_going)
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
  PetscCall(SNESComputeFunction(snes, W, G));
  *keep_going = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchApply_NLEQERR(SNESLineSearch linesearch)
{
  PetscBool               changed_y, changed_w;
  Vec                     X, F, Y, W, G;
  SNES                    snes;
  PetscReal               fnorm, xnorm, ynorm, gnorm, wnorm;
  PetscReal               lambda, minlambda, stol;
  PetscViewer             converged_monitor;
  PetscInt                max_its, it, snes_iteration;
  PetscReal               theta, mudash, lambdadash;
  SNESLineSearch_NLEQERR *nleqerr = (SNESLineSearch_NLEQERR *)linesearch->data;
  KSPConvergedReason      kspreason;
  const char **cdata = NULL;
  const char **csuccess = NULL;
  const char **cwarning = NULL;
  PetscBool          keep_going;
    char         lambda_fmt[PETSC_MONITOR_REAL_LENGTH];

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(NLEQERR_citation, &NLEQERR_cited));

  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G));
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  converged_monitor = linesearch->converged_monitor;
  if (converged_monitor) {
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_SUCCESS, &csuccess));
    PetscCall(PetscViewerASCIIGetColor(converged_monitor, PETSC_COLOR_WARNING, &cwarning));
  }
  PetscCall(SNESLineSearchGetTolerances(linesearch, &minlambda, NULL, NULL, NULL, NULL, &max_its));
  PetscCall(SNESGetTolerances(snes, NULL, NULL, &stol, NULL, NULL));

  /* reset the state of the Lipschitz estimates */
  PetscCall(SNESGetIterationNumber(snes, &snes_iteration));
  if (!snes_iteration) PetscCall(SNESLineSearchReset_NLEQERR(linesearch));

  /* precheck */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));

  PetscCall(VecNormBegin(Y, NORM_2, &ynorm));
  PetscCall(VecNormBegin(X, NORM_2, &xnorm));
  PetscCall(VecNormEnd(Y, NORM_2, &ynorm));
  PetscCall(VecNormEnd(X, NORM_2, &xnorm));

  /* Note: Y is *minus* the Newton step. For whatever reason PETSc doesn't solve with the minus on  the RHS. */

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

  /* At this point, we've solved the Newton system for delta_x, and we assume that
     its norm is greater than the solution tolerance (otherwise we wouldn't be in
     here). So let's go ahead and estimate the Lipschitz constant.

     W contains bar_delta_x_prev at this point. */

  PetscCall(PetscInfo(linesearch, "norm of Newton step: %14.12e\n", (double)ynorm));
  PetscCall(SNESLineSearchMonitor_NLEQERR(linesearch, 0, ynorm, 0.0));

  /* this needs information from a previous iteration, so can't do it on the first one */
  if (nleqerr->norm_delta_x_prev > 0 && nleqerr->norm_bar_delta_x_prev > 0) {
    PetscCall(VecWAXPY(G, +1.0, Y, W)); /* bar_delta_x - delta_x; +1 because Y is -delta_x */
    PetscCall(VecNormBegin(G, NORM_2, &gnorm));
    PetscCall(VecNormEnd(G, NORM_2, &gnorm));

    nleqerr->mu_curr = nleqerr->lambda_prev * (nleqerr->norm_delta_x_prev * nleqerr->norm_bar_delta_x_prev) / (gnorm * ynorm);
    lambda           = PetscMin(1.0, nleqerr->mu_curr);

    PetscCall(PetscInfo(linesearch, "Lipschitz estimate: %14.12e; lambda: %14.12e\n", (double)nleqerr->mu_curr, (double)lambda));
  } else {
    lambda = linesearch->damping;
  }

  /* The main while loop of the algorithm.
     At the end of this while loop, G should have the accepted new X in it. */

  for (it = 1; it <= max_its; it++) {
    PetscCall(PetscInfo(linesearch, "entering iteration with lambda: %14.12e\n", (double)lambda));

    /* Now comes the Regularity Test. */
    if (lambda <= minlambda) {
      /* This isn't what is suggested by Deuflhard, but it works better in my experience */
      PetscCall(PetscInfo(linesearch, "lambda has reached lambdamin, taking full Newton step\n"));
      PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it, lambda, SNES_LINESEARCH_SUCCEEDED, "regularity test failure"));
      lambda = 1.0;
      PetscCall(VecWAXPY(G, -lambda, Y, X));

      /* and clean up the state for next time */
      PetscCall(SNESLineSearchReset_NLEQERR(linesearch));
      /*
         The clang static analyzer detected a problem here; once the loop is broken the values
         nleqerr->norm_delta_x_prev     = ynorm;
         nleqerr->norm_bar_delta_x_prev = wnorm;
         are set, but wnorm has not even been computed.
         I don't know if this is the correct fix but by setting ynorm and wnorm to -1.0 at
         least the linesearch object is kept in the state set by the SNESLineSearchReset_NLEQERR() call above
      */
      ynorm = wnorm = -1.0;
      goto postcheck;
    }

    /* Compute new trial iterate */
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    PetscCall(SNESLineSearchComputeFunction_NLEQERR(linesearch, it, lambda, W, G, &keep_going));
    if (!keep_going) PetscFunctionReturn(PETSC_SUCCESS);

    /* Solve linear system for bar_delta_x_curr: old Jacobian, new RHS. Note absence of minus sign, compared to Deuflhard, in keeping with PETSc convention */
    PetscCall(KSPSolve(snes->ksp, G, W));
    PetscCall(KSPGetConvergedReason(snes->ksp, &kspreason));
    if (kspreason < 0) PetscCall(PetscInfo(snes, "Solution for \\bar{delta x}^{k+1} failed.\n"));

    /* W now contains -bar_delta_x_curr. */

    PetscCall(VecNorm(W, NORM_2, &wnorm));
    PetscCall(PetscInfo(linesearch, "norm of simplified Newton update: %14.12e\n", (double)wnorm));
    PetscCall(SNESLineSearchMonitor_NLEQERR(linesearch, it, wnorm, lambda));

    /* compute the monitoring quantities theta and mudash. */

    theta = wnorm / ynorm;

    PetscCall(VecWAXPY(G, -(1.0 - lambda), Y, W));
    PetscCall(VecNorm(G, NORM_2, &gnorm));

    mudash = (0.5 * ynorm * lambda * lambda) / gnorm;

    /* Check for termination of the linesearch */
    if (theta >= 1.0) {
      /* need to go around again with smaller lambda */
      PetscCall(PetscInfo(linesearch, "monotonicity check failed, ratio: %14.12e\n", (double)theta));
      lambda = PetscMin(mudash, 0.5 * lambda);
      lambda = PetscMax(lambda, minlambda);
      /* continue through the loop, i.e. go back to regularity test */
    } else {
      /* linesearch terminated */
      lambdadash = PetscMin(1.0, mudash);

      if (lambdadash == 1.0 && lambda == 1.0 && wnorm <= stol) {
        /* store the updated state, X - Y - W, in G:
           I need to keep W for the next linesearch */
        PetscCall(VecWAXPY(G, -1.0, Y, X));
        PetscCall(VecAXPY(G, -1.0, W));
      }

      /* Deuflhard suggests to add the following:
      else if (lambdadash >= 4.0 * lambda) {
        lambda = lambdadash;
      }
      to continue through the loop, i.e. go back to regularity test.
      I deliberately exclude this, as I have practical experience of this
      getting stuck in infinite loops (on e.g. an Allen--Cahn problem). */

      else {
        /* accept iterate without adding on, i.e. don't use bar_delta_x;
           again, I need to keep W for the next linesearch */
        PetscCall(VecWAXPY(G, -lambda, Y, X));
      }

      if (converged_monitor) {
        PetscCall(PetscViewerASCIIAddTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIFormatMonitorReal(converged_monitor, lambda, PETSC_REAL_FMT_SHORT, lambda_fmt));
        PetscCall(PetscViewerASCIIPrintf(converged_monitor, "SNESLineSearch succeeded due to " PetscColorFmt("%s") " iterations " PetscColorFmt("%" PetscInt_FMT) "\n", PetscColorArg(csuccess, "sufficient reduction"), PetscColorArg(cdata, it)));
        PetscCall(PetscViewerASCIISubtractTab(converged_monitor, ((PetscObject)linesearch)->tablevel));
      }
      PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
      goto postcheck;
    }
  }

  PetscCall(SNESLineSearchSetConvergenceReasonDefault_Internal(linesearch, it - 1, lambda, SNES_LINESEARCH_FAILED_REDUCT, "maximum iterations"));
  PetscFunctionReturn(PETSC_SUCCESS);

postcheck:
  if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, G));

  /* W currently contains -bar_delta_u. Scale it so that it contains bar_delta_u. */
  PetscCall(VecScale(W, -1.0));

  /* postcheck */
  PetscCall(SNESLineSearchPostCheck(linesearch, X, Y, G, &changed_y, &changed_w));
  if (changed_y || changed_w) {
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_USER));
    PetscCall(PetscInfo(snes, "Changing the search direction here doesn't make sense.\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* copy the solution and information from this iteration over */
  nleqerr->norm_delta_x_prev     = ynorm;
  nleqerr->norm_bar_delta_x_prev = wnorm;
  nleqerr->lambda_prev           = lambda;

  PetscCall(VecCopy(G, X));
  PetscCall(SNESComputeFunction(snes, X, F));
  PetscCall(VecNorm(X, NORM_2, &xnorm));
  PetscCall(VecNorm(F, NORM_2, &fnorm));
  PetscCall(SNESLineSearchSetLambda(linesearch, lambda));
  PetscCall(SNESLineSearchSetNorms(linesearch, xnorm, fnorm, ynorm < 0 ? PETSC_INFINITY : ynorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchView_NLEQERR(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscBool               iascii;
  SNESLineSearch_NLEQERR *nleqerr;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  nleqerr = (SNESLineSearch_NLEQERR *)linesearch->data;
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  NLEQ-ERR affine-covariant linesearch"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  current local Lipschitz estimate omega=%e\n", (double)nleqerr->mu_curr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchDestroy_NLEQERR(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(linesearch->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHNLEQERR - Error-oriented affine-covariant globalised Newton algorithm of Deuflhard {cite}`deuflhard2011`

   This linesearch is intended for Newton-type methods which are affine covariant. Affine covariance
   means that Newton's method will give the same iterations for F(x) = 0 and AF(x) = 0 for any nonsingular
   matrix A. This is a fundamental property; the philosophy of this linesearch is that globalisations
   of Newton's method should carefully preserve it.

   Options Database Keys:
+  -snes_linesearch_damping<1.0> - initial step length
-  -snes_linesearch_minlambda<1e-12> - minimum step length allowed

   Level: advanced

   Note:
   Contributed by Patrick Farrell <patrick.farrell@maths.ox.ac.uk>

.seealso: [](ch_snes), `SNESLineSearch`, `SNES`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_NLEQERR(SNESLineSearch linesearch)
{
  SNESLineSearch_NLEQERR *nleqerr;

  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_NLEQERR;
  linesearch->ops->destroy        = SNESLineSearchDestroy_NLEQERR;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = SNESLineSearchReset_NLEQERR;
  linesearch->ops->view           = SNESLineSearchView_NLEQERR;
  linesearch->ops->setup          = NULL;

  PetscCall(PetscNew(&nleqerr));

  linesearch->data    = (void *)nleqerr;
  linesearch->max_its = 40;
  PetscCall(SNESLineSearchReset_NLEQERR(linesearch));
  PetscFunctionReturn(PETSC_SUCCESS);
}
