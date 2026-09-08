#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/pslinesearch/pslinesearch.h>

static PetscErrorCode TaoLineSearchDestroy_PS(TaoLineSearch ls)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)ls, "TaoPSLineSearchSetTerms_C", NULL));
  PetscCall(PetscFree(armP->memory));
  PetscCall(VecDestroy(&armP->x));
  PetscCall(VecDestroy(&armP->work));
  PetscCall(VecDestroy(&armP->work2));
  PetscCall(PetscFree(ls->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchReset_PS(TaoLineSearch ls)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(armP->memory));
  armP->memorySetup = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchSetFromOptions_PS(TaoLineSearch ls, PetscOptionItems PetscOptionsObject)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PS linesearch options");
  PetscCall(PetscOptionsReal("-tao_ls_ps_eta", "decrease constant", "", armP->eta, &armP->eta, NULL));
  PetscCall(PetscOptionsInt("-tao_ls_ps_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchView_PS(TaoLineSearch ls, PetscViewer pv)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;
  PetscBool         isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  PS linesearch"));
    PetscCall(PetscViewerASCIIPrintf(pv, "eta=%g ", (double)armP->eta));
    PetscCall(PetscViewerASCIIPrintf(pv, "memsize=%" PetscInt_FMT "\n", armP->memorySize));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoPSLineSearchSetTerms(TaoLineSearch ls, TaoTermMapping f_term, Vec f_param, TaoTermMapping g_term, Vec g_param)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscUseMethod(ls, "TaoPSLineSearchSetTerms_C", (TaoLineSearch, TaoTermMapping, Vec, TaoTermMapping, Vec), (ls, f_term, f_param, g_term, g_param));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPSLineSearchSetTerms_PS(TaoLineSearch ls, TaoTermMapping f_term, Vec f_param, TaoTermMapping g_term, Vec g_param)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  armP->f_scale    = f_term.scale;
  armP->f_term     = f_term.term;
  armP->f_param    = f_param;
  armP->term_scale = g_term.scale;
  armP->prox_term  = g_term.term;
  armP->term_param = g_param;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate f at the trial point xnew and the certificate f(xnew) - (R + <g, xnew - xold> + |xnew - xold|^2 / (2 step)) */
static PetscErrorCode TaoLineSearchComputeCertificate_PS(TaoLineSearch ls, Vec xold, PetscReal *f, Vec g, Vec xnew)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;
  PetscReal         diffnorm;
  PetscScalar       inprod;

  PetscFunctionBegin;
  PetscCall(TaoTermComputeObjective(armP->f_term, xnew, armP->f_param, f));
  *f *= armP->f_scale;
  ls->nfeval++;
  PetscCall(VecWAXPY(armP->work2, -1., xold, xnew));
  PetscCall(VecDotNorm2(g, armP->work2, &inprod, &diffnorm));
  armP->cert = *f - (armP->ref + PetscRealPart(inprod) + diffnorm / (2 * ls->step));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchApply_PS(TaoLineSearch ls, Vec xold, PetscReal *f, Vec g, Vec xnew)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;
  PetscInt          i, its = 0;

  PetscFunctionBegin;
  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  if (!armP->work) {
    PetscCall(VecDuplicate(xold, &armP->work));
    PetscCall(VecDuplicate(xold, &armP->work2));
    PetscCall(PetscObjectReference((PetscObject)xold));
    armP->x = xold;
  } else if (xold != armP->x) {
    PetscCall(VecDestroy(&armP->work));
    PetscCall(VecDestroy(&armP->work2));
    PetscCall(VecDuplicate(xold, &armP->work));
    PetscCall(VecDuplicate(xold, &armP->work2));
    PetscCall(VecDestroy(&armP->x));
    PetscCall(PetscObjectReference((PetscObject)xold));
    armP->x = xold;
  }
  PetscCall(TaoLineSearchMonitor(ls, 0, *f, 0.0));

  /* Check linesearch parameters */
  if (armP->eta > 1) {
    PetscCall(PetscInfo(ls, "PS line search error: eta (%g) > 1\n", (double)armP->eta));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (armP->memorySize < 1) {
    PetscCall(PetscInfo(ls, "PS line search error: memory_size (%" PetscInt_FMT ") < 1\n", armP->memorySize));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "PS line search error: initial function inf or nan\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }
  if (ls->reason != TAOLINESEARCH_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  /* Nonmonotone reference: largest of the last memorySize values of f at the base points */
  if (armP->memorySize > 1) {
    if (!armP->memory) PetscCall(PetscMalloc1(armP->memorySize, &armP->memory));
    if (!armP->memorySetup) {
      for (i = 0; i < armP->memorySize; i++) armP->memory[i] = *f;
      armP->current     = 0;
      armP->memorySetup = PETSC_TRUE;
    } else {
      armP->current               = (armP->current + 1) % armP->memorySize;
      armP->memory[armP->current] = *f;
    }
    armP->ref = armP->memory[0];
    for (i = 1; i < armP->memorySize; i++) armP->ref = PetscMax(armP->ref, armP->memory[i]);
  } else armP->ref = *f;

  /* xnew already holds prox_g(xold - step g) for the initial step; shrink the step until the certificate holds */
  ls->step = ls->initstep;
  PetscCall(TaoLineSearchComputeCertificate_PS(ls, xold, f, g, xnew));
  while (armP->cert > ls->ftol && ls->nproxeval < ls->max_funcs && ls->step >= ls->stepmin) {
    ++its;
    ls->step *= armP->eta;
    PetscCall(VecWAXPY(armP->work, -ls->step, g, xold));
    PetscCall(TaoTermProximalMap(armP->prox_term, armP->term_param, armP->term_scale * ls->step, NULL, armP->work, 1.0, xnew));
    ls->nproxeval++;
    PetscCall(TaoLineSearchComputeCertificate_PS(ls, xold, f, g, xnew));
    PetscCall(TaoLineSearchMonitor(ls, its, *f, ls->step));
  }

  if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "Function is inf or nan.\n"));
    ls->reason = TAOLINESEARCH_FAILED_INFORNAN;
  } else if (armP->cert <= ls->ftol) {
    PetscCall(PetscInfo(ls, "%" PetscInt_FMT " prox evals in line search, step = %10.4f\n", ls->nproxeval, (double)ls->step));
    ls->reason = TAOLINESEARCH_SUCCESS;
  } else if (ls->nproxeval >= ls->max_funcs) {
    PetscCall(PetscInfo(ls, "Number of line search prox evals (%" PetscInt_FMT ") >= maximum allowed (%" PetscInt_FMT ")\n", ls->nproxeval, ls->max_funcs));
    ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
  } else {
    PetscCall(PetscInfo(ls, "Step length %g is below tolerance %g.\n", (double)ls->step, (double)ls->stepmin));
    ls->reason = TAOLINESEARCH_HALTED_RTOL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchSetUp_PS(TaoLineSearch ls)
{
  PetscBool is_fb = PETSC_FALSE;

  PetscFunctionBegin;
  if (ls->tao) PetscCall(PetscObjectTypeCompare((PetscObject)ls->tao, TAOFB, &is_fb));
  PetscCheck(is_fb, PetscObjectComm((PetscObject)ls), PETSC_ERR_SUP, "TAOLINESEARCHPS currently only supports TAOFB");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOLINESEARCHPS - Backtracking line search for proximal splitting algorithms such as `TAOFB`.

  Options Database Keys:
+ -tao_ls_ps_eta eta          - factor by which the step size shrinks on each backtrack (default 0.5)
. -tao_ls_ps_memory_size size - number of previous function values in the nonmonotone reference (default 1, monotone)
. -tao_ls_max_funcs n         - maximum number of proximal maps per line search; 0 disables the line search
- -tao_ls_ftol tol            - slack allowed in the descent condition, to absorb rounding (default 1e-12)

  Level: developer

  Notes:
  Given the base point x_k with f(x_k) and grad f(x_k), and the trial point x_{k+1} = prox_{step g}(x_k - step grad f(x_k)),
  the step is shrunk by eta (and the trial point recomputed) until f(x_{k+1}) <= R + <grad f(x_k), x_{k+1} - x_k> + |x_{k+1} - x_k|^2 / (2 step) + ftol,
  where R = f(x_k), or for the nonmonotone variant the largest of the last memory_size values of f at the base points.
  This line search evaluates only the smooth term of the `TAOFB` objective and cannot be used with other solvers.

.seealso: `TaoLineSearch`, `TAOFB`, `Tao`
M*/
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_PS(TaoLineSearch ls)
{
  TaoLineSearch_PS *armP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscCall(PetscNew(&armP));

  armP->memory     = NULL;
  armP->eta        = 0.5;
  armP->memorySize = 1;
  ls->data         = (void *)armP;
  ls->initstep     = 0;
  ls->ftol         = 1.e-12;

  ls->ops->monitor        = NULL;
  ls->ops->setup          = TaoLineSearchSetUp_PS;
  ls->ops->reset          = TaoLineSearchReset_PS;
  ls->ops->apply          = TaoLineSearchApply_PS;
  ls->ops->view           = TaoLineSearchView_PS;
  ls->ops->destroy        = TaoLineSearchDestroy_PS;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_PS;

  PetscCall(PetscObjectComposeFunction((PetscObject)ls, "TaoPSLineSearchSetTerms_C", TaoPSLineSearchSetTerms_PS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
