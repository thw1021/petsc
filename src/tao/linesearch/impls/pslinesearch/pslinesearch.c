#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/pslinesearch/pslinesearch.h>

static PetscErrorCode TaoLineSearchDestroy_PS(TaoLineSearch ls)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)ls, "TaoPSLineSearchSetDualWorkvec_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ls, "TaoPSLineSearchSetDualTestvec_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ls, "TaoPSLineSearchSetRegularizerTerm_C", NULL));
  PetscCall(PetscFree(armP->memory));
  if (armP->x) PetscCall(PetscObjectDereference((PetscObject)armP->x));
  if (armP->dualvec_work) PetscCall(PetscObjectDereference((PetscObject)armP->dualvec_work));
  if (armP->dualvec_test) PetscCall(PetscObjectDereference((PetscObject)armP->dualvec_test));
  if (armP->cj_orig_term) PetscCall(TaoTermDestroy(&armP->prox_term)); //Destroy conjugate one, only for TAOCV
  PetscCall(VecDestroy(&armP->work));
  PetscCall(VecDestroy(&armP->work2));
  PetscCall(PetscFree(ls->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchSetFromOptions_PS(TaoLineSearch ls, PetscOptionItems *PetscOptionsObject)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PS linesearch options");
  PetscCall(PetscOptionsReal("-tao_ls_PS_eta", "decrease constant", "", armP->eta, &armP->eta, NULL));
  PetscCall(PetscOptionsInt("-tao_ls_PS_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize, NULL));
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

PETSC_INTERN PetscErrorCode TaoPSLineSearchSetRegularizerTerm(TaoLineSearch ls, TaoMappedTerm reg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscUseMethod(ls, "TaoPSLineSearchSetRegularizerTerm_C", (TaoLineSearch, TaoMappedTerm), (ls, reg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPSLineSearchSetRegularizerTerm_PS(TaoLineSearch ls, TaoMappedTerm reg)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  armP->reg_term = reg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoPSLineSearchSetDualWorkvec(TaoLineSearch ls, Vec work)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(work, VEC_CLASSID, 2);
  PetscUseMethod(ls, "TaoPSLineSearchSetDualWorkvec_C", (TaoLineSearch, Vec), (ls, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPSLineSearchSetDualWorkvec_PS(TaoLineSearch ls, Vec work)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  armP->dualvec_work = work;
  PetscCall(PetscObjectReference((PetscObject)armP->dualvec_work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoPSLineSearchSetDualTestvec(TaoLineSearch ls, Vec work)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscValidHeaderSpecific(work, VEC_CLASSID, 2);
  PetscUseMethod(ls, "TaoPSLineSearchSetDualTestvec_C", (TaoLineSearch, Vec), (ls, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPSLineSearchSetDualTestvec_PS(TaoLineSearch ls, Vec test)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  armP->dualvec_test = test;
  PetscCall(PetscObjectReference((PetscObject)armP->dualvec_test));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* @ TaoApply_PS - This routine performs a linesearch. It
   backtracks until the (nonmonotone) PS conditions are satisfied.

   Input Parameters:
+  ls   - TaoLineSearch context
.  xold - z_k
.  f    - f(z_k)
.  g    - grad_f(z_k). same as tao->gradient
.  xnew - x_k
-  step - initial estimate of step length (Set via TaoLSSetInitialStep)

   Output parameters:
+  f    - f(x_{k+1})
.  xnew - x_{k+1} that satisfies the condition
-  step - final step length
@ */
static PetscErrorCode TaoLineSearchApply_PS(TaoLineSearch ls, Vec xold, PetscReal *f, Vec g, Vec xnew)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;
  PetscInt          i, its = 0;
  MPI_Comm          comm;
  Vec               vecin, vecout;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ls, &comm));
  ls->nfeval = 0;
  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  if (!armP->work) {
    PetscCall(VecDuplicate(xold, &armP->work));
    PetscCall(VecDuplicate(xold, &armP->work2));
    armP->x = xold;
    PetscCall(PetscObjectReference((PetscObject)armP->x));
  } else if (xold != armP->x) {
    PetscCall(VecDestroy(&armP->work));
    PetscCall(VecDestroy(&armP->work2));
    PetscCall(VecDuplicate(xold, &armP->work));
    PetscCall(VecDuplicate(xold, &armP->work2));
    PetscCall(PetscObjectDereference((PetscObject)armP->x));
    armP->x = xold;
    PetscCall(PetscObjectReference((PetscObject)armP->x));
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

  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function values. */
  if (armP->memorySize > 1) {
    if (!armP->memory) PetscCall(PetscMalloc1(armP->memorySize, &armP->memory));

    if (!armP->memorySetup) {
      for (i = 0; i < armP->memorySize; i++) armP->memory[i] = 0.;
      armP->current               = 0;
      armP->memorySetup           = PETSC_TRUE;
      armP->memory[armP->current] = *f;
    }

    /* Calculate reference value (MAX) */
    armP->ref = armP->memory[0];
    for (i = 1; i < armP->memorySize; i++) {
      if (armP->memory[i] > armP->ref) { armP->ref = armP->memory[i]; }
    }
  } else armP->ref = *f;

  ls->step = ls->initstep;

  if (ls->ops->preapply) PetscUseTypeMethod(ls, preapply, xold, f, xnew, g);

  while (armP->cert >= ls->ftol && ls->nproxeval < ls->max_funcs) {
    /* Calculate iterate */
    ++its;

    if (ls->ops->update) PetscUseTypeMethod(ls, update, xold, f, xnew, g);
    vecin  = (armP->lmap) ? armP->dualvec_work : armP->work;
    vecout = (armP->lmap) ? armP->dualvec_test : xnew;
    PetscCall(TaoTermProximalMap(armP->prox_term, armP->term_param, armP->term_scale * armP->test_step, armP->reg_term.term, vecin, armP->reg_term.scale, vecout));
    ls->nproxeval++;
    if (ls->ops->postupdate) PetscUseTypeMethod(ls, postupdate, xold, f, xnew, g);
    PetscCall(TaoLineSearchMonitor(ls, its, *f, ls->step));
  }

  /* Check termination */
  if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "Function is inf or nan.\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (ls->nproxeval >= ls->max_funcs) {
    PetscCall(PetscInfo(ls, "Number of line search prox evals (%" PetscInt_FMT ") > maximum allowed (%" PetscInt_FMT ")\n", ls->nproxeval, ls->max_funcs));
    ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
  } else if (ls->step < ls->stepmin) {
    PetscCall(PetscInfo(ls, "Step length is below tolernace.\n"));
    ls->reason = TAOLINESEARCH_HALTED_RTOL;
  }

  if (ls->ops->postapply) PetscUseTypeMethod(ls, postapply, xold, f, xnew, g);
  if (ls->reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* Successful termination, update memory. Only FIFO for PS */
  ls->reason = TAOLINESEARCH_SUCCESS;
  PetscCall(PetscInfo(ls, "%" PetscInt_FMT " prox evals in line search, step = %10.4f\n", ls->nproxeval, (double)ls->step));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchSetUp_PS(TaoLineSearch ls)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;
  Tao               tao;
  PetscBool         is_fb, is_cv;

  PetscFunctionBegin;
  tao = ls->tao;

  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &is_fb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &is_cv));

  PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, 1, NULL, &armP->f_scale, &armP->f_term, NULL));
  if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumSubParameters(tao->objective_parameters, 1, &armP->f_param));
  if (is_fb) {
    PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, 2, NULL, &armP->term_scale, &armP->prox_term, NULL));
    if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumSubParameters(tao->objective_parameters, 2, &armP->term_param));
  } else if (is_cv) {
    PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, 3, NULL, &armP->term_scale, &armP->cj_orig_term, &armP->lmap));
    //TODO technically cj term is created in cv.c but doing it again. fix later? or dont bother?
    PetscCall(TaoTermCreateConjugate(armP->cj_orig_term, &armP->prox_term));
    if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumSubParameters(tao->objective_parameters, 3, &armP->term_param));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "This routine only applies to TAOFB or TAOCV.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   TAOLINESEARCHPS - Special line-search type for proximal splittign algorithms.
   Should not be used with any other algorithm.

   Level: developer

seealso: `TaoLineSearch`, `TAOFB`, `TAOCV`, `Tao`
M*/
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_PS(TaoLineSearch ls)
{
  TaoLineSearch_PS *armP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscCall(PetscNew(&armP));

  armP->memory            = NULL;
  armP->eta               = 0.5;
  armP->memorySize        = 1;
  ls->data                = (void *)armP;
  ls->initstep            = 0;
  ls->ops->monitor        = NULL;
  ls->ops->setup          = TaoLineSearchSetUp_PS;
  ls->ops->reset          = NULL;
  ls->ops->apply          = TaoLineSearchApply_PS;
  ls->ops->view           = TaoLineSearchView_PS;
  ls->ops->destroy        = TaoLineSearchDestroy_PS;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_PS;

  PetscCall(PetscObjectComposeFunction((PetscObject)ls, "TaoPSLineSearchSetDualWorkvec_C", TaoPSLineSearchSetDualWorkvec_PS));
  PetscCall(PetscObjectComposeFunction((PetscObject)ls, "TaoPSLineSearchSetDualTestvec_C", TaoPSLineSearchSetDualTestvec_PS));
  PetscCall(PetscObjectComposeFunction((PetscObject)ls, "TaoPSLineSearchSetRegularizerTerm_C", TaoPSLineSearchSetRegularizerTerm_PS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
