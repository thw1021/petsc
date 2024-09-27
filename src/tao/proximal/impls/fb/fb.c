#include <../src/tao/proximal/impls/fb/fb.h> /*I "petsctao.h" I*/
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/dmimpl.h>
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/pslinesearch/pslinesearch.h>

static PetscBool  fasta_cited       = PETSC_FALSE;
static PetscBool  adapgm_cited      = PETSC_FALSE;
static const char fasta_citation[]  = "@article{goldstein2015fasta,\n"
                                      "title={FASTA: A generalized implementation of forward-backward splitting},\n"
                                      "author={Goldstein, Tom and Studer, Christoph and Baraniuk, Richard},\n"
                                      "journal={arXiv preprint arXiv:1501.04979},\n"
                                      " year={2015}\n"
                                      "}\n";
static const char adapgm_citation[] = "@article{latafat2023convergence,\n"
                                      "title={On the convergence of adaptive first order methods: proximal gradient and alternating minimization algorithms},\n"
                                      "author={Latafat, Puya and Themelis, Andreas and Patrinos, Panagiotis},\n"
                                      "booktitle={6th Annual Learning for Dynamics and Control Conference},\n"
                                      "pages={197--208},\n"
                                      "year={2024},\n"
                                      "organization={PMLR},\n"
                                      "}\n";

static PetscErrorCode TaoFB_LineSearch_PreApply_Private(TaoLineSearch ls, Vec in, PetscReal *f, Vec out, Vec g)
{
  PetscReal         temp, diffnorm, inprod;
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  /* Input is prox_g(x- step * gradf(x)) *
   * Calculate function at new iterate i */
  PetscCall(TaoLineSearchComputeObjective(ls, out, &temp));
  /* Check criteria */
  PetscCall(VecWAXPY(armP->work2, -1., in, out));
  PetscCall(VecTDot(armP->work2, armP->work2, &diffnorm));
  PetscCall(VecTDot(armP->work2, g, &inprod));
  armP->cert = temp - (inprod + (1 / (2 * ls->step)) * diffnorm + armP->ref);

  /* accept xnew */
  if (armP->cert < ls->rtol) {
    PetscCall(TaoLineSearchComputeObjective(ls, out, f));
    ls->reason = TAOLINESEARCH_SUCCESS;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFB_LineSearch_PostApply_Private(TaoLineSearch ls, Vec in, PetscReal *f, Vec out, Vec g)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  if (armP->memorySize > 1) {
    armP->current++;
    if (armP->current >= armP->memorySize) armP->current = 0;
    armP->memory[armP->current] = *f;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFB_LineSearch_Update_Private(TaoLineSearch ls, Vec in, PetscReal *f, Vec out, Vec g)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  ls->step = ls->step * armP->eta;
  /* FB: input to prox: x_k - step*gradf(x_k) */
  PetscCall(VecWAXPY(armP->work, -ls->step, g, in));
  armP->test_step = ls->step * ls->prox_scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFB_LineSearch_PostUpdate_Private(TaoLineSearch ls, Vec xold, PetscReal *f, Vec xnew, Vec g)
{
  PetscReal         inprod, diffnorm;
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;

  PetscFunctionBegin;
  PetscCall(TaoLineSearchComputeObjective(ls, xnew, f));
  PetscCall(VecWAXPY(armP->work2, -1., xold, xnew));
  PetscCall(VecTDot(armP->work2, armP->work2, &diffnorm));
  PetscCall(VecTDot(armP->work2, g, &inprod));

  armP->cert = *f - (inprod + (1 / (2 * ls->step)) * diffnorm + armP->ref);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFB_ADAPGM_Update_Stepsize_Private(Tao tao)
{
  TAO_FB   *fb = (TAO_FB *)tao->data;
  PetscReal grad_x_dot, graddiff, xdiff, L, C, min1, min2, temp;

  PetscFunctionBegin;
  /* workvec: v, which is x - step*gradf(x)        *
   * temp \gets sqrt(norm(xnew - xold)             *
   * Gradient eval happens before prox for adaPGM, *
   * but after prox for FISTA-variants             *
   * workvec: (z-x)/step + gradf(x)                */

  /* step update                          *
   * workvec  : gradf(xnew) - gradf(xold) *
   * workvec2 : xnew - xold               */
  PetscCall(VecWAXPY(fb->workvec, -1., fb->grad_old, tao->gradient));
  PetscCall(VecWAXPY(fb->workvec2, -1., fb->x_old, tao->solution));
  PetscCall(VecTDot(fb->workvec, fb->workvec2, &grad_x_dot));
  PetscCall(VecTDot(fb->workvec2, fb->workvec2, &xdiff));
  PetscCall(VecTDot(fb->workvec, fb->workvec, &graddiff));

  L            = grad_x_dot / xdiff;
  C            = graddiff / grad_x_dot;
  min1         = tao->step * PetscSqrtReal(1 + tao->step / fb->step_old);
  temp         = PetscMax(tao->step * L * (tao->step * C - 1), 0);
  min2         = (temp == 0) ? PETSC_INFINITY : (tao->step / (2 * PetscSqrtReal(temp)));
  fb->step_old = tao->step;
  tao->step    = PetscMin(min1, min2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFB_ComputeResidual_And_LogConv_Private(Tao tao, PetscReal f)
{
  TAO_FB   *fb = (TAO_FB *)tao->data;
  PetscReal gnorm0, gradnorm;

  PetscFunctionBegin;
  PetscCall(VecNorm(fb->grad_old, NORM_2, &gradnorm));
  PetscCall(VecWAXPY(fb->workvec2, -1., tao->solution, fb->x_old));
  PetscCall(VecNorm(fb->workvec2, NORM_2, &tao->residual));

  if (PetscIsInfOrNanReal(tao->residual)) {
    PetscCall(PetscInfo(tao, "Failed to converged, residual value is Inf or NaN\n"));
    tao->reason = TAO_DIVERGED_NAN;
  }
  tao->residual /= tao->step;

  if (!fb->use_accel && tao->linesearch->max_funcs == 0 && (fb->step_old == tao->step)) {
    /* Fixed cases, and for backtracking case if step_old == step_new */
    PetscCall(VecNorm(fb->dualvec, NORM_2, &gnorm0));
  } else {
    PetscCall(VecWAXPY(fb->workvec2, -tao->step, fb->grad_old, fb->x_old));
    PetscCall(VecNorm(fb->workvec2, NORM_2, &gnorm0));
  }
  gnorm0 /= tao->step;
  tao->gnorm0 = PetscMax(gradnorm, gnorm0) + tao->gttol;
  if (PetscIsInfOrNanReal(tao->gnorm0)) {
    PetscCall(PetscInfo(tao, "Failed to converged, relative residual norm is Inf or NaN\n"));
    tao->reason = TAO_DIVERGED_NAN;
  }

  PetscCall(TaoLogConvergenceHistory(tao, f, tao->residual, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, tao->residual, 0.0, tao->step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  tao->niter++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPSSetSmoothTerm_FB(Tao tao, PetscInt idx)
{
  TAO_FB *fb = (TAO_FB *)tao->data;
  DMTao   tdm;

  PetscFunctionBegin;
  PetscCheck(idx < tao->num_terms, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Index number exceeds number of DMs in Tao object");
  PetscCall(DMGetDMTao(tao->dms[idx], &tdm));
  PetscCheck(tdm->ops->computeobjectiveandgradient || tdm->ops->computegradient, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "DMTaoSetObjective/Gradient() has not been called");
  fb->smoothterm = tao->dms[idx];
  fb->f_scale    = tao->dm_scales[idx];
  PetscCall(PetscObjectReference((PetscObject)fb->smoothterm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPSSetNonSmoothTerm_FB(Tao tao, PetscInt idx)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscCheck(idx < tao->num_terms, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Index number exceeds number of DMs in Tao object");
  fb->proxterm   = tao->dms[idx];
  fb->prox_scale = tao->dm_scales[idx];
  PetscCall(PetscObjectReference((PetscObject)fb->proxterm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPSUseAdaptiveStep_FB(Tao tao, PetscBool flg)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveBool(tao, flg, 2);
  fb->use_adapt = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPSUseAcceleration_FB(Tao tao, PetscBool flg)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveBool(tao, flg, 2);
  fb->use_accel = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_FB(Tao tao)
{
  TAO_FB                      *fb = (TAO_FB *)tao->data;
  PetscReal                    f, f_prox;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;
  PetscCheck(tao->step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize cannot be negative");
  PetscCheck(fb->xi >= 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Backtracing scale factor needs to be equal or greater than 1");
  PetscCheck(!(fb->use_accel && fb->use_adapt), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoFB only supports either acceleration or adaptive step, not both");
  PetscCheck(fb->smoothterm, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoPSSetSmoothTerm needs to be called");
  PetscCheck(fb->proxterm, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoPSSetNonSmoothTerm needs to be called");
  PetscCall(DMTaoGetLipschitz(fb->smoothterm, &fb->lip));

  if (fb->lip == 0) {
    /* Approximating initial Lipschitz number via two random vectors */
    PetscReal   gradnorm, xnorm;
    PetscRandom rctx;

    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(fb->workvec, rctx));
    PetscCall(VecSetRandom(fb->workvec2, rctx));
    PetscCall(DMTaoComputeGradient(fb->smoothterm, fb->workvec, fb->x_old));
    PetscCall(DMTaoComputeGradient(fb->smoothterm, fb->workvec2, fb->grad_old));
    if (fb->f_scale != 1) PetscCall(VecScale(fb->x_old, fb->f_scale));
    if (fb->f_scale != 1) PetscCall(VecScale(fb->grad_old, fb->f_scale));
    PetscCall(VecAXPY(fb->grad_old, -1., fb->x_old));
    PetscCall(VecAXPY(fb->workvec, -1., fb->workvec2));
    PetscCall(VecNorm(fb->grad_old, NORM_2, &gradnorm));
    gradnorm *= fb->f_scale;
    PetscCall(VecNorm(fb->workvec, NORM_2, &xnorm));

    fb->lip   = gradnorm / xnorm;
    fb->lip   = PetscMax(fb->lip, 1.e-6);
    tao->step = 2. / fb->lip / 10;

    PetscCall(PetscRandomDestroy(&rctx));
  } else {
    tao->step = 1. / fb->lip;
  }

  fb->step_old = tao->step;
  tao->reason  = TAO_CONTINUE_ITERATING;

  if (fb->use_accel) {
    PetscCall(PetscCitationsRegister(fasta_citation, &fasta_cited));
    PetscCall(VecCopy(tao->solution, fb->x_old));
  }

  if (fb->use_adapt) PetscCall(PetscCitationsRegister(adapgm_citation, &adapgm_cited));
  PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, tao->solution, &f, tao->gradient));
  if (fb->f_scale != 1) f *= fb->f_scale;
  if (fb->f_scale != 1) PetscCall(VecScale(tao->gradient, fb->f_scale));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    if (!fb->use_accel) {
      PetscCall(VecCopy(tao->solution, fb->x_old));
      PetscCall(VecCopy(tao->gradient, fb->grad_old));
    }

    /* Backtrackig PG stepsize scaling */
    if (!fb->use_adapt && tao->linesearch->max_funcs > 0) tao->step *= fb->xi;
    /* Note: DMTaoApplyProximalMap's scale is 1/(2*step) */
    PetscCall(VecWAXPY(fb->dualvec, -tao->step, tao->gradient, tao->solution));
    PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, tao->step * fb->prox_scale, fb->dualvec, tao->solution, PETSC_FALSE));
    tao->nproxs++;

    /* -tao_ls_max_funcs 0 -> no linesearch, but constant stepsize
       In this case, constant stepsize needs to be properly chosen for the algorithm to converge.
       -tao_ls_PS_memory_size  1 -> monotonic linesearch
       -tao_ls_PS_memory_size >1 -> nonmonotonic linesearch */
    if (!fb->use_adapt && tao->linesearch->max_funcs > 0) {
      fb->step_old = tao->step;
      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, tao->step));
      PetscCall(TaoLineSearchApply(tao->linesearch, fb->x_old, &f, tao->gradient, tao->solution, &tao->step, &ls_status));
      PetscCall(TaoAddLineSearchCounts(tao));
      /* Linesearch failure. Abort */
      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        tao->step   = 0.;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      }
      PetscCall(TaoLineSearchGetStepLength(tao->linesearch, &tao->step));
    }

    /* Post-processings */
    /* Fixed PGM  and adaPGM */
    if (fb->use_adapt) {
      PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, tao->solution, &f, tao->gradient));
      if (fb->f_scale != 1) f *= fb->f_scale;
      if (fb->f_scale != 1) PetscCall(VecScale(tao->gradient, fb->f_scale));
    }

    PetscCall(DMTaoComputeObjective(fb->proxterm, tao->solution, &f_prox));
    f_prox *= fb->prox_scale;
    PetscCall(TaoFB_ComputeResidual_And_LogConv_Private(tao, f + f_prox));

    if (!fb->use_accel && !fb->use_adapt) {
      /* fixed and backtracking PGM */
      PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, tao->solution, &f, tao->gradient));
      if (fb->f_scale != 1) f *= fb->f_scale;
      if (fb->f_scale != 1) PetscCall(VecScale(tao->gradient, fb->f_scale));
    } else if (fb->use_accel) {
      /* Nesterov-type */
      fb->t_fista_old = fb->t_fista;
      fb->t_fista     = (1. + PetscSqrtReal(1. + 4. * fb->t_fista_old * fb->t_fista_old)) / 2.;
      fb->fista_beta  = (fb->t_fista_old - 1) / (fb->t_fista);

      PetscCall(VecCopy(tao->solution, fb->x_old));
      PetscCall(VecCopy(tao->gradient, fb->grad_old));
      PetscCall(VecAXPBY(tao->solution, -fb->fista_beta, 1 + fb->fista_beta, fb->x_old));
      PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, tao->solution, &f, tao->gradient));
      if (fb->f_scale != 1) f *= fb->f_scale;
      if (fb->f_scale != 1) PetscCall(VecScale(tao->gradient, fb->f_scale));
    } else if (fb->use_adapt) {
      PetscCall(TaoFB_ADAPGM_Update_Stepsize_Private(tao));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_FB(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Forward backward problem that solves f(x)+g(x), where you have gradient of f(x), and proximal operator of g(x).");
  PetscCall(PetscOptionsReal("-tao_fb_initial_step", "Initial stepsize for forward-backward algorithm", "", tao->step, &tao->step, NULL));
  PetscCall(PetscOptionsReal("-tao_fb_ls_scale", "Scaling parameter for backtracking proximal gradient", "", fb->xi, &fb->xi, NULL));
  PetscCall(PetscOptionsBool("-tao_fb_accel", "Use Acceleration (Nesterov-type)", "", fb->use_accel, &fb->use_accel, NULL));
  PetscCall(PetscOptionsBool("-tao_fb_adaptive", "Use adaptive stepsize (adaPGM)", "", fb->use_adapt, &fb->use_adapt, NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_FB(Tao tao, PetscViewer viewer)
{
  DMTao     tdm;
  PetscBool isascii;
  TAO_FB   *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Backtracking linesearch scaling parameter: xi=%g\n", (double)fb->xi));
    if (fb->use_accel) PetscCall(PetscViewerASCIIPrintf(viewer, "Using Nesterov-type acceleration\n"));
    else if (fb->use_adapt) PetscCall(PetscViewerASCIIPrintf(viewer, "Using adaPGM-type adaptive stepsize\n"));
    if (fb->smoothterm) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Smooth Term:\n"));
      PetscCall(DMGetDMTao(fb->smoothterm, &tdm));
      PetscCall(DMTaoView(tdm, viewer));
    }
    if (fb->proxterm) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Proximal Term:\n"));
      PetscCall(DMGetDMTao(fb->proxterm, &tdm));
      PetscCall(DMTaoView(tdm, viewer));
    }
    if (fb->reg) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Regularizer Term:\n"));
      PetscCall(DMGetDMTao(fb->reg, &tdm));
      PetscCall(DMTaoView(tdm, viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_FB(Tao tao)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!fb->workvec) PetscCall(VecDuplicate(tao->solution, &fb->workvec));
  if (!fb->workvec2) PetscCall(VecDuplicate(tao->solution, &fb->workvec2));
  if (!fb->dualvec) PetscCall(VecDuplicate(tao->solution, &fb->dualvec));
  if (!fb->x_old) PetscCall(VecDuplicate(tao->solution, &fb->x_old));
  if (!fb->grad_old) PetscCall(VecDuplicate(tao->solution, &fb->grad_old));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)tao), &fb->reg));
  PetscCall(DMTaoSetType(fb->reg, DMTAOL2));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  if (fb->smoothterm) PetscCall(TaoLineSearchUseDM(tao->linesearch, fb->smoothterm));
  PetscCall(TaoLineSearchSetProxAndLinearMap(tao->linesearch, fb->proxterm, fb->prox_scale, fb->reg, NULL, 0.));

  tao->linesearch->ops->preapply   = TaoFB_LineSearch_PreApply_Private;
  tao->linesearch->ops->postapply  = TaoFB_LineSearch_PostApply_Private;
  tao->linesearch->ops->update     = TaoFB_LineSearch_Update_Private;
  tao->linesearch->ops->postupdate = TaoFB_LineSearch_PostUpdate_Private;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_FB(Tao tao)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&fb->workvec));
  PetscCall(VecDestroy(&fb->workvec2));
  PetscCall(VecDestroy(&fb->dualvec));
  PetscCall(VecDestroy(&fb->x_old));
  PetscCall(VecDestroy(&fb->grad_old));
  PetscCall(DMDestroy(&fb->reg));
  PetscCall(DMDestroy(&fb->smoothterm));
  PetscCall(DMDestroy(&fb->proxterm));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSSetNonSmoothTerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSSetSmoothTerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSUseAdaptiveStep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSUseAcceleration_C", NULL));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     TAOFB -   Forward-Backward proximal splitting algorithm.

   Options Database Keys:
+      -tao_fb_ls_scale <r> - Linesearch scaling parameter
.      -tao_fb_accel - Use Nesterov-type acceleration
-      -tao_fb_adaptive - Use adaPGM-type adaptive stepsize

   Level: beginner

   Note:
   See {cite}`goldstein2015fasta`, {cite}`latafat2024convergence`.

.seealso: `Tao`, `TaoType, `TAOCV`
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_FB(Tao tao)
{
  TAO_FB     *fb;
  const char *ls_type = TAOLINESEARCHPS;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fb));

  tao->gttol = 1.e-8;

  tao->ops->destroy         = TaoDestroy_FB;
  tao->ops->setup           = TaoSetUp_FB;
  tao->ops->setfromoptions  = TaoSetFromOptions_FB;
  tao->ops->view            = TaoView_FB;
  tao->ops->solve           = TaoSolve_FB;
  tao->ops->convergencetest = TaoDefaultConvergenceTest;

  PetscCall(TaoParametersInitialize(tao));
  PetscObjectParameterSetDefault(tao, max_it, 1000);

  tao->data = (void *)fb;

  fb->lip         = 0.;
  fb->f_scale     = 1.;
  fb->prox_scale  = 1.;
  fb->t_fista     = 1;
  fb->t_fista_old = 1;
  fb->fista_beta  = 0.;
  fb->xi          = 1.;
  fb->smoothterm  = NULL;
  fb->proxterm    = NULL;
  fb->use_accel   = PETSC_TRUE;
  fb->use_adapt   = PETSC_FALSE;

  /* Non-monotonic linesearch
   *
   * \hat{f}_k = max(f_{k-1},f_{k-2}, ... , f_{k-min(M,k)}), where M is linesearch history size
   * f(x_{k+1}) < \hat{f}_k + \langle x_{k+1} - x_k, grad_f(x_k) \rangle + (1/(2 step)) |x_{k+1} - x_k |^2
   *
   */
  PetscCall(TaoLineSearchCreate(PetscObjectComm((PetscObject)tao), &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, ls_type));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSSetNonSmoothTerm_C", TaoPSSetNonSmoothTerm_FB));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSSetSmoothTerm_C", TaoPSSetSmoothTerm_FB));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSUseAdaptiveStep_C", TaoPSUseAdaptiveStep_FB));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSUseAcceleration_C", TaoPSUseAcceleration_FB));
  PetscFunctionReturn(PETSC_SUCCESS);
}
