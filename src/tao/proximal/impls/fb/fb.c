#include <../src/tao/proximal/impls/fb/fb.h> /*I "petsctao.h" I*/
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/pslinesearch/pslinesearch.h>

static PetscBool  fasta_cited       = PETSC_FALSE;
static PetscBool  adapgm_cited      = PETSC_FALSE;
static const char fasta_citation[]  = "@article{goldstein2015fasta,\n"
                                      "  title={FASTA: A generalized implementation of forward-backward splitting},\n"
                                      "  author={Goldstein, Tom and Studer, Christoph and Baraniuk, Richard},\n"
                                      "  journal={arXiv preprint arXiv:1501.04979},\n"
                                      "  year={2015}\n"
                                      "}\n";
static const char adapgm_citation[] = "@inproceedings{latafat2024convergence,\n"
                                      "  title={On the convergence of adaptive first order methods: proximal gradient and alternating minimization algorithms},\n"
                                      "  author={Latafat, Puya and Themelis, Andreas and Patrinos, Panagiotis},\n"
                                      "  booktitle={6th Annual Learning for Dynamics and Control Conference},\n"
                                      "  pages={197--208},\n"
                                      "  year={2024},\n"
                                      "  organization={PMLR}\n"
                                      "}\n";

/* Evaluate the smooth term f (scaled by its TaoAddTerm() weight) and count it against the Tao objective */
static PetscErrorCode TaoFBComputeSmooth_Private(Tao tao, Vec x, PetscReal *f, Vec g)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  if (f && g) {
    PetscCall(TaoTermMappingComputeObjectiveAndGradient(&fb->f_term, x, fb->f_param, INSERT_VALUES, f, g));
    tao->objective_term.term->nobjgrad++;
  } else if (f) {
    PetscCall(TaoTermMappingComputeObjective(&fb->f_term, x, fb->f_param, INSERT_VALUES, f));
    tao->objective_term.term->nobj++;
  } else {
    PetscCall(TaoTermMappingComputeGradient(&fb->f_term, x, fb->f_param, INSERT_VALUES, g));
    tao->objective_term.term->ngrad++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Step-size rule of adaPGM (Latafat, Themelis, Patrinos 2024), see stepsize(rule::OurRule, ...) in AdaProx.jl */
static PetscErrorCode TaoFB_ADAPGM_Update_Stepsize_Private(Tao tao)
{
  TAO_FB     *fb   = (TAO_FB *)tao->data;
  PetscReal   step = tao->step, dot, graddiff, xdiff, L, C, min1, min2, temp;
  PetscScalar dp;

  PetscFunctionBegin;
  /* workvec: grad f(x_{k+1}) - grad f(x_k), workvec2: x_{k+1} - x_k */
  PetscCall(VecWAXPY(fb->workvec, -1., fb->grad_old, tao->gradient));
  PetscCall(VecWAXPY(fb->workvec2, -1., fb->x_old, tao->solution));
  PetscCall(VecDotNorm2(fb->workvec2, fb->workvec, &dp, &graddiff));
  PetscCall(VecNorm(fb->workvec2, NORM_2, &xdiff));
  dot   = PetscRealPart(dp);
  xdiff = xdiff * xdiff;
  /* Guard the 0/0 cases (unchanged gradient or repeated iterate) like nan_to_zero() in AdaProx.jl */
  L    = (xdiff > 0) ? dot / xdiff : 0;
  C    = (dot != 0) ? graddiff / dot : 0;
  min1 = step * PetscSqrtReal(1 + step / fb->step_old);
  temp = PetscMax(step * L * (step * C - 1), 0);
  min2 = (temp == 0) ? PETSC_INFINITY : step / (2 * PetscSqrtReal(temp));

  fb->step_old = step;
  tao->step    = PetscMin(min1, min2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFB_ComputeResidual_And_LogConv_Private(Tao tao, PetscReal f)
{
  TAO_FB   *fb = (TAO_FB *)tao->data;
  PetscReal gradnorm, mapnorm;

  PetscFunctionBegin;
  /* The residual is the norm of the gradient mapping |x_{k+1} - x_k| / step. As in FASTA it is normalized
     by max(|grad f(x_k)|, |x_{k+1} - (x_k - step grad f(x_k))| / step) for the relative test -tao_gttol */
  PetscCall(VecWAXPY(fb->workvec2, -1., fb->x_old, tao->solution));
  PetscCall(VecNorm(fb->workvec2, NORM_2, &tao->residual));
  PetscCall(VecAXPY(fb->workvec2, tao->step, fb->grad_old));
  PetscCall(VecNorm(fb->workvec2, NORM_2, &mapnorm));
  PetscCall(VecNorm(fb->grad_old, NORM_2, &gradnorm));
  tao->residual /= tao->step;
  tao->gnorm0 = PetscMax(gradnorm, mapnorm / tao->step) + PETSC_SQRT_MACHINE_EPSILON;

  tao->niter++;
  PetscCall(TaoLogConvergenceHistory(tao, f, tao->residual, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, tao->residual, 0.0, tao->step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  if (tao->reason == TAO_CONTINUE_ITERATING && PetscIsInfOrNanReal(tao->residual)) {
    PetscCall(PetscInfo(tao, "Failed to converge, residual is Inf or NaN\n"));
    tao->reason = TAO_DIVERGED_NAN;
  }
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
  PetscInt                     nfeval, ngeval, nfgeval;
  PetscBool                    use_ls;
  TaoLineSearchConvergedReason ls_status;

  PetscFunctionBegin;
  PetscCheck(fb->initial_step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Initial stepsize cannot be negative");
  PetscCheck(fb->xi >= 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Backtracking scale factor must be at least 1");
  PetscCheck(!(fb->use_accel && fb->use_adapt), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoFB only supports either acceleration or adaptive step, not both");
  use_ls = (PetscBool)(!fb->use_adapt && tao->linesearch->max_funcs > 0);

  PetscCall(TaoTermGetLipschitz(fb->f_term.term, &fb->lip));
  fb->lip *= fb->f_term.scale;
  if (fb->initial_step > 0) tao->step = fb->initial_step;
  else if (fb->lip == 0) {
    /* Approximate the Lipschitz constant of grad f from two random points, as FASTA does */
    PetscReal   gradnorm, xnorm;
    PetscRandom rctx;

    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(fb->workvec, rctx));
    PetscCall(VecSetRandom(fb->workvec2, rctx));
    PetscCall(TaoFBComputeSmooth_Private(tao, fb->workvec, NULL, fb->x_old));
    PetscCall(TaoFBComputeSmooth_Private(tao, fb->workvec2, NULL, fb->grad_old));
    PetscCall(VecAXPY(fb->grad_old, -1., fb->x_old));
    PetscCall(VecAXPY(fb->workvec, -1., fb->workvec2));
    PetscCall(VecNorm(fb->grad_old, NORM_2, &gradnorm));
    PetscCall(VecNorm(fb->workvec, NORM_2, &xnorm));
    PetscCall(PetscRandomDestroy(&rctx));
    fb->lip   = PetscMax(gradnorm / xnorm, 1.e-6);
    tao->step = 2. / fb->lip / 10;
  } else tao->step = 1. / fb->lip;

  fb->step_old    = tao->step;
  fb->t_fista     = 1.;
  fb->t_fista_old = 1.;
  fb->fista_beta  = 0.;
  tao->reason     = TAO_CONTINUE_ITERATING;
  if (fb->use_accel) {
    PetscCall(PetscCitationsRegister(fasta_citation, &fasta_cited));
    PetscCall(VecCopy(tao->solution, fb->x_accel));
  }
  if (fb->use_adapt) PetscCall(PetscCitationsRegister(adapgm_citation, &adapgm_cited));

  /* x_old is the base point of the forward step: x_k, or the extrapolated point y_k with acceleration */
  PetscCall(VecCopy(tao->solution, fb->x_old));
  PetscCall(TaoFBComputeSmooth_Private(tao, fb->x_old, &f, tao->gradient));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    PetscCall(VecCopy(tao->gradient, fb->grad_old));
    if (use_ls) tao->step *= fb->xi;
    /* Forward-backward step: solution = prox_{step g}(x_old - step grad f(x_old)) */
    PetscCall(VecWAXPY(fb->dualvec, -tao->step, fb->grad_old, fb->x_old));
    PetscCall(TaoTermProximalMap(fb->g_term.term, fb->g_param, tao->step * fb->g_term.scale, NULL, fb->dualvec, 1.0, tao->solution));
    if (use_ls) {
      /* Shrink the step (and recompute the solution) until f(solution) <= R + <grad f(x_old), solution - x_old> + |solution - x_old|^2 / (2 step) */
      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, tao->step));
      PetscCall(TaoLineSearchApply(tao->linesearch, fb->x_old, &f, fb->grad_old, tao->solution, &tao->step, &ls_status));
      PetscCall(TaoLineSearchGetNumberFunctionEvaluations(tao->linesearch, &nfeval, &ngeval, &nfgeval));
      tao->objective_term.term->nobj += nfeval;
      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        PetscCall(PetscInfo(tao, "Line search failed: %s\n", TaoLineSearchConvergedReasons[ls_status]));
        PetscCall(VecCopy(fb->use_accel ? fb->x_accel : fb->x_old, tao->solution));
        tao->reason = TAO_DIVERGED_LS_FAILURE;
        break;
      }
    }

    /* Objective at the new iterate; with acceleration the gradient is needed at the extrapolated point instead */
    if (fb->use_accel) {
      if (!use_ls) PetscCall(TaoFBComputeSmooth_Private(tao, tao->solution, &f, NULL));
    } else PetscCall(TaoFBComputeSmooth_Private(tao, tao->solution, &f, tao->gradient));
    PetscCall(TaoTermMappingComputeObjective(&fb->g_term, tao->solution, fb->g_param, INSERT_VALUES, &f_prox));
    PetscCall(TaoFB_ComputeResidual_And_LogConv_Private(tao, f + f_prox));
    if (tao->reason != TAO_CONTINUE_ITERATING) break;

    if (fb->use_accel) {
      /* Nesterov-type acceleration: next base point y_{k+1} = x_{k+1} + beta (x_{k+1} - x_k) */
      fb->t_fista_old = fb->t_fista;
      fb->t_fista     = (1. + PetscSqrtReal(1. + 4. * fb->t_fista_old * fb->t_fista_old)) / 2.;
      fb->fista_beta  = (fb->t_fista_old - 1.) / fb->t_fista;
      PetscCall(VecWAXPY(fb->workvec2, -1., fb->x_accel, tao->solution));
      PetscCall(VecCopy(tao->solution, fb->x_accel));
      PetscCall(VecWAXPY(fb->x_old, fb->fista_beta, fb->workvec2, tao->solution));
      PetscCall(TaoFBComputeSmooth_Private(tao, fb->x_old, &f, tao->gradient));
    } else {
      if (fb->use_adapt) PetscCall(TaoFB_ADAPGM_Update_Stepsize_Private(tao));
      PetscCall(VecCopy(tao->solution, fb->x_old));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_FB(Tao tao, PetscOptionItems PetscOptionsObject)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Forward backward problem that solves f(x)+g(x), where you have gradient of f(x), and proximal operator of g(x).");
  PetscCall(PetscOptionsReal("-tao_fb_initial_step", "Initial stepsize for forward-backward algorithm (0 means the inverse Lipschitz constant)", "", fb->initial_step, &fb->initial_step, NULL));
  PetscCall(PetscOptionsReal("-tao_fb_ls_scale", "Scaling parameter for backtracking proximal gradient", "", fb->xi, &fb->xi, NULL));
  PetscCall(PetscOptionsBool("-tao_fb_accel", "Use Acceleration (Nesterov-type)", "", fb->use_accel, &fb->use_accel, NULL));
  PetscCall(PetscOptionsBool("-tao_fb_adaptive", "Use adaptive stepsize (adaPGM)", "", fb->use_adapt, &fb->use_adapt, NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_FB(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;
  TAO_FB   *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Backtracking linesearch scaling parameter: xi=%g\n", (double)fb->xi));
    if (fb->use_accel) PetscCall(PetscViewerASCIIPrintf(viewer, "Using Nesterov-type acceleration\n"));
    else if (fb->use_adapt) PetscCall(PetscViewerASCIIPrintf(viewer, "Using adaPGM-type adaptive stepsize\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "f Term:\n"));
    PetscCall(TaoTermView(fb->f_term.term, viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "g Term:\n"));
    PetscCall(TaoTermView(fb->g_term.term, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFBSetUpTerms(Tao tao, TaoTermMapping *f_term, TaoTermMapping *g_term)
{
  TAO_FB   *fb = (TAO_FB *)tao->data;
  PetscBool is_sum, found_f = PETSC_FALSE, found_g = PETSC_FALSE;
  PetscInt  i, nterms;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tao->objective_term.term, TAOTERMSUM, &is_sum));
  PetscCheck(is_sum, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOFB requires an objective sum with terms prefixed \"f_\" and \"g_\"");
  PetscCall(TaoTermSumGetNumberTerms(tao->objective_term.term, &nterms));
  for (i = 0; i < nterms; i++) {
    TaoTerm     term;
    PetscReal   scale;
    Mat         map;
    const char *prefix;
    PetscBool   is_f, is_g;

    PetscCall(TaoTermSumGetTerm(tao->objective_term.term, i, &prefix, &scale, &term, &map));
    PetscCall(PetscStrcmp(prefix, "f_", &is_f));
    PetscCall(PetscStrcmp(prefix, "g_", &is_g));
    PetscCheck(!(is_f && found_f), PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOFB objective contains more than one term prefixed \"f_\"");
    PetscCheck(!(is_g && found_g), PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOFB objective contains more than one term prefixed \"g_\"");
    if (is_f) {
      PetscCall(TaoTermMappingSetData(f_term, prefix, scale, term, map));
      if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumParameters(tao->objective_parameters, i, &fb->f_param));
      found_f = PETSC_TRUE;
    } else if (is_g) {
      PetscCall(TaoTermMappingSetData(g_term, prefix, scale, term, map));
      if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumParameters(tao->objective_parameters, i, &fb->g_param));
      found_g = PETSC_TRUE;
    }
  }
  PetscCheck(found_f, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOFB objective has no smooth term prefixed \"f_\"");
  PetscCheck(found_g, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOFB objective has no proximal term prefixed \"g_\"");
  PetscCheck(!f_term->map && !g_term->map, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "TAOFB currently requires identity maps for its \"f_\" and \"g_\" terms");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_FB(Tao tao)
{
  TAO_FB   *fb = (TAO_FB *)tao->data;
  PetscBool is_ps;

  PetscFunctionBegin;
  PetscCall(TaoFBSetUpTerms(tao, &fb->f_term, &fb->g_term));
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!fb->workvec) PetscCall(VecDuplicate(tao->solution, &fb->workvec));
  if (!fb->workvec2) PetscCall(VecDuplicate(tao->solution, &fb->workvec2));
  if (!fb->dualvec) PetscCall(VecDuplicate(tao->solution, &fb->dualvec));
  if (!fb->x_old) PetscCall(VecDuplicate(tao->solution, &fb->x_old));
  if (!fb->grad_old) PetscCall(VecDuplicate(tao->solution, &fb->grad_old));
  if (!fb->x_accel) PetscCall(VecDuplicate(tao->solution, &fb->x_accel));

  PetscCall(PetscObjectTypeCompare((PetscObject)tao->linesearch, TAOLINESEARCHPS, &is_ps));
  PetscCheck(is_ps, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "TAOFB requires the %s line search, not %s", TAOLINESEARCHPS, ((PetscObject)tao->linesearch)->type_name);
  PetscCall(TaoPSLineSearchSetTerms(tao->linesearch, fb->f_term, fb->f_param, fb->g_term, fb->g_param));
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
  PetscCall(VecDestroy(&fb->x_accel));
  PetscCall(TaoTermMappingReset(&fb->f_term));
  PetscCall(TaoTermMappingReset(&fb->g_term));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSUseAdaptiveStep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSUseAcceleration_C", NULL));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOFB - Forward-backward (proximal gradient) splitting for min f(x) + g(x), where f is smooth and g has a computable proximal map.

  Options Database Keys:
+ -tao_fb_initial_step step - initial step size; the default is the inverse Lipschitz constant of the gradient of f, estimated if unknown
. -tao_fb_ls_scale scale    - factor by which the step size grows before each backtracking line search
. -tao_fb_accel             - use Nesterov-type acceleration (FISTA)
- -tao_fb_adaptive          - use the adaptive step size of adaPGM

  Level: beginner

  Notes:
  The objective must be a sum of two terms added with `TaoAddTerm()` using the prefixes "f_" for the smooth term and "g_" for the proximal term.
  The line search is `TAOLINESEARCHPS`; use `-tao_ls_max_funcs 0` for a fixed step size and `-tao_ls_ps_memory_size` for a nonmonotone line search.
  See {cite}`goldstein2015fasta` and {cite}`latafat2024convergence`.

.seealso: `Tao`, `TaoType`, `TaoAddTerm()`, `TAOLINESEARCHPS`
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_FB(Tao tao)
{
  TAO_FB *fb;

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

  fb->lip          = 0.;
  fb->initial_step = 0.;
  fb->t_fista      = 1.;
  fb->t_fista_old  = 1.;
  fb->fista_beta   = 0.;
  fb->xi           = 1.;
  fb->use_accel    = PETSC_TRUE;
  fb->use_adapt    = PETSC_FALSE;

  PetscCall(TaoLineSearchCreate(PetscObjectComm((PetscObject)tao), &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, TAOLINESEARCHPS));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSUseAdaptiveStep_C", TaoPSUseAdaptiveStep_FB));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoPSUseAcceleration_C", TaoPSUseAcceleration_FB));
  PetscFunctionReturn(PETSC_SUCCESS);
}
