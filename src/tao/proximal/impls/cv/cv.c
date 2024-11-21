#include <../src/tao/proximal/impls/cv/cv.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petscdm.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/pslinesearch/pslinesearch.h>

static PetscBool cited = PETSC_FALSE;

static const char citation[] = "@article{latafat2024adaptive,\n"
                               "title={Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient},\n"
                               "author={Latafat, Puya and Themelis, Andreas and Stella, Lorenzo and Patrinos, Panagiotis},\n"
                               "journal={Mathematical Programming},\n"
                               "pages={1--39},\n"
                               "year={2024},\n"
                               "publisher={Springer}\n";

static PetscErrorCode TaoCV_LineSearch_PreApply_Private(TaoLineSearch ls, Vec in, PetscReal *f, Vec out, Vec g)
{
  PetscReal         grad_x_dot, xdiffnorm, graddiffnorm;
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;
  TAO_CV           *cv   = (TAO_CV *)ls->tao->data;

  PetscFunctionBegin;
  armP->xi = cv->pd_ratio * ls->tao->step * cv->eta * (1 + ls->tao->gatol);
  armP->xi *= armP->xi;

  PetscCall(VecWAXPY(cv->workvec, -1., cv->grad_old, ls->tao->gradient));
  PetscCall(VecWAXPY(cv->workvec2, -1., cv->x_old, ls->tao->solution));
  PetscCall(VecTDot(cv->workvec, cv->workvec2, &grad_x_dot));
  PetscCall(VecNorm(cv->workvec2, NORM_2, &xdiffnorm));
  PetscCall(VecNorm(cv->workvec, NORM_2, &graddiffnorm));

  armP->L = (xdiffnorm == 0) ? 0 : grad_x_dot / (xdiffnorm * xdiffnorm);
  armP->C = (grad_x_dot == 0) ? 0 : (graddiffnorm * graddiffnorm) / grad_x_dot;
  armP->D = ls->tao->step * armP->L * (ls->tao->step * armP->C - 1);

  cv->eta *= cv->R;
  /* For TAOCV, linesearch needs to go at least once */
  armP->cert = PETSC_INFINITY;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoCV_LineSearch_Update_Private(TaoLineSearch ls, Vec in, PetscReal *f, Vec out, Vec g)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;
  TAO_CV           *cv   = (TAO_CV *)ls->tao->data;
  PetscReal         min1, min2, min3, rho, temp, temp2, temp3;

  PetscFunctionBegin;
  min1           = ls->tao->step * PetscSqrtReal(1 + ls->tao->step / cv->step_old);
  min2           = 1 / (2 * cv->Theta * cv->pd_ratio * cv->eta);
  temp           = 1 - 4 * armP->xi;
  temp2          = cv->pd_ratio * cv->eta * ls->tao->step; //Unlike no linesearch, this "xi" uses updated norm estimate
  temp3          = PetscSqrtReal(armP->D * armP->D + temp * temp2 * temp2);
  min3           = ls->tao->step * PetscSqrtReal(temp / (2 * (1 + ls->tao->gatol) * (temp3 + armP->D)));
  armP->step_new = PetscMin(min1, PetscMin(min2, min3));
  rho            = armP->step_new / ls->tao->step;
  cv->sigma      = cv->pd_ratio * cv->pd_ratio * armP->step_new;

  /* dualvec_work: w = y + sigma *((1+rho) * Ax - rho * Ax_old) */
  PetscCall(VecWAXPY(cv->dualvec_work, -cv->sigma * rho, cv->Ax_old, ls->tao->dualvec));
  PetscCall(VecAXPY(cv->dualvec_work, cv->sigma * (1 + rho), cv->Ax));

  armP->test_step = cv->sigma * cv->h_scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoCV_LineSearch_PostUpdate_Private(TaoLineSearch ls, Vec in, PetscReal *f, Vec out, Vec g)
{
  TaoLineSearch_PS *armP = (TaoLineSearch_PS *)ls->data;
  TAO_CV           *cv   = (TAO_CV *)ls->tao->data;
  PetscReal         norm1, norm2;

  PetscFunctionBegin;
  /* workvec : A^T * y_test */
  PetscCall(MatMultTranspose(cv->h_lmap, cv->dualvec_test, cv->workvec));
  /* norm1 = norm(ATy_test - ATy) */
  PetscCall(VecWAXPY(cv->workvec2, -1., cv->ATy, cv->workvec));
  PetscCall(VecNorm(cv->workvec2, NORM_2, &norm1));
  /* norm2 = norm(y_test - y) */
  PetscCall(VecWAXPY(cv->dualvec_work2, -1., cv->dualvec_test, ls->tao->dualvec));
  PetscCall(VecNorm(cv->dualvec_work2, NORM_2, &norm2));
  armP->cert = -cv->eta + norm1 / norm2;
  *f         = armP->cert;
  if (armP->cert <= ls->ftol) {
    cv->step_old  = ls->tao->step;
    ls->tao->step = armP->step_new;
    ls->step      = armP->step_new;
    PetscCall(VecCopy(cv->dualvec_test, ls->tao->dualvec));
    PetscCall(VecCopy(cv->workvec, cv->ATy));
    ls->reason = TAOLINESEARCH_SUCCESS;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  cv->eta *= cv->r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoCV_Stepsize_No_LS_Private(Tao tao)
{
  TAO_CV   *cv = (TAO_CV *)tao->data;
  PetscReal xi, grad_x_dot, xdiffnorm, graddiffnorm, L, C, D, min1, min2, min3, temp, temp2;

  PetscFunctionBegin;
  xi = cv->pd_ratio * tao->step * cv->h_lmap_norm;
  xi *= xi;

  PetscCall(VecWAXPY(cv->workvec, -1., cv->grad_old, tao->gradient));
  PetscCall(VecWAXPY(cv->workvec2, -1., cv->x_old, tao->solution));
  PetscCall(VecTDot(cv->workvec, cv->workvec2, &grad_x_dot));
  PetscCall(VecNorm(cv->workvec2, NORM_2, &xdiffnorm));
  PetscCall(VecNorm(cv->workvec, NORM_2, &graddiffnorm));

  L = (xdiffnorm == 0) ? 0 : grad_x_dot / (xdiffnorm * xdiffnorm);
  C = (grad_x_dot == 0) ? 0 : (graddiffnorm * graddiffnorm) / grad_x_dot;
  D = tao->step * L * (tao->step * C - 1);

  min1         = tao->step * PetscSqrtReal(1 + tao->step / cv->step_old);
  min2         = 1 / (2 * cv->Theta * cv->pd_ratio * cv->h_lmap_norm);
  temp         = 1 - 4 * xi * (1 + tao->gatol) * (1 + tao->gatol);
  temp2        = PetscSqrtReal(D * D + xi * temp);
  min3         = tao->step * PetscSqrtReal(temp / (2 * (1 + tao->gatol) * (temp2 + D)));
  cv->step_old = tao->step;
  tao->step    = PetscMin(min1, PetscMin(min2, min3));
  cv->sigma    = cv->pd_ratio * cv->pd_ratio * tao->step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_CV(Tao tao)
{
  TAO_CV                      *cv = (TAO_CV *)tao->data;
  TaoTerm                      fterm, gterm, hterm;
  PetscBool                    issum;
  PetscReal                    f, gnorm, rho, temp;
  PetscReal                    pri_res_norm, dual_res_norm, g_val, h_val;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;
  PetscCheck(tao->step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize cannot be negative");
  PetscCheck(cv->R <= 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Scale factor needs to be equal or less than 1");
  PetscCheck(cv->r > 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Backtracking factor needs to be greater than 1");
  PetscCall(PetscCitationsRegister(citation, &cited));

  /* Note: Assuming 1,2,3 index orders for f, g, h term. (0 is callbacks) */
  PetscCall(PetscObjectTypeCompare((PetscObject)tao->objective_term.term, TAOTERMSUM, &issum));
  PetscCheck(issum, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TAOCV Main objective term needs to be of type TAOTERMSUM");
  PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, 1, NULL, &cv->f_scale, &fterm, NULL));
  PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, 2, NULL, &cv->g_scale, &gterm, NULL));
  /* Note: h_lmap is non-owning to avoid unnecessary data movement */
  PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, 3, NULL, &cv->h_scale, &hterm, &cv->h_lmap));
  /* If a user wants f = NULL, use TAOTERMZERO */
  PetscCheck(fterm, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TAOCV f-term is NULL");
  PetscCheck(gterm, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TAOCV g-term is NULL");
  PetscCheck(hterm, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TAOCV h-term is NULL");
  PetscCheck(cv->f_scale >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TAOCV f-term scale cannot be negative");
  PetscCheck(cv->g_scale >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TAOCV g-term scale cannot be negative");
  PetscCheck(cv->h_scale >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TAOCV h-term scale cannot be negative");
  PetscCheck(cv->h_lmap, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Linear mapping cannot be NULL");

  /* If initial stepsize is zero, compute some estimate */
  if (tao->step == 0 && cv->h_lmap_norm > 0) tao->step = 1 / (2 * cv->Theta * cv->pd_ratio * cv->h_lmap_norm);
  else if (tao->step == 0 && cv->h_lmap_norm == 0) tao->step = 1 / (2 * cv->Theta * cv->pd_ratio * cv->eta);
  /* Checking whether the estimate is not too large */
  if (cv->h_lmap_norm > 0) {
    temp      = tao->step - 1 / (2 * cv->Theta * cv->pd_ratio * cv->h_lmap_norm);
    tao->step = (temp <= 0) ? tao->step : 1 / (2 * cv->Theta * cv->pd_ratio * cv->h_lmap_norm);

    if (temp > 0) PetscCall(PetscInfo(tao, "TAOCV initial stepsize too large. Using lower acceptable value\n"));
  } else {
    temp      = tao->step - 1 / (2 * cv->Theta * cv->pd_ratio * cv->eta);
    tao->step = (temp <= 0) ? tao->step : 1 / (2 * cv->Theta * cv->pd_ratio * cv->eta);

    if (temp > 0) PetscCall(PetscInfo(tao, "TAOCV initial stepsize too large. Using lower acceptable value\n"));
  }
  cv->sigma = tao->step * cv->pd_ratio * cv->pd_ratio;

  cv->step_old = tao->step;

  PetscCall(TaoTermObjectiveAndGradient(fterm, tao->solution, NULL, &f, tao->gradient));
  if (cv->f_scale != 1) f *= cv->f_scale;
  if (cv->f_scale != 1) PetscCall(VecScale(tao->gradient, cv->f_scale));
  PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  /* set initial dualvec, y as zero, so skip initial ATy computation */
  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(VecCopy(tao->solution, cv->x_old));
  PetscCall(VecCopy(tao->gradient, cv->grad_old));
  PetscCall(VecCopy(cv->Ax, cv->Ax_old));
  PetscCall(VecSet(cv->ATy, 0.));
  PetscCall(VecSet(tao->dualvec, 0.));
  PetscCall(MatMult(cv->h_lmap, tao->solution, cv->Ax));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* workvec: v = x - step * (grad_x + ATy) */
    PetscCall(VecWAXPY(cv->workvec, -tao->step, tao->gradient, tao->solution));
    PetscCall(VecAXPY(cv->workvec, -tao->step, cv->ATy));

    /* x = prog_g(v, step) */
    PetscCall(VecCopy(tao->solution, cv->x_old));
    PetscCall(TaoTermProximalMap(cv->g_term.term, cv->g_param, tao->step * cv->g_scale, cv->reg_term.term, cv->workvec, cv->reg_term.scale, tao->solution));
    /* update Ax, and grad */
    PetscCall(VecCopy(cv->Ax, cv->Ax_old));
    PetscCall(VecCopy(tao->gradient, cv->grad_old));
    PetscCall(MatMult(cv->h_lmap, tao->solution, cv->Ax));
    PetscCall(TaoTermObjectiveAndGradient(fterm, tao->solution, NULL, &f, tao->gradient));
    if (cv->f_scale != 1) f *= cv->f_scale;
    if (cv->f_scale != 1) PetscCall(VecScale(tao->gradient, cv->f_scale));
    /* workvec = (v - x)/step + grad_x + ATy */
    PetscCall(VecAXPY(cv->workvec, -1., tao->solution));
    PetscCall(VecScale(cv->workvec, 1 / tao->step));
    PetscCall(VecAXPBYPCZ(cv->workvec, 1., 1., 1., tao->gradient, cv->ATy));
    PetscCall(VecNorm(cv->workvec, NORM_2, &pri_res_norm));

    // update stepsize. no vanilla Condat-Vu
    if (tao->linesearch->max_funcs == 0) {
      /* No linesearch */
      PetscCall(TaoCV_Stepsize_No_LS_Private(tao));
      rho = tao->step / cv->step_old;

      /* dualvec_work: w = y + sigma *((1+rho) * Ax - rho * Ax_old) */
      PetscCall(VecWAXPY(cv->dualvec_work, -cv->sigma * rho, cv->Ax_old, tao->dualvec));
      PetscCall(VecAXPY(cv->dualvec_work, cv->sigma * (1 + rho), cv->Ax));

      /* dualvec: y = prox_h*(w, sigma) */
      PetscCall(TaoTermProximalMap(cv->h_cj_mapped_term.term, cv->h_param, cv->sigma * cv->h_cj_mapped_term.scale, cv->reg_term.term, cv->dualvec_work, cv->reg_term.scale, tao->dualvec));
    } else {
      // LS needs: x1, x0, grad_1, grad_0, Ax_old, dualvec(y), sigma, pd_ratio, Theta, eta
      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, tao->step));
      PetscCall(TaoLineSearchApply(tao->linesearch, cv->x_old, &f, tao->gradient, tao->solution, &tao->step, &ls_status));
      PetscCall(TaoAddLineSearchCounts(tao));
      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        tao->step   = 0.;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      }
    }

    PetscCall(VecAXPY(cv->dualvec_work, -1., tao->dualvec));
    PetscCall(VecScale(cv->dualvec_work, 1 / cv->sigma));
    PetscCall(VecAXPY(cv->dualvec_work, -1., cv->Ax));
    PetscCall(VecNorm(cv->dualvec_work, NORM_2, &dual_res_norm));

    tao->residual = PetscSqrtReal(pri_res_norm * pri_res_norm) + PetscSqrtReal(dual_res_norm * dual_res_norm);

    PetscCall(TaoTermObjective(cv->g_term.term, tao->solution, cv->g_param, &g_val));
    g_val *= cv->g_scale;
    PetscCall(TaoTermObjective(cv->h_term.term, cv->Ax, cv->h_param, &h_val));
    h_val *= cv->h_scale;
    /* convergence test */
    PetscCall(TaoLogConvergenceHistory(tao, f + g_val + h_val, tao->residual, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f + g_val + h_val, tao->residual, 0.0, tao->step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    tao->niter++;

    /* post-processing */
    if (tao->linesearch->max_funcs == 0) PetscCall(MatMultTranspose(cv->h_lmap, tao->dualvec, cv->ATy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_CV(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Forward backward problem that solves f(x)+g(Ax)+h(x), where you have gradient of f(x), and proximal operator of g(x) and h(x), and some linear map A.");
  PetscCall(PetscOptionsReal("-tao_cv_initial_step", "Initial stepsize for Condat-Vu algorithm", "", tao->step, &tao->step, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_norm_estimate_factor", "Scale factor for estimating operator norm of linear map. Must be <= 1.", "", cv->R, &cv->R, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_primal_dual_ratio", "Primal-dual Ratio factor for balancing solution. Must be non-negative", "", cv->pd_ratio, &cv->pd_ratio, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_backtrack_parameter", "Backtracking parameter r. Must be  >1.", "", cv->r, &cv->r, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_theta", "Stepsize scale parameter theta. Must be  >1+tol.", "", cv->Theta, &cv->Theta, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_norm_estimate", "Initial matrix norm estimate. Must be > 0.", "", cv->h_lmap_norm, &cv->h_lmap_norm, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_eta", "Initial linear map norm estimate. Must be nonnegative", "", cv->eta, &cv->eta, NULL));
  //TODO is this correct?
  PetscCall(PetscOptionsReal("-tao_cv_regularizer_scale", "Scale of HALFL2SQUARED regularizer. Must be nonnegative", "", cv->reg_term.scale, &cv->reg_term.scale, NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_CV(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;
  TAO_CV   *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Norm estimate factor: R=%g\n", (double)cv->R));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Primal-dual ratio: ratio=%g\n", (double)cv->pd_ratio));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Backtracking paramter: r=%g\n", (double)cv->r));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stepsize scale parameter: Theta=%g\n", (double)cv->Theta));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Using adaPDM-type adaptive stepsize\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "f Term:\n"));
    PetscCall(TaoTermView(cv->f_term.term, viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "g Term:\n"));
    PetscCall(TaoTermView(cv->g_term.term, viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "h Term:\n"));
    PetscCall(TaoTermView(cv->h_term.term, viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Regularizer Term:\n"));
    PetscCall(TaoTermView(cv->reg_term.term, viewer));
    //TODO should I view h_conjugate?
    {
      PetscCall(PetscViewerASCIIPrintf(viewer, "h Term Map:\n"));
      {
        PetscViewerFormat format;
        PetscBool         pop = PETSC_FALSE;

        PetscCall(PetscViewerGetFormat(viewer, &format));
        if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
          PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
          pop = PETSC_TRUE;
        }
        PetscCall(MatView(cv->h_term.map, viewer));
        if (pop) PetscCall(PetscViewerPopFormat(viewer));
      }
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoCVSetUpTerms(Tao tao, TaoMappedTerm *f_term, TaoMappedTerm *g_term, TaoMappedTerm *h_term, TaoMappedTerm *h_conjugate_term)
{
  TAO_CV   *cv = (TAO_CV *)tao->data;
  PetscBool is_sum;

  PetscFunctionBegin;
  // here we do logic to determine which terms in tao->objective_term.term correspond to which terms in the CV solver
  PetscCheck(g_term->map == NULL, PETSC_COMM_SELF, PETSC_ERR_SUP, "TAOCV: g term cannot have a nontrivial map");
  PetscCall(PetscObjectTypeCompare((PetscObject)tao->objective_term.term, TAOTERMSUM, &is_sum));
  if (is_sum) {
    PetscInt f_idx = 1;
    PetscInt g_idx = 2;
    PetscInt h_idx = 3;

    { // f
      TaoTerm     f;
      PetscReal   scale;
      Mat         map;
      const char *prefix;

      PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, f_idx, &prefix, &scale, &f, &map));
      PetscCall(TaoMappedTermSetData(f_term, prefix, scale, f, map));
      //No param for f term
    }
    { // g
      TaoTerm     g;
      PetscReal   scale;
      Mat         map;
      const char *prefix;

      PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, g_idx, &prefix, &scale, &g, &map));
      PetscCall(TaoMappedTermSetData(g_term, prefix, scale, g, map));
      if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumSubParameters(tao->objective_parameters, g_idx, &cv->g_param));
    }
    { // h
      TaoTerm     h;
      PetscReal   scale;
      Mat         map;
      const char *prefix;

      PetscCall(TaoTermSumGetSubterm(tao->objective_term.term, h_idx, &prefix, &scale, &h, &map));
      PetscCall(TaoMappedTermSetData(h_term, prefix, scale, h, map));
      PetscCall(TaoTermCreateConjugate(h, &cv->h_cj_term));
      PetscCall(TaoMappedTermSetData(h_conjugate_term, prefix, scale, cv->h_cj_term, map));
      if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumSubParameters(tao->objective_parameters, h_idx, &cv->h_param));
    }
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "TAOCV requires objective term to be of TAOTERMSUM type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_CV(Tao tao)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoCVSetUpTerms(tao, &cv->f_term, &cv->g_term, &cv->h_term, &cv->h_cj_mapped_term));
  /* sol sized vectors */
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!cv->workvec) PetscCall(VecDuplicate(tao->solution, &cv->workvec));
  if (!cv->workvec2) PetscCall(VecDuplicate(tao->solution, &cv->workvec2));
  if (!cv->x_old) PetscCall(VecDuplicate(tao->solution, &cv->x_old));
  if (!cv->grad_old) PetscCall(VecDuplicate(tao->solution, &cv->grad_old));
  if (!cv->ATy) PetscCall(VecDuplicate(tao->solution, &cv->ATy));
  /* dual sized vectors */
  if (!cv->Ax) PetscCall(MatCreateVecs(cv->h_term.map, NULL, &cv->Ax));
  if (!cv->Ax_old) PetscCall(MatCreateVecs(cv->h_term.map, NULL, &cv->Ax_old));
  if (!tao->dualvec) PetscCall(MatCreateVecs(cv->h_term.map, NULL, &tao->dualvec));
  if (!cv->dualvec_test || !cv->lmap_norm_set) PetscCall(MatCreateVecs(cv->h_term.map, NULL, &cv->dualvec_test));
  if (!cv->dualvec_work) PetscCall(MatCreateVecs(cv->h_term.map, NULL, &cv->dualvec_work));
  if (!cv->dualvec_work2) PetscCall(MatCreateVecs(cv->h_term.map, NULL, &cv->dualvec_work2));

  tao->linesearch->ops->preapply   = TaoCV_LineSearch_PreApply_Private;
  tao->linesearch->ops->update     = TaoCV_LineSearch_Update_Private;
  tao->linesearch->ops->postupdate = TaoCV_LineSearch_PostUpdate_Private;

  PetscCall(TaoPSLineSearchSetDualWorkvec(tao->linesearch, cv->dualvec_work));
  PetscCall(TaoPSLineSearchSetDualTestvec(tao->linesearch, cv->dualvec_test));
  PetscCall(TaoPSLineSearchSetRegularizerTerm(tao->linesearch, cv->reg_term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_CV(Tao tao)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cv->workvec));
  PetscCall(VecDestroy(&cv->workvec2));
  PetscCall(VecDestroy(&cv->x_old));
  PetscCall(VecDestroy(&cv->grad_old));
  PetscCall(VecDestroy(&cv->Ax));
  PetscCall(VecDestroy(&cv->Ax_old));
  PetscCall(VecDestroy(&cv->ATy));
  PetscCall(VecDestroy(&cv->dualvec_work));
  PetscCall(VecDestroy(&cv->dualvec_work2));
  PetscCall(VecDestroy(&cv->dualvec_test));
  PetscCall(TaoTermDestroy(&cv->h_cj_term));
  PetscCall(TaoMappedTermReset(&cv->reg_term));
  PetscCall(TaoMappedTermReset(&cv->f_term));
  PetscCall(TaoMappedTermReset(&cv->g_term));
  PetscCall(TaoMappedTermReset(&cv->h_term));
  PetscCall(TaoMappedTermReset(&cv->h_cj_mapped_term));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     TAOCV -   Condat-Vu proximal splitting algorithm.

   Options Database Keys:
+      -tao_cv_norm_estimate_factor <r> - Matrix norm estimate factor
.      -tao_cv_primal_dual_ratio <r> - Primal-dual ratio factor
.      -tao_cv_backtracking_parameter <r> - Linesearch backtracking parameter
-      -tao_cv_theta <r> - Stepsize scaling parameter

   Level: beginner

   Note:
   See {cite}`latafat2024adaptive`.

.seealso: `Tao`, `TaoType`, `TAOFB`
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_CV(Tao tao)
{
  TAO_CV     *cv;
  const char *ls_type = TAOLINESEARCHPS;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cv));

  tao->ops->destroy         = TaoDestroy_CV;
  tao->ops->setup           = TaoSetUp_CV;
  tao->ops->setfromoptions  = TaoSetFromOptions_CV;
  tao->ops->view            = TaoView_CV;
  tao->ops->solve           = TaoSolve_CV;
  tao->ops->convergencetest = TaoDefaultConvergenceTest;

  PetscCall(TaoParametersInitialize(tao));
  PetscObjectParameterSetDefault(tao, max_it, 1000);

  tao->data = (void *)cv;

  cv->h_scale     = 1.;
  cv->g_scale     = 1.;
  cv->gnorm_norm  = 0.;
  cv->R           = 0.95;
  cv->r           = 2.;
  cv->pd_ratio    = 0.01;
  cv->Theta       = 1.2;
  cv->eta         = 1.;
  cv->h_lmap_norm = 0.;

  PetscCall(TaoLineSearchCreate(PetscObjectComm((PetscObject)tao), &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, ls_type));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));
  {
    TaoTerm reg;

    PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), &reg));
    PetscCall(TaoTermSetType(reg, TAOTERMHALFL2SQUARED));
    PetscCall(TaoMappedTermSetData(&cv->reg_term, "reg_", 1.0, reg, NULL));
    PetscCall(TaoTermDestroy(&reg));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoCVSetInitialNormEstimate(Tao tao, PetscReal norm)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscCheck(norm >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Norm cannot be negative");
  cv->h_lmap_norm = norm;
  PetscFunctionReturn(PETSC_SUCCESS);
}
