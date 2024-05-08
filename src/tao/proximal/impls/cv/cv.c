#include <../src/tao/proximal/impls/cv/cv.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsctaolinesearch.h>
#include <petscdm.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>

static PetscBool  cited      = PETSC_FALSE;
static const char citation[] = "@article{latafat2023adaptive,\n"
                               "title={Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient},\n"
                               "author={Latafat, Puya and Themelis, Andreas and Stella, Lorenzo and Patrinos, Panagiotis},\n"
                               "journal={arXiv preprint arXiv:2301.04431},\n"
                               "pages={4},\n"
                               "year={2023}"
                               "}\n";

static PetscErrorCode TaoCV_ObjGrad_Private(Tao tao, Vec x, PetscReal *f, Vec g)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  if (cv->smoothterm) {
    PetscCall(DMTaoComputeObjectiveAndGradient(cv->smoothterm, x, f, g));
    *f *= cv->f_scale;
    PetscCall(VecScale(g, cv->f_scale));
  } else {
    PetscCall(TaoComputeObjectiveAndGradient(tao, x, f, g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO I have this implemented in PSARMIJO too, but...
//this needs pd_ratio, eta, x, x_old, grad, grad_old,  step, step_old, nu, sigma, lmap, dualvecs? Aty
static PetscErrorCode TaoCV_Stepsize_With_LS_Private(Tao tao)
{
  TAO_CV   *cv = (TAO_CV *)tao->data;
  PetscReal xi, grad_x_dot, xdiffnorm, graddiffnorm, L, C, D, min1, min2, min3, temp, temp2, temp3, step_new, rho, norm1, norm2;

  PetscFunctionBegin;
  // TODO should we use eta for norm estimate, or just use h_map_norm? set bool is still false
  xi = cv->pd_ratio * tao->step * cv->eta * (1 + tao->gatol);
  xi *= xi;

  PetscCall(VecWAXPY(cv->workvec, -1., cv->grad_old, tao->gradient));
  PetscCall(VecWAXPY(cv->workvec2, -1., cv->x_old, tao->solution));
  PetscCall(VecTDot(cv->workvec, cv->workvec2, &grad_x_dot));
  PetscCall(VecNorm(cv->workvec2, NORM_2, &xdiffnorm));
  PetscCall(VecNorm(cv->workvec, NORM_2, &graddiffnorm));

  L = (xdiffnorm == 0) ? 0 : grad_x_dot / (xdiffnorm * xdiffnorm);
  C = (grad_x_dot == 0) ? 0 : (graddiffnorm * graddiffnorm) / grad_x_dot;
  D = tao->step * L * (tao->step * C - 1);

  //  cv->h_lmap_norm *= cv->R;
  cv->eta *= cv->R;

  PetscInt iter = 0;
  while (1) {
    min1      = tao->step * PetscSqrtReal(1 + tao->step / cv->step_old);
    min2      = 1 / (2 * cv->nu * cv->pd_ratio * cv->eta);
    temp      = 1 - 4 * xi;
    temp2     = cv->pd_ratio * cv->eta * tao->step; //Unlike no linesearch, this "xi" uses updated norm estimate
    temp3     = PetscSqrtReal(D * D + temp * temp2 * temp2);
    min3      = tao->step * PetscSqrtReal(temp / (2 * (1 + tao->gatol) * (temp3 + D)));
    step_new  = PetscMin(min1, PetscMin(min2, min3));
    rho       = step_new / tao->step;
    cv->sigma = cv->pd_ratio * cv->pd_ratio * step_new;

    /* dualvec_work: w = y + sigma *((1+rho) * Ax - rho * Ax_old) */
    PetscCall(VecWAXPY(cv->dualvec_work, -cv->sigma * rho, cv->Ax_old, tao->dualvec));
    PetscCall(VecAXPY(cv->dualvec_work, cv->sigma * (1 + rho), cv->Ax));
    /* dualvec: y = prox_h*(w, sigma) */
    PetscCall(DMTaoApplyProximalMap(cv->h_prox, cv->reg, cv->sigma * cv->h_scale, cv->dualvec_work, cv->dualvec_test, PETSC_TRUE));
    /* workvec : A^T * y_test */
    PetscCall(MatMultTranspose(cv->h_lmap, cv->dualvec_test, cv->workvec));
    /* norm1 = norm(ATy_test - ATy) */
    PetscCall(VecWAXPY(cv->workvec2, -1., cv->ATy, cv->workvec));
    PetscCall(VecNorm(cv->workvec2, NORM_2, &norm1));
    /* norm2 = norm(y_test - y) */
    PetscCall(VecWAXPY(cv->dualvec_work2, -1., cv->dualvec_test, tao->dualvec));
    PetscCall(VecNorm(cv->dualvec_work2, NORM_2, &norm2));
    if (cv->eta >= norm1 / norm2) {
      cv->step_old = tao->step;
      tao->step    = step_new;
      PetscCall(VecCopy(cv->dualvec_test, tao->dualvec));
      PetscCall(VecCopy(cv->workvec, cv->ATy));
      break;
    }
    cv->eta *= cv->r;
    iter++;
    PetscCheck(iter < 100, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "LS loop stuck");
  }
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
  min2         = 1 / (2 * cv->nu * cv->pd_ratio * cv->h_lmap_norm);
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
  TAO_CV   *cv = (TAO_CV *)tao->data;
  PetscReal f, gnorm, lip, rho;
  PetscReal pri_res_norm, dual_res_norm, g_val, h_val;
  //  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;
  PetscCheck(tao->step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize cannot be negative");
  PetscCheck(cv->R <= 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Scale factor needs to be equal or less than 1");
  PetscCheck(cv->r > 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Backtracking factor needs to be greater than 1");
  if (cv->smoothterm) PetscCall(DMTaoGetLipschitz(cv->smoothterm, &lip));
  PetscCall(PetscCitationsRegister(citation, &cited));

  //f can be missing (becomes PDHG), but both g and h need to be present - (i.e., does not suppoer LV/PAPC)
  //does not support missing lmap, that is, does not support Douglas-Rachford
  //TODO bunch of check/assert about combinations...
  cv->lip = lip;

  if (tao->step == 0 && cv->h_lmap_norm > 0) tao->step = 1 / (2 * cv->nu * cv->pd_ratio * cv->h_lmap_norm);
  else if (tao->step == 0 && cv->h_lmap_norm == 0) tao->step = 1 / (2 * cv->nu * cv->pd_ratio * cv->eta);
  cv->sigma = tao->step * cv->pd_ratio * cv->pd_ratio;

  cv->step_old = tao->step;

  PetscCheck(tao->step > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize has to be greater than zero");
  PetscCall(TaoCV_ObjGrad_Private(tao, tao->solution, &f, tao->gradient));
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
  PetscCall(TaoCV_ObjGrad_Private(tao, tao->solution, &f, tao->gradient));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* workvec: v = x - step * (grad_x + ATy) */
    PetscCall(VecWAXPY(cv->workvec, -tao->step, tao->gradient, tao->solution));
    PetscCall(VecAXPY(cv->workvec, -tao->step, cv->ATy));

    /* x = prog_g(v, step) */
    PetscCall(VecCopy(tao->solution, cv->x_old)); //TODO Axprev, gradprev copy here
    PetscCall(DMTaoApplyProximalMap(cv->g_prox, cv->reg, tao->step * cv->g_scale, cv->workvec, tao->solution, PETSC_FALSE));
    /* update Ax, and grad */
    PetscCall(VecCopy(cv->Ax, cv->Ax_old));
    PetscCall(VecCopy(tao->gradient, cv->grad_old));
    PetscCall(MatMult(cv->h_lmap, tao->solution, cv->Ax));
    PetscCall(TaoCV_ObjGrad_Private(tao, tao->solution, &f, tao->gradient));
    /* workvec = (v - x)/step + grad_x + ATy */
    PetscCall(VecAXPY(cv->workvec, -1., tao->solution));
    PetscCall(VecScale(cv->workvec, 1 / tao->step));
    PetscCall(VecAXPBYPCZ(cv->workvec, 1., 1., 1., tao->gradient, cv->ATy));
    PetscCall(VecNorm(cv->workvec, NORM_2, &pri_res_norm));

    // update stepsize. no vanilla Condat-Vu
    if (cv->lmap_norm_set) {
      PetscCall(TaoCV_Stepsize_No_LS_Private(tao));
      rho = tao->step / cv->step_old;

      /* dualvec_work: w = y + sigma *((1+rho) * Ax - rho * Ax_old) */
      PetscCall(VecWAXPY(cv->dualvec_work, -cv->sigma * rho, cv->Ax_old, tao->dualvec));
      PetscCall(VecAXPY(cv->dualvec_work, cv->sigma * (1 + rho), cv->Ax));

      /* dualvec: y = prox_h*(w, sigma) */
      PetscCall(DMTaoApplyProximalMap(cv->h_prox, cv->reg, cv->sigma * cv->h_scale, cv->dualvec_work, tao->dualvec, PETSC_TRUE));
    } else {
      PetscCall(TaoCV_Stepsize_With_LS_Private(tao)); //prox_h is done inside this routine for linesearch version
      //calling this a linesearch a-la Armijo is a strech....
      // this LS inputs are not really "real" in that all things are done internally...
      // LS needs: x1, x0, grad_1, grad_0, Ax_old, dualvec(y), sigma, pd_ratio, nu, eta
      //PetscCall(TaoLineSearchApply(tao->linesearch, cv->dualvec, &f, tao->gradient, tao->solution, &tao->step, &ls_status));
    }

    PetscCall(VecAXPY(cv->dualvec_work, -1., tao->dualvec));
    PetscCall(VecScale(cv->dualvec_work, 1 / cv->sigma));
    PetscCall(VecAXPY(cv->dualvec_work, -1., cv->Ax));
    PetscCall(VecNorm(cv->dualvec_work, NORM_2, &dual_res_norm));

    tao->residual = PetscSqrtReal(pri_res_norm * pri_res_norm) + PetscSqrtReal(dual_res_norm * dual_res_norm);

    PetscCall(DMTaoComputeObjective(cv->g_prox, tao->solution, &g_val));
    g_val *= cv->g_scale;
    PetscCall(DMTaoComputeObjective(cv->h_prox, cv->Ax, &h_val));
    h_val *= cv->h_scale;
    /* convergence test */
    PetscCall(TaoLogConvergenceHistory(tao, f + g_val + h_val, tao->residual, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f + g_val + h_val, tao->residual, 0.0, tao->step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    tao->niter++;

    /* post-processing */
    PetscCall(MatMultTranspose(cv->h_lmap, tao->dualvec, cv->ATy)); //TODO dont need this for LS version.
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
  PetscCall(PetscOptionsReal("-tao_cv_nu", "Stepsize scale parameter nu. Must be  >1+tol.", "", cv->nu, &cv->nu, NULL));
  //TODO TaoCVSetInitialNormEstimate(Tao, PetscReal) ?
  PetscCall(PetscOptionsReal("-tao_cv_eta", "Initial linear map norm estimate. Must be nonnegative", "", cv->eta, &cv->eta, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_tol", "Stepsize tolerance parameter", "", cv->tol, &cv->tol, NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_CV(Tao tao, PetscViewer viewer)
{
  DMTao     tdm;
  PetscBool isascii;
  TAO_CV   *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Norm estimate factor: R=%g\n", (double)cv->R));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Primal-dual ratio: ratio=%g\n", (double)cv->pd_ratio));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Backtracking paramter: r=%g\n", (double)cv->r));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stepsize scale parameter: nu=%g\n", (double)cv->nu));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Using adaPDM-type adaptive stepsize\n"));
    if (cv->smoothterm) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Smooth Term:\n"));
      PetscCall(DMGetDMTao(cv->smoothterm, &tdm));
      PetscCall(DMTaoView(tdm, viewer));
    }
    if (cv->g_prox) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Non-smooth Term g(x):\n"));
      PetscCall(DMGetDMTao(cv->g_prox, &tdm));
      PetscCall(DMTaoView(tdm, viewer));
    }
    if (cv->h_prox) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Non-smooth Term h(Ax):\n"));
      PetscCall(DMGetDMTao(cv->h_prox, &tdm));
      PetscCall(DMTaoView(tdm, viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_CV(Tao tao)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  /* sol sized vectors */
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!cv->workvec) PetscCall(VecDuplicate(tao->solution, &cv->workvec));
  if (!cv->workvec2) PetscCall(VecDuplicate(tao->solution, &cv->workvec2));
  if (!cv->x_old) PetscCall(VecDuplicate(tao->solution, &cv->x_old));
  if (!cv->grad_old) PetscCall(VecDuplicate(tao->solution, &cv->grad_old));
  if (!cv->ATy) PetscCall(VecDuplicate(tao->solution, &cv->ATy));
  /* dual sized vectors */
  if (!cv->Ax) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->Ax));
  if (!cv->Ax_old) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->Ax_old));
  if (!tao->dualvec) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &tao->dualvec));
  if (!cv->dualvec_test || !cv->lmap_norm_set) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->dualvec_test));
  if (!cv->dualvec_work) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->dualvec_work));
  if (!cv->dualvec_work2) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->dualvec_work2));
  //TODO option to set regularizer type??
  //if L is set, should default stepsize be that? not for sparsa...
  PetscCall(DMCreate(PetscObjectComm((PetscObject)tao), &cv->reg));
  PetscCall(DMTaoSetType(cv->reg, DMTAOL2));
  if (cv->smoothterm) {
    PetscCall(TaoLineSearchUseDM(tao->linesearch, cv->smoothterm));
  } else {
    PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  }
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
  PetscCall(MatDestroy(&cv->h_lmap));
  PetscCall(DMDestroy(&cv->reg));
  PetscCall(DMDestroy(&cv->smoothterm));
  PetscCall(DMDestroy(&cv->g_prox));
  PetscCall(DMDestroy(&cv->h_prox));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoCreate_CV(Tao tao)
{
  TAO_CV     *cv;
  const char *armijo_type = TAOLINESEARCHPSARMIJO;

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

  cv->h_scale    = 1.;
  cv->g_scale    = 1.;
  cv->gnorm_norm = 0.;
  cv->tol        = 1.e-6;
  cv->R          = 0.95;
  cv->r          = 2.;
  cv->pd_ratio   = 0.01;
  cv->nu         = 1.2;
  cv->eta        = 1.;
  cv->smoothterm = NULL;

  PetscCall(TaoLineSearchCreate(PetscObjectComm((PetscObject)tao), &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, armijo_type));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}
