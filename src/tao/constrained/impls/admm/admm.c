#include <../src/tao/constrained/impls/admm/admm.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <../src/tao/term/impls/sum/taotermsum.h> // TaoTermSumVecNestGetSubVecsRead(), TaoTermSumVecSetRestoreSubVecsRead()
#include <petscsf.h>
#include <../src/tao/util/softthreshold.h> // TaoIsSoftThreshold(), TaoSolve_SoftThreshold()

const char *const TaoADMMUpdateTypes[] = {"basic", "adaptive", "TaoADMMUpdateType", "TAO_ADMM_UPDATE", NULL};

static PetscErrorCode TaoADMMInitializeSubproblemSolutions(Tao tao)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;
  Vec       x, z;

  PetscFunctionBegin;
  switch (am->initialize_type) {
  case ADMM_INITIALIZE_SCATTER:
    PetscCall(TaoGetSolution(am->x_subsolver, &x));
    PetscCall(TaoGetSolution(am->z_subsolver, &z));
    PetscCall(VecScatterBegin(am->x_scatter, tao->solution, x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(am->x_scatter, tao->solution, x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(am->z_scatter, tao->solution, z, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(am->z_scatter, tao->solution, z, INSERT_VALUES, SCATTER_FORWARD));
    break;
  case ADMM_INITIALIZE_Z_AX:
    PetscCall(TaoGetSolution(am->z_subsolver, &z));
    PetscCall(TaoSetSolution(am->x_subsolver, tao->solution));
    PetscCall(MatMult(am->A, tao->solution, z));
    break;
  case ADMM_INITIALIZE_X_BZ:
    PetscCall(TaoGetSolution(am->x_subsolver, &x));
    PetscCall(TaoSetSolution(am->z_subsolver, tao->solution));
    PetscCall(MatMult(am->B, tao->solution, x));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMComputeOuterSolution(Tao tao)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;
  Vec       x, z;

  PetscFunctionBegin;
  switch (am->initialize_type) {
  case ADMM_INITIALIZE_SCATTER:
    PetscCall(TaoGetSolution(am->x_subsolver, &x));
    PetscCall(TaoGetSolution(am->z_subsolver, &z));
    PetscCall(VecScatterBegin(am->x_scatter, x, tao->solution, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(am->x_scatter, x, tao->solution, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterBegin(am->z_scatter, z, tao->solution, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(am->z_scatter, z, tao->solution, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterBegin(am->x_scatter, am->d_x, tao->gradient, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(am->x_scatter, am->d_x, tao->gradient, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterBegin(am->z_scatter, am->d_z, tao->gradient, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(am->z_scatter, am->d_z, tao->gradient, INSERT_VALUES, SCATTER_REVERSE));
    break;
  case ADMM_INITIALIZE_Z_AX:
    PetscCall(TaoGetSolution(am->x_subsolver, &x));
    PetscCall(VecCopy(x, tao->solution));
    break;
  case ADMM_INITIALIZE_X_BZ:
    PetscCall(TaoGetSolution(am->z_subsolver, &z));
    PetscCall(VecCopy(z, tao->solution));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TODO: remove
PETSC_INTERN PetscErrorCode TaoADMMVecDuplicateAndCopy(Vec x, Vec *y)
{
  PetscFunctionBegin;
  if (!*y) PetscCall(VecDuplicate(x, y));
  PetscCall(VecCopy(x, *y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMUpdateXSubproblem(Tao tao)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  PetscCall((*am->updatexsubproblem)(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMUpdateZSubproblem(Tao tao)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  PetscCall((*am->updatezsubproblem)(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMSolveSubproblem(Tao tao, Tao subsolver, PetscReal *fx)
{
  PetscReal fc;

  PetscFunctionBegin;
  PetscCall(TaoSolve(subsolver)); // computes indivudal components of the objective and puts them in subsolver->objective_values;
  PetscCall(TaoGetSolutionStatus(subsolver, NULL, &fc, NULL, NULL, NULL, NULL));
  *fx = fc - subsolver->objective_values[subsolver->num_terms - 1]; // subtract the metric penalty from the objective value
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMUpdateY(Tao tao, Vec r, Vec r_halfstep, Vec y)
{
  Tao_ADMM *am    = (Tao_ADMM *)tao->data;
  PetscReal mu    = am->mu;
  PetscReal gamma = am->relaxation_gamma;

  PetscFunctionBegin;
  PetscCall(VecAXPY(y, mu, r));
  if (gamma != 1.0) PetscCall(VecAXPY(y, mu * (gamma - 1.0), r_halfstep));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMPrimalResidual(Vec Ax, Vec Bz, Vec c, Vec r)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(Ax, r));
  PetscCall(VecAXPY(r, 1.0, Bz));
  if (c) PetscCall(VecAXPY(r, 1.0, c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMDualResidualDebug(Tao tao)
{
  Tao_ADMM   *am = (Tao_ADMM *)tao->data;
  TaoTerm     f, g;
  Vec         f_params, g_params;
  PetscInt    f_n_terms, g_n_terms;
  Vec         df_x, dg_z;
  Vec         Aty, Bty;
  Vec         x, z;
  PetscInt    tabs;
  PetscViewer viewer = am->debug_viewer;

  PetscFunctionBegin;
  PetscCall(TaoGetTerm(am->x_subsolver, NULL, &f, &f_params, NULL));
  PetscCall(TaoGetTerm(am->z_subsolver, NULL, &g, &g_params, NULL));
  PetscCall(TaoGetSolution(am->x_subsolver, &x));
  PetscCall(TaoGetSolution(am->z_subsolver, &z));

  PetscCall(TaoTermSumGetNumSubterms(f, &f_n_terms));
  PetscCall(TaoTermSumSetSubtermMask(f, f_n_terms - 1, TAOTERM_MASK_OBJECTIVE));
  PetscCall(TaoTermSumGetNumSubterms(g, &g_n_terms));
  PetscCall(TaoTermSumSetSubtermMask(g, g_n_terms - 1, TAOTERM_MASK_OBJECTIVE));

  PetscCall(VecDuplicate(am->d_x, &df_x));
  PetscCall(VecDuplicate(am->d_x, &Aty));
  PetscCall(VecCopy(am->d_x, df_x));
  PetscCall(MatMultTranspose(am->A, am->y, Aty));
  PetscCall(VecAXPY(df_x, -1.0, Aty));
  PetscCall(VecDestroy(&Aty));

  PetscCall(VecDuplicate(am->d_z, &dg_z));
  PetscCall(VecDuplicate(am->d_z, &Bty));
  PetscCall(VecCopy(am->d_z, dg_z));
  PetscCall(MatMultTranspose(am->B, am->y, Bty));
  PetscCall(VecAXPY(dg_z, -1.0, Bty));
  PetscCall(VecDestroy(&Bty));

  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "TaoADMM, x dual residual test:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(TaoTestGradient_Internal(am->x_subsolver, x, df_x, am->debug_viewer, NULL));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "TaoADMM, z dual residual test:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(TaoTestGradient_Internal(am->z_subsolver, z, dg_z, am->debug_viewer, NULL));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));

  PetscCall(VecDestroy(&dg_z));
  PetscCall(VecDestroy(&df_x));
  PetscCall(TaoTermSumSetSubtermMask(f, f_n_terms - 1, TAOTERM_MASK_NONE));
  PetscCall(TaoTermSumSetSubtermMask(g, g_n_terms - 1, TAOTERM_MASK_NONE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMDualResidual(Tao tao)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  if (am->x_inexact && am->x_subsolver->gradient) {
    PetscCall(VecCopy(am->x_subsolver->gradient, am->d_x));
  } else {
    PetscCall(VecZeroEntries(am->d_x));
  }
  if (am->z_inexact && am->z_subsolver->gradient) {
    PetscCall(VecCopy(am->z_subsolver->gradient, am->d_z));
  } else {
    PetscCall(VecZeroEntries(am->d_z));
  }
  if (am->dualresidual) PetscCall((*am->dualresidual)(tao));
  if (am->debug_viewer) {
    PetscBool is_ascii;

    PetscCall(PetscObjectTypeCompare((PetscObject)am->debug_viewer, PETSCVIEWERASCII, &is_ascii));
    if (is_ascii) {
      PetscCall(PetscViewerPushFormat(am->debug_viewer, am->debug_viewer_format));
      PetscCall(TaoADMMDualResidualDebug(tao));
      PetscCall(PetscViewerPopFormat(am->debug_viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMSetUpdateType_ADMM(Tao tao, TaoADMMUpdateType type)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  am->mu_update = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMGetUpdateType_ADMM(Tao tao, TaoADMMUpdateType *type)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  *type = am->mu_update;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoConvergenceTest_ADMM(Tao tao, PETSC_UNUSED void *ctx)
{
  Tao_ADMM          *am    = (Tao_ADMM *)tao->data;
  PetscInt           niter = tao->niter, nfuncs = PetscMax(tao->nfuncs, tao->nfuncgrads);
  PetscInt           max_funcs = tao->max_funcs;
  PetscReal          gnorm     = tao->residual;
  PetscReal          f         = tao->fc;
  PetscReal          gatol = tao->gatol, grtol = tao->grtol;
  PetscReal          catol = tao->catol, crtol = tao->crtol;
  PetscReal          cnorm  = tao->cnorm;
  TaoConvergedReason reason = tao->reason;
  PetscInt           n, n_y;
  PetscReal          primal_scale, primal_abs_tol, primal_rel_tol, primal_tol;
  PetscReal          dual_scale, dual_abs_tol, dual_rel_tol, dual_tol;

  PetscFunctionBegin;
  if (reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetSize(tao->solution, &n));
  PetscCall(VecGetSize(am->y, &n_y));

  primal_scale   = PetscMax(am->c_norm, am->Ax_norm);
  primal_scale   = PetscMax(primal_scale, am->Bz_norm);
  primal_abs_tol = PetscSqrtReal((PetscReal)n_y) * catol;
  primal_rel_tol = primal_scale * crtol;
  primal_tol     = PetscMax(primal_abs_tol, primal_rel_tol);

  dual_scale   = PetscSqrtReal(am->Aty_norm * am->Aty_norm + am->Bty_norm * am->Bty_norm);
  dual_abs_tol = PetscSqrtReal((PetscReal)n) * gatol;
  dual_rel_tol = dual_scale * grtol;
  dual_tol     = PetscMax(dual_abs_tol, dual_rel_tol);

  if (PetscIsInfOrNanReal(f)) {
    PetscCall(PetscInfo(tao, "Failed to converged, Lagrangian value is Inf or NaN\n"));
    reason = TAO_DIVERGED_NAN;
  } else if (gnorm <= dual_tol && cnorm <= primal_tol) {
    if (cnorm == 0.0) PetscCall(PetscInfo(tao, "Converged due to: primal residual ||Ax + Bz + c|| = 0\n"));
    else if (cnorm <= primal_rel_tol) PetscCall(PetscInfo(tao, "Converged due to: primal residual ||Ax + Bz + c|| / max(||Ax||,||Bz||,||c||) = %g < %g\n", (double)(cnorm / primal_scale), (double)crtol));
    else PetscCall(PetscInfo(tao, "Converged due to: primal residual ||Ax + Bz + c|| = %g < %g\n", (double)primal_scale, (double)primal_abs_tol));

    if (gnorm == 0.0) PetscCall(PetscInfo(tao, "                  dual residual ||(grad f(x) + A^T y, grad g(z) + B^T y)|| = 0\n"));
    else if (gnorm <= dual_rel_tol) PetscCall(PetscInfo(tao, "                  dual residual ||(grad f(x) + A^T y, grad g(z) + B^T y)|| / ||(A^T y, B^T y)|| = %g < %g\n", (double)(gnorm / dual_scale), (double)grtol));
    else PetscCall(PetscInfo(tao, "                  dual residual ||(grad f(x) + A^T y, grad g(z) + A^T y)|| = %g < %g\n", (double)dual_scale, (double)dual_abs_tol));

    reason = gnorm <= dual_rel_tol ? TAO_CONVERGED_GRTOL : TAO_CONVERGED_GATOL;
  } else if (max_funcs != PETSC_UNLIMITED && nfuncs > max_funcs) {
    PetscCall(PetscInfo(tao, "Exceeded maximum number of function evaluations: %" PetscInt_FMT " > %" PetscInt_FMT "\n", nfuncs, max_funcs));
    reason = TAO_DIVERGED_MAXFCN;
  } else if (niter >= tao->max_it) {
    PetscCall(PetscInfo(tao, "Exceeded maximum number of iterations: %" PetscInt_FMT " > %" PetscInt_FMT "\n", niter, tao->max_it));
    reason = TAO_DIVERGED_MAXITS;
  } else {
    reason = TAO_CONTINUE_ITERATING;
  }
  tao->reason = reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solve f(x) + g(z) s.t. Ax + Bz + c = 0 */
static PetscErrorCode TaoSolve_ADMM(Tao tao)
{
  Tao_ADMM   *am = (Tao_ADMM *)tao->data;
  Vec         x, z;
  PetscScalar y_dot_r;
  PetscReal   fg, fx, gz, c_norm, c_norm2, d_norm, d_x_norm, d_z_norm, lagrangian;

  PetscFunctionBegin;
  PetscCall(TaoADMMInitializeSubproblemSolutions(tao));
  PetscCall(TaoGetSolution(am->x_subsolver, &x));
  PetscCall(TaoGetSolution(am->z_subsolver, &z));
  if (!tao->recycle) PetscCall(VecZeroEntries(am->y));
  PetscCall(MatMult(am->A, x, am->Ax));
  PetscCall(MatMult(am->B, z, am->Bz));
  PetscCall(TaoComputeObjective(tao, tao->solution, &fg));
  PetscCall(TaoADMMPrimalResidual(am->Ax, am->Bz, am->c, am->r));
  PetscCall(VecDotNorm2(am->y, am->r, &y_dot_r, &c_norm2));
  c_norm     = PetscSqrtReal(c_norm2);
  lagrangian = fg + PetscRealPart(y_dot_r);
  PetscCall(TaoMonitor(tao, tao->niter, lagrangian, PETSC_DEFAULT, c_norm, 1.0 / am->mu));

  tao->reason = TAO_CONTINUE_ITERATING;
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);
    PetscCall(TaoADMMUpdateXSubproblem(tao));
    PetscCall(TaoADMMSolveSubproblem(tao, am->x_subsolver, &fx));
    PetscCall(MatMult(am->A, x, am->Ax));
    PetscCall(TaoADMMPrimalResidual(am->Ax, am->Bz, am->c, am->r_halfstep));
    PetscCall(TaoADMMUpdateZSubproblem(tao));
    PetscCall(TaoADMMSolveSubproblem(tao, am->z_subsolver, &gz));
    PetscCall(MatMult(am->B, z, am->Bz));
    PetscCall(TaoADMMPrimalResidual(am->Ax, am->Bz, am->c, am->r));
    PetscCall(TaoADMMUpdateY(tao, am->r, am->r_halfstep, am->y));
    PetscCall(TaoADMMDualResidual(tao));
    PetscCall(VecNorm(am->d_x, NORM_2, &d_x_norm));
    PetscCall(VecNorm(am->d_z, NORM_2, &d_z_norm));
    PetscCall(PetscInfo(tao, "x component of dual residual: %8.2e, z component of dual residual: %8.2e\n", (double)d_x_norm, (double)d_z_norm));
    d_norm = PetscSqrtReal(d_x_norm * d_x_norm + d_z_norm * d_z_norm);
    PetscCall(VecDotNorm2(am->y, am->r, &y_dot_r, &c_norm2));
    c_norm     = PetscSqrtReal(c_norm2);
    lagrangian = fx + gz + PetscRealPart(y_dot_r);
    tao->niter++;
    PetscCall(TaoMonitor(tao, tao->niter, lagrangian, d_norm, c_norm, 1.0 / am->mu));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  }
  PetscCall(TaoADMMComputeOuterSolution(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_ADMM(Tao tao, PetscOptionItems PetscOptionsObject)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "ADMM solves f(x) + g(z) subject to Ax + Bz + c = 0");
  PetscCall(PetscOptionsReal("-tao_admm_spectral_penalty", "Constant for Augmented Lagrangian term", "", am->mu, &am->mu, NULL));
  PetscCall(PetscOptionsReal("-tao_admm_relaxation_parameter", "relaxation parameter for z update", "", am->relaxation_gamma, &am->relaxation_gamma, NULL));
  PetscCall(PetscOptionsInt("-tao_admm_adptivity_period", "iterations per parameter update", "", am->adaptivity_period, &am->adaptivity_period, NULL));
  PetscCall(PetscOptionsEnum("-tao_admm_update_type", "Lagrangian spectral penalty update policy", "TaoADMMUpdateType", TaoADMMUpdateTypes, (PetscEnum)am->mu_update, (PetscEnum *)&am->mu_update, NULL));
  PetscOptionsHeadEnd();
  PetscCall(PetscOptionsCreateViewer(PetscObjectComm((PetscObject)tao), ((PetscObject)tao)->options, ((PetscObject)tao)->prefix, "-tao_admm_debug", &am->debug_viewer, &am->debug_viewer_format, NULL));
  am->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_ADMM(Tao tao, PetscViewer viewer)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;
  PetscBool is_ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &is_ascii));
  if (is_ascii) {
    Vec       *sub_params = NULL;
    PetscBool *is_dummy   = NULL;
    PetscInt   n;

    if (!tao->setupcalled) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "ADMM type: unknown [setup not called yet]\n"));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (tao->objective_parameters) PetscCall(TaoTermSumVecNestGetSubVecsRead(tao->objective_parameters, &n, &sub_params, &is_dummy));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ADMM type: min_{x,z} "));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    if (am->f_num_terms > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "["));
      for (PetscInt i = 0; i < am->f_num_terms; i++)
        PetscCall(TaoTermViewSumPrintSubterm(tao->objective_term.term, viewer, PETSC_TRUE, TaoTermSumGetSubVec(tao->objective_parameters, sub_params, is_dummy, am->f_terms[i]), am->f_terms[i], i == 0 ? PETSC_TRUE : PETSC_FALSE, am->f_mapped == PETSC_BOOL3_TRUE ? PETSC_FALSE : PETSC_TRUE, "f", "A", "x", "p"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "]"));
      if (am->g_num_terms > 0) PetscCall(PetscViewerASCIIPrintf(viewer, " + "));
    }
    if (am->g_num_terms > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "["));
      for (PetscInt i = 0; i < am->g_num_terms; i++)
        PetscCall(TaoTermViewSumPrintSubterm(tao->objective_term.term, viewer, PETSC_TRUE, TaoTermSumGetSubVec(tao->objective_parameters, sub_params, is_dummy, am->g_terms[i]), am->g_terms[i], i == 0 ? PETSC_TRUE : PETSC_FALSE, am->g_mapped == PETSC_BOOL3_TRUE ? PETSC_FALSE : PETSC_TRUE, "f", "A", "z", "p"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "]"));
    }
    if (tao->objective_parameters) PetscCall(TaoTermSumVecNestRestoreSubVecsRead(tao->objective_parameters, &n, &sub_params, &is_dummy));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
    PetscCall(PetscViewerASCIIPrintf(viewer, "           such that "));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    switch (am->initialize_type) {
    case ADMM_INITIALIZE_SCATTER:
      PetscCall(PetscViewerASCIIPrintf(viewer, "A x + B z = %s", am->c ? "c" : "0"));
      break;
    case ADMM_INITIALIZE_Z_AX: {
      PetscBool is_constdiag;
      PetscCall(PetscObjectTypeCompare((PetscObject)am->A, MATCONSTANTDIAGONAL, &is_constdiag));
      if (is_constdiag) {
        PetscScalar scale;

        PetscCall(MatConstantDiagonalGetConstant(am->A, &scale));
        if (scale == 1.0) PetscCall(PetscViewerASCIIPrintf(viewer, "x = z"));
        else if (PetscImaginaryPart(scale) == 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, "%g x = z", (double)PetscRealPart(scale)));
        else PetscCall(PetscViewerASCIIPrintf(viewer, "(%g + i %g) x = z", (double)PetscRealPart(scale), (double)PetscImaginaryPart(scale)));
      } else {
        PetscCall(TaoTermViewSumPrintMapName(viewer, am->A, am->g_terms[0], "A", PETSC_FALSE));
        PetscCall(PetscViewerASCIIPrintf(viewer, " x = z"));
      }
      break;
    }
    case ADMM_INITIALIZE_X_BZ: {
      PetscBool is_constdiag;
      PetscCall(PetscObjectTypeCompare((PetscObject)am->B, MATCONSTANTDIAGONAL, &is_constdiag));
      if (is_constdiag) {
        PetscScalar scale;

        PetscCall(MatConstantDiagonalGetConstant(am->B, &scale));
        if (scale == 1.0) PetscCall(PetscViewerASCIIPrintf(viewer, "x = z"));
        else if (PetscImaginaryPart(scale) == 0.0) PetscCall(PetscViewerASCIIPrintf(viewer, "x = %g z", (double)PetscRealPart(scale)));
        else PetscCall(PetscViewerASCIIPrintf(viewer, "x = (%g + i %g)", (double)PetscRealPart(scale), (double)PetscImaginaryPart(scale)));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "x = "));
        PetscCall(TaoTermViewSumPrintMapName(viewer, am->B, am->f_terms[0], "A", PETSC_FALSE));
        PetscCall(PetscViewerASCIIPrintf(viewer, " z"));
      }
      break;
    }
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "x subsolver:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(TaoView(am->x_subsolver, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));

  PetscCall(PetscViewerASCIIPrintf(viewer, "z subsolver:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(TaoView(am->z_subsolver, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUpADMMComputeIndicesFromComplement(Tao tao, PetscInt n_terms, PetscInt f_num_terms, PetscInt g_num_terms, const PetscInt *g_terms, PetscInt **f_terms)
{
  PetscInt term;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(f_num_terms, f_terms));
  term = 0;
  for (PetscInt i = 0; i < f_num_terms; i++) {
    for (PetscInt t = term; t < n_terms; t++) {
      PetscBool t_in_g = PETSC_FALSE;
      for (PetscInt j = 0; j < g_num_terms; j++) {
        if (g_terms[j] == t) {
          t_in_g = PETSC_TRUE;
          break;
        }
      }
      if (!t_in_g) {
        term = t;
        break;
      } else {
        term++;
      }
    }
    PetscAssert(term < n_terms, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Couldn't find index in the complement of g_terms");
    (*f_terms)[i] = term;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumGetSubsetMap(TaoTerm objective, PetscBool3 f_mapped, PetscInt f_num_terms, const PetscInt *f_terms, Mat *f_map)
{
  PetscFunctionBegin;
  if (!f_num_terms) {
    *f_map = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermSumGetSubterm(objective, f_terms[0], NULL, NULL, NULL, f_map));
  for (PetscInt i = 1; i < f_num_terms; i++) {
    Mat map_i;

    PetscCall(TaoTermSumGetSubterm(objective, f_terms[i], NULL, NULL, NULL, &map_i));
    if (map_i != *f_map) {
      PetscCheck(f_mapped == PETSC_BOOL3_UNKNOWN, PetscObjectComm((PetscObject)objective), PETSC_ERR_ARG_INCOMP, "subterm %" PetscInt_FMT " = %" PetscInt_FMT " does not have the same map as subterm 0 = %" PetscInt_FMT ", cannot separate the map", i, f_terms[i], f_terms[0]);
      *f_map = NULL;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumCreateSubset_ADMM(TaoTerm objective, Mat f_map, const char tao_prefix[], const char sub_prefix[], PetscInt f_num_terms, const PetscInt f_terms[], TaoTerm *f)
{
  PetscFunctionBegin;
  if (f_map == NULL) {
    PetscCall(TaoTermDuplicate(objective, TAOTERM_DUPLICATE_TYPE, f));
  } else {
    Vec f_map_output;

    PetscCall(MatCreateVecs(f_map, NULL, &f_map_output));
    PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)objective), f));
    PetscCall(TaoTermSetSolutionTemplate(*f, f_map_output));
    PetscCall(VecDestroy(&f_map_output));
    PetscCall(TaoTermSetType(*f, TAOTERMSUM));
  }
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*f, tao_prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*f, sub_prefix));
  PetscCall(TaoTermSumSetNumSubterms(*f, f_num_terms));
  for (PetscInt i = 0; i < f_num_terms; i++) {
    PetscReal   scale;
    TaoTerm     term;
    Mat         map;
    TaoTermMask mask;
    Mat         uH, uHpre, mH, mHpre;

    PetscCall(TaoTermSumGetSubterm(objective, f_terms[i], NULL, &scale, &term, &map));
    PetscCall(TaoTermSumSetSubterm(*f, i, NULL, scale, term, (f_map == NULL) ? map : NULL)); // only use the map if it is not being separated out as f_map
    PetscCall(TaoTermSumGetSubtermMask(objective, f_terms[i], &mask));
    PetscCall(TaoTermSumSetSubtermMask(*f, i, mask));
    PetscCall(TaoTermSumGetSubtermHessianMatrices(objective, f_terms[i], &uH, &uHpre, &mH, &mHpre));
    if (f_map == NULL) {
      PetscCall(TaoTermSumSetSubtermHessianMatrices(*f, i, uH, uHpre, mH, mHpre));
    } else {
      // because the input space has changed, the mapped Hessians in objective are the unmapped Hessians in am->f
      PetscCall(TaoTermSumSetSubtermHessianMatrices(*f, i, mH, mHpre, NULL, NULL));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUpADMMCreateSubParams(Vec params, PetscInt f_num_terms, const PetscInt f_terms[], TaoTerm f, Vec *f_params)
{
  Vec       *sub_params;
  Vec       *f_sub_params;
  PetscBool *is_dummy, any_f = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(TaoTermSumVecNestGetSubVecsRead(params, NULL, &sub_params, &is_dummy));

  PetscCall(PetscCalloc1(f_num_terms, &f_sub_params));
  for (PetscInt i = 0; i < f_num_terms; i++) f_sub_params[i] = TaoTermSumGetSubVec(params, sub_params, is_dummy, f_terms[i]);
  for (PetscInt i = 0; i < f_num_terms; i++) any_f = f_sub_params[i] ? PETSC_TRUE : any_f;
  if (any_f) {
    PetscCall(TaoTermSumParametersPack(f, f_sub_params, f_params));
    PetscCall(TaoTermSetParametersTemplate(f, *f_params));
  }
  PetscCall(PetscFree(f_sub_params));
  PetscCall(TaoTermSumVecNestRestoreSubVecsRead(params, NULL, &sub_params, &is_dummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// test for the tao being a min alpha * ||x - a||_1 + beta * ||x - b||_2, in which case we will use soft thresholding
PETSC_INTERN PetscErrorCode TaoADMMConfigureSubTao_Default(Tao tao, Tao subtao)
{
  PetscBool is_softthreshold;

  PetscFunctionBegin;
  PetscCall(TaoIsSoftThreshold(subtao, &is_softthreshold));
  if (!is_softthreshold) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoSetType(subtao, TAOSHELL));
  PetscCall(TaoShellSetSolve(subtao, TaoSolve_SoftThreshold));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrixColumnScatter(Mat J, Mat matscatter, Mat *J_sub)
{
  PetscSF            sf;
  PetscInt           m, rstart, n_roots, n_leaves;
  const PetscInt    *leaves;
  const PetscSFNode *remotes;
  PetscInt          *sorted_perm;
  PetscLayout        col_map;
  const PetscInt    *ranges;
  PetscInt          *is_indices;
  IS                 col_is;
  IS                 row_is;

  PetscFunctionBegin;
  PetscCall(MatScatterGetVecScatter(matscatter, &sf));
  PetscCall(PetscSFGetGraph(sf, &n_roots, &n_leaves, &leaves, &remotes));

  PetscCall(PetscMalloc1(n_leaves, &sorted_perm));
  PetscCall(PetscMalloc1(n_leaves, &is_indices));
  if (leaves == NULL) {
    for (PetscInt i = 0; i < n_leaves; i++) sorted_perm[i] = i;
  } else {
    PetscCall(PetscSortIntWithPermutation(n_leaves, leaves, sorted_perm));
  }
  PetscCall(MatGetLayouts(J, NULL, &col_map));
  PetscCall(PetscLayoutGetRanges(col_map, &ranges));
  for (PetscInt i = 0; i < n_leaves; i++) {
    PetscSFNode remote = remotes[sorted_perm[i]];

    is_indices[i] = ranges[remote.rank] + remote.index;
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)J), n_leaves, is_indices, PETSC_OWN_POINTER, &col_is));
  PetscCall(MatGetLocalSize(J, &m, NULL));
  PetscCall(MatGetOwnershipRange(J, &rstart, NULL));
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)J), m, rstart, 1, &row_is));
  PetscCall(MatCreateSubMatrix(J, row_is, col_is, MAT_INITIAL_MATRIX, J_sub));
  PetscCall(ISDestroy(&row_is));
  PetscCall(ISDestroy(&col_is));
  PetscCall(PetscFree(sorted_perm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_ADMM(Tao tao)
{
  Tao_ADMM   *am = (Tao_ADMM *)tao->data;
  VecType     vec_type;
  PetscLayout layout;
  PetscInt    n_terms;
  PetscBool   objective_is_sum;
  TaoTerm     objective = tao->objective_term.term;
  Vec         params    = tao->objective_parameters;
  Mat         f_map     = NULL;
  Mat         g_map     = NULL;
  Vec         f_params  = NULL;
  Vec         g_params  = NULL;
  MPI_Comm    comm;
  const char *tao_prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCall(PetscObjectTypeCompare((PetscObject)objective, TAOTERMSUM, &objective_is_sum));
  PetscCheck(objective_is_sum, comm, PETSC_ERR_SUP, "Objective function must be TAOTERMSUM to use TAOADMM");

  // Get the sizes of the groups
  PetscCall(TaoTermSumGetNumSubterms(objective, &n_terms));
  if (am->f_num_terms == PETSC_DECIDE && am->g_num_terms == PETSC_DECIDE) am->f_num_terms = n_terms / 2;
  if (am->f_num_terms == PETSC_DECIDE) am->f_num_terms = n_terms - am->g_num_terms;
  if (am->g_num_terms == PETSC_DECIDE) am->g_num_terms = n_terms - am->f_num_terms;
  PetscCheck(am->f_num_terms >= 0 && am->g_num_terms >= 0 && am->f_num_terms + am->g_num_terms == n_terms, comm, PETSC_ERR_ARG_SIZ, "Term group sizes %" PetscInt_FMT " + %" PetscInt_FMT " are not a valid split of % " PetscInt_FMT " objective terms", am->f_num_terms, am->g_num_terms, n_terms);

  // Get the indices in the groups
  if (!am->f_terms && !am->g_terms) {
    PetscInt term;

    PetscCall(PetscMalloc1(am->f_num_terms, &am->f_terms));
    PetscCall(PetscMalloc1(am->g_num_terms, &am->g_terms));

    term = 0;
    for (PetscInt i = 0; i < am->f_num_terms; i++) am->f_terms[i] = term++;
    for (PetscInt i = 0; i < am->g_num_terms; i++) am->g_terms[i] = term++;
  } else if (!am->f_terms) {
    PetscCall(TaoSetUpADMMComputeIndicesFromComplement(tao, n_terms, am->f_num_terms, am->g_num_terms, am->g_terms, &am->f_terms));
  } else if (!am->g_terms) {
    PetscCall(TaoSetUpADMMComputeIndicesFromComplement(tao, n_terms, am->g_num_terms, am->f_num_terms, am->f_terms, &am->g_terms));
  }
  for (PetscInt i = 0; i < am->f_num_terms; i++) PetscCheck(am->f_terms[i] >= 0 && am->f_terms[i] < n_terms, comm, PETSC_ERR_ARG_OUTOFRANGE, "f term %" PetscInt_FMT " = %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", i, am->f_terms[i], n_terms);
  for (PetscInt i = 0; i < am->g_num_terms; i++) PetscCheck(am->g_terms[i] >= 0 && am->g_terms[i] < n_terms, comm, PETSC_ERR_ARG_OUTOFRANGE, "g term %" PetscInt_FMT " = %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", i, am->g_terms[i], n_terms);

  PetscCall(VecGetLayout(tao->solution, &layout));
  PetscCall(VecGetType(tao->solution, &vec_type));

  //// determine the initialization_type and the constraints
  // first find the maps for the f and g terms, and then decide whether to use them or not
  if (am->f_mapped != PETSC_BOOL3_FALSE && am->f_num_terms > 0) PetscCall(TaoTermSumGetSubsetMap(objective, am->f_mapped, am->f_num_terms, am->f_terms, &f_map));
  if (am->g_mapped != PETSC_BOOL3_FALSE && am->g_num_terms > 0) PetscCall(TaoTermSumGetSubsetMap(objective, am->g_mapped, am->g_num_terms, am->g_terms, &g_map));

  if (tao->eq_constrained) {
    // if the user provides equality constraints, it can only mean that the initalization type is ADMM_INITIALIZE_SCATTER
    PetscBool f_map_is_scatter = PETSC_FALSE;
    PetscBool g_map_is_scatter = PETSC_FALSE;
    Vec       zero_solution;
    PetscReal c_norm;

    if (f_map) PetscCall(PetscObjectTypeCompare((PetscObject)f_map, MATSCATTER, &f_map_is_scatter));
    if (g_map) PetscCall(PetscObjectTypeCompare((PetscObject)g_map, MATSCATTER, &g_map_is_scatter));
    PetscCheck(f_map_is_scatter && g_map_is_scatter, comm, PETSC_ERR_ARG_INCOMP, "ADMM can only be used with equality constraints if the maps are scatters");
    am->initialize_type = ADMM_INITIALIZE_SCATTER;

    PetscCall(VecDuplicate(tao->constraints_equality, &am->c));
    PetscCall(VecDuplicate(tao->solution, &zero_solution));
    PetscCall(VecZeroEntries(zero_solution));
    PetscCall(TaoComputeEqualityConstraints(tao, zero_solution, am->c));
    PetscCall(VecNorm(am->c, NORM_2, &c_norm));
    if (c_norm == 0.0) PetscCall(VecDestroy(&am->c));
    PetscCall(TaoComputeJacobianEquality(tao, zero_solution, tao->jacobian_equality, tao->jacobian_equality_pre));
    PetscCall(VecDestroy(&zero_solution));
    PetscCall(MatCreateSubMatrixColumnScatter(tao->jacobian_equality, f_map, &am->A));
    PetscCall(MatCreateSubMatrixColumnScatter(tao->jacobian_equality, g_map, &am->B));
    PetscCall(PetscObjectReference((PetscObject)tao->constraints_equality));
    am->r = tao->constraints_equality;
  } else {
    am->c_norm = 0.0;
    if (f_map && g_map) {
      PetscCheck(!(am->f_mapped == PETSC_BOOL3_TRUE && am->g_mapped == PETSC_BOOL3_TRUE), comm, PETSC_ERR_ARG_INCOMP, "ADMM algorithm cannot separate the map from both terms");
      if (am->f_mapped == PETSC_BOOL3_TRUE) g_map = NULL;      // if the user specifically requested f_map to be separated, do not separate g_map
      else if (am->g_mapped == PETSC_BOOL3_TRUE) f_map = NULL; // and vice versa
      else {
        // an f_map and a g_map have been found, but the user has provided no guidance about whether to separate either, default to using neither
        PetscCall(PetscInfo(tao, "Maps found for both ADMM terms, using neither: call TaoADMMSetTermGroups() to specify whether either map should be used\n"));
        f_map = NULL;
        g_map = NULL;
      }
    }
    if (!f_map && g_map) {
      PetscLayout A_row_layout;

      // the user specified f(u) + g(A(u)), so they want Ax = z
      // A = g_map, B = -I
      am->initialize_type = ADMM_INITIALIZE_Z_AX;
      PetscCall(PetscObjectReference((PetscObject)g_map));
      PetscCall(MatDestroy(&am->A));
      am->A = g_map;
      PetscCall(MatGetLayouts(am->A, &A_row_layout, NULL));
      PetscCall(MatDestroy(&am->B));
      PetscCall(MatCreate(comm, &am->B));
      PetscCall(MatSetLayouts(am->B, A_row_layout, A_row_layout));
      PetscCall(MatSetVecType(am->B, vec_type));
      PetscCall(MatSetType(am->B, MATCONSTANTDIAGONAL));
      PetscCall(MatZeroEntries(am->B));
      PetscCall(MatShift(am->B, -1.0));
    } else if (f_map && !g_map) {
      PetscLayout B_row_layout;

      // the user specified f(B(u)) + g(u), so they want x = Bz
      // A = -I, B = f_map
      am->initialize_type = ADMM_INITIALIZE_X_BZ;
      PetscCall(PetscObjectReference((PetscObject)f_map));
      PetscCall(MatDestroy(&am->B));
      am->B = f_map;
      PetscCall(MatGetLayouts(am->B, &B_row_layout, NULL));
      PetscCall(MatDestroy(&am->A));
      PetscCall(MatCreate(comm, &am->A));
      PetscCall(MatSetLayouts(am->A, B_row_layout, B_row_layout));
      PetscCall(MatSetVecType(am->A, vec_type));
      PetscCall(MatSetType(am->A, MATCONSTANTDIAGONAL));
      PetscCall(MatZeroEntries(am->A));
      PetscCall(MatShift(am->A, -1.0));
    } else {
      PetscAssert(f_map == NULL && g_map == NULL, comm, PETSC_ERR_PLIB, "Invalid map state");
      // the user specified f(x) + g(x), so they want x = z
      // A = -I, B = I
      am->initialize_type = ADMM_INITIALIZE_X_BZ;
      PetscCall(MatDestroy(&am->A));
      PetscCall(MatCreate(comm, &am->A));
      PetscCall(MatSetLayouts(am->A, layout, layout));
      PetscCall(MatSetVecType(am->A, vec_type));
      PetscCall(MatSetType(am->A, MATCONSTANTDIAGONAL));
      PetscCall(MatZeroEntries(am->A));
      PetscCall(MatShift(am->A, -1.0));
      PetscCall(MatDestroy(&am->B));
      PetscCall(MatCreate(comm, &am->B));
      PetscCall(MatSetLayouts(am->B, layout, layout));
      PetscCall(MatSetVecType(am->B, vec_type));
      PetscCall(MatSetType(am->B, MATCONSTANTDIAGONAL));
      PetscCall(MatZeroEntries(am->B));
      PetscCall(MatShift(am->B, 1.0));
    }
    PetscCall(VecDestroy(&am->r));
    PetscCall(MatCreateVecs(am->A, NULL, &am->r));
  }

  PetscCall(VecDestroy(&am->r_halfstep));
  PetscCall(VecDuplicate(am->r, &am->r_halfstep));
  PetscCall(VecDestroy(&am->y));
  PetscCall(VecDuplicate(am->r, &am->y));
  PetscCall(VecDestroy(&am->Ax));
  PetscCall(VecDuplicate(am->r, &am->Ax));
  PetscCall(VecDestroy(&am->Bz));
  PetscCall(VecDuplicate(am->r, &am->Bz));
  PetscCall(MatCreateVecs(am->A, &am->d_x, NULL));
  PetscCall(MatCreateVecs(am->B, &am->d_z, NULL));

  PetscCall(TaoGetOptionsPrefix(tao, &tao_prefix));

  // construct the f and g terms
  PetscCall(TaoTermSumCreateSubset_ADMM(objective, f_map, tao_prefix, "admm_sub_0_", am->f_num_terms, am->f_terms, &am->f));
  PetscCall(TaoTermSumCreateSubset_ADMM(objective, g_map, tao_prefix, "admm_sub_1_", am->g_num_terms, am->g_terms, &am->g));

  // get the parameter vectors
  if (params) {
    PetscCall(TaoSetUpADMMCreateSubParams(params, am->f_num_terms, am->f_terms, am->f, &f_params));
    PetscCall(TaoSetUpADMMCreateSubParams(params, am->g_num_terms, am->g_terms, am->g, &g_params));
  }

  if (!am->x_subsolver) {
    PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &am->x_subsolver));
    PetscCall(TaoSetOptionsPrefix(am->x_subsolver, tao_prefix));
    PetscCall(TaoAppendOptionsPrefix(am->x_subsolver, "admm_sub_0_"));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)am->x_subsolver, (PetscObject)tao, 1));
    PetscCall(TaoSetType(am->x_subsolver, TAONLS));
    if (am->initialize_type == ADMM_INITIALIZE_Z_AX) PetscCall(TaoSetSolution(am->x_subsolver, tao->solution)); // x subproblem can share a solution with the outer tao
  }
  PetscCall(TaoSetTerm(am->x_subsolver, 1.0, am->f, f_params, NULL));
  PetscCall(VecDestroy(&f_params));

  if (!am->z_subsolver) {
    PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &am->z_subsolver));
    PetscCall(TaoSetOptionsPrefix(am->z_subsolver, tao_prefix));
    PetscCall(TaoAppendOptionsPrefix(am->z_subsolver, "admm_sub_1_"));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)am->z_subsolver, (PetscObject)tao, 1));
    PetscCall(TaoSetType(am->z_subsolver, TAONLS));
    if (am->initialize_type == ADMM_INITIALIZE_X_BZ) PetscCall(TaoSetSolution(am->z_subsolver, tao->solution)); // z subproblem can share a solution with the outer tao
  }
  PetscCall(TaoSetTerm(am->z_subsolver, 1.0, am->g, g_params, NULL));
  PetscCall(VecDestroy(&g_params));

  /* we have constructed the constraints (Ax + Bz + c == 0) and initialized x_subsolver with f(x) and z_subsolver with g(z):
     it is now the implementation's job to set up the metric terms */
  if (am->mu_update == TAO_ADMM_UPDATE_ADAPTIVE) {
    PetscCall(TaoADMMSetUp_ARADMM(tao));
  } else {
    PetscCall(TaoADMMSetUp_Basic(tao));
  }

  if (am->setfromoptionscalled) {
    PetscCall(TaoSetFromOptions(am->x_subsolver));
    PetscCall(TaoSetFromOptions(am->z_subsolver));
  }
  PetscCall(TaoSetUp(am->x_subsolver));
  PetscCall(TaoSetUp(am->z_subsolver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_ADMM(Tao tao)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  if (am->destroy) PetscCall((*am->destroy)(tao));
  PetscCall(VecDestroy(&am->Ax));
  PetscCall(VecDestroy(&am->Bz));
  PetscCall(VecDestroy(&am->y));
  PetscCall(VecDestroy(&am->d_x));
  PetscCall(VecDestroy(&am->d_z));
  PetscCall(VecDestroy(&am->r));
  PetscCall(VecDestroy(&am->r_halfstep));
  PetscCall(VecDestroy(&am->c));
  PetscCall(VecDestroy(&am->y));

  PetscCall(MatDestroy(&am->A));
  PetscCall(MatDestroy(&am->B));
  PetscCall(TaoDestroy(&am->x_subsolver));
  PetscCall(TaoDestroy(&am->z_subsolver));
  PetscCall(TaoTermDestroy(&am->f));
  PetscCall(TaoTermDestroy(&am->g));
  PetscCall(PetscFree(am->f_terms));
  PetscCall(PetscFree(am->g_terms));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoADMMSetUpdateType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoADMMGetUpdateType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoADMMSetTermGroups_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoADMMGetTermGroups_C", NULL));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMSetTermGroups_ADMM(Tao tao, PetscInt f_num_terms, const PetscInt f_terms[], PetscBool f_mapped, PetscInt g_num_terms, const PetscInt g_terms[], PetscBool g_mapped)
{
  Tao_ADMM *admm        = (Tao_ADMM *)tao->data;
  PetscInt *f_terms_new = NULL, *g_terms_new = NULL;

  PetscFunctionBegin;
  if (f_terms) {
    PetscCall(PetscMalloc1(f_num_terms, &f_terms_new));
    PetscCall(PetscArraycpy(f_terms_new, f_terms, f_num_terms));
  }
  PetscCall(PetscFree(admm->f_terms));
  admm->f_num_terms = f_num_terms;
  admm->f_terms     = f_terms_new;
  admm->f_mapped    = PetscBoolToBool3(f_mapped);
  if (g_terms) {
    PetscCall(PetscMalloc1(g_num_terms, &g_terms_new));
    PetscCall(PetscArraycpy(g_terms_new, g_terms, g_num_terms));
  }
  PetscCall(PetscFree(admm->g_terms));
  admm->g_num_terms = g_num_terms;
  admm->g_terms     = g_terms_new;
  admm->g_mapped    = PetscBoolToBool3(g_mapped);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMGetTermGroups_ADMM(Tao tao, PetscInt *f_num_terms, const PetscInt *f_terms[], PetscBool *f_mapped, PetscInt *g_num_terms, const PetscInt *g_terms[], PetscBool *g_mapped)
{
  Tao_ADMM *admm = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  if (f_num_terms) *f_num_terms = admm->f_num_terms;
  if (f_terms) *f_terms = admm->f_terms;
  if (f_mapped) *f_mapped = PetscBool3ToBool(admm->f_mapped);
  if (g_num_terms) *g_num_terms = admm->g_num_terms;
  if (g_terms) *g_terms = admm->g_terms;
  if (g_mapped) *g_mapped = PetscBool3ToBool(admm->g_mapped);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOADMM - Alternating direction method of multipliers method for solving linear problems with
            constraints. in a $ \min_x f(x) + g(z)$  s.t. $Ax+Bz=c$.
            This algorithm employs two sub Tao solvers, of which type can be specified
            by the user. User need to provide ObjectiveAndGradient routine, and/or HessianRoutine for both subsolvers.
            Hessians can be given boolean flag determining whether they change with respect to a input vector. This can be set via
            `TaoADMMSet{Misfit,Regularizer}HessianChangeStatus()`.
            Second subsolver does support `TAOSHELL`. It should be noted that L1-norm is used for objective value for `TAOSHELL` type.
            There is option to set regularizer option, and currently soft-threshold is implemented. For spectral penalty update,
            currently there are basic option and adaptive option.
            Constraint is set at Ax+Bz=c, and A and B can be set with `TaoADMMSet{Misfit,Regularizer}ConstraintJacobian()`.
            c can be set with `TaoADMMSetConstraintVectorRHS()`.
            The user can also provide regularizer weight for second subsolver. {cite}`xu2017adaptive`

  Options Database Keys:
+ -tao_admm_regularizer_coefficient        - regularizer constant (default 1.e-6)
. -tao_admm_spectral_penalty               - Constant for Augmented Lagrangian term (default 1.)
. -tao_admm_relaxation_parameter           - relaxation parameter for Z update (default 1.)
. -tao_admm_tolerance_update_factor        - ADMM dynamic tolerance update factor (default 1.e-12)
. -tao_admm_spectral_penalty_update_factor - ADMM spectral penalty update curvature safeguard value (default 0.2)
. -tao_admm_minimum_spectral_penalty       - Set ADMM minimum spectral penalty (default 0)
. -tao_admm_dual_update                    - Lagrangian dual update policy ("basic","adaptive","adaptive-relaxed") (default "basic")
- -tao_admm_regularizer_type               - ADMM regularizer update rule ("user","soft-threshold") (default "soft-threshold")

  Level: beginner

.seealso: `TaoADMMGetSpectralPenalty()`, `TaoADMMGetMisfitSubsolver()`, `TaoADMMGetRegularizationSubsolver()`, `TaoADMMSetConstraintVectorRHS()`,
          `TaoADMMSetMinimumSpectralPenalty()`, `TaoADMMSetRegularizerCoefficient()`, `TaoADMMGetRegularizerCoefficient()`,
          `TaoADMMGetDualVector()`, `TaoADMMSetRegularizerType()`,
          `TaoADMMGetRegularizerType()`, `TaoADMMSetUpdateType()`, `TaoADMMGetUpdateType()`
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_ADMM(Tao tao)
{
  Tao_ADMM *am;

  PetscFunctionBegin;
  PetscCall(PetscNew(&am));

  tao->ops->destroy         = TaoDestroy_ADMM;
  tao->ops->setup           = TaoSetUp_ADMM;
  tao->ops->setfromoptions  = TaoSetFromOptions_ADMM;
  tao->ops->view            = TaoView_ADMM;
  tao->ops->solve           = TaoSolve_ADMM;
  tao->ops->convergencetest = TaoConvergenceTest_ADMM;

  PetscCall(TaoParametersInitialize(tao));
  PetscCall(TaoSetConvergenceTest(tao, TaoConvergenceTest_ADMM, NULL));

  tao->data               = (void *)am;
  am->mu                  = 1.;
  am->mu_update           = TAO_ADMM_UPDATE_BASIC;
  am->relaxation_gamma    = 1.5;
  am->f_num_terms         = PETSC_DECIDE;
  am->f_mapped            = PETSC_BOOL3_UNKNOWN;
  am->g_num_terms         = PETSC_DECIDE;
  am->g_mapped            = PETSC_BOOL3_UNKNOWN;
  am->x_inexact           = PETSC_TRUE;
  am->z_inexact           = PETSC_TRUE;
  am->adaptivity_period   = 2;
  am->correlation_epsilon = 0.2;

  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &am->x_subsolver));
  PetscCall(TaoSetOptionsPrefix(am->x_subsolver, "admm_sub_0_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)am->x_subsolver, (PetscObject)tao, 1));
  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &am->z_subsolver));
  PetscCall(TaoSetOptionsPrefix(am->z_subsolver, "admm_sub_1_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)am->z_subsolver, (PetscObject)tao, 1));

  PetscCall(TaoSetType(am->x_subsolver, TAONLS));
  PetscCall(TaoSetType(am->z_subsolver, TAONLS));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoADMMSetUpdateType_C", TaoADMMSetUpdateType_ADMM));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoADMMGetUpdateType_C", TaoADMMGetUpdateType_ADMM));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoADMMSetTermGroups_C", TaoADMMSetTermGroups_ADMM));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoADMMGetTermGroups_C", TaoADMMGetTermGroups_ADMM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMSetSpectralPenalty - Set the spectral penalty (mu) value

  Collective

  Input Parameters:
+ tao - the `Tao` solver context
- mu  - spectral penalty

  Level: advanced

.seealso: `TaoADMMSetMinimumSpectralPenalty()`, `TAOADMM`
@*/
PetscErrorCode TaoADMMSetSpectralPenalty(Tao tao, PetscReal mu)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  am->mu = mu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMGetSpectralPenalty - Get the spectral penalty (mu) value

  Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. mu - spectral penalty

  Level: advanced

.seealso: `TaoADMMSetMinimumSpectralPenalty()`, `TaoADMMSetSpectralPenalty()`, `TAOADMM`
@*/
PetscErrorCode TaoADMMGetSpectralPenalty(Tao tao, PetscReal *mu)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(mu, 2);
  *mu = am->mu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMGetMisfitSubsolver - Get the pointer to the misfit subsolver inside `TAOADMM`

  Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. misfit - the `Tao` subsolver context

  Level: advanced

.seealso: `TAOADMM`, `Tao`
@*/
PetscErrorCode TaoADMMGetMisfitSubsolver(Tao tao, Tao *misfit)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  *misfit = am->x_subsolver;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMGetRegularizationSubsolver - Get the pointer to the regularization subsolver inside `TAOADMM`

  Collective

  Input Parameter:
. tao - the `Tao` solver context

  Output Parameter:
. reg - the `Tao` subsolver context

  Level: advanced

.seealso: `TAOADMM`, `Tao`
@*/
PetscErrorCode TaoADMMGetRegularizationSubsolver(Tao tao, Tao *reg)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  *reg = am->z_subsolver;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMSetMinimumSpectralPenalty - Set the minimum value for the spectral penalty

  Collective

  Input Parameters:
+ tao - the `Tao` solver context
- mu  - minimum spectral penalty value

  Level: advanced

.seealso: `TaoADMMGetSpectralPenalty()`, `TAOADMM`
@*/
PetscErrorCode TaoADMMSetMinimumSpectralPenalty(Tao tao, PetscReal mu)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  am->mu_min = mu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMGetDualVector - Returns the dual vector associated with the current `TAOADMM` state

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. Y - the current solution

  Level: intermediate

.seealso: `TAOADMM`
@*/
PetscErrorCode TaoADMMGetDualVector(Tao tao, Vec *Y)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  *Y = am->y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMSetUpdateType - Set update routine for `TAOADMM` routine

  Not Collective

  Input Parameters:
+ tao  - the `Tao` context
- type - spectral parameter update type

  Level: intermediate

.seealso: `TaoADMMGetUpdateType()`, `TaoADMMUpdateType`, `TAOADMM`
@*/
PetscErrorCode TaoADMMSetUpdateType(Tao tao, TaoADMMUpdateType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(tao, type, 2);
  PetscTryMethod(tao, "TaoADMMSetUpdateType_C", (Tao, TaoADMMUpdateType), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMGetUpdateType - Gets the type of spectral penalty update routine for `TAOADMM`

  Not Collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameter:
. type - the type of spectral penalty update routine

  Level: intermediate

.seealso: `TaoADMMSetUpdateType()`, `TaoADMMUpdateType`, `TAOADMM`
@*/
PetscErrorCode TaoADMMGetUpdateType(Tao tao, TaoADMMUpdateType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscUseMethod(tao, "TaoADMMGetUpdateType_C", (Tao, TaoADMMUpdateType *), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMSetTermGroups - Split the terms in the objective function of a `Tao` into groups for the ADMM subsolvers

  Logically collective

  Input Parameters:
+ tao         - the `Tao` context
. f_num_terms - the size of `f_terms`
. f_terms     - (optional) the indices of the terms that will be grouped into the $f$ ADMM term (if `NULL`, this will be taken to be the first `f_num_terms` terms)
. f_mapped    - whether the linear map should be separated from the $f$ terms (see Note)
. g_num_terms - the size of `g_terms`
. g_terms     - (optional) the indices of the terms that will be grouped into the $g$ ADMM term (if `NULL`, this will be taken ot be the first `g_num_terms` terms that are not in `f_terms`)
- g_mapped    - whether the linear map should be separated from the $f$ terms (see Note)

  Level: intermediate

  Notes:
  The ADMM algorithm operates on an optimization problem with the form

  ```{math}
  \begin{aligned}
    &\min_{x,z} f(x) + g(z) \\
    &\text{such that} A x + B z = c.
  \end{aligned}
  ```

  This function controls how a problem of this form is constructed from `tao`.

  When the objective function of `tao` is $f(x) + g(x)$, there is no need to call `TaoADMMSetTermGroups()`: this will automatically be transformed into

  ```{math}
  \min_{x,z} f(x) + g(z) \quad \text{such that} x = z.
  ```

  When the objective function of `tao` is $f(x) + g(Ax)$, where $A$ is a linear map, there is also no need to call `TaoADMMSetTermGroups()`: $A$ is separated from $g$ by default, so ADMM will solve

  ```{math}
  \min_{x,z} f(x) + g(z) \quad \text{such that} A x = z.
  ```

  If you would like to keep $g$ and $A$ together, you should call `TaoADMMSetTermGroups()` with `g_mapped = PETSC_FALSE`, so that ADMM will solve

  ```{math}
  \min_{x,z} f(x) + g(A z) \quad \text{such that} x = z.
  ```

  Similarly if the objective function of `tao` is $f(Ax) + g(x)$, by default this will become

  ```{math}
  \min_{x,z} f(x) + g(z) \quad \text{such that} x = A z,
  ```

  and you should call `TaoADMMSetTermGroups()` with `f_mapped = PETSC_FALSE` to keep $f$ and $A$ together.

  If you have an objective function with more than two terms, like $f(x) + g(x) + h(x)$, and you want to group $f(x) + h(x)$ into one function for the ADMM algorithm, you should call

.vb
  PetscInt f_terms[2] = {0, 2};

  TaoADMMSetTermGroups(tao, 2, f_terms, PETSC_FALSE, 1, NULL, PETSC_FALSE);
.ve

  and ADMM will solve

  ```{math}
  \min_{x,z} \{f(x) + h(x)\} + g(z) \quad \text{such that} x = z.
  ```

.seealso: [](ch_tao), `TAOADMM`
@*/
PetscErrorCode TaoADMMSetTermGroups(Tao tao, PetscInt f_num_terms, const PetscInt f_terms[], PetscBool f_mapped, PetscInt g_num_terms, const PetscInt g_terms[], PetscBool g_mapped)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCheck(f_num_terms >= 0 && g_num_terms >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "f_num_terms = %" PetscInt_FMT ", g_num_terms = %" PetscInt_FMT ", both should be nonnegative", f_num_terms, g_num_terms);
  PetscTryMethod(tao, "TaoADMMSetTermGroups_C", (Tao, PetscInt, const PetscInt[], PetscBool, PetscInt, const PetscInt[], PetscBool), (tao, f_num_terms, f_terms, f_mapped, g_num_terms, g_terms, g_mapped));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoADMMGetTermGroups - Get the splits set with `TaoADMMSetTermGroups`

  Logically collective

  Input Parameter:
. tao - the `Tao` context

  Output Parameters:
+ f_num_terms - the size of `f_terms`
. f_terms     - the indices of the terms that will be grouped into the $f$ ADMM term
. f_mapped    - whether the linear map should be separated from the $f$ terms
. g_num_terms - the size of `g_terms`
. g_terms     - the indices of the terms that will be grouped into the $g$ ADMM term
- g_mapped    - wheter the linear map should be separated from the $g$ terms

  Level: intermediate

.seealso: [](ch_tao), `TAOADMM`
@*/
PetscErrorCode TaoADMMGetTermGroups(Tao tao, PetscInt *f_num_terms, const PetscInt *f_terms[], PetscBool *f_mapped, PetscInt *g_num_terms, const PetscInt *g_terms[], PetscBool *g_mapped)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod(tao, "TaoADMMGetTermGroups_C", (Tao, PetscInt *, const PetscInt *[], PetscInt *, const PetscInt *[]), (tao, f_num_terms, f_terms, g_num_terms, g_terms));
  PetscFunctionReturn(PETSC_SUCCESS);
}
