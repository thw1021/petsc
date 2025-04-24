#include "admm.h"

PETSC_UNUSED static PetscErrorCode TaoADMMCopyOldVecs_Linearized(Tao tao, Vec x, Vec z, PETSC_UNUSED Vec y, Vec Ax, Vec Bz, Vec r)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  // x_old, z_old, Ax_old, and Bz_old are needed for TaoADMMComputeDualResidualNorm_Linearized
  PetscCall(TaoADMMVecDuplicateAndCopy(am->x_subsolver->solution, &am->x_old));
  PetscCall(TaoADMMVecDuplicateAndCopy(am->z_subsolver->solution, &am->z_old));
  PetscCall(TaoADMMVecDuplicateAndCopy(am->Ax, &am->Ax_old));
  PetscCall(TaoADMMVecDuplicateAndCopy(am->Bz, &am->Bz_old));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// the subproblem objective is f(x) + (mu_x/2) || x - (x_k - (mu / mu_x) * A'(gamma * r_k + (1/mu) * y)) ||_2^2
PETSC_UNUSED static PetscErrorCode TaoADMMUpdateXSubproblem_Linearized_Internal(Tao tao, Vec x, Vec z, Vec y, Vec Ax, Vec Bz, Vec r, Tao x_subsolver, PetscBool swapped)
{
  Tao_ADMM   *am = (Tao_ADMM *)tao->data;
  PetscReal   mu    = am->mu;
  Vec         w     = NULL; // TODO
  PetscReal   mu_x  = swapped ? am->linearized_mu_x : am->linearized_mu_z;
  PetscReal   gamma = swapped ? am->relaxation_gamma : 1.0;
  Mat         A     = swapped ? am->A : am->B;
  TaoTerm     objective, metric;
  PetscReal   mu_old;
  Vec         params, v;
  Mat         map;
  PetscInt    n_terms;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(TaoGetTerm(x_subsolver, NULL, &objective, &params, NULL));
  PetscCall(TaoTermSumGetNumSubterms(objective, &n_terms));
  // update the spectral penalty
  PetscCall(TaoTermSumGetSubterm(objective, n_terms - 1, &prefix, &mu_old, &metric, &map));
  PetscCall(TaoTermSumSetSubterm(objective, n_terms - 1, prefix, mu_x, metric, map));
  // update the bias vector v
  PetscCall(VecNestGetSubVec(params, n_terms - 1, &v));
  PetscCall(VecCopy(r, w));
  PetscCall(VecScale(w, gamma));
  PetscCall(VecAXPY(w, 1.0 / mu, y));
  PetscCall(MatMultTranspose(A, w, v));
  PetscCall(VecAYPX(v, - mu / mu_x, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode TaoADMMUpdateXSubproblem_Linearized(Tao tao, Vec x, Vec z, Vec y, Vec Ax, Vec Bz, Vec c, Tao x_subsolver)
{
  PetscFunctionBegin;
  PetscCall(TaoADMMUpdateXSubproblem_Linearized_Internal(tao, x, z, y, Ax, Bz, c, x_subsolver, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode TaoADMMUpdateZSubproblem_Linearized(Tao tao, Vec x, Vec z, Vec y, Vec Ax, Vec Bz, Vec c, Tao z_subsolver)
{
  PetscFunctionBegin;
  // the is symmetric with the linearized x update with x and z terms swapped
  PetscCall(TaoADMMUpdateXSubproblem_Linearized_Internal(tao, z, x, y, Bz, Ax, c, z_subsolver, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode TaoADMMComputeDualResidualNorm_Linearized(Tao tao, Vec x, Vec z, Vec y, Vec Ax, Vec Bz, Vec c, PetscReal *dual_norm)
{
  Tao_ADMM   *am      = (Tao_ADMM *)tao->data;
  Vec         x_old   = am->x_old;
  Vec         Ax_old  = am->Ax_old;
  Vec         Bz_old  = am->Bz_old;
  Mat         A       = am->A;
  PetscReal   mu      = am->mu;
  PetscReal   mu_x    = am->linearized_mu_x;

  PetscFunctionBegin;
  PetscCall(VecAXPY(x_old, -1.0, x));
  PetscCall(VecScale(x, -mu_x));
  PetscCall(VecAXPY(Ax_old, -1.0, Ax));
  PetscCall(VecAXPY(Bz_old, -1.0, Bz));
  PetscCall(VecAXPY(Ax_old, +1.0, Bz_old));
  PetscCall(VecScale(Ax_old, mu));
  PetscCall(MatMultTransposeAdd(A, Ax_old, x_old, x_old));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoADMMSetUp_Linearized(Tao tao)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

