#include "admm_basic.h"

/* the subproblem objective is \min_u h(u) + (mu/2) || C u - (C u_k - gamma * r_k - (1/mu) * y_k) ||_2^2

   for the x subproblem:

     u     = x
     h     = f
     C     = A
     r_k   = A x_k + B z_k - c
     gamma = 1

   for the z subproblem:

     u     = z
     h     = g
     C     = B
     r_k   = A x_{k+1} + B z_k - c (aka r_{k+1/2})
     gamma = relaxation_gamma
*/
static PetscErrorCode TaoADMMUpdateSubproblem_Basic(Tao tao, Vec Cu_k, Vec r_k, Vec y_k, Tao u_subsolver, PetscReal gamma, PetscReal scale)
{
  Tao_ADMM   *am    = (Tao_ADMM *)tao->data;
  PetscReal   mu    = am->mu;
  TaoTerm     objective, metric;
  PetscReal   mu_old;
  Vec         params;
  Vec         v;
  Mat         map;
  PetscInt    n_terms;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(TaoGetTerm(u_subsolver, NULL, &objective, &params, NULL));
  PetscCall(TaoTermSumGetNumSubterms(objective, &n_terms));
  // update the spectral penalty
  PetscCall(TaoTermSumGetSubterm(objective, n_terms - 1, &prefix, &mu_old, &metric, &map));
  PetscCall(TaoTermSumSetSubterm(objective, n_terms - 1, prefix, mu, metric, map));
  // update the bias vector v
  PetscCall(VecNestGetSubVec(params, n_terms - 1, &v));
  PetscCall(VecCopy(Cu_k, v));
  PetscCall(VecAXPY(v, -gamma, r_k));
  PetscCall(VecAXPY(v, -1.0 / mu, y_k));
  PetscCall(VecScale(v, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoADMMUpdateXSubproblem_Basic(Tao tao)
{
  Tao_ADMM      *am    = (Tao_ADMM *)tao->data;
  TaoADMM_Basic *basic = (TaoADMM_Basic *)am->data;

  PetscFunctionBegin;
  PetscCall(VecCopy(am->y, basic->y_old)); // store y_old before the x update to use in dual residual calculation
  PetscCall(TaoADMMUpdateSubproblem_Basic(tao, am->Ax, am->r, am->y, am->x_subsolver, 1.0, basic->A_scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoADMMUpdateZSubproblem_Basic(Tao tao)
{
  Tao_ADMM      *am    = (Tao_ADMM *)tao->data;
  TaoADMM_Basic *basic = (TaoADMM_Basic *)am->data;

  PetscFunctionBegin;
  PetscCall(TaoADMMUpdateSubproblem_Basic(tao, am->Bz, am->r_halfstep, am->y, am->z_subsolver, am->relaxation_gamma, basic->B_scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Starting with d_x being the gradient / a vector in the subgradient of the x_subsolver when it terminates:

     d_x \in \partial f(x_{k+1}) + \mu A'(A x_{k+1} - (A x_k - r_k - (1/\mu) * y_k))

     d_x \in \partial f(x_{k+1}) + A'y_k + \mu A'(A x_{k+1} - (A x_k - r_k))

     d_x \in \partial f(x_{k+1}) + A'y_{k+1} + A'(y_k - y_{k+1} + \mu (A x_{k+1} - (A x_k - r_k)))

     d_x \in \partial f(x_{k+1}) + A'y_{k+1} + A'(y_k - y_{k+1} + \mu (r_k + A x_{k+1} - A x_k))

     d_x \in \partial f(x_{k+1}) + A'y_{k+1} + A'(y_k - y_{k+1} + \mu r_{k+1/2})

     d_x - A'(y_k - y_{k+1} + \mu r_{k+1/2}) \in \partial f(x_{k+1}) + A'y_{k+1}
                                                 \_____________________________/
                                           this is the x component of the dual residual

   Starting with d_z being the gradient / a vector in the subgradient of the z_subsolver when it terminates:

     d_z \in \partial \partial g(x_{k+1}) + \mu B'( B z_{k+1} - (B z_k - gamma * r_{k+1/2} - (1/\mu) * y_k))

     d_z \in \partial \partial g(x_{k+1}) + B'(y_k + \mu(B z_{k+1} - (B z_k - gamma * r_{k+1/2}))

     d_z \in \partial \partial g(x_{k+1}) + B'(y_k + \mu(gamma * r_{k+1/2} + B z_{k+1} - B z_k))

     d_z \in \partial \partial g(x_{k+1}) + B'(y_k + \mu(r_{k+1/2} + B z_{k+1} - B z_k + (gamma - 1)r_{k+1/2}))

     d_z \in \partial \partial g(x_{k+1}) + B'(y_k + \mu(r_k + (gamma - 1)r_{k+1/2}))

     d_z \in \partial \partial g(x_{k+1}) + B'(y_k + \mu(r_k + (gamma - 1)r_{k+1/2}))
                                      \___________________________________/
                                          this is the update for y_k+1

     d_z \in \partial \partial g(x_{k+1}) + B'y_k+1
                      \___________________________/
                  this is the z component of the dual residual

 */
static PetscErrorCode TaoADMMDualResidual_Basic(Tao tao)
{
  Tao_ADMM *am     = (Tao_ADMM *)tao->data;
  TaoADMM_Basic *basic = (TaoADMM_Basic *)am->data;
  Vec       y_old  = basic->y_old;
  Vec       d_x    = am->d_x;
  Mat       A      = am->A;
  PetscReal mu     = am->mu;

  PetscFunctionBegin;
  PetscCall(VecAXPY(y_old, -1.0, am->y));
  PetscCall(VecAXPY(y_old, mu, am->r_halfstep));
  PetscCall(VecScale(y_old, -1.0));
  PetscCall(MatMultTransposeAdd(A, y_old, d_x, d_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoADMMDestroy_Basic_Internal(Tao tao, TaoADMM_Basic *basic)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&basic->y_old));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMDestroy_Basic(Tao tao)
{
  Tao_ADMM      *am    = (Tao_ADMM *)tao->data;
  TaoADMM_Basic *basic = (TaoADMM_Basic *)am->data;

  PetscFunctionBegin;
  PetscCall(TaoADMMDestroy_Basic_Internal(tao, basic));
  PetscCall(PetscFree(basic));
  am->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMBasicAddMetric(Tao subsolver, Mat A, Vec y, PetscReal mu, PetscReal *A_scale)
{
  PetscBool is_constdiag;
  TaoTerm   x_metric;
  Vec       v;
  const char *prefix;

  PetscFunctionBegin;
  *A_scale = 1.0;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATCONSTANTDIAGONAL, &is_constdiag));
  if (is_constdiag) {
    PetscReal scale;

    PetscCall(MatConstantDiagonalGetConstant(A, &scale));
    if (scale == 1.0 || scale == -1.0) {
      *A_scale = scale;
    } else {
      is_constdiag = PETSC_FALSE;
    }
  }
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)subsolver), &x_metric));
  PetscCall(TaoTermSetSolutionTemplate(x_metric, y));
  PetscCall(TaoTermSetType(x_metric, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetParametersMode(x_metric, TAOTERM_PARAMETERS_REQUIRED));
  PetscCall(VecDuplicate(y, &v));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)subsolver, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)x_metric, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)x_metric, "metric_"));
  PetscCall(TaoAddTerm(subsolver, "metric_", mu, x_metric, v, is_constdiag ? NULL: A));
  PetscCall(VecDestroy(&v));
  PetscCall(TaoTermDestroy(&x_metric));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoADMMSetUp_Basic_Internal(Tao tao, TaoADMM_Basic *basic)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(am->y, &basic->y_old));
  PetscCall(TaoADMMBasicAddMetric(am->x_subsolver, am->A, am->y, am->mu, &basic->A_scale));
  PetscCall(TaoADMMBasicAddMetric(am->z_subsolver, am->B, am->y, am->mu, &basic->B_scale));
  am->updatexsubproblem = TaoADMMUpdateXSubproblem_Basic;
  am->updatezsubproblem = TaoADMMUpdateZSubproblem_Basic;
  am->dualresidual      = TaoADMMDualResidual_Basic;
  am->destroy           = TaoADMMDestroy_Basic;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoADMMSetUp_Basic(Tao tao)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;
  TaoADMM_Basic *basic = NULL;

  PetscFunctionBegin;
  PetscCall(PetscNew(&basic));
  am->data = (void *)basic;
  PetscCall(TaoADMMSetUp_Basic_Internal(tao, basic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
