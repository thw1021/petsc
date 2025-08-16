#include "admm_basic.h"

typedef struct _n_TaoADMM_ARADMM {
  TaoADMM_Basic basic;
  Vec           y_halfstep;
  Vec           y_halfstep_0;
  Vec           y_0;
  Vec           Ax_0;
  Vec           Bz_0;
} TaoADMM_ARADMM;

static PetscBool  cited      = PETSC_FALSE;
static const char citation[] = "@misc{xu2017adaptive,\n"
                               "   title={Adaptive Relaxed ADMM: Convergence Theory and Practical Implementation},\n"
                               "   author={Zheng Xu and Mario A. T. Figueiredo and Xiaoming Yuan and Christoph Studer and Tom Goldstein},\n"
                               "   year={2017},\n"
                               "   eprint={1704.02712},\n"
                               "   archivePrefix={arXiv},\n"
                               "   primaryClass={cs.CV}\n"
                               "}  \n";

static PetscErrorCode TaoADMMComputeUpdateScales_ARADMM(Tao tao, Vec dy, Vec dh, PetscReal *alpha_inv, PetscReal *alpha_cor)
{
  PetscScalar dots[2];
  Vec         vecs[2] = {dy, dh};
  PetscScalar dy_norm_2, dh_norm_2, dy_dot_dh;
  PetscReal   alpha_inv_SD, alpha_inv_MG;

  PetscFunctionBegin;
  PetscCall(VecMDot(dy, 2, vecs, dots));
  dy_norm_2 = dots[0];
  /* signs are reversed from the paper because the paper defines r = c - A x - Bz and we use r = A x + B z + c
     which means that our multipliers y and y_hat have opposite signs from the paper */
  dy_dot_dh = -dots[1];
  PetscCall(VecDot(dh, dh, &dh_norm_2));

  alpha_inv_SD = PetscRealPart(dy_norm_2 / dy_dot_dh);
  alpha_inv_MG = PetscRealPart(dy_dot_dh / dh_norm_2);
  *alpha_inv   = (2 * alpha_inv_MG > alpha_inv_SD) ? alpha_inv_MG : (alpha_inv_SD - alpha_inv_MG / 2.0);
  PetscCall(PetscInfo(tao, "  SD % 9.2e (%8.2e / % 9.2e) MG % 9.2e (% 9.2e / %8.2e) final % 9.2e\n", (double)alpha_inv_SD, (double)PetscRealPart(dy_norm_2), (double)PetscRealPart(dy_dot_dh), (double)alpha_inv_MG, (double)PetscRealPart(dy_dot_dh), (double)PetscRealPart(dh_norm_2), (double)*alpha_inv));
  *alpha_cor = PetscRealPart(dy_dot_dh) / PetscSqrtReal(dy_norm_2 * dh_norm_2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMUpdateParameters_ARADMM(Tao tao)
{
  Tao_ADMM       *am      = (Tao_ADMM *)tao->data;
  TaoADMM_ARADMM *aradmm  = (TaoADMM_ARADMM *)am->data;
  Vec             y_hat   = aradmm->y_halfstep;
  Vec             y_hat_0 = aradmm->y_halfstep_0;
  Vec             Ax      = am->Ax;
  Vec             Ax_0    = aradmm->Ax_0;
  Vec             Bz      = am->Bz;
  Vec             Bz_0    = aradmm->Bz_0;
  Vec             y       = am->y;
  Vec             y_0     = aradmm->y_0;
  PetscReal       eps_cor = am->correlation_epsilon;
  PetscScalar     alpha_inv, alpha_cor;
  PetscScalar     beta_inv, beta_cor;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation, &cited));
  PetscCall(VecAXPY(y_0, -1.0, y));
  PetscCall(VecAXPY(y_hat_0, -1.0, y_hat));
  PetscCall(VecAXPY(Ax_0, -1.0, Ax));
  PetscCall(VecAXPY(Bz_0, -1.0, Bz));
  PetscCall(PetscInfo(tao, "x subproblem curvature estimate:\n"));
  PetscCall(TaoADMMComputeUpdateScales_ARADMM(tao, y_hat_0, Ax_0, &alpha_inv, &alpha_cor));
  PetscCall(PetscInfo(tao, "z subproblem curvature estimate:\n"));
  PetscCall(TaoADMMComputeUpdateScales_ARADMM(tao, y_0, Bz_0, &beta_inv, &beta_cor));

  if (alpha_cor > eps_cor && beta_cor > eps_cor) {
    am->mu               = PetscSqrtReal(alpha_inv * beta_inv);
    am->relaxation_gamma = 1.0 + 2.0 * PetscSqrtReal(alpha_inv * beta_inv) / (alpha_inv + beta_inv);
    PetscCall(PetscInfo(tao, "case alpha %8.2e, beta %8.2e > eps %8.2e, new ADMM parameters:   mu %8.2e gamma %8.2e\n", (double)alpha_cor, (double)beta_cor, (double)eps_cor, (double)am->mu, (double)am->relaxation_gamma));
  } else if (alpha_cor > eps_cor) {
    am->mu               = alpha_inv;
    am->relaxation_gamma = 1.9;
    PetscCall(PetscInfo(tao, "case alpha %8.2e > eps %8.2e > beta % 9.2e, new ADMM parameters: mu %8.2e gamma %8.2e\n", (double)alpha_cor, (double)eps_cor, (double)beta_cor, (double)am->mu, (double)am->relaxation_gamma));
  } else if (beta_cor > eps_cor) {
    am->mu               = beta_inv;
    am->relaxation_gamma = 1.1;
    PetscCall(PetscInfo(tao, "case beta %8.2e > eps %8.2e > alpha % 9.2e, new ADMM parameters: mu %8.2e gamma %8.2e\n", (double)beta_cor, (double)eps_cor, (double)alpha_cor, (double)am->mu, (double)am->relaxation_gamma));
  } else {
    am->relaxation_gamma = 1.5;
    PetscCall(PetscInfo(tao, "case eps %8.2e > alpha % 9.2e, beta % 9.2e, new ADMM parameters: mu %8.2e gamma %8.2e\n", (double)eps_cor, (double)alpha_cor, (double)beta_cor, (double)am->mu, (double)am->relaxation_gamma));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMUpdateXSubproblem_ARADMM(Tao tao)
{
  Tao_ADMM       *am     = (Tao_ADMM *)tao->data;
  TaoADMM_ARADMM *aradmm = (TaoADMM_ARADMM *)am->data;
  PetscInt        T      = am->adaptivity_period;
  PetscInt        iter;

  PetscFunctionBegin;
  PetscCall(TaoGetIterationNumber(tao, &iter));
  if (iter > 0 && T > 0 && ((iter + T - 1) % T) == 0) {
    if (iter > 1) PetscCall(TaoADMMUpdateParameters_ARADMM(tao));
    PetscCall(VecCopy(am->y, aradmm->y_0));
    PetscCall(VecCopy(aradmm->y_halfstep, aradmm->y_halfstep_0));
    PetscCall(VecCopy(am->Ax, aradmm->Ax_0));
    PetscCall(VecCopy(am->Bz, aradmm->Bz_0));
  }
  PetscCall(TaoADMMUpdateXSubproblem_Basic(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMUpdateZSubproblem_ARADMM(Tao tao)
{
  Tao_ADMM       *am     = (Tao_ADMM *)tao->data;
  TaoADMM_ARADMM *aradmm = (TaoADMM_ARADMM *)am->data;
  PetscInt        T      = am->adaptivity_period;
  PetscInt        iter;

  PetscFunctionBegin;
  PetscCall(TaoGetIterationNumber(tao, &iter));
  if (T > 0 && (iter % T) == 0) {
    // compute y_halfstep = am->y + r_halfstep
    PetscCall(VecWAXPY(aradmm->y_halfstep, am->mu, am->r_halfstep, am->y));
  }
  PetscCall(TaoADMMUpdateZSubproblem_Basic(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoADMMDestroy_ARADMM(Tao tao)
{
  Tao_ADMM       *am     = (Tao_ADMM *)tao->data;
  TaoADMM_ARADMM *aradmm = (TaoADMM_ARADMM *)am->data;

  PetscFunctionBegin;
  PetscCall(TaoADMMDestroy_Basic_Internal(tao, &aradmm->basic));
  PetscCall(VecDestroy(&aradmm->y_0));
  PetscCall(VecDestroy(&aradmm->y_halfstep_0));
  PetscCall(VecDestroy(&aradmm->y_halfstep));
  PetscCall(VecDestroy(&aradmm->Ax_0));
  PetscCall(VecDestroy(&aradmm->Bz_0));
  PetscCall(PetscFree(aradmm));
  am->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoADMMSetUp_ARADMM(Tao tao)
{
  Tao_ADMM       *am     = (Tao_ADMM *)tao->data;
  TaoADMM_ARADMM *aradmm = NULL;

  PetscFunctionBegin;
  PetscCall(PetscNew(&aradmm));
  am->data = (void *)aradmm;
  PetscCall(TaoADMMSetUp_Basic_Internal(tao, &aradmm->basic));
  PetscCall(VecDuplicate(am->y, &aradmm->y_0));
  PetscCall(VecDuplicate(am->y, &aradmm->y_halfstep));
  PetscCall(VecDuplicate(am->y, &aradmm->y_halfstep_0));
  PetscCall(VecDuplicate(am->Ax, &aradmm->Ax_0));
  PetscCall(VecDuplicate(am->Bz, &aradmm->Bz_0));
  am->destroy           = TaoADMMDestroy_ARADMM;
  am->updatexsubproblem = TaoADMMUpdateXSubproblem_ARADMM;
  am->updatezsubproblem = TaoADMMUpdateZSubproblem_ARADMM;
  PetscFunctionReturn(PETSC_SUCCESS);
}
