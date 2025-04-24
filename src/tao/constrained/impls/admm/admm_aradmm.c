#include "admm.h"

static PetscBool  cited      = PETSC_FALSE;
static const char citation[] = "@misc{xu2017adaptive,\n"
                               "   title={Adaptive Relaxed ADMM: Convergence Theory and Practical Implementation},\n"
                               "   author={Zheng Xu and Mario A. T. Figueiredo and Xiaoming Yuan and Christoph Studer and Tom Goldstein},\n"
                               "   year={2017},\n"
                               "   eprint={1704.02712},\n"
                               "   archivePrefix={arXiv},\n"
                               "   primaryClass={cs.CV}\n"
                               "}  \n";

static PetscErrorCode TaoADMMComputeUpdateScales_ARADMM(Vec dy, Vec dh, PetscReal *alpha_inv, PetscReal *alpha_cor)
{
  PetscScalar dots[2];
  Vec         vecs[2] = {dy, dh};
  PetscScalar dy_norm_2, dh_norm_2, dy_dot_dh;
  PetscReal   alpha_inv_SD, alpha_inv_MG;

  PetscFunctionBegin;
  PetscCall(VecMDot(dy, 2, vecs, dots));
  dy_norm_2 = dots[0];
  dy_dot_dh = dots[1];
  PetscCall(VecDot(dh, dh, &dh_norm_2));

  alpha_inv_SD = PetscRealPart(dy_norm_2 / dy_dot_dh);
  alpha_inv_MG = PetscRealPart(dy_dot_dh / dh_norm_2);
  *alpha_inv = (2 * alpha_inv_MG > alpha_inv_SD) ? alpha_inv_MG : (alpha_inv_SD - alpha_inv_MG / 2.0);
  *alpha_cor = PetscRealPart(dy_dot_dh) / PetscSqrtReal(dy_norm_2 * dh_norm_2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode TaoADMMUpdateParameters_ARADMM(Tao tao)
{
  Tao_ADMM *am = (Tao_ADMM *)tao->data;
  Vec       y_hat = am->y_halfstep;
  Vec       y_hat_old = am->y_halfstep_0;
  Vec       Ax = am->Ax;
  Vec       Ax_old = am->Ax_old;
  Vec       Bz = am->Bz_old;
  Vec       Bz_old = am->Bz_old;
  Vec       y = am->y;
  Vec       y_old = am->y_old;
  PetscReal eps_cor = am->correlation_epsilon;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation, &cited));
  if (tao->niter > 0) {
    PetscScalar alpha_inv, alpha_cor;
    PetscScalar beta_inv, beta_cor;
    // *_old values are initialized, use the vectors to store diff values

    PetscCall(VecAXPY(y_old, -1.0, y));
    PetscCall(VecAXPY(y_hat_old, -1.0, y_hat));
    PetscCall(VecAXPY(Ax_old, -1.0, Ax));
    // Bz_old already contains Bz_{k+1} - Bz_k
    PetscCall(TaoADMMComputeUpdateScales_ARADMM(y_hat_old, Ax_old, &alpha_inv, &alpha_cor));
    PetscCall(TaoADMMComputeUpdateScales_ARADMM(y_old, Bz_old, &beta_inv, &beta_cor));

    if (alpha_cor > eps_cor && beta_cor > eps_cor) {
      am->mu               = PetscSqrtReal(alpha_inv * beta_inv);
      am->relaxation_gamma = 1.0 + 2.0 * PetscSqrtReal(alpha_inv * beta_inv) / (alpha_inv + beta_inv);
    } else if (alpha_cor > eps_cor) {
      am->mu               = alpha_inv;
      am->relaxation_gamma = 1.9;
    } else if (beta_cor > eps_cor) {
      am->mu = beta_inv;
      am->relaxation_gamma = 1.1;
    } else {
      am->relaxation_gamma = 1.5;
    }
  }
  PetscCall(VecCopy(y, y_old));
  PetscCall(VecCopy(y_hat, y_hat_old));
  PetscCall(VecCopy(Ax, Ax_old));
  PetscCall(VecCopy(Bz, Bz_old));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoADMMSetUp_ARADMM(Tao tao)
{
  PetscFunctionBegin;
  PetscCall(TaoADMMSetUp_Basic(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}
