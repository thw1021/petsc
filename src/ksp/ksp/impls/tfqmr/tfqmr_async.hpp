#include <petsc/private/kspimpl.h>
#include <petscmanagedmemory.hpp>

#include <vector>

struct KSP_TFQMR_Async {
  PetscInt                          check_every{1};
  std::vector<Petsc::ManagedScalar> work_scalars{};
  std::vector<Petsc::ManagedReal>   work_reals{};
};

PETSC_INTERN PetscErrorCode KSPSetUp_TFQMR(KSP);
static PetscErrorCode       KSPSetUp_TFQMR_Async(KSP ksp)
{
  const auto         tfqmr = static_cast<KSP_TFQMR_Async *>(ksp->data);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(KSPSetUp_TFQMR(ksp));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCallCXX(tfqmr->work_scalars.resize(6));
  for (auto &&scal : tfqmr->work_scalars) PetscCall(scal.Reserve(dctx, 1));
  PetscCallCXX(tfqmr->work_reals.resize(4));
  for (auto &&real : tfqmr->work_reals) PetscCall(real.Reserve(dctx, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_TFQMR_Async(KSP ksp)
{
  const auto         tfqmr = static_cast<KSP_TFQMR_Async *>(ksp->data);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  for (auto &&scal : tfqmr->work_scalars) PetscCall(scal.Destroy(dctx));
  for (auto &&scal : tfqmr->work_reals) PetscCall(scal.Destroy(dctx));
  delete tfqmr;
  ksp->data = nullptr;
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_TFQMR_Async(KSP ksp)
{
  using namespace Petsc;

  PetscInt i;
  auto     tfqmr        = static_cast<KSP_TFQMR_Async *>(ksp->data);
  auto    &work_reals   = tfqmr->work_reals;
  auto    &dp           = work_reals[0];
  auto    &dpold        = work_reals[1];
  auto    &tau          = work_reals[2];
  auto    &work_scalars = tfqmr->work_scalars;
  auto    &rhoold       = work_scalars[0];
  auto     X            = ksp->vec_sol;
  auto     B            = ksp->vec_rhs;
  auto     R            = ksp->work[0];
  auto     RP           = ksp->work[1];
  auto     V            = ksp->work[2];
  auto     T            = ksp->work[3];
  auto     Q            = ksp->work[4];
  auto     P            = ksp->work[5];
  auto     U            = ksp->work[6];
  auto     D            = ksp->work[7];
  auto     T1           = ksp->work[8];
  auto     AUQ          = V;

  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  /* Compute initial preconditioned residual */
  PetscCall(KSPInitialResidual(ksp, X, V, T, R, B));
  /* Test for nothing to do */
  PetscCall(VecNormAsync(R, NORM_2, &dp, dctx));
  PetscCall(PetscObjectSAWsTakeAccess(PetscObjectCast(ksp)));
  if (ksp->normtype == KSP_NORM_NONE) {
    ksp->rnorm = 0.0;
  } else {
    ksp->rnorm = dp.cfront(dctx);
  }
  ksp->its = 0;
  PetscCall(PetscObjectSAWsGrantAccess(PetscObjectCast(ksp)));
  PetscCall(KSPMonitor(ksp, 0, ksp->rnorm));
  PetscCall((*ksp->converged)(ksp, 0, ksp->rnorm, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* Make the initial Rp == R */
  PetscCall(VecCopyAsync(R, RP, dctx));

  /* Set the initial conditions */
  // clang-format off
  PetscCall(
    MultiEval(
      dctx,
      expr::make_expr_pair(tau, dp),
      expr::make_expr_pair(dpold, dp)
    )
  );
  // clang-format on

  PetscCall(VecDotAsync(R, RP, &rhoold, dctx)); /* rhoold = (r,rp)     */
  PetscCall(VecCopyAsync(R, U, dctx));
  PetscCall(VecCopyAsync(R, P, dctx));
  PetscCall(KSP_PCApplyBAorAB(ksp, P, V, T));
  PetscCall(VecSetAsync(D, MANAGED_SCAL_ZERO(), dctx));

  for (i = 0; i < ksp->max_it; ++i) {
    auto &a   = work_scalars[1];
    auto &s   = work_scalars[2];
    auto &rho = work_scalars[3];

    PetscCall(PetscObjectSAWsTakeAccess(PetscObjectCast(ksp)));
    ksp->its++;
    PetscCall(PetscObjectSAWsGrantAccess(PetscObject(ksp)));
    PetscCall(VecDotAsync(V, RP, &s, dctx)); /* s <- (v,rp)          */
    // KSPCheckDot(ksp, s);
    a = Eval(-rhoold / s, dctx);                                 /* a <- -rho / s         */
    PetscCall(VecWAXPYAsync(Q, a, V, U, dctx));                  /* q <- u - a v         */
    PetscCall(VecWAXPYAsync(T, MANAGED_SCAL_ONE(), U, Q, dctx)); /* t <- u + q           */
    PetscCall(KSP_PCApplyBAorAB(ksp, T, AUQ, T1));
    PetscCall(VecAXPYAsync(R, a, AUQ, dctx)); /* r <- r - a K (u + q) */
    PetscCall(VecNormAsync(R, NORM_2, &dp, dctx));
    // KSPCheckNorm(ksp, dp);
    for (auto m = 0; m < 2; ++m) {
      auto &psi    = work_reals[3];
      auto &psiold = work_scalars[4];
      auto &eta    = work_scalars[5];

      if (m) {
        psi = Eval(dp / tau, dctx);
      } else {
        if (!i) {
          psiold.at(dctx, 0) = 0.0;
          eta.at(dctx, 0)    = 0.0;
        }
        psi = Eval(PetscSqrtReal(dp * dpold) / tau, dctx);
      }
      const auto cm = 1.0 / PetscSqrtReal(1.0 + psi * psi);

      // clang-format off
      PetscCall(
        MultiEval(
          dctx,
          expr::make_expr_pair(tau, tau * psi * cm),
          expr::make_expr_pair(psiold, -psiold * psiold * eta / a),
          expr::make_expr_pair(eta, -cm * cm * a)
        )
      );
      // clang-format on
      PetscCall(VecAYPXAsync(D, psiold, m ? Q : U, dctx));
      PetscCall(VecAXPYAsync(X, eta, D, dctx));
      psiold = Eval(psi, dctx);

      if (((i + 1) % tfqmr->check_every) == 0) {
        if (ksp->normtype == KSP_NORM_NONE) {
          ksp->rnorm = 0.0;
        } else {
          ksp->rnorm = PetscSqrtReal(2 * i + m + 2.0) * tau.cfront(dctx);
        }

        PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
        PetscCall(KSPMonitor(ksp, i + 1, ksp->rnorm));
        PetscCall((*ksp->converged)(ksp, i + 1, ksp->rnorm, &ksp->reason, ksp->cnvP));
        if (ksp->reason) break;
      }
    }
    if (ksp->reason) break;

    PetscCall(VecDotAsync(R, RP, &rho, dctx));       /* rho <- (r,rp)       */
    rhoold = Eval(rho / rhoold, dctx);               /* b <- rho / rhoold   */
    PetscCall(VecWAXPYAsync(U, rhoold, Q, R, dctx)); /* u <- r + b q        */
    PetscCall(VecAXPYAsync(Q, rhoold, P, dctx));
    PetscCall(VecWAXPYAsync(P, rhoold, Q, U, dctx)); /* p <- u + b(q + b p) */
    PetscCall(KSP_PCApplyBAorAB(ksp, P, V, Q));      /* v <- K p  */

    // clang-format off
    PetscCall(
      MultiEval(
        dctx,
        expr::make_expr_pair(rhoold, rho),
        expr::make_expr_pair(dpold, dp)
      )
    );
    // clang-format on
  }
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscCall(KSPUnwindPreconditioner(ksp, X, T));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode KSPCreate_TFQMR(KSP);

/*MC
  KSPTFQMRASYNC - A transpose free QMR (quasi minimal residual),

  Level: beginner

  Notes:
  Supports left and right preconditioning, but not symmetric

  The "residual norm" computed in this algorithm is actually just an upper bound on the actual
  residual norm. That is for left preconditioning it is a bound on the preconditioned residual
  and for right preconditioning it is a bound on the true residual.

  References:
. * - Freund, 1993

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPTCQMR`
M*/
PETSC_INTERN PetscErrorCode KSPCreate_TFQMR_Async(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPCreate_TFQMR(ksp));
  PetscCall(PetscFree(ksp->data));
  ksp->data         = new KSP_TFQMR_Async{};
  ksp->ops->destroy = KSPDestroy_TFQMR_Async;
  ksp->ops->setup   = KSPSetUp_TFQMR_Async;
  ksp->ops->solve   = KSPSolve_TFQMR_Async;
  PetscFunctionReturn(PETSC_SUCCESS);
}
