#include <../src/ksp/ksp/impls/cg/cgimpl.h> /*I "petscksp.h" I*/
#include <petscmanagedmemory.hpp>

#include <vector>

struct KSP_CG_Async {
  KSP_CG                            cg;
  std::vector<Petsc::ManagedScalar> work_scalars;
  std::vector<Petsc::ManagedReal>   work_reals;
  PetscDeviceContext                work_ctx;
};

PETSC_EXTERN PetscErrorCode KSPSetUp_CG(KSP);
static PetscErrorCode       KSPSetUp_CG_Async(KSP ksp)
{
  const auto         cga = static_cast<KSP_CG_Async *>(ksp->data);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(KSPSetUp_CG(ksp));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  if (!cga->work_ctx) {
    PetscCall(PetscDeviceContextDuplicate(dctx, &cga->work_ctx));
    PetscCall(PetscObjectSetName(PetscObjectCast(cga->work_ctx), "work ctx"));
  }

  PetscCallCXX(cga->work_scalars.resize(5));
  PetscCallCXX(cga->work_reals.resize(1));

  for (auto &&scal : cga->work_scalars) PetscCall(scal.Reserve(dctx, 1));
  for (auto &&real : cga->work_reals) PetscCall(real.Reserve(dctx, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     KSPSolve_CG - This routine actually applies the conjugate gradient method

     Note : this routine can be replaced with another one (see below) which implements
            another variant of CG.

   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
static PetscErrorCode KSPSolve_CG_Async(KSP ksp)
{
  using namespace Petsc;

  PetscInt           i;
  const auto         cga = static_cast<KSP_CG_Async *>(ksp->data);
  const auto         cg  = &cga->cg;
  const auto         X = ksp->vec_sol, B = ksp->vec_rhs;
  const auto         R = ksp->work[0], Z = ksp->work[1], P = ksp->work[2], W = Z;
  auto              &dp   = cga->work_reals[0];
  auto              &beta = cga->work_scalars[0];
  Mat                Amat, Pmat;
  PetscDeviceContext dctx, sub_ctx = cga->work_ctx;
  const auto         VecXDotAsync = [&cg](Vec x, Vec y, ManagedScalar *a, PetscDeviceContext d) { return cg->type == KSP_CG_HERMITIAN ? VecDotAsync(x, y, a, d) : VecTDotAsync(x, y, a, d); };

  PetscFunctionBegin;
  {
    PetscBool diagonalscale;

    PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
    PetscCheck(!diagonalscale, PetscObjectComm(ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", PetscObjectCast(ksp)->type_name);
  }
  PetscCheck(!ksp->calc_sings, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));

  ksp->its = 0;
  if (ksp->guess_zero) {
    PetscCall(VecCopyAsync(B, R, dctx)); // r <- b (x is 0)
  } else {
    PetscCall(KSP_MatMult(ksp, Amat, X, R)); // r <- b - Ax
    PetscCall(VecAYPXAsync(R, MANAGED_SCAL_MINUS_ONE(), B, dctx));
  }
  /* This may be true only on a subset of MPI ranks; setting it here so it will be detected by
     the first norm computation below */
  if (ksp->reason == KSP_DIVERGED_PC_FAILED) PetscCall(VecSetInf(R));

  switch (ksp->normtype) {
  case KSP_NORM_PRECONDITIONED:
    PetscCall(KSP_PCApply(ksp, R, Z));             // z <- Br
    PetscCall(VecNormAsync(Z, NORM_2, &dp, dctx)); // dp <- z'*z = e'*A'*B'*B*A*e
    break;
  case KSP_NORM_UNPRECONDITIONED:
    PetscCall(VecNormAsync(R, NORM_2, &dp, dctx)); // dp <- r'*r = e'*A'*A*e
    break;
  case KSP_NORM_NATURAL:
    PetscCall(KSP_PCApply(ksp, R, Z));                    // z <- Br
    PetscCall(VecXDotAsync(Z, R, &beta, dctx));           // beta <- z'*r
    dp = Eval(PetscSqrtReal(PetscAbsScalar(beta)), dctx); // dp <- r'*z = r'*B*r = e'*A'*B*A*e
    break;
  case KSP_NORM_NONE:
    dp.front(dctx) = 0.0;
    break;
  default:
    SETERRQ(PetscObjectComm(ksp), PETSC_ERR_SUP, "%s", KSPNormTypes[ksp->normtype]);
  }
  ksp->rnorm = dp.cfront(dctx);
  PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
  PetscCall(KSPMonitor(ksp, 0, ksp->rnorm));
  PetscCall((*ksp->converged)(ksp, 0, ksp->rnorm, &ksp->reason, ksp->cnvP)); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  if (ksp->normtype != KSP_NORM_NATURAL) {
    if (ksp->normtype != KSP_NORM_PRECONDITIONED) PetscCall(KSP_PCApply(ksp, R, Z)); // z <- Br
    PetscCall(VecXDotAsync(Z, R, &beta, dctx));                                      // beta <- z'*r
  }

  for (i = 0; i < ksp->max_it; ++i) {
    auto &betaold = cga->work_scalars[1];

    ksp->its = i + 1;
    if (beta.KnownAndEqual(0.0)) {
      ksp->reason = KSP_CONVERGED_ATOL;
      PetscCall(PetscInfo(ksp, "converged due to beta = 0\n"));
      break;
    }

    if (i) {
      auto &b = cga->work_scalars[2];

      b = Eval(beta / betaold, dctx);
      PetscCall(VecAYPXAsync(P, b, Z, dctx)); // p <- z + b* p
    } else {
      PetscCall(VecCopyAsync(Z, P, dctx)); // p <- z
    }

    auto &a    = cga->work_scalars[3];
    auto &mina = cga->work_scalars[4];

    PetscCall(KSP_MatMult(ksp, Amat, P, W)); // w <- Ap
    PetscCall(VecXDotAsync(P, W, &a, dctx)); // a <- p'w

    // clang-format off
    PetscCall(
      MultiEval(
        dctx,
        expr::make_expr_pair(a, beta / a), // a = beta/p'w
        expr::make_expr_pair(betaold, beta),
        expr::make_expr_pair(mina, -a)
      )
    );
    // clang-format on

    PetscCall(VecAXPYAsync(X, a, P, dctx));       // x <- x + ap
    PetscCall(VecAXPYAsync(R, mina, W, sub_ctx)); // r <- r - aw

    if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->chknorm < i + 2) {
      PetscCall(KSP_PCApply(ksp, R, Z));             // z <- Br
      PetscCall(VecNormAsync(Z, NORM_2, &dp, dctx)); // dp <- z'*z
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i + 2) {
      PetscCall(VecNormAsync(R, NORM_2, &dp, dctx)); // dp <- r'*r
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      PetscCall(KSP_PCApply(ksp, R, Z));          // z <- Br
      PetscCall(VecXDotAsync(Z, R, &beta, dctx)); // beta <- r'*z
      dp = Eval(PetscSqrtReal(PetscAbsScalar(beta)), dctx);
    } else {
      dp.front(dctx) = 0.0;
    }

    if (((i + 1) % cg->check_every) == 0) {
      ksp->rnorm = dp.cfront(dctx);
      PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm));
      PetscCall(KSPMonitor(ksp, i + 1, ksp->rnorm));
      PetscCall((*ksp->converged)(ksp, i + 1, ksp->rnorm, &ksp->reason, ksp->cnvP));
    }
    if (ksp->reason) break;

    if ((ksp->normtype != KSP_NORM_NATURAL) || (ksp->chknorm >= i + 2)) {
      if (ksp->normtype != KSP_NORM_PRECONDITIONED) PetscCall(KSP_PCApply(ksp, R, Z)); // z <- Br
      PetscCall(VecXDotAsync(Z, R, &beta, dctx));                                      // beta <- z'*r
    }
  }
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_CG_Async(KSP ksp)
{
  KSP_CG *old_cg;
  auto    cga = static_cast<KSP_CG_Async *>(ksp->data);

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(1, &old_cg));
  *old_cg = cga->cg;
  for (auto &&scal : cga->work_scalars) PetscCall(scal.Destroy(cga->work_ctx));
  for (auto &&real : cga->work_reals) PetscCall(real.Destroy(cga->work_ctx));
  PetscCall(PetscDeviceContextDestroy(&cga->work_ctx));
  delete cga;
  ksp->data = static_cast<void *>(old_cg);
  PetscCall(KSPDestroy_CG(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPView_CG_Async(KSP ksp, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(KSPView_CG(ksp, viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "kind: ASYNC\n"));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    KSPCGUseSingleReduction_CG

    This routine sets a flag to use a variant of CG. Note that (in somewhat
    atypical fashion) it also swaps out the routine called when KSPSolve()
    is invoked.
*/
static PetscErrorCode KSPCGUseSingleReduction_CG_Async(KSP ksp, PetscBool flg)
{
  PetscFunctionBegin;
  PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_SUP, "Single reduction not implemented");
  static_cast<KSP_CG *>(ksp->data)->singlereduction = flg;
  ksp->ops->solve                                   = KSPSolve_CG_Async;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode KSPCreate_CG(KSP);

/*MC
     KSPCG - The Preconditioned Conjugate Gradient (PCG) iterative method

   Options Database Keys:
+   -ksp_cg_type Hermitian - (for complex matrices only) indicates the matrix is Hermitian, see `KSPCGSetType()`
.   -ksp_cg_type symmetric - (for complex matrices only) indicates the matrix is symmetric
-   -ksp_cg_single_reduction - performs both inner products needed in the algorithm with a single `MPI_Allreduce()` call, see `KSPCGUseSingleReduction()`

   Level: beginner

   Notes:
    The PCG method requires both the matrix and preconditioner to be symmetric positive (or negative) (semi) definite.

   Only left preconditioning is supported; there are several ways to motivate preconditioned CG, but they all produce the same algorithm.
   One can interpret preconditioning A with B to mean any of the following\:
.n  (1) Solve a left-preconditioned system BAx = Bb, using inv(B) to define an inner product in the algorithm.
.n  (2) Solve a right-preconditioned system ABy = b, x = By, using B to define an inner product in the algorithm.
.n  (3) Solve a symmetrically-preconditioned system, E^TAEy = E^Tb, x = Ey, where B = EE^T.
.n  (4) Solve Ax=b with CG, but use the inner product defined by B to define the method [2].
.n  In all cases, the resulting algorithm only requires application of B to vectors.

   For complex numbers there are two different CG methods, one for Hermitian symmetric matrices and one for non-Hermitian symmetric matrices. Use
   `KSPCGSetType()` to indicate which type you are using.

   One can use `KSPSetComputeEigenvalues()` and `KSPComputeEigenvalues()` to compute the eigenvalues of the (preconditioned) operator

   Developer Notes:
    KSPSolve_CG() should actually query the matrix to determine if it is Hermitian symmetric or not and NOT require the user to
   indicate it to the `KSP` object.

   References:
+  * - Magnus R. Hestenes and Eduard Stiefel, Methods of Conjugate Gradients for Solving Linear Systems,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379
-  * - Josef Malek and Zdenek Strakos, Preconditioning and the Conjugate Gradient Method in the Context of Solving PDEs,
    SIAM, 2014.

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPSetComputeEigenvalues()`, `KSPComputeEigenvalues()`
          `KSPCGSetType()`, `KSPCGUseSingleReduction()`, `KSPPIPECG`, `KSPGROPPCG`
M*/

/*
    KSPCreate_CG - Creates the data structure for the Krylov method CG and sets the
       function pointers for all the routines it needs to call (KSPSolve_CG() etc)

    It must be labeled as PETSC_EXTERN to be dynamically linkable in C++
*/
PETSC_INTERN PetscErrorCode KSPCreate_CG_Async(KSP ksp)
{
  KSP_CG_Async *cg;

  PetscFunctionBegin;
  PetscCall(KSPCreate_CG(ksp));
  cg           = new KSP_CG_Async;
  cg->cg       = *static_cast<KSP_CG *>(ksp->data);
  cg->work_ctx = nullptr;
  PetscCall(PetscFree(ksp->data));
  ksp->data = cg;

  ksp->ops->setup   = KSPSetUp_CG_Async;
  ksp->ops->solve   = KSPSolve_CG_Async;
  ksp->ops->destroy = KSPDestroy_CG_Async;
  ksp->ops->view    = KSPView_CG_Async;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPCGUseSingleReduction_C", KSPCGUseSingleReduction_CG_Async));
  PetscFunctionReturn(PETSC_SUCCESS);
}
