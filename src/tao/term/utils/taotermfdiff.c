#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petscsnes.h>
#include <petscdmshell.h>
#include "taotermhessianshell.h"

typedef struct _n_TaoTermWithParameters {
  TaoTerm term; // weak-reference
  Vec     params;
} TaoTermWithParameters;

static PetscErrorCode TaoTermWithParametersDestroy(void **ctx)
{
  TaoTermWithParameters *t = (TaoTermWithParameters *)*ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&t->params));
  PetscCall(PetscFree(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESFunction_TaoTerm(SNES snes, Vec X, Vec G, void *ctx)
{
  TaoTermWithParameters *t = (TaoTermWithParameters *)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(t->term, TAOTERM_CLASSID, 4);
  PetscCall(TaoTermComputeGradient(t->term, X, t->params, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeGradientFD - Approximate the gradient of a `TaoTerm` using finite differences

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. x      - a solution vector
- params - parameters vector (may be `NULL`, see `TaoTermParametersMode()`)

  Output Parameter:
. g - the computed finite difference approximation to the gradient

  Options Database Keys:
+ -tao_term_fd_delta <delta>       - change in `x` used to calculate finite differences
- -tao_term_gradient_use_fd <bool> - Use `TaoTermComputeGradientFD()` in `TaoTermComputeGradient()`

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not optimized to take advantage of
  sparsity in the problem.  Although not recommended for general use in
  large-scale applications, it can be useful in checking the correctness of a
  user-provided gradient.  Call `TaoTermComputeGradientUseFDPush()` to start using
  this routine in `TaoTermComputeGradient()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeGradientUseFDPush()`,
          `TaoTermComputeGradientUseFDPop()`,
          `TaoTermComputeHessianFD()`,
@*/
PetscErrorCode TaoTermComputeGradientFD(TaoTerm term, Vec x, Vec params, Vec g)
{
  Vec          x_perturbed;
  PetscScalar *_g;
  PetscReal    f, f2;
  PetscInt     low, high, N, i;
  PetscReal    h = term->fd_delta;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(x, &x_perturbed));
  PetscCall(VecCopy(x, x_perturbed));
  PetscCall(VecGetSize(x_perturbed, &N));
  PetscCall(VecGetOwnershipRange(x_perturbed, &low, &high));
  PetscCall(VecSetOption(x_perturbed, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(VecGetArray(g, &_g));
  for (i = 0; i < N; i++) {
    PetscCall(VecSetValue(x_perturbed, i, -h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    PetscCall(TaoTermComputeObjective(term, x_perturbed, params, &f));
    PetscCall(VecSetValue(x_perturbed, i, 2.0 * h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    PetscCall(TaoTermComputeObjective(term, x_perturbed, params, &f2));
    PetscCall(VecSetValue(x_perturbed, i, -h, ADD_VALUES));
    PetscCall(VecAssemblyBegin(x_perturbed));
    PetscCall(VecAssemblyEnd(x_perturbed));
    if (i >= low && i < high) _g[i - low] = (f2 - f) / (2.0 * h);
  }
  PetscCall(VecRestoreArray(g, &_g));
  PetscCall(VecDestroy(&x_perturbed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGetStashedHessianColoring(TaoTerm term, Vec params, Mat H, MatFDColoring *mat_coloring)
{
  PetscFunctionBegin;
  *mat_coloring = NULL;
  PetscCall(PetscObjectQuery((PetscObject)H, "__TaoTermHessianMatFDColoring", (PetscObject *)mat_coloring));
  if (*mat_coloring == NULL) {
    ISColoring coloring;

    PetscCall(TaoTermGetHessianColoring(term, &coloring));
    if (coloring) {
      MatFDColoring          _mat_coloring;
      PetscContainer         twp_container;
      TaoTermWithParameters *twp;

      PetscCall(PetscNew(&twp));
      twp->term = term;

      PetscCall(MatFDColoringCreate(H, coloring, &_mat_coloring));
      PetscCall(MatFDColoringSetFunction(_mat_coloring, (MatFDColoringFn *)SNESFunction_TaoTerm, (void *)twp));
      PetscCall(MatFDColoringSetUp(H, coloring, _mat_coloring));

      // stash the TaoTermWithParameters in a recoverable way in the MatFDColoring
      PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &twp_container));
      PetscCall(PetscContainerSetPointer(twp_container, (void *)twp));
      PetscCall(PetscContainerSetCtxDestroy(twp_container, TaoTermWithParametersDestroy));
      PetscCall(PetscObjectCompose((PetscObject)_mat_coloring, "__TaoTermWithParameters", (PetscObject)twp_container));
      PetscCall(PetscContainerDestroy(&twp_container));

      // stash the MatFDColoring in a recoverable way in the matrix
      PetscCall(PetscObjectCompose((PetscObject)H, "__TaoTermHessianMatFDColoring", (PetscObject)_mat_coloring));
      *mat_coloring = _mat_coloring;
      PetscCall(MatFDColoringDestroy(&_mat_coloring));
    }
  }

  if (*mat_coloring) {
    PetscContainer twp_container = NULL;

    // the coloring could have originated outside of this function (in TaoComputeHessianDefaultColoring()), so
    // detect if it originated in this function and update the parameters
    PetscCall(PetscObjectQuery((PetscObject)*mat_coloring, "__TaoTermWithParameters", (PetscObject *)&twp_container));
    if (twp_container) {
      TaoTermWithParameters *twp = NULL;

      PetscCall(PetscContainerGetPointer(twp_container, (void **)&twp));
      PetscCall(PetscObjectReference((PetscObject)params));
      twp->term = term;
      PetscCall(VecDestroy(&twp->params));
      twp->params = params;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeHessianSingleFD - Approximate the Hessian of a `TaoTerm` using finite differences

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. x      - a solution vector
- params - parameters vector (may be `NULL`, see `TaoTermParametersMode()`)

  Output Parameter:
. H - the computed finite difference approximation to the Hessian

  Options Database Keys:
+ -tao_term_fd_delta <delta>      - change in X used to calculate finite differences
- -tao_term_hessian_use_fd <bool> - Use `TaoTermComputeHessianFD()` in `TaoTermComputeHessian()`

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not optimized to take advantage of
  sparsity in the problem.  Although not recommended for general use in
  large-scale applications, it can be useful in checking the correctness of a
  user-provided gradient.  Call `TaoTermComputeHessianUseFDPush()` to start using
  this routine in `TaoTerm()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeGradientUseFDPush()`,
          `TaoTermComputeGradientUseFDPop()`,
@*/
static PetscErrorCode TaoTermComputeHessianSingleFD(TaoTerm term, Vec x, Vec params, Mat H)
{
  SNES                  snes;
  TaoTermWithParameters t;
  DM                    dm;
  MatFDColoring         mat_coloring;

  PetscFunctionBegin;
  PetscCall(TaoTermGetStashedHessianColoring(term, params, H, &mat_coloring));
  if (!mat_coloring) {
    PetscCall(PetscInfo(term, "TaoTerm using finite differences w/o coloring to compute Hessian matrix\n"));
    PetscCall(SNESCreate(PetscObjectComm((PetscObject)H), &snes));
    t.term   = term;
    t.params = params;
    PetscCall(SNESSetFunction(snes, NULL, SNESFunction_TaoTerm, (void *)&t));
    PetscCall(SNESGetDM(snes, &dm));
    PetscCall(DMShellSetGlobalVector(dm, x));
    PetscCall(SNESSetUp(snes));
    {
      PetscInt n, N;

      PetscCall(VecGetSize(x, &N));
      PetscCall(VecGetLocalSize(x, &n));
      PetscCall(MatSetSizes(H, n, n, N, N));
      PetscCall(MatSetUp(H));
    }
    PetscCall(SNESComputeJacobianDefault(snes, x, H, H, NULL));
    PetscCall(SNESDestroy(&snes));
  } else {
    PetscCall(PetscInfo(term, "TAO computing matrix using finite differences Hessian and coloring\n"));
    PetscCall(MatFDColoringApply(H, mat_coloring, x, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermComputeHessianFD - Use finite difference to compute Hessian matrix.

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. x      - a solution vector
- params - parameters vector (may be `NULL`, see `TaoTermParametersMode()`)

  Output Parameter:
+ H    - (optional) Hessian matrix
- Hpre - (optional) Hessian preconditioning matrix

  Options Database Keys:
+ -tao_term_fd_delta <delta>      - change in X used to calculate finite differences
- -tao_term_hessian_use_fd <bool> - Use `TaoTermComputeHessianFD()` in `TaoTermComputeHessian()`

  Level: advanced

  Notes:
  If either matrix is a `MATSHELL` created with `TaoTermCreateHessianShell()`,
  then finite difference  will be used for a matrix-free finite
  difference approximation (`MATMFFD`).

  This routine is slow and expensive, and is not optimized to take advantage of
  sparsity in the problem.  Although not recommended for general use in
  large-scale applications, it can be useful in checking the correctness of a
  user-provided Hessian.  Call `TaoTermComputeHessianUseFDPush()` to start using
  this routine in `TaoTermComputeHessian()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermComputeHessian()`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermComputeHessianUseFDPush()`,
          `TaoTermComputeHessianUseFDPop()`,
@*/
PetscErrorCode TaoTermComputeHessianFD(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  PetscCall(TaoTermUpdateHessianShells(term, x, params, &H, &Hpre));
  //TODO just cccp hessiansingle code for now
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  if (params) PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
  if (H) PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  if (Hpre) PetscValidHeaderSpecific(Hpre, MAT_CLASSID, 5);
  if (H) PetscCall(TaoTermComputeHessianSingleFD(term, x, params, H));
  if (Hpre && Hpre != H) {
    if (H) PetscCall(MatCopy(H, Hpre, UNKNOWN_NONZERO_PATTERN));
    else PetscCall(TaoTermComputeHessianSingleFD(term, x, params, Hpre));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMFFDFunction_TaoTermHessianShell(void *ctx, Vec x, Vec g)
{
  TaoTermHessianShell *hess = (TaoTermHessianShell *)ctx;

  PetscFunctionBegin;
  // we expect the solution to move around in a finite difference method, but not the parameters
  PetscCall(TaoTermHessianShellCheck(hess, PETSC_FALSE, PETSC_TRUE));
  PetscCall(TaoTermComputeGradient(hess->term, x, hess->params, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermInitializeHessianMFFD(TaoTerm term, Mat mffd)
{
  TaoTermHessianShell *hess;
  PetscLayout          sol_layout;
  VecType              sol_vec_type;
  PetscContainer       container;

  PetscFunctionBegin;
  PetscCall(TaoTermGetSolutionLayout(term, &sol_layout));
  PetscCall(MatSetLayouts(mffd, sol_layout, sol_layout));
  PetscCall(TaoTermGetSolutionVecType(term, &sol_vec_type));
  PetscCall(MatSetVecType(mffd, sol_vec_type));
  PetscCall(MatSetType(mffd, MATMFFD));
  PetscCall(PetscNew(&hess));
  hess->term = term;
  PetscCall(MatMFFDSetFunction(mffd, MatMFFDFunction_TaoTermHessianShell, hess));
  PetscCall(MatSetOption(mffd, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(mffd, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &container));
  PetscCall(PetscContainerSetPointer(container, (void *)hess));
  PetscCall(PetscContainerSetCtxDestroy(container, TaoTermHessianShellDestroy));
  PetscCall(PetscObjectCompose((PetscObject)mffd, "__TaoTermHessianShell", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianMFFD - Create a `MATMFFD` for a matrix-free finite-difference approximation of the Hessian of a `TaoTerm`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. mffd - a `Mat` of type `MATMFFD`

  Level: advanced

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianFD()`
@*/
PetscErrorCode TaoTermCreateHessianMFFD(TaoTerm term, Mat *mffd)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)term), mffd));
  PetscCall(TaoTermInitializeHessianMFFD(term, *mffd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoTermComputeHessianMFFD(TaoTerm term, Vec x, Vec params, Mat H, Mat B)
{
  PetscContainer       container;
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)H, "__TaoTermHessianShell", (PetscObject *)&container));
  if (!container) {
    PetscCall(TaoTermInitializeHessianMFFD(term, H));
    PetscCall(PetscObjectQuery((PetscObject)H, "__TaoTermHessianShell", (PetscObject *)&container));
    PetscCheck(container, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "failed to initialize mffd matrix");
  }
  PetscCall(PetscContainerGetPointer(container, (void **)&hess));
  PetscCheck(hess->term == term, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Hessian shell matrix does not come from this TaoTerm");
  PetscCall(PetscObjectReference((PetscObject)params));
  PetscCall(VecDestroy(&hess->params));
  hess->params = params;
  PetscCall(MatMFFDSetBase(H, x, NULL));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermUpdateHessianMFFD - Update the solution and parameter vectors for a shell matrix constructed with `TaoTermCreateHessianMFFD()`

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. mffd   - the matrix created with `TaoTermCreateHessianMFFD()`
. x      - a solution vector
- params - a parameters vector

  Level: advanced

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianFD()`, `TaoTermCreateHessianMFFD()`
@*/
PetscErrorCode TaoTermUpdateHessianMFFD(TaoTerm term, Mat mffd, Vec x, Vec params)
{
  PetscContainer       container;
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)mffd, "__TaoTermHessianShell", (PetscObject *)&container));
  if (!container) {
    PetscCall(TaoTermInitializeHessianMFFD(term, mffd));
    PetscCall(PetscObjectQuery((PetscObject)mffd, "__TaoTermHessianShell", (PetscObject *)&container));
    PetscCheck(container, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "failed to initialize mffd matrix");
  }
  PetscCall(PetscContainerGetPointer(container, (void **)&hess));
  PetscCheck(hess->term == term, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Hessian shell matrix does not come from this TaoTerm");
  PetscCall(PetscObjectReference((PetscObject)x));
  PetscCall(VecDestroy(&hess->x));
  hess->x = x;
  PetscCall(PetscObjectStateGet((PetscObject)x, &hess->x_state));
  PetscCall(PetscObjectReference((PetscObject)params));
  PetscCall(VecDestroy(&hess->params));
  hess->params = params;
  if (params) PetscCall(PetscObjectStateGet((PetscObject)params, &hess->params_state));
  else hess->params_state = 0;
  hess->x_state_change_warning      = PETSC_FALSE;
  hess->params_state_change_warning = PETSC_FALSE;
  PetscCall(MatMFFDSetBase(mffd, x, NULL));
  PetscCall(MatAssemblyBegin(mffd, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mffd, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
