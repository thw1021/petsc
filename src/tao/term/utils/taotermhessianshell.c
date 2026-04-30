#include <petsc/private/taoimpl.h> /*I "petsctaoterm.h" I*/
#include "taotermhessianshell.h"

static PetscErrorCode TaoTermHessianShellDestroy(PetscCtxRt ctx)
{
  TaoTermHessianShell *hess = *(TaoTermHessianShell **)ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&hess->x));
  PetscCall(VecDestroy(&hess->params));
  PetscCall(TaoTermDestroy(&hess->term));
  PetscCall(PetscFree(hess));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermHessianShellCheck(TaoTermHessianShell *hess, PetscBool check_solution, PetscBool check_parameters)
{
  PetscFunctionBegin;
  if (check_solution) {
    PetscObjectState x_state;

    PetscCall(PetscObjectStateGet((PetscObject)hess->x, &x_state));
    if (!hess->x_state_change_warning && x_state != hess->x_state) {
      hess->x_state_change_warning = PETSC_TRUE;
      PetscCall(PetscInfo(hess->term, "x vector may have changed since TaoTermUpdateHessianShell() was called\n"));
    }
  }
  if (check_parameters) {
    PetscObjectState params_state = 0;

    if (hess->params) PetscCall(PetscObjectStateGet((PetscObject)hess->params, &params_state));
    if (!hess->params_state_change_warning && params_state != hess->params_state) {
      hess->params_state_change_warning = PETSC_TRUE;
      PetscCall(PetscInfo(hess->term, "parameter vector may have changed since TaoTermUpdateHessianShell() was called\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_TaoTermHessianShell(Mat shell, Vec v, Vec y)
{
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(shell, &hess));
  PetscCall(TaoTermHessianShellCheck(hess, PETSC_TRUE, PETSC_TRUE));
  PetscCall(TaoTermComputeHessianMult(hess->term, hess->x, hess->params, v, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianShell - Create a `MATSHELL` for `TaoTermComputeHessianMult()`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. shell - a `Mat` of type `MATSHELL`

  Level: advanced

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianMult()`, `TaoTermUpdateHessianShell()`
@*/
PetscErrorCode TaoTermCreateHessianShell(TaoTerm term, Mat *shell)
{
  TaoTermHessianShell *hess;
  PetscLayout          sol_layout;
  VecType              sol_vec_type;
  PetscContainer       container;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(shell, 2);
  PetscCall(MatCreate(PetscObjectComm((PetscObject)term), shell));
  PetscCall(TaoTermGetSolutionLayout(term, &sol_layout));
  PetscCall(MatSetLayouts(*shell, sol_layout, sol_layout));
  PetscCall(TaoTermGetSolutionVecType(term, &sol_vec_type));
  PetscCall(MatSetVecType(*shell, sol_vec_type));
  PetscCall(MatSetType(*shell, MATSHELL));
  PetscCall(PetscNew(&hess));
  PetscCall(PetscObjectReference((PetscObject)term));
  hess->term = term;
  PetscCall(MatShellSetContext(*shell, hess));
  PetscCall(MatShellSetOperation(*shell, MATOP_MULT, (void (*)(void))MatMult_TaoTermHessianShell));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &container));
  PetscCall(PetscContainerSetPointer(container, hess));
  PetscCall(PetscContainerSetCtxDestroy(container, TaoTermHessianShellDestroy));
  PetscCall(PetscObjectCompose((PetscObject)*shell, "__TaoTermHessianShell", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermUpdateHessianShell - Update the solution and parameter vectors for a shell matrix constructed with `TaoTermCreateHessianShell()`

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. shell  - the matrix created with `TaoTermCreateHessianShell()`
. x      - a solution vector
- params - a parameters vector (may be `NULL`)

  Level: advanced

  Note:
  After this is called, `MatMult()` will perform `TaoTermComputeHessianMult()` with the given solution and parameter vectors.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianMult()`, `TaoTermCreateHessianShell()`
@*/
PetscErrorCode TaoTermUpdateHessianShell(TaoTerm term, Mat shell, Vec x, Vec params)
{
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(shell, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscCall(MatShellGetContext(shell, &hess));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MaybeUpdateOneShell(TaoTerm term, Vec x, Vec params, Mat *M)
{
  PetscContainer marker = NULL;

  PetscFunctionBegin;
  if (*M) PetscCall(PetscObjectQuery((PetscObject)*M, "__TaoTermHessianShell", (PetscObject *)&marker));
  if (marker) {
    PetscCall(TaoTermUpdateHessianShell(term, *M, x, params));
    *M = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermPreprocessHessianShells - For any of `*H`, `*Hpre` that is a `TaoTermCreateHessianShell()`-produced
  matrix, update its stored solution and parameter vectors and replace the local pointer with `NULL` so the
  caller can fall through to assemble the remaining (non-shell) matrices via the regular `term->ops->hessian` path.
*/
PETSC_INTERN PetscErrorCode TaoTermPreprocessHessianShells(TaoTerm term, Vec x, Vec params, Mat *H, Mat *Hpre)
{
  PetscBool Hpre_is_H = (*Hpre == *H) ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MaybeUpdateOneShell(term, x, params, H));
  if (Hpre_is_H && !*H) *Hpre = NULL;
  PetscCall(MaybeUpdateOneShell(term, x, params, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}
