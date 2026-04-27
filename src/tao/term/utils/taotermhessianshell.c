#include <petsc/private/taoimpl.h> /*I "petsctaoterm.h" I*/
#include "taotermhessianshell.h"

static PetscErrorCode TaoTermHessianShellDestroy(void *ctx)
{
  TaoTermHessianShell *hess = *(TaoTermHessianShell **)ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&hess->x));
  PetscCall(VecDestroy(&hess->params));
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
  PetscCall(MatShellGetContext(shell, (void *)&hess));
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

  Note:
  The term is stored as a weak reference to avoid reference cycles. The caller must ensure
  the `TaoTerm` outlives the shell matrix.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianMult()`, `TaoTermUpdateHessianShell()`
@*/
PetscErrorCode TaoTermCreateHessianShell(TaoTerm term, Mat *shell)
{
  TaoTermHessianShell *hess;
  PetscLayout          sol_layout;
  VecType              sol_vec_type;

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
  hess->term = term;
  PetscCall(MatShellSetContext(*shell, (void *)hess));
  PetscCall(MatShellSetContextDestroy(*shell, TaoTermHessianShellDestroy));
  PetscCall(MatShellSetOperation(*shell, MATOP_MULT, (void (*)(void))MatMult_TaoTermHessianShell));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
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
  PetscCall(MatShellGetContext(shell, (void *)&hess));
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

static PetscErrorCode TaoTermUpdateHessianShellSingle(TaoTerm term, Vec x, Vec params, Mat *H)
{
  PetscFunctionBegin;
  if (*H) {
    PetscBool is_shell;

    PetscCall(PetscObjectTypeCompare((PetscObject)*H, MATSHELL, &is_shell));
    if (is_shell) {
      PetscCall(TaoTermUpdateHessianShell(term, *H, x, params));
      *H = NULL;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermUpdateHessianShells - Handle `MATSHELL` matrices in `TaoTermComputeHessian()`

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. x      - a solution vector
. params - a parameters vector (may be `NULL`)
. H      - pointer to the `H` argument of `TaoTermComputeHessian()`; if it is a `MATSHELL`, it will be updated and the pointer will then point to `NULL`
- Hpre   - pointer to the `Hpre` argument of `TaoTermComputeHessian()`; if it is a `MATSHELL`, it will be updated and the pointer will then point to `NULL`

  Level: developer

  Developer Note:
  This function is to simplify implementing `TaoTermComputeHessian()` when an
  implementation can optionally use a `MATSHELL` or a `MATMFFD` for its Hessian matrices: call
  this function at the start of `TaoTermComputeHessian()`, and then only proceed to
  assemble `H` and/or `Hpre` if they are not `NULL`.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermComputeHessianMult()`, `TaoTermCreateHessianShell()`
@*/
PetscErrorCode TaoTermUpdateHessianShells(TaoTerm term, Vec x, Vec params, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(H, 4);
  PetscAssertPointer(Hpre, 5);
  if (*Hpre == *H) *Hpre = NULL;
  PetscCall(TaoTermUpdateHessianShellSingle(term, x, params, H));
  PetscCall(TaoTermUpdateHessianShellSingle(term, x, params, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}
