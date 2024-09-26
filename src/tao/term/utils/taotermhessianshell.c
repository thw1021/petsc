#include <petsc/private/taoimpl.h> /*I "petsctaoterm.h" I*/

typedef struct _n_TaoTermHessianShell TaoTermHessianShell;

struct _n_TaoTermHessianShell {
  TaoTerm          term; // has to be a weak reference to avoid a cycle
  Vec              x;
  Vec              params;
  PetscObjectState x_state;
  PetscObjectState params_state;
  PetscBool        x_state_change_warning;
  PetscBool        params_state_change_warning;
};

static PetscErrorCode TaoTermHessianShellDestroy(void *ctx)
{
  TaoTermHessianShell *hess = (TaoTermHessianShell *)ctx;

  PetscFunctionBegin;
  // Do not destroy hess->term, it is a weak reference that does not increment the reference count
  PetscCall(VecDestroy(&hess->x));
  PetscCall(VecDestroy(&hess->params));
  PetscCall(PetscFree(hess));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_TaoTermHessianShell(Mat shell, Vec v, Vec y)
{
  PetscObjectState     x_state, params_state = 0;
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(shell, (void *)&hess));
  PetscCall(PetscObjectStateGet((PetscObject)hess->x, &x_state));
  if (hess->params) PetscCall(PetscObjectStateGet((PetscObject)hess->params, &params_state));
  if (!hess->x_state_change_warning && x_state != hess->x_state) {
    hess->x_state_change_warning = PETSC_TRUE;
    PetscCall(PetscInfo(hess->term, "x vector may have changed since TaoTermUpdateHessianShell() was called\n"));
  }
  if (!hess->params_state_change_warning && params_state != hess->params_state) {
    hess->params_state_change_warning = PETSC_TRUE;
    PetscCall(PetscInfo(hess->term, "parameter vector may have changed since TaoTermUpdateHessianShell() was called\n"));
  }
  PetscCall(TaoTermHessianMult(hess->term, hess->x, hess->params, v, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianShell - Create a `MATSHELL` for `TaoTermHessianMult()`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. shell - a `Mat` of type `MATSHELL`

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermHessianMult()`, `TaoTermUpdateHessianShell()`
@*/
PetscErrorCode TaoTermCreateHessianShell(TaoTerm term, Mat *shell)
{
  TaoTermHessianShell *hess;
  PetscLayout          sol_layout;
  VecType              sol_vec_type;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)term), shell));
  PetscCall(TaoTermGetLayouts(term, &sol_layout, NULL));
  PetscCall(MatSetLayouts(*shell, sol_layout, sol_layout));
  PetscCall(TaoTermGetVecTypes(term, &sol_vec_type, NULL));
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
  TaoTermUpdateHessianShell - Update the solution and parameter vectors for a shell matrix constructed with `TaoTermHessianMult()`

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. shell  - the matrix created with `TaoTermCreateHessianShell()`
. x      - a solution vector
- params - a parameters vector

  Level: advanced

  Note:
  After this is called `MatMult()` will perform `TaoTermHessianMult()` with the give solution and parameter vectors.

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermHessianMult()`, `TaoTermCreateHessianShell()`
@*/
PetscErrorCode TaoTermUpdateHessianShell(TaoTerm term, Mat shell, Vec x, Vec params)
{
  TaoTermHessianShell *hess;

  PetscFunctionBegin;
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

/*@
  TaoTermUpdateHessianShells - Handle shell matrices in `TaoTermHessian()`

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
. x      - a solution vector
. params - a parameters vector
. H      - pointer to the `H` argument of `TaoTermHessian()`; if it is a `MATSHELL`, it will be updated and the pointer will then point to `NULL`
- Hpre   - pointer to the `Hpre` argument of `TaoTermHessian()`; if it is a `MATSHELL`, it will be updated and the pointer will then point to `NULL`

  Level: developer

  Developer Note:
  This function is to simplify implementing `TaoTermHessian()` when an
  implementation can optionally use a `MATSHELL` for its Hessian matrices: call
  this function at the start of `TaoTermHessian()`, and then only proceed to
  assemby `H` and/or `Hpre` if they are not `NULL`.

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermHessianMult()`, `TaoTermCreateHessianShell()`
@*/
PetscErrorCode TaoTermUpdateHessianShells(TaoTerm term, Vec x, Vec params, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  if (*Hpre == *H) *Hpre = NULL;
  if (*H) {
    PetscBool is_shell;

    PetscCall(PetscObjectTypeCompare((PetscObject)*H, MATSHELL, &is_shell));
    if (is_shell) {
      PetscCall(TaoTermUpdateHessianShell(term, *H, x, params));
      *H = NULL;
    }
  }
  if (*Hpre) {
    PetscBool is_shell;

    PetscCall(PetscObjectTypeCompare((PetscObject)*Hpre, MATSHELL, &is_shell));
    if (is_shell) {
      PetscCall(TaoTermUpdateHessianShell(term, *Hpre, x, params));
      *Hpre = NULL;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
