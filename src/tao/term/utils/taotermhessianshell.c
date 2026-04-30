#include <petsc/private/taoimpl.h> /*I "petsctaoterm.h" I*/
#include "taotermhessianshell.h"

static PetscErrorCode TaoTermHessianShellDestroy(PetscCtxRt ctx)
{
  TaoTermHessianShell *hess = *(TaoTermHessianShell **)ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&hess->x));
  PetscCall(VecDestroy(&hess->params));
  PetscCall(VecDestroy(&hess->Ax));
  PetscCall(TaoTermDestroy(&hess->term));
  /* hess->mt is borrowed and not destroyed here */
  PetscCall(PetscFree(hess));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermHessianShellCheck - Warn when the solution or parameter vectors cached in a
  `TaoTermHessianShell` have changed from last `TaoTermUpdateHessianShell()` call.

  Input Parameters:
+ hess             - the hessian-shell context retrieved from the shell `Mat`
. check_solution   - if `PETSC_TRUE`, compare the current state of `hess->x` against the cached state
- check_parameters - if `PETSC_TRUE`, compare the current state of `hess->params` against the cached state

*/
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
  PetscCheck(hess->x, PetscObjectComm((PetscObject)shell), PETSC_ERR_ARG_WRONGSTATE, "TaoTermUpdateHessianShell() must be called before applying this Hessian shell");
  PetscCall(TaoTermHessianShellCheck(hess, PETSC_TRUE, PETSC_TRUE));
  if (hess->mt) {
    Vec Ax = hess->x;

    /* mapped term: y = map^H (grad^2 f)(map x) map v, using the summand's Hessian-vector product */
    if (hess->mt->map) {
      if (!hess->Ax) PetscCall(MatCreateVecs(hess->mt->map, NULL, &hess->Ax));
      PetscCall(MatMult(hess->mt->map, hess->x, hess->Ax));
      Ax = hess->Ax;
    }
    PetscCall(TaoTermMappingComputeHessianMult(hess->mt, Ax, hess->params, NULL, v, INSERT_VALUES, y));
  } else PetscCall(TaoTermComputeHessianMult(hess->term, hess->x, hess->params, v, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianShell - Create a `MATSHELL` for `TaoTermComputeHessianMult()`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. shell - a `Mat` of type `MATSHELL`

  Level: developer

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

/*
  TaoTermMappingCreateHessianShell - Create a `MATSHELL` that applies the mapped Hessian
  `map^H (grad^2 f)(map x) map` of a `TaoTermMapping` matrix-free through
  `TaoTermMappingComputeHessianMult()`.

  Collective

  Input Parameter:
. mt - a `TaoTermMapping` whose mapping matrix `map` is set

  Output Parameter:
. shell - a `Mat` of type `MATSHELL` sized in the solution (outer) space

  Level: developer

  Note:
  Unlike `TaoTermCreateHessianShell()`, whose `MatMult()` calls `TaoTermComputeHessianMult()` on a
  bare term, this shell routes through the mapping so that `map`/`map^H` are applied around the
  summand's Hessian-vector product.  Use it for a mapped objective term whose outer Hessian is
  requested as `MATSHELL`, so the framework does not attempt to assemble `map^H H map` (`PtAP`).

.seealso: [](sec_tao_term), `TaoTermMapping`, `TaoTermCreateHessianShell()`, `TaoTermMappingComputeHessianMult()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingCreateHessianShell(TaoTermMapping *mt, Mat *shell)
{
  TaoTermHessianShell *hess;
  PetscLayout          sol_layout;
  VecType              sol_vec_type;
  PetscContainer       container;

  PetscFunctionBegin;
  PetscAssertPointer(shell, 2);
  PetscCheck(mt->map, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_ARG_WRONGSTATE, "TaoTermMappingCreateHessianShell() requires a mapping matrix");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)mt->term), shell));
  PetscCall(MatGetLayouts(mt->map, NULL, &sol_layout)); /* solution (outer) space is the domain of map */
  PetscCall(MatSetLayouts(*shell, sol_layout, sol_layout));
  PetscCall(MatGetVecType(mt->map, &sol_vec_type));
  PetscCall(MatSetVecType(*shell, sol_vec_type));
  PetscCall(MatSetType(*shell, MATSHELL));
  PetscCall(PetscNew(&hess));
  PetscCall(PetscObjectReference((PetscObject)mt->term));
  hess->term = mt->term;
  hess->mt   = mt;
  PetscCall(MatShellSetContext(*shell, hess));
  PetscCall(MatShellSetOperation(*shell, MATOP_MULT, (void (*)(void))MatMult_TaoTermHessianShell));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*shell, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)mt->term), &container));
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

  Level: developer

  Note:
  After this is called, `MatMult()` will perform `TaoTermComputeHessianMult()` with the given solution and parameter vectors.
  Any scaling or shift previously applied to the shell (for example a `MatShift()` added by a solver to regularize the
  Newton system) is discarded, so the shell represents the unperturbed Hessian at the new `(x, params)`.

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
  /* Reset any accumulated MatShift()/MatScale() the solver applied to the shell at the previous evaluation
     point.  MATSHELL accumulates these in its scaling/shift bookkeeping and only clears them on assembly, so
     without this a solver that regularizes with an in-place MatShift(H, pert) every iteration (e.g. TAONLS)
     would leave the shift compounding across iterations.  This mirrors TaoTermComputeHessianMFFD(),
     which assembles the MATMFFD Hessian on every refresh for the same reason. */
  PetscCall(MatAssemblyBegin(shell, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(shell, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  After checking whether the input matrix is a valid one, refresh its
  cached (x, params) snapshot and NULL out `*M`.

  Input/Output Parameters:
+ term   - the parent `TaoTerm` whose Hessian shell is being updated
. x      - the current solution vector to record in the shell
. params - the current parameter vector (or `NULL`)
- M      - on entry, the candidate matrix to inspect; on exit, NULL when `*M` was a shell
*/
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
  TaoTermPreprocessHessianShells - Detect any of `*H`, `*Hpre` that were produced by
  `TaoTermCreateHessianShell()`, refresh their stored (x, params) snapshot, and point
  caller's local pointer to `NULL` so the standard `term->ops->hessian` path skips the
  shell during assembly.

  Input Parameters:
+ term   - the parent `TaoTerm`
. x      - the current solution vector
- params - the current parameter vector (may be `NULL`)

  Input/Output Parameters:
+ H    - on entry, the requested Hessian matrix; on exit, NULL if it was a Hessian shell
         created by `TaoTermCreateHessianShell()` (in which case its cached (x, params)
         snapshot has been refreshed)
- Hpre - on entry, the requested preconditioning matrix; on exit, NULL under the same
         condition as `H`.  Additionally, if `*Hpre == *H` and `*H` were a shell, `*Hpre`
         is set to NULL to match, without refreshing the shared shell a second time.

  Notes:
  After this preprocessing the standard implementation of `term->ops->hessian` only has to
  assemble the remaining non-shell matrices.  Shells are state-only: `MatMult()` will lazily
  invoke `TaoTermComputeHessianMult()` at the refreshed (x, params).
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
