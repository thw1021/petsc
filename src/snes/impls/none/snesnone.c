#include <petsc/private/snesimpl.h> /*I   "petscsnes.h"   I*/

static PetscErrorCode SNESSolve_None(SNES snes)
{
  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITS;
  snes->iter                   = 0;
  snes->norm                   = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
      SNESNONE - Nonlinear solver that is the identity map.

   Level: beginner

.seealso: `SNES`, `SNESType`, `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESNEWTONLS`, `SNESNEWTONTR`
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_None(SNES snes)
{
  PetscFunctionBegin;
  snes->ops->solve = SNESSolve_None;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  snes->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
