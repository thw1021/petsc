#include <petsc/private/regressorimpl.h>

PETSC_EXTERN PetscErrorCode PetscRegressorCreate_Linear(PetscRegressor);

PetscErrorCode PetscRegressorRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscRegressorRegisterAllCalled) PetscFunctionReturn(0);
  PetscRegressorRegisterAllCalled = PETSC_TRUE;
  // TODO: Register all of the sub-types
  PetscCall(PetscRegressorRegister(PETSCREGRESSORLINEAR, PetscRegressorCreate_Linear));
  PetscFunctionReturn(0);
}
