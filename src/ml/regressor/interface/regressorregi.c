#include <petsc/private/regressorimpl.h>

PETSC_EXTERN PetscErrorCode PetscRegressorCreate_Linear(PetscRegressor);

PetscErrorCode PetscRegressorRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscRegressorRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscRegressorRegisterAllCalled = PETSC_TRUE;
  // Register all of the types of PetscRegressor
  PetscCall(PetscRegressorRegister(PETSCREGRESSORLINEAR, PetscRegressorCreate_Linear));
  PetscFunctionReturn(PETSC_SUCCESS);
}
