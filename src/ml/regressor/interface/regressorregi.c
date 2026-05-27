#include <petsc/private/regressorimpl.h>

PETSC_EXTERN PetscErrorCode PetscRegressorCreate_Linear(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorCreate_NLLS(PetscRegressor);

PetscErrorCode PetscRegressorRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscRegressorRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscRegressorRegisterAllCalled = PETSC_TRUE;
  // Register all of the types of PetscRegressor
#if !PetscDefined(USE_COMPLEX)
  PetscCall(PetscRegressorRegister(PETSCREGRESSORLINEAR, PetscRegressorCreate_Linear));
  PetscCall(PetscRegressorRegister(PETSCREGRESSORNLLS, PetscRegressorCreate_NLLS));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
