#include <petsc/private/dasimpl.h>

PETSC_EXTERN PetscErrorCode PetscDASCreate_ETKF(PetscDAS);

/*@C
  PetscDASRegisterAll - Registers all of the data assimilation methods in the PetscDAS package.

  Not Collective

  Level: advanced

.seealso: `PetscDASRegister()`, `PetscDASRegisterDestroy()`
@*/
PetscErrorCode PetscDASRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDASRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDASRegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscDASRegister(PETSCDASETKF, PetscDASCreate_ETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}