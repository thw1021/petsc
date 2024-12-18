#include <petsc/private/nmfimpl.h> /*I "petscnmf.h" I*/

PETSC_EXTERN PetscErrorCode PetscNMFCreate_AOADMM(PetscNMF);
PETSC_EXTERN PetscErrorCode PetscNMFCreate_HALS(PetscNMF);

/*@C
  PetscNMFRegisterAll - Registers all of the optimization methods in the PetscNMF
  package.

  Not Collective

  Level: developer

.seealso: `PetscNMF`, `PetscNMFRegister()`, `PetscNMFRegisterDestroy()`
@*/
PetscErrorCode PetscNMFRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscNMFRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscNMFRegisterAllCalled = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscNMFRegister(PETSCNMFAOADMM, PetscNMFCreate_AOADMM));
  PetscCall(PetscNMFRegister(PETSCNMFHALS, PetscNMFCreate_HALS));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
