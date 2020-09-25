#include <petsc/private/imimpl.h>

PetscBool IMRegisterAllCalled = PETSC_FALSE;

PETSC_EXTERN PetscErrorCode IMCreate_Trivial(IM);

PetscErrorCode IMRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (IMRegisterAllCalled) PetscFunctionReturn(0);
  IMRegisterAllCalled = PETSC_TRUE;

  ierr = IMRegister(IMTRIVIAL, IMCreate_Trivial);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
