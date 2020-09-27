#include <petsc/private/imimpl.h>

PetscBool IMRegisterAllCalled = PETSC_FALSE;

PETSC_EXTERN PetscErrorCode IMCreate_Map(IM);

PetscErrorCode IMRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (IMRegisterAllCalled) PetscFunctionReturn(0);
  IMRegisterAllCalled = PETSC_TRUE;

  ierr = IMRegister(IMMAP, IMCreate_Map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
