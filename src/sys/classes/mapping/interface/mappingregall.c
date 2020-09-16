#include <petsc/private/mappingimpl.h>

PetscBool PetscMappingRegisterAllCalled = PETSC_FALSE;

PETSC_EXTERN PetscErrorCode PetscMappingCreate_Trivial(PetscMapping);

PetscErrorCode PetscMappingRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscMappingRegisterAllCalled) PetscFunctionReturn(0);
  PetscMappingRegisterAllCalled = PETSC_TRUE;

  ierr = PetscMappingRegister(PETSCMAPPINGTRIVIAL, PetscMappingCreate_Trivial);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
