#include "petscda.h"

PETSC_EXTERN PetscErrorCode PetscDALETKFInitialize(PetscDA);

PETSC_EXTERN PetscErrorCode PetscDALETKFRegister(void)
{
  PetscFunctionBegin;
  PetscCall(PetscDARegister("letkf", PetscDALETKFInitialize));
  PetscCall(PetscDALETKFInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}