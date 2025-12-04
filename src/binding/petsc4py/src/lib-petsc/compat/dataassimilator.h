#ifndef PETSC4PY_COMPAT_DATAASSIMILATOR_H
#define PETSC4PY_COMPAT_DATAASSIMILATOR_H

#include <petscdataassimilator.h>

#if defined(PETSC_USE_COMPLEX)

#define PetscDASError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PetscDASETKFSetSqrtType(PETSC_UNUSED PetscDAS da,PETSC_UNUSED PetscDASETKFSqrtType type) {PetscDASError;}
PetscErrorCode PetscDASETKFGetSqrtType(PETSC_UNUSED PetscDAS da,PETSC_UNUSED PetscDASETKFSqrtType *type) {PetscDASError;}
#undef PetscDASError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_DATAASSIMILATOR_H*/
