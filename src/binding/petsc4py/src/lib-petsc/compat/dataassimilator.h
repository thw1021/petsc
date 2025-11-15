#ifndef PETSC4PY_COMPAT_DATAASSIMILATOR_H
#define PETSC4PY_COMPAT_DATAASSIMILATOR_H

#include <petscdataassimilator.h>

#if defined(PETSC_USE_COMPLEX)

#define PetscDataAssimilatorError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PetscDataAssimilatorETKFSetSqrtType(PETSC_UNUSED PetscDataAssimilator da,PETSC_UNUSED PetscDataAssimilatorETKFSqrtType type) {PetscDataAssimilatorError;}
PetscErrorCode PetscDataAssimilatorETKFGetSqrtType(PETSC_UNUSED PetscDataAssimilator da,PETSC_UNUSED PetscDataAssimilatorETKFSqrtType *type) {PetscDataAssimilatorError;}
#undef PetscDataAssimilatorError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_DATAASSIMILATOR_H*/
