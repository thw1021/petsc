#ifndef PETSC4PY_COMPAT_DA_H
#define PETSC4PY_COMPAT_DA_H

#include <petscda.h>

#if defined(PETSC_USE_COMPLEX)

#define PetscDAError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PetscDAETKFSetSqrtType(PETSC_UNUSED PetscDA da,PETSC_UNUSED PetscDASqrtType type) {PetscDAError;}
PetscErrorCode PetscDAETKFGetSqrtType(PETSC_UNUSED PetscDA da,PETSC_UNUSED PetscDASqrtType *type) {PetscDAError;}
#undef PetscDAError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_DA_H*/
