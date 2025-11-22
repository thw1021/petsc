#ifndef PETSC4PY_COMPAT_DAS_H
#define PETSC4PY_COMPAT_DAS_H
#if defined(PETSC_USE_COMPLEX)

#define PetscDASError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PetscDASETKFSetInflation(PETSC_UNUSED PetscDAS das,PETSC_UNUSED PetscReal inflation) {PetscDASError;}
#undef PetscDASError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_DAS_H*/