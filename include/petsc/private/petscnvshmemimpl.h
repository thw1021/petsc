#ifndef PETSCNVSHMEMIMPL_H
#define PETSCNVSHMEMIMPL_H
#include <petscsys.h>

#if PetscDefined(HAVE_NVSHMEM)
PETSC_INTERN PetscErrorCode PetscNvshmemInitializeCheck(void);
PETSC_INTERN PetscErrorCode PetscNvshmemMalloc(size_t, void **);
PETSC_INTERN PetscErrorCode PetscNvshmemCalloc(size_t, void **);
PETSC_INTERN PetscErrorCode PetscNvshmemFree_Private(void *);
  #define PetscNvshmemFree(ptr) ((PetscErrorCode)((ptr) && (PetscNvshmemFree_Private(ptr) || ((ptr) = PETSC_NULLPTR, PETSC_SUCCESS))))
PETSC_INTERN PetscErrorCode PetscNvshmemSum(PetscInt, PetscScalar *, const PetscScalar *);
PETSC_INTERN PetscErrorCode PetscNvshmemMax(PetscInt, PetscReal *, const PetscReal *);
#else
  #define PetscNvshmemFree(ptr) PETSC_SUCCESS
#endif

#endif
