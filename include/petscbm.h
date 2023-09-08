#pragma once

#include <petscis.h>

/* SUBMANSEC = Sys */

/*S
     PetscBM - Abstract PETSc object that manages a benchmark test

   Level: intermediate

.seealso: `PetscBMCreate()`, `PetscBMDestroy()`, `PetscBMSetType()`, `PetscBMType`
S*/
typedef struct _p_PetscBM *PetscBM;

/*J
    PetscBMType - String with the name of a PETSc benchmark test

   Level: intermediate

.seealso: `PetscBMCreate()`, `PetscBMDestroy()`, `PetscBMSetType()`, `PetscBM`
J*/
typedef const char *PetscBMType;

PETSC_EXTERN PetscClassId PetscBM_CLASSID;

PETSC_EXTERN PetscErrorCode PetscBMInitializePackage(void);

PETSC_EXTERN PetscErrorCode PetscBMCreate(MPI_Comm, PetscBM *);
PETSC_EXTERN PetscErrorCode PetscBMSetFromOptions(PetscBM);
PETSC_EXTERN PetscErrorCode PetscBMSetUp(PetscBM);
PETSC_EXTERN PetscErrorCode PetscBMRun(PetscBM);
PETSC_EXTERN PetscErrorCode PetscBMReset(PetscBM);
PETSC_EXTERN PetscErrorCode PetscBMSetOptionsPrefix(PetscBM, const char[]);
PETSC_EXTERN PetscErrorCode PetscBMView(PetscBM, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscBMViewFromOptions(PetscBM, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscBMDestroy(PetscBM *);
PETSC_EXTERN PetscErrorCode PetscBMSetType(PetscBM, PetscBMType);
PETSC_EXTERN PetscErrorCode PetscBMGetType(PetscBM, PetscBMType *);
PETSC_EXTERN PetscErrorCode PetscBMRegister(const char[], PetscErrorCode (*)(PetscBM));
PETSC_EXTERN PetscErrorCode PetscBMSetSize(PetscBM, PetscInt);
PETSC_EXTERN PetscErrorCode PetscBMGetSize(PetscBM, PetscInt *);
