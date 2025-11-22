#pragma once

#include <petscksp.h>
#include <petsctao.h>

/* MANSEC = ML */
/* SUBMANSEC = PetscDAS */

typedef struct _p_PetscDAS *PetscDAS;

typedef const char *PetscDASType;
#define PETSCDASETKF "etkf"

PETSC_EXTERN PetscFunctionList PetscDASList;
PETSC_EXTERN PetscClassId      PETSCDAS_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDASInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDASFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscDASRegister(const char[], PetscErrorCode (*)(PetscDAS));

PETSC_EXTERN PetscErrorCode PetscDASCreate(MPI_Comm, PetscDAS *);

PETSC_EXTERN PetscErrorCode PetscDASDestroy(PetscDAS *);

PETSC_EXTERN PetscErrorCode PetscDASReset(PetscDAS);

PETSC_EXTERN PetscErrorCode PetscDASSetType(PetscDAS, PetscDASType);

PETSC_EXTERN PetscErrorCode PetscDASGetType(PetscDAS, PetscDASType *);

PETSC_EXTERN PetscErrorCode PetscDASSetFromOptions(PetscDAS);

PETSC_EXTERN PetscErrorCode PetscDASSetUp(PetscDAS);

PETSC_EXTERN PetscErrorCode PetscDASView(PetscDAS, PetscViewer);

PETSC_EXTERN PetscErrorCode PetscDASViewFromOptions(PetscDAS, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscDASSetOptionsPrefix(PetscDAS, const char[]);
PETSC_EXTERN PetscErrorCode PetscDASAppendOptionsPrefix(PetscDAS, const char[]);
PETSC_EXTERN PetscErrorCode PetscDASGetOptionsPrefix(PetscDAS, const char *[]);

PETSC_EXTERN PetscErrorCode PetscDASSetEnsembleSize(PetscDAS, PetscInt);

PETSC_EXTERN PetscErrorCode PetscDASSetStateSize(PetscDAS, PetscInt);

PETSC_EXTERN PetscErrorCode PetscDASSetObservationOperator(PetscDAS, Mat);

PETSC_EXTERN PetscErrorCode PetscDASSetObservationErrorCovarianceDiagonal(PetscDAS, Vec);

PETSC_EXTERN PetscErrorCode PetscDASSetEnsemble(PetscDAS, Vec *);

PETSC_EXTERN PetscErrorCode PetscDASGetEnsemble(PetscDAS, Vec **);

PETSC_EXTERN PetscErrorCode PetscDASGetEnsembleMean(PetscDAS, Vec *);

PETSC_EXTERN PetscErrorCode PetscDASAssimilate(PetscDAS, Vec);

PETSC_EXTERN PetscErrorCode PetscDASForecast(PetscDAS, Mat);

PETSC_EXTERN PetscErrorCode PetscDASETKFSetInflation(PetscDAS, PetscReal);

PETSC_EXTERN PetscErrorCode VecSetGaussianRandom(Vec, PetscRandom, PetscReal, PetscReal);