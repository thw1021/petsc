#pragma once

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

/* MANSEC = DA */

typedef struct _p_PetscDA *PetscDA;

typedef enum {
  PETSCDAETKF_SQRT_CHOLESKY = 0,
  PETSCDAETKF_SQRT_EIGEN    = 1
} PetscDAETKFSqrtType;

typedef const char *PetscDAType;
#define PETSCDAETKF "petscdaetkf"

/* Logging support */
PETSC_EXTERN PetscClassId PETSCDA_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDAInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDAFinalizePackage(void);

PETSC_EXTERN PetscErrorCode PetscDARegister(const char[], PetscErrorCode (*)(PetscDA));
PETSC_EXTERN PetscErrorCode PetscDARegisterAll(void);

PETSC_EXTERN PetscErrorCode PetscDACreate(MPI_Comm, PetscDA *);
PETSC_EXTERN PetscErrorCode PetscDADestroy(PetscDA *);
PETSC_EXTERN PetscErrorCode PetscDASetType(PetscDA, PetscDAType);
PETSC_EXTERN PetscErrorCode PetscDAGetType(PetscDA, PetscDAType *);
PETSC_EXTERN PetscErrorCode PetscDAView(PetscDA, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDAViewFromOptions(PetscDA, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscDASetFromOptions(PetscDA);

PETSC_EXTERN PetscErrorCode PetscDASetSizes(PetscDA, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDAGetSizes(PetscDA, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDASetUp(PetscDA);

PETSC_EXTERN PetscErrorCode PetscDASetObsErrorVariance(PetscDA, Vec);
PETSC_EXTERN PetscErrorCode PetscDAGetObsErrorVariance(PetscDA, Vec *);

PETSC_EXTERN PetscErrorCode PetscDASetOrthogonalTransform(PetscDA, Mat);
PETSC_EXTERN PetscErrorCode PetscDAGetOrthogonalTransform(PetscDA, Mat *);

PETSC_EXTERN PetscErrorCode PetscDAGetEnsembleMember(PetscDA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDARestoreEnsembleMember(PetscDA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDASetEnsembleMember(PetscDA, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode PetscDAComputeMean(PetscDA, Vec);
PETSC_EXTERN PetscErrorCode PetscDAComputeAnomalies(PetscDA, Mat *);
PETSC_EXTERN PetscErrorCode PetscDAAnalysis(PetscDA, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode PetscDAApplyModel(PetscDA, PetscErrorCode (*)(Vec, Vec, void *), void *);

PETSC_EXTERN PetscErrorCode VecSetRandomGaussian(Vec, PetscRandom, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode PetscDAETKFRegister(void);
PETSC_EXTERN PetscErrorCode PetscDAETKFInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDAETKFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscDAETKFSetSqrtType(PetscDA, PetscDAETKFSqrtType);
PETSC_EXTERN PetscErrorCode PetscDAETKFGetSqrtType(PetscDA, PetscDAETKFSqrtType *);
