#pragma once

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

/*I <petscts.h> I*/

/* MANSEC = ML */
/* SUBMANSEC = PetscDataAssimilator */

typedef struct _p_PetscDataAssimilator *PetscDataAssimilator;

typedef enum {
  PETSCDAETKF_SQRT_CHOLESKY = 0,
  PETSCDAETKF_SQRT_EIGEN    = 1
} PetscDataAssimilatorETKFSqrtType;

typedef const char *PetscDataAssimilatorType;
#define PETSCDAETKF "etkf"

/* Logging support */
PETSC_EXTERN PetscClassId PETSCDA_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorFinalizePackage(void);

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorRegister(const char[], PetscErrorCode (*)(PetscDataAssimilator));
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorRegisterAll(void);

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorCreate(MPI_Comm, PetscDataAssimilator *);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorDestroy(PetscDataAssimilator *);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorSetType(PetscDataAssimilator, PetscDataAssimilatorType);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorGetType(PetscDataAssimilator, PetscDataAssimilatorType *);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorView(PetscDataAssimilator, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorViewFromOptions(PetscDataAssimilator, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorSetFromOptions(PetscDataAssimilator);

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorSetSizes(PetscDataAssimilator, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorGetSizes(PetscDataAssimilator, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorSetUp(PetscDataAssimilator);

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorSetObsErrorVariance(PetscDataAssimilator, Vec);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorGetObsErrorVariance(PetscDataAssimilator, Vec *);

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorSetOrthogonalTransform(PetscDataAssimilator, Mat);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorGetOrthogonalTransform(PetscDataAssimilator, Mat *);

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorGetEnsembleMember(PetscDataAssimilator, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorRestoreEnsembleMember(PetscDataAssimilator, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorSetEnsembleMember(PetscDataAssimilator, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorComputeMean(PetscDataAssimilator, Vec);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorComputeAnomalies(PetscDataAssimilator, Mat *);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorAnalysis(PetscDataAssimilator, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorApplyModel(PetscDataAssimilator, PetscErrorCode (*)(Vec, Vec, void *), void *);

PETSC_EXTERN PetscErrorCode VecSetRandomGaussian(Vec, PetscRandom, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorETKFRegister(void);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorETKFInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorETKFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorETKFSetSqrtType(PetscDataAssimilator, PetscDataAssimilatorETKFSqrtType);
PETSC_EXTERN PetscErrorCode PetscDataAssimilatorETKFGetSqrtType(PetscDataAssimilator, PetscDataAssimilatorETKFSqrtType *);
