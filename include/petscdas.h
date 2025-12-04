#pragma once

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

/* MANSEC = ML */
/* SUBMANSEC = PetscDAS */

typedef struct _p_PetscDAS *PetscDAS;

typedef enum {
  PETSCDASETKF_SQRT_CHOLESKY = 0,
  PETSCDASETKF_SQRT_EIGEN    = 1
} PetscDASETKFSqrtType;

typedef const char *PetscDASType;
#define PETSCDASETKF "etkf"

PETSC_EXTERN PetscErrorCode PetscDASInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDASFinalizePackage(void);

PETSC_EXTERN PetscErrorCode PetscDASRegister(const char[], PetscErrorCode (*)(PetscDAS));
PETSC_EXTERN PetscErrorCode PetscDASRegisterAll(void);

PETSC_EXTERN PetscErrorCode PetscDASCreate(MPI_Comm, PetscDAS *);
PETSC_EXTERN PetscErrorCode PetscDASDestroy(PetscDAS *);
PETSC_EXTERN PetscErrorCode PetscDASSetType(PetscDAS, PetscDASType);
PETSC_EXTERN PetscErrorCode PetscDASGetType(PetscDAS, PetscDASType *);
PETSC_EXTERN PetscErrorCode PetscDASView(PetscDAS, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDASViewFromOptions(PetscDAS, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscDASSetFromOptions(PetscDAS);

PETSC_EXTERN PetscErrorCode PetscDASSetSizes(PetscDAS, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDASGetSizes(PetscDAS, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDASSetUp(PetscDAS);

PETSC_EXTERN PetscErrorCode PetscDASSetObsErrorVariance(PetscDAS, Vec);
PETSC_EXTERN PetscErrorCode PetscDASGetObsErrorVariance(PetscDAS, Vec *);

PETSC_EXTERN PetscErrorCode PetscDASSetOrthogonalTransform(PetscDAS, Mat);
PETSC_EXTERN PetscErrorCode PetscDASGetOrthogonalTransform(PetscDAS, Mat *);

PETSC_EXTERN PetscErrorCode PetscDASGetEnsembleMember(PetscDAS, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDASRestoreEnsembleMember(PetscDAS, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDASSetEnsembleMember(PetscDAS, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode PetscDASComputeMean(PetscDAS, Vec);
PETSC_EXTERN PetscErrorCode PetscDASComputeAnomalies(PetscDAS, Mat *);
PETSC_EXTERN PetscErrorCode PetscDASAnalysis(PetscDAS, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode PetscDASApplyModel(PetscDAS, PetscErrorCode (*)(Vec, Vec, void *), void *);

PETSC_EXTERN PetscErrorCode PetscDASSetOptionsPrefix(PetscDAS, const char[]);
PETSC_EXTERN PetscErrorCode PetscDASAppendOptionsPrefix(PetscDAS, const char[]);
PETSC_EXTERN PetscErrorCode PetscDASGetOptionsPrefix(PetscDAS, const char *[]);

PETSC_EXTERN PetscErrorCode VecSetRandomGaussian(Vec, PetscRandom, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode PetscDASETKFRegister(void);
PETSC_EXTERN PetscErrorCode PetscDASETKFInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDASETKFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscDASETKFSetSqrtType(PetscDAS, PetscDASETKFSqrtType);
PETSC_EXTERN PetscErrorCode PetscDASETKFGetSqrtType(PetscDAS, PetscDASETKFSqrtType *);
