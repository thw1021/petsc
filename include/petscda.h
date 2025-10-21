#pragma once

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

/* MANSEC = TS */
/* SUBMANSEC = DA */

typedef struct _p_DA *DA;

typedef enum {
  DAETKF_SQRT_CHOLESKY = 0,
  DAETKF_SQRT_EIGEN    = 1
} DAETKFSqrtType;

typedef const char *DAType;
#define DAETKF "daetkf"

/* Logging support */
PETSC_EXTERN PetscClassId DA_CLASSID;

PETSC_EXTERN PetscErrorCode DAInitializePackage(void);
PETSC_EXTERN PetscErrorCode DAFinalizePackage(void);

PETSC_EXTERN PetscErrorCode DARegister(const char[], PetscErrorCode (*)(DA));
PETSC_EXTERN PetscErrorCode DARegisterAll(void);

PETSC_EXTERN PetscErrorCode DACreate(MPI_Comm, DA *);
PETSC_EXTERN PetscErrorCode DADestroy(DA *);
PETSC_EXTERN PetscErrorCode DASetType(DA, DAType);
PETSC_EXTERN PetscErrorCode DAGetType(DA, DAType *);
PETSC_EXTERN PetscErrorCode DAView(DA, PetscViewer);
PETSC_EXTERN PetscErrorCode DAViewFromOptions(DA, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode DASetFromOptions(DA);

PETSC_EXTERN PetscErrorCode DASetSizes(DA, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DAGetSizes(DA, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DASetUp(DA);

PETSC_EXTERN PetscErrorCode DASetObsErrorVariance(DA, Vec);
PETSC_EXTERN PetscErrorCode DAGetObsErrorVariance(DA, Vec *);

PETSC_EXTERN PetscErrorCode DASetOrthogonalTransform(DA, Mat);
PETSC_EXTERN PetscErrorCode DAGetOrthogonalTransform(DA, Mat *);

PETSC_EXTERN PetscErrorCode DAGetEnsembleMember(DA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode DARestoreEnsembleMember(DA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode DASetEnsembleMember(DA, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode DAComputeMean(DA, Vec);
PETSC_EXTERN PetscErrorCode DAComputeAnomalies(DA, Mat *);
PETSC_EXTERN PetscErrorCode DAAnalysis(DA, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode DAApplyModel(DA, PetscErrorCode (*)(Vec, Vec, void *), void *);

PETSC_EXTERN PetscErrorCode VecSetRandomGaussian(Vec, PetscRandom, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode DAETKFRegister(void);
PETSC_EXTERN PetscErrorCode DAETKFInitializePackage(void);
PETSC_EXTERN PetscErrorCode DAETKFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode DAETKFSetSqrtType(DA, DAETKFSqrtType);
PETSC_EXTERN PetscErrorCode DAETKFGetSqrtType(DA, DAETKFSqrtType *);
