#pragma once

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

/* MANSEC = ML */
/* SUBMANSEC = PetscDA */

typedef struct _p_PetscDA *PetscDA;

typedef enum {
  PETSCDA_SQRT_CHOLESKY = 0,
  PETSCDA_SQRT_EIGEN    = 1
} PetscDASqrtType;

typedef const char *PetscDAType;
#define PETSCDAETKF  "etkf"
#define PETSCDALETKF "letkf"

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
PETSC_EXTERN PetscErrorCode PetscDASetLocalSizes(PetscDA, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDAGetSizes(PetscDA, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDASetNDOF(PetscDA, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDAGetNDOF(PetscDA, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDASetUp(PetscDA);

PETSC_EXTERN PetscErrorCode PetscDASetObsErrorVariance(PetscDA, Vec);
PETSC_EXTERN PetscErrorCode PetscDAGetObsErrorVariance(PetscDA, Vec *);

PETSC_EXTERN PetscErrorCode PetscDASetInflation(PetscDA, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDAGetInflation(PetscDA, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDAGetEnsembleMember(PetscDA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDARestoreEnsembleMember(PetscDA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDASetEnsembleMember(PetscDA, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode PetscDAComputeEnsembleMean(PetscDA, Vec);
PETSC_EXTERN PetscErrorCode PetscDAComputeAnomalies(PetscDA, Vec, Mat *);
PETSC_EXTERN PetscErrorCode PetscDAAnalysis(PetscDA, Vec, Mat);
PETSC_EXTERN PetscErrorCode PetscDAApplyModel(PetscDA, PetscErrorCode (*)(Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode PetscDAInitializeEnsemble(PetscDA, Vec, PetscInt, PetscReal, PetscRandom);

PETSC_EXTERN PetscErrorCode PetscDASetOptionsPrefix(PetscDA, const char[]);
PETSC_EXTERN PetscErrorCode PetscDAAppendOptionsPrefix(PetscDA, const char[]);
PETSC_EXTERN PetscErrorCode PetscDAGetOptionsPrefix(PetscDA, const char *[]);

PETSC_EXTERN PetscErrorCode PetscDAVecSetRandomGaussian(Vec, PetscRandom, PetscReal, PetscReal);

/* T-matrix factorization functions (base class) */
PETSC_EXTERN PetscErrorCode PetscDASetSqrtType(PetscDA, PetscDASqrtType);
PETSC_EXTERN PetscErrorCode PetscDAGetSqrtType(PetscDA, PetscDASqrtType *);
PETSC_EXTERN PetscErrorCode PetscDATFactor(PetscDA, Mat);
PETSC_EXTERN PetscErrorCode PetscDAApplyTInverse(PetscDA, Vec, Vec);
PETSC_EXTERN PetscErrorCode PetscDAApplySqrtTInverse(PetscDA, Mat, Mat);

PETSC_EXTERN PetscErrorCode PetscDAETKFInitialize(PetscDA);
PETSC_EXTERN PetscErrorCode PetscDAETKFInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDAETKFFinalizePackage(void);

PETSC_EXTERN PetscErrorCode PetscDALETKFInitialize(PetscDA);
PETSC_EXTERN PetscErrorCode PetscDALETKFInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDALETKFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscDALETKFSetLocalization(PetscDA, Mat, Mat);
PETSC_EXTERN PetscErrorCode PetscDALETKFSetObsPerVertex(PetscDA, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDALETKFGetObsPerVertex(PetscDA, PetscInt *);

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_EXTERN PetscErrorCode PetscDALETKFGetLocalizationMatrix(const PetscInt, const PetscInt, Vec[3], PetscReal[3], Mat, Mat *);
#endif
