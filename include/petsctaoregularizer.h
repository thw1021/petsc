#pragma once

/* SUBMANSEC = Tao */

/*S
   TaoRegularizer - PETSc object that manages regularizer for the `Tao` optimization solves

   Level: intermediate

.seealso: [](ch_tao), `TaoRegularizerType`, `Tao`, `TaoCreate()`, `TaoDestroy()`, `TaoSetType()`, `TaoType`
S*/
typedef struct _p_TaoRegularizer *TaoRegularizer;

#include <petsctao.h>

/*J
        TaoRegularizerType - String with the name of a `TaoRegularizer` method

   Values:
+   `TAOREGULARIZERL1`   - L1 regularizer. \|x\|_1. Not valid for Metric where y has been set.
.   `TAOREGULARIZERL2`   - L2 regularizer. \|x-y\|_2^2.  y = 0 if not set or NULL.
/   `TAOREGULARIZERKL`   - KL Divergence. Only for Metric version where y vector has been set.
-   `TAOREGULARIZERUSER` - User-set version. The routines are stored within Tao

  Options Database Key:
.  -tao_reg_type <type> - select which method Tao should use at runtime

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerSetType()`, `TaoCreate()`, `TaoSetType()`
J*/
typedef const char *TaoRegularizerType;
#define TAOREGULARIZERL1 "l1"
#define TAOREGULARIZERL2 "l2"
#define TAOREGULARIZERKL "kl"

PETSC_EXTERN PetscClassId      TAOREGULARIZER_CLASSID;
PETSC_EXTERN PetscFunctionList TaoRegularizerList;

PETSC_EXTERN PetscErrorCode TaoRegularizerCreate(MPI_Comm, TaoRegularizer *);
PETSC_EXTERN PetscErrorCode TaoRegularizerSetFromOptions(TaoRegularizer);
PETSC_EXTERN PetscErrorCode TaoRegularizerSetUp(TaoRegularizer);
PETSC_EXTERN PetscErrorCode TaoRegularizerDestroy(TaoRegularizer *);
PETSC_EXTERN PetscErrorCode TaoRegularizerMonitor(TaoRegularizer, PetscInt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoRegularizerView(TaoRegularizer, PetscViewer);
PETSC_EXTERN PetscErrorCode TaoRegularizerViewFromOptions(TaoRegularizer, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode TaoRegularizerSetOptionsPrefix(TaoRegularizer, const char prefix[]);
PETSC_EXTERN PetscErrorCode TaoRegularizerReset(TaoRegularizer);
PETSC_EXTERN PetscErrorCode TaoRegularizerAppendOptionsPrefix(TaoRegularizer, const char[]);
PETSC_EXTERN PetscErrorCode TaoRegularizerGetOptionsPrefix(TaoRegularizer, const char *[]);
PETSC_EXTERN PetscErrorCode TaoRegularizerGetStepLength(TaoRegularizer, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoRegularizerGetCentralVector(TaoRegularizer, Vec *);
PETSC_EXTERN PetscErrorCode TaoRegularizerSetCentralVector(TaoRegularizer, Vec);
PETSC_EXTERN PetscErrorCode TaoRegularizerSetInitialStepLength(TaoRegularizer, PetscReal);

PETSC_EXTERN PetscErrorCode TaoRegularizerGetType(TaoRegularizer, TaoRegularizerType *);
PETSC_EXTERN PetscErrorCode TaoRegularizerSetType(TaoRegularizer, TaoRegularizerType);

PETSC_EXTERN PetscErrorCode TaoRegularizerIsUsingTaoRoutines(TaoRegularizer, PetscBool *);

PETSC_EXTERN PetscErrorCode TaoRegularizerSetObjective(TaoRegularizer, PetscErrorCode (*)(TaoRegularizer, Vec, PetscReal *, void *), void *);
PETSC_EXTERN PetscErrorCode TaoRegularizerSetGradient(TaoRegularizer, PetscErrorCode (*)(TaoRegularizer, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoRegularizerSetObjectiveAndGradient(TaoRegularizer, PetscErrorCode (*)(TaoRegularizer, Vec, PetscReal *, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoRegularizerUseTaoRoutines(TaoRegularizer, Tao);

PETSC_EXTERN PetscErrorCode TaoRegularizerComputeObjective(TaoRegularizer, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoRegularizerComputeObjectiveAndGradient(TaoRegularizer, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoRegularizerComputeGradient(TaoRegularizer, Vec, Vec);

PETSC_EXTERN PetscErrorCode TaoRegularizerSetScale(TaoRegularizer, PetscReal);
PETSC_EXTERN PetscErrorCode TaoRegularizerGetScale(TaoRegularizer, PetscReal *);

PETSC_EXTERN PetscErrorCode TaoRegularizerInitializePackage(void);
PETSC_EXTERN PetscErrorCode TaoRegularizerFinalizePackage(void);

PETSC_EXTERN PetscErrorCode TaoRegularizerRegister(const char[], PetscErrorCode (*)(TaoRegularizer));
PETSC_EXTERN PetscErrorCode TaoSetRegularizer(Tao, TaoRegularizer);
PETSC_EXTERN PetscErrorCode TaoGetRegularizer(Tao, TaoRegularizer *);
PETSC_EXTERN PetscErrorCode TaoRegularizerSetTao(TaoRegularizer, Tao);
