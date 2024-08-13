#ifndef PETSCREGRESSOR_H
#define PETSCREGRESSOR_H

#include <petsctao.h>

/* SUBMANSEC = Regressor */

/*S
     PetscRegressor - Abstract PETSc object that manages regression and classification problems

   Level: beginner

.seealso: `PetscRegressorCreate()`, `PetscRegressorSetType()`, `PetscRegressorType()`, `PetscRegressorDestroy()`
S*/

typedef struct _p_PetscRegressor *PetscRegressor;

/*J
     PetscRegressorType - String with the name of a PETSc regression method.

   Level: beginner

.seealso: PetscRegressorSetType(), PetscRegressor, PetscRegressorRegister(), PetscRegressorCreate(), PetscRegressorSetFromOptions()
J*/
typedef const char *PetscRegressorType;
#define PETSCREGRESSORLINEAR "linear"

/* Note that PetscRegressorLinearType is not a proper "type" in PETSc; it is more analogous to something like MatProductAlgorithm.
   PetscOptionsEList() should be used to ensure that the user picks a valid linear regression type from the possible options here.

   If the list of PetscRegressorLinearTypes changes, be sure to update the list at the top of linear.c as well! */
typedef const char *PetscRegressorLinearType;
#define PETSCREGRESSORLINEARDEFAULT "ols"
#define PETSCREGRESSORLINEAROLS   "ols"
#define PETSCREGRESSORLINEARLASSO "lasso"
#define PETSCREGRESSORLINEARRIDGE "ridge"

PETSC_EXTERN PetscFunctionList PetscRegressorList;
PETSC_EXTERN PetscClassId      PETSCREGRESSOR_CLASSID;

PETSC_EXTERN PetscErrorCode PetscRegressorInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscRegressorFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscRegressorRegister(const char[], PetscErrorCode (*)(PetscRegressor));
PETSC_EXTERN PetscErrorCode PetscRegressorCreate(MPI_Comm, PetscRegressor *);
PETSC_EXTERN PetscErrorCode PetscRegressorReset(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorDestroy(PetscRegressor *);
PETSC_EXTERN PetscErrorCode PetscRegressorSetType(PetscRegressor, PetscRegressorType);
PETSC_EXTERN PetscErrorCode PetscRegressorSetRegularizerWeight(PetscRegressor, PetscReal);
PETSC_EXTERN PetscErrorCode PetscRegressorSetUp(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorView(PetscRegressor, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscRegressorFit(PetscRegressor, Mat, Vec);
PETSC_EXTERN PetscErrorCode PetscRegressorPredict(PetscRegressor, Mat, Vec);
PETSC_EXTERN PetscErrorCode PetscRegressorGetTao(PetscRegressor, Tao *);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearSetFitIntercept(PetscRegressor, PetscBool);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearSetUseKSP(PetscRegressor, PetscBool);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetKSP(PetscRegressor, KSP *);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetCoefficients(PetscRegressor, Vec *);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetIntercept(PetscRegressor, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearSetType(PetscRegressor, PetscRegressorLinearType);
#endif
