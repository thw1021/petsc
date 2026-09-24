#pragma once

#include <petsctao.h>

/* MANSEC = ML */
/* SUBMANSEC = PetscRegressor */

/*S
   PetscRegressor - Abstract PETSc object that manages regression and classification problems

   Level: beginner

   Notes:
   For linear problems `PetscRegressor` supports ordinary least squares, lasso, and ridge regression using the `PetscRegressorType` of `PETSCREGRESSORLINEAR`
   and `PetscRegressorLinearType` of `REGRESSOR_LINEAR_OLS`, `REGRESSOR_LINEAR_LASSO`, and `REGRESSOR_LINEAR_RIDGE`.
   For models that depend nonlinearly on their parameters, `PetscRegressor` supports nonlinear least squares curve fitting
   using the `PetscRegressorType` of `PETSCREGRESSORNLLS`.

   We have slightly abused the term "regressor" in the naming of this component of PETSc.
   Statisticians would say that we are doing "regression", and a "regressor", in this context, strictly means an
   independent (or "predictor") variable in the regression analysis. However, "regressor" has taken on an informal
   meaning in the machine-learning community of something along the lines of "algorithm or implementation used to fit
   a regression model". Examples are `MLPRegressor` (multi-layer perceptron regressor) or `RandomForestRegressor`
   from the scikit-learn toolkit (which is itself not consistent about the use of the term "regressor", since it has a
   `LinearRegression` component instead of a `LinearRegressor` component).

.seealso: `PetscRegressorCreate()`, `PetscRegressorLinearType`, `PetscRegressorSetType()`, `PetscRegressorType`, `PetscRegressorDestroy()`,
          `PETSCREGRESSORLINEAR`, `PETSCREGRESSORNLLS`, `REGRESSOR_LINEAR_OLS`, `REGRESSOR_LINEAR_LASSO`, `REGRESSOR_LINEAR_RIDGE`
S*/
typedef struct _p_PetscRegressor *PetscRegressor;

/*J
  PetscRegressorType - String with the name of a PETSc regression method.

  Level: beginner

.seealso: [](ch_regressor), `PetscRegressorSetType()`, `PetscRegressor`, `PetscRegressorRegister()`, `PetscRegressorCreate()`, `PetscRegressorSetFromOptions()`,
          `PETSCREGRESSORLINEAR`, `PETSCREGRESSORNLLS`
J*/
typedef const char *PetscRegressorType;
#define PETSCREGRESSORLINEAR "linear"
#define PETSCREGRESSORNLLS   "nlls"

/*E
  PetscRegressorLinearType - Type of linear regression

  Values:
+  `REGRESSOR_LINEAR_OLS`    - ordinary least squares
.  `REGRESSOR_LINEAR_LASSO`  - lasso
-  `REGRESSOR_LINEAR_RIDGE`  - ridge

  Level: advanced

  Note:
  One can perform binary classification using the ridge regressor type by converting labels into the
  values -1 and +1, corresponding to the two classes, and then performing a ridge regression.
  Observations with a negative prediction value are then placed in the -1 class, while those with positive values
  are placed in the +1 class.
  This is the approach used in the RidgeClassifer implementation provided by the scikit-learn library.

.seealso: `PetscRegressor`, `PETSCREGRESSORLINEAR`
E*/

typedef enum {
  REGRESSOR_LINEAR_OLS,
  REGRESSOR_LINEAR_LASSO,
  REGRESSOR_LINEAR_RIDGE
} PetscRegressorLinearType;
PETSC_EXTERN const char *const PetscRegressorLinearTypes[];

PETSC_EXTERN PetscFunctionList PetscRegressorList;
PETSC_EXTERN PetscClassId      PETSCREGRESSOR_CLASSID;

PETSC_EXTERN PetscErrorCode PetscRegressorInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscRegressorFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscRegressorRegister(const char[], PetscErrorCode (*)(PetscRegressor));

PETSC_EXTERN PetscErrorCode PetscRegressorCreate(MPI_Comm, PetscRegressor *);
PETSC_EXTERN PetscErrorCode PetscRegressorReset(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorDestroy(PetscRegressor *);

PETSC_EXTERN PetscErrorCode PetscRegressorSetOptionsPrefix(PetscRegressor, const char[]);
PETSC_EXTERN PetscErrorCode PetscRegressorAppendOptionsPrefix(PetscRegressor, const char[]);
PETSC_EXTERN PetscErrorCode PetscRegressorGetOptionsPrefix(PetscRegressor, const char *[]);

PETSC_EXTERN PetscErrorCode PetscRegressorSetType(PetscRegressor, PetscRegressorType);
PETSC_EXTERN PetscErrorCode PetscRegressorGetType(PetscRegressor, PetscRegressorType *);
PETSC_EXTERN PetscErrorCode PetscRegressorSetRegularizerWeight(PetscRegressor, PetscReal);
PETSC_EXTERN PetscErrorCode PetscRegressorSetUp(PetscRegressor);
PETSC_EXTERN PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor);

PETSC_EXTERN PetscErrorCode PetscRegressorView(PetscRegressor, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscRegressorViewFromOptions(PetscRegressor, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode PetscRegressorFit(PetscRegressor, Mat, Vec);
PETSC_EXTERN PetscErrorCode PetscRegressorPredict(PetscRegressor, Mat, Vec);
PETSC_EXTERN PetscErrorCode PetscRegressorGetTao(PetscRegressor, Tao *);

/*S
  PetscRegressorNLLSFunctionFn - A prototype of a `PETSCREGRESSORNLLS` model evaluation function that would be passed to `PetscRegressorNLLSSetFunction()`

  Calling Sequence:
+ regressor - the `PetscRegressor` context
. X         - data matrix of independent variables (one row per sample)
. p         - vector of model parameters
. f         - output vector of model evaluations $f(X, p)$
- ctx       - [optional] user-defined function context

  Level: beginner

.seealso: `PetscRegressor`, `PETSCREGRESSORNLLS`, `PetscRegressorNLLSSetFunction()`, `PetscRegressorNLLSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscRegressorNLLSFunctionFn(PetscRegressor regressor, Mat X, Vec p, Vec f, void *ctx);

/*S
  PetscRegressorNLLSJacobianFn - A prototype of a `PETSCREGRESSORNLLS` Jacobian evaluation function that would be passed to `PetscRegressorNLLSSetJacobian()`

  Calling Sequence:
+ regressor - the `PetscRegressor` context
. X         - data matrix of independent variables (one row per sample)
. p         - vector of model parameters
. J         - Jacobian matrix $\partial f/\partial p$
. Jpre      - matrix used to construct the preconditioner, often the same as `J`
- ctx       - [optional] user-defined function context

  Level: beginner

.seealso: `PetscRegressor`, `PETSCREGRESSORNLLS`, `PetscRegressorNLLSSetJacobian()`, `PetscRegressorNLLSFunctionFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscRegressorNLLSJacobianFn(PetscRegressor regressor, Mat X, Vec p, Mat J, Mat Jpre, void *ctx);

PETSC_EXTERN PetscErrorCode PetscRegressorNLLSSetFunction(PetscRegressor, Vec, PetscRegressorNLLSFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSGetFunction(PetscRegressor, Vec *, PetscRegressorNLLSFunctionFn **, void **);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSSetJacobian(PetscRegressor, Mat, Mat, PetscRegressorNLLSJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSGetJacobian(PetscRegressor, Mat *, Mat *, PetscRegressorNLLSJacobianFn **, void **);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSSetInitialParameters(PetscRegressor, Vec);
PETSC_EXTERN PetscErrorCode PetscRegressorNLLSGetParameters(PetscRegressor, Vec *);

PETSC_EXTERN PetscErrorCode PetscRegressorLinearSetFitIntercept(PetscRegressor, PetscBool);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearSetUseKSP(PetscRegressor, PetscBool);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetKSP(PetscRegressor, KSP *);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetCoefficients(PetscRegressor, Vec *);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetIntercept(PetscRegressor, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearSetType(PetscRegressor, PetscRegressorLinearType);
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetType(PetscRegressor, PetscRegressorLinearType *);
