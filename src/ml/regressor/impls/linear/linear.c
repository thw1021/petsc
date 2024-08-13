#include <../src/ml/regressor/impls/linear/linearimpl.h> /*I "petscregressor.h" I*/

// Note: If the list of PetscRegressorLinearTypes changes, it must also be updated in petscregressor.h
#define PETSCREGRESSOR_LINEAR_NUM_TYPES 3
static const char *PetscRegressor_Linear_Types_Table[PETSCREGRESSOR_LINEAR_NUM_TYPES] = {PETSCREGRESSORLINEAROLS, PETSCREGRESSORLINEARLASSO, PETSCREGRESSORLINEARRIDGE};

PetscErrorCode EvaluateResidual(Tao tao, Vec x, Vec f, void *ptr)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)ptr;

  PetscFunctionBegin;
  /* Evaluate f = A * x - b */
  PetscCall(MatMult(linear->X, x, f));
  PetscCall(VecAXPY(f, -1.0, linear->rhs));
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *ptr)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)ptr;

  PetscFunctionBegin;
  J    = linear->X;
  Jpre = linear->X;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorSetUp_Linear(PetscRegressor regressor)
{
  //MPI_Comm comm;
  PetscInt               M, N;
  PetscBool              flg;
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;
  KSP                    ksp;
  Tao                    tao;

  PetscFunctionBegin;

  PetscCall(MatGetSize(regressor->training, &M, &N));

  if (linear->fit_intercept) {
    /* If we are fitting the intercept, we need to make A a composite matrix using MATCENTERING to preserve sparsity.
     * Though there might be some cases we don't want to do this for, depending on what kind of matrix is passed in. (Probably bad idea for dense?)
     * We will also need to ensure that the right-hand side passed to the KSP is also mean-centered, since we
     * intend to compute the intercept separately from regression coefficients (that is, we will not be adding a
     * column of all 1s to our design matrix). */
    PetscCall(MatCreateCentering(PetscObjectComm((PetscObject)regressor), PETSC_DECIDE, M, &linear->C));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)regressor), &linear->X));
    PetscCall(MatSetSizes(linear->X, PETSC_DECIDE, PETSC_DECIDE, M, N));
    PetscCall(MatSetType(linear->X, MATCOMPOSITE));
    PetscCall(MatCompositeSetType(linear->X, MAT_COMPOSITE_MULTIPLICATIVE));
    PetscCall(MatCompositeAddMat(linear->X, regressor->training));
    PetscCall(MatCompositeAddMat(linear->X, linear->C));
    PetscCall(VecDuplicate(regressor->target, &linear->rhs));
    PetscCall(MatMult(linear->C, regressor->target, linear->rhs));
  } else {
    /* When not fitting intercept, we assume that the input data are already centered.
     * TODO: Perhaps revisit exactly what options should exist around this. */
    linear->X   = regressor->training;
    linear->rhs = regressor->target;
  }

  if (linear->coefficients) PetscCall(VecDestroy(&linear->coefficients));

  if (linear->use_ksp) {
    // TODO: Make this error out if the linear type is not OLS.

    if (!linear->ksp) {
      PetscCall(PetscRegressorLinearGetKSP(regressor, &linear->ksp));
      // TODO: Figure out if I need to set operators for the KSP here or set the operator X.
      // I think maybe I can just do this stuff in the Fit() routine.
    }
    ksp = linear->ksp;

    PetscCall(MatCreateVecs(linear->X, &linear->coefficients, NULL));
    /* Set up the KSP to solve the least squares problem (without solving for intercept, as this is done separately) using KSPLSQR.
     * TODO: Add options to use other methods. */
    PetscCall(MatCreateNormal(linear->X, &linear->XtX));
    PetscCall(KSPSetType(ksp, KSPLSQR));
    PetscCall(KSPSetOperators(ksp, linear->X, linear->XtX));
    PetscCall(KSPSetFromOptions(ksp)); // TODO: Does this have the right option prefixes set?
  } else {                             /* Use TAO */
    if (!regressor->tao) { PetscCall(PetscRegressorGetTao(regressor, &tao)); }

    PetscCall(MatCreateVecs(linear->X, &linear->coefficients, &linear->residual));
    /* Set up the TAO object to solve the (regularized) least squares problem (without solving for intercept, which is done separately) using TAOBRGN. */
    PetscCall(TaoSetType(tao, TAOBRGN));
    PetscCall(TaoSetSolution(tao, linear->coefficients));
    PetscCall(TaoSetResidualRoutine(tao, linear->residual, EvaluateResidual, linear));
    PetscCall(TaoSetJacobianResidualRoutine(tao, linear->X, linear->X, EvaluateJacobian, linear));
    // Set the regularization type and weight for the BRGN as linear->type dictates:
    PetscCall(PetscStrcmp(linear->type, PETSCREGRESSORLINEAROLS, &flg));
    if (flg && !linear->use_ksp) PetscCall(TaoBRGNSetRegularizerWeight(tao, 0.0));
    PetscCall(PetscStrcmp(linear->type, PETSCREGRESSORLINEARLASSO, &flg));
    if (flg) {
      PetscCall(PetscOptionsSetValue(NULL, "-tao_brgn_regularization_type", "l1dict"));
      if (regressor->regularizer_weight_is_set) PetscCall(TaoBRGNSetRegularizerWeight(tao, regressor->regularizer_weight));
      else PetscCall(TaoBRGNSetRegularizerWeight(tao, 1.0));  // Set the default regularization weight to 1.0, the default for LASSO in SciKit-learn
    }
    PetscCall(PetscStrcmp(linear->type, PETSCREGRESSORLINEARRIDGE, &flg));
    if (flg) {
      PetscCall(PetscOptionsSetValue(NULL, "-tao_brgn_regularization_type", "l2pure"));
      if (regressor->regularizer_weight_is_set) PetscCall(TaoBRGNSetRegularizerWeight(tao, regressor->regularizer_weight));
      else PetscCall(TaoBRGNSetRegularizerWeight(tao, 1.0));  // Set the default regularization weight to 1.0, the default for ridge regression in SciKit-learn
    }
    PetscCall(TaoSetFromOptions(tao));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorReset_Linear(PetscRegressor regressor)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;

  PetscFunctionBegin;
  /* Destroy the PETSc objects associated with the linear regressor implementation. */
  PetscCall(MatDestroy(&linear->X));
  PetscCall(MatDestroy(&linear->XtX));
  PetscCall(MatDestroy(&linear->C));
  PetscCall(KSPDestroy(&linear->ksp));
  PetscCall(VecDestroy(&linear->coefficients));
  PetscCall(VecDestroy(&linear->rhs));
  PetscCall(VecDestroy(&linear->residual));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorDestroy_Linear(PetscRegressor regressor)
{
  PetscFunctionBegin;
  PetscCall(PetscRegressorReset_Linear(regressor));
  PetscCall(PetscFree(regressor->data));

  PetscFunctionReturn(0);
}

/*@
   PetscRegressorLinearSetFitIntercept - Set a flag to indicate that the intercept (also known as the "bias" or "offset") should
   be calculated; data are assumed to be mean-centered if false.

   Logically Collective on PetscRegressor

   Input Parameters:
+  regressor - the regressor context
-  flg - PETSC_TRUE to calculate the intercept, PETSC_FALSE to assume centered data (default is true)

   Level: intermediate
@*/
/* TODO: Add companion PetscRegressorLinearGetFitIntercept(), and put it in the .seealso: */
PetscErrorCode PetscRegressorLinearSetFitIntercept(PetscRegressor regressor, PetscBool flg)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;

  PetscFunctionBegin;
  linear->fit_intercept = flg;
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorLinearSetUseKSP - Set a flag to indicate that a KSP object, instead of a Tao one, should be used
   to fit the regressor

   Logically Collective on PetscRegressor

   Input Parameters:
+  regressor - the regressor context
-  flg - PETSC_TRUE to use a KSP, PETSC_FALSE to use a Tao object (default is false)

   Level: intermediate
@*/
/* TODO: Add companion PetscRegressorLinearGetUseKSP(), and put it in the .seealso: */
PetscErrorCode PetscRegressorLinearSetUseKSP(PetscRegressor regressor, PetscBool flg)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;

  PetscFunctionBegin;
  linear->use_ksp = flg;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorSetFromOptions_Linear(PetscOptionItems *PetscOptionsObject, PetscRegressor regressor)
{
  PetscBool set, flg = PETSC_FALSE;
  PetscInt i;
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscRegressor options for linear regressors");
  PetscCall(PetscOptionsBool("-regressor_linear_fit_intercept", "Calculate intercept for linear model", "PetscRegressorLinearSetFitIntercept", flg, &flg, &set));
  if (set) PetscCall(PetscRegressorLinearSetFitIntercept(regressor, flg));
  PetscCall(PetscOptionsBool("-regressor_linear_use_ksp", "Use KSP instead of TAO for linear model fitting problem", "PetscRegressorLinearSetFitIntercept", flg, &flg, &set));
  if (set) PetscCall(PetscRegressorLinearSetUseKSP(regressor, flg));
  PetscCall(PetscOptionsEList("-regressor_linear_type", "Linear regression method", "", PetscRegressor_Linear_Types_Table, PETSCREGRESSOR_LINEAR_NUM_TYPES, linear->type, &i, &set));
  if (set) PetscCall(PetscRegressorLinearSetType(regressor, PetscRegressor_Linear_Types_Table[i]));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRegressorView_Linear(PetscRegressor regressor, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorLinearGetKSP - Returns the KSP context for a PETSCREGRESSORLINEAR object.

   Not Collective, but if the PetscRegressor is parallel, then the KSP object is parallel

   Input Parameter:
.  regressor - the regressor context

   Output Parameter:
.  ksp - the KSP context

   Notes:
   This routine will always return a KSP, but, depending on the type of the linear regressor and the options that are set, the regressor may actually use a Tao object instead of this KSP.

   Level: beginner

.seealso: PetscRegressorGetTao()
@*/
PetscErrorCode PetscRegressorLinearGetKSP(PetscRegressor regressor, KSP *ksp)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(ksp, 2);
  /* Analogous to how SNESGetKSP() operates, this routine should create the KSP if it doesn't exist.
   * TODO: Follow what SNESGetKSP() does when setting this up. */
  if (!linear->ksp) {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)regressor), &linear->ksp));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)linear->ksp, (PetscObject)regressor, 1));
    PetscCall(PetscObjectSetOptions((PetscObject)linear->ksp, ((PetscObject)regressor)->options));
  }
  *ksp = linear->ksp;
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorLinearGetCoefficients - Get a vector of the fitted coefficients from a linear regression model

   Not Collective

   Input Parameter:
.  regressor - the regressor context

   Output Parameter:
.  coefficients - the vector of the coefficients

   Level: beginner

.seealso: PetscRegressorLinearGetIntercept(), PETSCREGRESSORLINEAR
@*/
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetCoefficients(PetscRegressor regressor, Vec *coefficients)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(coefficients, 2);
  *coefficients = linear->coefficients;
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorLinearGetIntercept - Get the intercept from a linear regression model

   Not Collective

   Input Parameter:
.  regressor - the regressor context

   Output Parameter
.  intercept - the intercept

.seealso: PetscRegressorLinearGetCoefficients(), PETSCREGRESSORLINEAR
@*/
PETSC_EXTERN PetscErrorCode PetscRegressorLinearGetIntercept(PetscRegressor regressor, PetscScalar *intercept)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(intercept, 2);
  *intercept = linear->intercept;
  PetscFunctionReturn(0);
}

/*@C
   PetscRegressorLinearSetType - Sets the type of linear regression to be performed

   Logically Collective

   Input Parameters:
+  regressor - the `PetscRegressor` context (should be of type `PETSCREGRESSORLINEAR`)
-  type - a known linear regression method

   Options Database Key:
.  -regressor_linear_type - Sets the linear regression method; use -help for a list of available methods
   (for instance "-regressor_linear_type ols" or "-regressor_linear_type lasso")

   Level: intermediate

.seealso: `PetscRegressorLinearGetType()`, `PetscRegressorLinearType`, `PetscRegressorSetType()`
@*/
PetscErrorCode PetscRegressorLinearSetType(PetscRegressor regressor, PetscRegressorLinearType type)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)regressor, PETSCREGRESSORLINEAR, &match));
  PetscCheck(match, PetscObjectComm((PetscObject)regressor), PETSC_ERR_ARG_WRONG, "regressor is not of type PETSCREGRESSORLINEAR");
  PetscCall(PetscFree(linear->type));
  PetscCall(PetscStrallocpy(type, (char**)&linear->type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscRegressorFit_Linear(PetscRegressor regressor)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;
  KSP                    ksp;
  PetscScalar            target_mean, *column_means_global, *column_means_local, column_means_dot_coefficients;
  Vec                    column_means;
  PetscInt               m, N, istart, i;

  PetscFunctionBegin;
  if (linear->use_ksp && !linear->ksp) PetscCall(PetscRegressorLinearGetKSP(regressor, &linear->ksp));
  ksp = linear->ksp;

  /* Solve the least-squares problem (previously set up in PetscRegressorSetUp_Linear()) without finding the intercept. */
  if (linear->use_ksp) {
    PetscCall(KSPSolve(ksp, linear->rhs, linear->coefficients));
  } else {
    PetscCall(TaoSolve(regressor->tao));
  }

  /* Calculate the intercept. */
  if (linear->fit_intercept) {
    PetscCall(MatGetSize(regressor->training, NULL, &N));
    PetscCall(PetscMalloc1(N, &column_means_global));
    PetscCall(VecMean(regressor->target, &target_mean));
    /* We need the means of all columns of regressor->training, placed into a Vec compatible with linear->coefficients.
     * Note the potential scalability issue: MatGetColumnMeans() computes means of ALL colummns. */
    PetscCall(MatGetColumnMeans(regressor->training, column_means_global));
    /* TODO: Calculation of the Vec and matrix column means should probably go into the SetUp phase, and also be placed
     *       into a routine that is callable from outside of PetscRegressorFit_Linear(), because we'll want to do the same
     *       thing for other models, such as ridge and LASSO regression, and should avoid code duplication.
     *       What we are calling 'target_mean' and 'column_means' should be stashed in the base linear regressor struct,
     *       and perhaps renamed to make it clear they are offsets that should be applied (though the current naming
     *       makes sense since it makes it clear where these come from.) */
    PetscCall(VecDuplicate(linear->coefficients, &column_means));
    PetscCall(VecGetLocalSize(column_means, &m));
    PetscCall(VecGetOwnershipRange(column_means, &istart, NULL));
    PetscCall(VecGetArrayWrite(column_means, &column_means_local));
    for (i = 0; i < m; i++) { column_means_local[i] = column_means_global[istart + i]; }
    PetscCall(VecRestoreArrayWrite(column_means, &column_means_local));
    PetscCall(VecDot(column_means, linear->coefficients, &column_means_dot_coefficients));
    PetscCall(VecDestroy(&column_means));
    linear->intercept = target_mean - column_means_dot_coefficients;
  } else {
    linear->intercept = 0.0;
  }

  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscRegressorPredict_Linear(PetscRegressor regressor, Mat X, Vec y)
{
  PETSCREGRESSOR_LINEAR *linear = (PETSCREGRESSOR_LINEAR *)regressor->data;

  PetscFunctionBegin;
  PetscCall(MatMult(X, linear->coefficients, y));
  PetscCall(VecShift(y, linear->intercept));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PETSCREGRESSORLINEAR - Linear regression model (ordinary least squares or regularized variants)

   Options Database:
+  -regressor_linear_fit_intercept - Calculate the intercept for the linear model
-  -regressor_linear_use_ksp - Use KSP instead of TAO for linear model fitting (non-regularized variants only)

   Notes:
   This is the default regressor in PetscRegressor

   Level: beginner

.seealso: PetscRegressorCreate(), PetscRegressor, PetscRegressorSetType()
M*/
PETSC_EXTERN PetscErrorCode PetscRegressorCreate_Linear(PetscRegressor regressor)
{
  PETSCREGRESSOR_LINEAR *linear;

  PetscFunctionBegin;
  PetscCall(PetscNew(&linear));
  regressor->data = (void *)linear;

  regressor->ops->setup          = PetscRegressorSetUp_Linear;
  regressor->ops->reset          = PetscRegressorReset_Linear;
  regressor->ops->destroy        = PetscRegressorDestroy_Linear;
  regressor->ops->setfromoptions = PetscRegressorSetFromOptions_Linear;
  regressor->ops->view           = PetscRegressorView_Linear;
  regressor->ops->fit            = PetscRegressorFit_Linear;
  regressor->ops->predict        = PetscRegressorPredict_Linear;

  linear->intercept     = 0.0;
  linear->fit_intercept = PETSC_TRUE;  /* Default to calculating the intercept. */
  linear->use_ksp       = PETSC_FALSE; /* Do not default to using KSP for solving the model-fitting problem (use TAO instead). */
  PetscCall(PetscStrallocpy(PETSCREGRESSORLINEARDEFAULT, (char**)&linear->type));
    /* Above, manually set the default linear regressor type.
       We don't use PetscRegressorLinearSetType() here, because that expects the SetUp event to already have happened. */
  PetscFunctionReturn(0);
}
