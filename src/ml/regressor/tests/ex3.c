static char help[] = "Tests some linear PetscRegressor types with different regularizers.\n\n";

#include <petscregressor.h>

int main(int argc, char **args)
{
  PetscRegressor     regressor;
  PetscMPIInt        rank;
  PetscInt           i, N = 10;
  Mat                X;
  Vec                y, y_predicted, coefficients;
  PetscScalar        intercept, mean;
  PetscBool          flg_string, flg_ascii, flg_view_sol, match;
  PetscRegressorType check_type;

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Options for PetscRegressor ex3:", "");
  PetscCall(PetscOptionsInt("-N", "Dimension of the N x N data matrix", NULL, N, &N, NULL));
  flg_string   = PETSC_FALSE;
  flg_ascii    = PETSC_FALSE;
  flg_view_sol = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_string_viewer", &flg_string, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_ascii_viewer", &flg_ascii, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_sols", &flg_view_sol, NULL));
  PetscOptionsEnd();

  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecDuplicate(y, &y_predicted));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
  PetscCall(MatSetSizes(X, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(X));
  PetscCall(MatSetUp(X));

  /* Set up a training data matrix that is the identity.
   * We do this because this gives us a special case in which we can analytically determine what the regression
   * coefficients should be for ordinary least squares, LASSO (L1 regularized), and ridge (L2 regularized) regression.
   * See details in section 6.2 of James et al.'s An Introduction to Statistical Learning (ISLR), in the subsection
   * titled "A Simple Special Case for Ridge Regression and the Lasso".
   * Note that the coefficients we generate with ridge regression (-regressor_linear_type ridge -regressor_regularizer_weight <lambda>, or, equivalently,
   * -tao_brgn_regularization_type l2pure -tao_brgn_regularizer_weight <lambda>) match those of the ISLR formula exactly.
   * For LASSO it does not match the ISLR formula: where they use lambda/2, we need to use lambda.
   * It also doesn't match what Scikit-learn does; in that case their lambda is 1/n_samples of our lambda. Apparently everyone is scaling
   * their loss function by a different value, hence the need to change what "lambda" is. But it's clear that ISLR, Scikit-learn, and we
   * are basically doing the same thing otherwise. */
  if (!rank) {
    for (i = 0; i < N; i++) {
      PetscCall(VecSetValue(y, i, (PetscScalar)i, INSERT_VALUES));
      PetscCall(MatSetValue(X, i, i, 1.0, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  PetscCall(MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY));
  /* Center the target vector we will train with. */
  PetscCall(VecMean(y, &mean));
  PetscCall(VecShift(y, -1.0 * mean));

  PetscCall(PetscRegressorCreate(PETSC_COMM_WORLD, &regressor));
  PetscCall(PetscRegressorSetType(regressor, PETSCREGRESSORLINEAR));
  PetscCall(PetscRegressorLinearSetType(regressor, REGRESSOR_LINEAR_OLS));
  PetscCall(PetscRegressorLinearSetFitIntercept(regressor, PETSC_FALSE));
  PetscCall(PetscRegressorSetFromOptions(regressor));
  PetscCall(PetscRegressorFit(regressor, X, y));
  PetscCall(PetscRegressorPredict(regressor, X, y_predicted));
  PetscCall(PetscRegressorLinearGetIntercept(regressor, &intercept));
  PetscCall(PetscRegressorLinearGetCoefficients(regressor, &coefficients));

  if (flg_view_sol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Training target vector is\n"));
    PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Predicted values are\n"));
    PetscCall(VecView(y_predicted, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coefficients are\n"));
    PetscCall(VecView(coefficients, PETSC_VIEWER_STDOUT_WORLD));
  }

  if (flg_string) {
    PetscViewer stringviewer;
    char        string[512];
    const char *outstring;

    PetscCall(PetscViewerStringOpen(PETSC_COMM_WORLD, string, sizeof(string), &stringviewer));
    PetscCall(PetscRegressorView(regressor, stringviewer));
    PetscCall(PetscViewerStringGetStringRead(stringviewer, &outstring, NULL));
    PetscCheck((char *)outstring == (char *)string, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "String returned from viewer does not equal original string");
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Output from string viewer:%s\n", outstring));
    PetscCall(PetscViewerDestroy(&stringviewer));
  } else if (flg_ascii) PetscCall(PetscRegressorView(regressor, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscRegressorGetType(regressor, &check_type));
  PetscCall(PetscStrcmp(check_type, PETSCREGRESSORLINEAR, &match));
  PetscCheck(match, PETSC_COMM_WORLD, PETSC_ERR_ARG_NOTSAMETYPE, "Regressor type is not Linear");

  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&y_predicted));
  PetscCall(PetscRegressorDestroy(&regressor));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: asciiview
      args: -test_ascii_viewer

   test:
       suffix: stringview
       args: -test_string_viewer

   test:
      suffix: ksp_intercept
      args: -regressor_linear_use_ksp -regressor_linear_fit_intercept -regressor_view

   test:
      suffix: ksp_no_intercept
      args: -regressor_linear_use_ksp -regressor_view

   test:
      suffix: lasso_1
      nsize: 1
      args: -regressor_type linear -regressor_linear_type lasso -regressor_regularizer_weight 2 -regressor_linear_fit_intercept -view_sols

   test:
      suffix: lasso_2
      nsize: 2
      args: -regressor_type linear -regressor_linear_type lasso -regressor_regularizer_weight 2 -regressor_linear_fit_intercept -view_sols

   test:
      suffix: ridge_1
      nsize: 1
      args: -regressor_type linear -regressor_linear_type ridge -regressor_regularizer_weight 2 -regressor_linear_fit_intercept -view_sols

   test:
      suffix: ridge_2
      nsize: 2
      args: -regressor_type linear -regressor_linear_type ridge -regressor_regularizer_weight 2 -regressor_linear_fit_intercept -view_sols

   test:
      suffix: ols_1
      nsize: 1
      args: -regressor_type linear -regressor_linear_type ols -regressor_linear_fit_intercept -view_sols

   test:
      suffix: ols_2
      nsize: 2
      args: -regressor_type linear -regressor_linear_type ols -regressor_linear_fit_intercept -view_sols

TEST*/
