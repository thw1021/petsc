static char help[] = "Tests basic creation and destruction of PetscRegressor objects.\n\n";

#include <petscregressor.h>

int main(int argc, char **args)
{
  PetscRegressor regressor;
  PetscMPIInt    rank;
  Mat            X;
  Vec            y, y_predicted, coefficients;
  PetscScalar    intercept;
  /* y_array[] and X_array[] are mean-centered; in ex2.c they are not! */
  PetscScalar y_array[5]  = {0.20000, -0.30000, -0.80000, -0.30000, 1.20000};
  PetscScalar X_array[10] = {-1.00000, 0.50000, -0.50000, -0.25000, 0.00000, -0.50000, 0.50000, -0.25000, 1.00000, 0.50000};
  PetscInt    rows_ix[5]  = {0, 1, 2, 3, 4};
  PetscInt    cols_ix[2]  = {0, 1};

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, 5));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecDuplicate(y, &y_predicted));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
  PetscCall(MatSetSizes(X, PETSC_DECIDE, PETSC_DECIDE, 5, 2));
  PetscCall(MatSetFromOptions(X));
  PetscCall(MatSetUp(X));

  if (!rank) {
    PetscCall(VecSetValues(y, 5, rows_ix, y_array, INSERT_VALUES));
    PetscCall(MatSetValues(X, 5, rows_ix, 2, cols_ix, X_array, ADD_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  PetscCall(MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscRegressorCreate(PETSC_COMM_WORLD, &regressor));
  PetscCall(PetscRegressorSetType(regressor, PETSCREGRESSORLINEAR));
  PetscRegressorSetFromOptions(regressor);
  PetscCall(PetscRegressorFit(regressor, X, y));
  PetscCall(PetscRegressorPredict(regressor, X, y_predicted));
  PetscCall(PetscRegressorLinearGetIntercept(regressor, &intercept));
  PetscCall(PetscRegressorLinearGetCoefficients(regressor, &coefficients));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Intercept is %lf\n", intercept));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coefficients are\n"));
  PetscCall(VecView(coefficients, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Predicted values are\n"));
  PetscCall(VecView(y_predicted, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscRegressorDestroy(&regressor));

  PetscCall(PetscFinalize());
  return 0;
}
