static char help[] = "Tests PETSCREGRESSORNLLS by fitting an exponential model y = p0 * exp(p1 * x) + p2.\n\n";

#include <petscregressor.h>

static PetscErrorCode Model(PetscRegressor regressor, Mat X, Vec p, Vec f, void *ctx)
{
  const PetscScalar *p_array, *x_array;
  PetscScalar       *f_array;
  PetscInt           m, i;

  PetscFunctionBeginUser;
  PetscCall(MatDenseGetArrayRead(X, &x_array));
  PetscCall(VecGetArrayRead(p, &p_array));
  PetscCall(VecGetArrayWrite(f, &f_array));
  PetscCall(VecGetLocalSize(f, &m));
  for (i = 0; i < m; i++) f_array[i] = p_array[0] * PetscExpScalar(p_array[1] * x_array[i]) + p_array[2];
  PetscCall(MatDenseRestoreArrayRead(X, &x_array));
  PetscCall(VecRestoreArrayRead(p, &p_array));
  PetscCall(VecRestoreArrayWrite(f, &f_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Jacobian(PetscRegressor regressor, Mat X, Vec p, Mat J, Mat Jpre, void *ctx)
{
  const PetscScalar *p_array, *x_array;
  PetscInt           m, i, rstart;

  PetscFunctionBeginUser;
  PetscCall(MatDenseGetArrayRead(X, &x_array));
  PetscCall(VecGetArrayRead(p, &p_array));
  PetscCall(MatGetLocalSize(J, &m, NULL));
  PetscCall(MatGetOwnershipRange(J, &rstart, NULL));
  for (i = 0; i < m; i++) {
    PetscInt    row     = rstart + i;
    PetscInt    cols[3] = {0, 1, 2};
    PetscScalar e       = PetscExpScalar(p_array[1] * x_array[i]);
    PetscScalar vals[3] = {e, p_array[0] * x_array[i] * e, 1.0};

    PetscCall(MatSetValues(J, 1, &row, 3, cols, vals, INSERT_VALUES));
  }
  PetscCall(MatDenseRestoreArrayRead(X, &x_array));
  PetscCall(VecRestoreArrayRead(p, &p_array));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  if (Jpre != J) {
    PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscRegressor regressor;
  PetscMPIInt    rank;
  Mat            X;
  Vec            y, y_predicted, p0, p_fit;
  PetscInt       M       = 20;
  PetscInt       N       = 3;
  PetscScalar    p_true[3] = {2.5, -0.3, 0.5};
  PetscScalar    p0_array[3] = {1.0, 0.0, 0.0};
  PetscBool      use_analytic_jac = PETSC_TRUE;
  PetscInt       i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_analytic_jacobian", &use_analytic_jac, NULL));

  /* Build the M x 1 data matrix X (rows hold x_i in [0, 4]) and target y = f(x; p_true). */
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, 1, NULL, &X));
  PetscCall(MatCreateVecs(X, NULL, &y));
  PetscCall(VecDuplicate(y, &y_predicted));

  if (!rank) {
    PetscScalar *x_array, *y_array;

    PetscCall(MatDenseGetArrayWrite(X, &x_array));
    PetscCall(VecGetArrayWrite(y, &y_array));
    for (i = 0; i < M; i++) {
      x_array[i] = 4.0 * i / (M - 1);
      y_array[i] = p_true[0] * PetscExpScalar(p_true[1] * x_array[i]) + p_true[2];
    }
    PetscCall(MatDenseRestoreArrayWrite(X, &x_array));
    PetscCall(VecRestoreArrayWrite(y, &y_array));
  }
  PetscCall(MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));

  /* Initial parameter guess. */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &p0));
  PetscCall(VecSetSizes(p0, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(p0));
  if (!rank) {
    PetscInt ix[3] = {0, 1, 2};

    PetscCall(VecSetValues(p0, 3, ix, p0_array, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(p0));
  PetscCall(VecAssemblyEnd(p0));

  PetscCall(PetscRegressorCreate(PETSC_COMM_WORLD, &regressor));
  PetscCall(PetscRegressorSetType(regressor, PETSCREGRESSORNLLS));
  PetscCall(PetscRegressorSetRegularizerWeight(regressor, 0.0));
  PetscCall(PetscRegressorSetFromOptions(regressor));
  PetscCall(PetscRegressorNLLSSetFunction(regressor, NULL, Model, NULL));
  PetscCall(PetscRegressorNLLSSetInitialParameters(regressor, p0));
  if (use_analytic_jac) PetscCall(PetscRegressorNLLSSetJacobian(regressor, NULL, NULL, Jacobian, NULL));
  PetscCall(PetscRegressorFit(regressor, X, y));
  PetscCall(PetscRegressorNLLSGetParameters(regressor, &p_fit));
  PetscCall(PetscRegressorPredict(regressor, X, y_predicted));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Fitted parameters:\n"));
  PetscCall(VecView(p_fit, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Predicted values:\n"));
  PetscCall(VecView(y_predicted, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&y_predicted));
  PetscCall(VecDestroy(&p0));
  PetscCall(PetscRegressorDestroy(&regressor));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

  testset:
    output_file: output/ex4.out

    test:
      suffix: analytic_jac

    test:
      suffix: fd_jac
      args: -use_analytic_jacobian false

TEST*/
