static char help[] = "Solves a simple constrained optimization problem with TaoSNESSetObjectiveAndGradients(). See Users Guide chapter on Tao\n";

// solves the same problem as src/snes/tutorials/exadj.c

#include <petsctao.h>
#include <petscsnes.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal p, q;
};

static PetscErrorCode evaluateFunction(SNES snes, Vec U, Vec F, void *ctx)
{
  User               user = (User)ctx;
  PetscReal          p = user->p, q = user->q;
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));
  f[0] = u[0] * u[0] + u[0] * u[1] + p * p * p - 3;
  f[1] = u[0] * u[1] + u[1] * u[1] + q * q * q - 6;
  PetscCall(VecRestoreArray(F, &f));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode evaluateJacobian(SNES snes, Vec U, Mat A, Mat B, void *ctx)
{
  PetscInt           rowcol[] = {0, 1};
  PetscScalar        J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  J[0][0] = 2 * u[0] + u[1];
  J[1][0] = u[1];
  J[0][1] = u[0];
  J[1][1] = u[0] + 2 * u[1];
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatSetValues(A, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode evaluateJacobianP(SNES snes, Vec U, Mat Jp, void *ctx)
{
  User               user = (User)ctx;
  PetscReal          p = user->p, q = user->q;
  PetscInt           rowcol[] = {0, 1};
  PetscScalar        J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  J[0][0] = 3 * p * p;
  J[1][0] = 0;
  J[0][1] = 0;
  J[1][1] = 3 * q * q;
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatSetValues(Jp, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Jp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode evaluateObjectiveAndGradients(Tao tao, SNES snes, Vec U, Vec P, PetscReal *f, Vec Lambda, Vec Mu, void *ctx)
{
  const PetscScalar *p, *u;
  PetscScalar       *mu, *lambda;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(P, &p));
  PetscCall(VecGetArrayRead(U, &u));

  // evaluate objective function
  *f = u[0] * u[0] + u[1] * u[1] + p[0] * p[0] + p[1] * p[1];

  // evaluate gradients
  PetscCall(VecGetArray(Mu, &mu));
  PetscCall(VecGetArray(Lambda, &lambda));
  mu[0]     = 2 * p[0];
  mu[1]     = 2 * p[1];
  lambda[0] = 2 * u[0];
  lambda[1] = 2 * u[1];
  PetscCall(VecRestoreArray(Lambda, &lambda));
  PetscCall(VecRestoreArray(Mu, &mu));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(P, &p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode inputParameters(SNES snes, Vec P, void *ctx)
{
  User               user = (User)ctx;
  const PetscScalar *p;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(P, &p));
  user->p = p[0];
  user->q = p[1];
  PetscCall(VecRestoreArrayRead(P, &p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscMPIInt    size;
  struct _n_User user;
  Tao            tao;
  SNES           snes;
  Vec            U, P;
  Mat            J, Jp;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBQNLS));
  PetscCall(TaoSetFromOptions(tao));

  /* Create necessary matrix and vectors */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));
  PetscCall(MatCreateVecs(J, &U, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &Jp));
  PetscCall(MatSetSizes(Jp, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(Jp));
  PetscCall(MatSetUp(Jp));
  PetscCall(MatCreateVecs(Jp, NULL, &P));

  /* Create nonlinear solver */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, NULL, evaluateFunction, &user));
  PetscCall(SNESSetJacobian(snes, J, J, evaluateJacobian, &user));
  PetscCall(SNESSetJacobianP(snes, Jp, evaluateJacobianP, &user));
  PetscCall(VecSet(U, .5));
  PetscCall(SNESSetSolution(snes, U));
  PetscCall(SNESSetInputParameters(snes, inputParameters, &user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(TaoSNESSetObjectiveAndGradients(tao, snes, evaluateObjectiveAndGradients, &user));
  PetscCall(VecSet(P, .5));
  PetscCall(TaoSetSolution(tao, P));
  PetscCall(TaoSolve(tao));

  PetscCall(TaoDestroy(&tao));
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&Jp));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&P));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
    build:
      requires: !complex !single

    test:
      args: -tao_monitor

TEST*/
