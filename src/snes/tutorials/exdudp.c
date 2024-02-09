static char help[] = "Computed du/dp with SNES. See Users Guide chapter on SNES\n";

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
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(F, &f));
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
  PetscCall(MatSetValues(A, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(U, &u));
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
  PetscCall(MatSetValues(Jp, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Jp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jp, MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscMPIInt    size;
  struct _n_User user;
  Vec            U;
  Mat            J, Jp, DuDp;
  SNES           snes;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.p = PetscRealConstant(0.5);
  user.q = PetscRealConstant(0.5);
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-p", &user.p, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-q", &user.q, NULL));

  /* Create necessary matrix and vectors */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));
  PetscCall(MatCreateVecs(J, &U, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &Jp));
  PetscCall(MatSetType(Jp, MATDENSE));
  PetscCall(MatSetSizes(Jp, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(Jp));
  PetscCall(MatSetUp(Jp));

  /* Create nonlinear solver */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, NULL, evaluateFunction, &user));
  PetscCall(SNESSetJacobian(snes, J, J, evaluateJacobian, &user));
  PetscCall(SNESSetJacobianP(snes, Jp, evaluateJacobianP, &user));
  PetscCall(SNESSetSolution(snes, U));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(VecSet(U, .5));

  PetscCall(SNESSolve(snes, NULL, U));
  PetscCall(SNESComputeDuDp(snes, MAT_INITIAL_MATRIX, &DuDp));
  PetscCall(MatView(DuDp, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&Jp));
  PetscCall(MatDestroy(&DuDp));
  PetscCall(VecDestroy(&U));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
    build:
      requires: !complex !single

    test:
      args: -ksp_monitor

TEST*/
