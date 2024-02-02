static char help[] = "Solves a simple constrained optimization problem with SNES and Tao. See Users Guide chapter on SNES\n";

#include <petsctao.h>
#include <petscsnes.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal p, q;
  SNES      snes;
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

PetscErrorCode evaluateObjectiveAndGradient(Tao tao, Vec P, PetscReal *f, Vec G, void *ctx)
{
  User                user = (User)ctx;
  SNES                snes = user->snes;
  const PetscScalar  *p, *u;
  PetscScalar        *mu, *lambda;
  Vec                 U, *Mu, *Lambda;
  SNESConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(P, &p));
  user->p = p[0];
  user->q = p[1];

  PetscCall(SNESSolve(snes, NULL, NULL));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  PetscCheck(reason > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to solve nonlinear system");
  PetscCall(SNESGetSolution(snes, &U));
  PetscCall(VecGetArrayRead(U, &u));

  // evaluate objective function
  *f = u[0] * u[0] + u[1] * u[1] + p[0] * p[0] + p[1] * p[1];

  // evaluate gradient
  PetscCall(SNESGetCostGradients(snes, NULL, &Lambda, &Mu));
  PetscCall(VecGetArray(Mu[0], &mu));
  PetscCall(VecGetArray(Lambda[0], &lambda));
  mu[0]     = 2 * p[0];
  mu[1]     = 2 * p[1];
  lambda[0] = 2 * u[0];
  lambda[1] = 2 * u[1];
  PetscCall(VecRestoreArray(Mu[0], &mu));
  PetscCall(VecRestoreArray(Lambda[0], &lambda));
  //  VecView(Lambda[0],0);
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(P, &p));
  PetscCall(SNESAdjointSolve(snes));
  PetscCall(VecCopy(Mu[0], G));
  //VecView(G,0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscMPIInt    size;
  struct _n_User user;
  Tao            tao;
  Vec            U, Mu[1], Lambda[1], P;
  Mat            J, Jp;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBQNLS));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, evaluateObjectiveAndGradient, &user));
  PetscCall(TaoSetFromOptions(tao));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.p = PetscRealConstant(0.0);
  user.q = PetscRealConstant(0.0);
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-p", &user.p, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-q", &user.q, NULL));

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
  PetscCall(MatCreateVecs(Jp, &Lambda[0], &Mu[0]));
  PetscCall(MatCreateVecs(Jp, NULL, &P));

  /* Create nonlinear solver */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &user.snes));
  PetscCall(SNESSetFunction(user.snes, NULL, evaluateFunction, &user));
  PetscCall(SNESSetJacobian(user.snes, J, J, evaluateJacobian, &user));
  PetscCall(SNESSetJacobianP(user.snes, Jp, evaluateJacobianP, &user));
  PetscCall(SNESSetCostGradients(user.snes, 1, Lambda, Mu));
  PetscCall(VecSet(U, .5));
  PetscCall(SNESSetSolution(user.snes, U));
  PetscCall(SNESSetFromOptions(user.snes));

  PetscCall(VecSet(P, .5));
  PetscCall(TaoSetSolution(tao, P));
  PetscCall(TaoSolve(tao));

  PetscCall(TaoDestroy(&tao));
  PetscCall(SNESDestroy(&user.snes));
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&Jp));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&Lambda[0]));
  PetscCall(VecDestroy(&Mu[0]));
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
