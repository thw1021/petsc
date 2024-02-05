static char help[] = "Solves a  DAE-constrained optimization problem with with TaoTSSetObjectiveAndGradients(). See Users Guide chapter on Tao\n";

// solves the same problem as src/ts/tutorials/ex20opt_p.c

#include <petsctao.h>
#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal p;     /* parameter in ODE */
  PetscReal ob[2]; /* observation used by the objective/cost function */
};

static PetscErrorCode evaluateRHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  User               user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));
  f[0] = u[1];
  f[1] = user->p * ((1. - u[0] * u[0]) * u[1] - u[0]);
  PetscCall(VecRestoreArray(F, &f));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode evaluateRHSJacobian(TS ts, PetscReal t, Vec U, Mat A, Mat B, void *ctx)
{
  User               user     = (User)ctx;
  PetscReal          p        = user->p;
  PetscInt           rowcol[] = {0, 1};
  PetscScalar        J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  J[0][0] = 0;
  J[1][0] = -p * (2.0 * u[1] * u[0] + 1.);
  J[0][1] = 1.0;
  J[1][1] = p * (1.0 - u[0] * u[0]);
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatSetValues(A, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (B && A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode evaluateRHSJacobianP(TS ts, PetscReal t, Vec U, Mat A, void *ctx)
{
  PetscInt           row[] = {0, 1}, col[] = {0};
  PetscScalar        J[2][1];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  J[0][0] = 0;
  J[1][0] = (1. - u[0] * u[0]) * u[1] - u[0];
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatSetValues(A, 2, row, 1, col, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode inputParameters(TS ts, Vec P, void *ctx)
{
  User               user = (User)ctx;
  const PetscScalar *p;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(P, &p));
  user->p = p[0];
  PetscCall(VecRestoreArrayRead(P, &p));
  PetscCall(TSSetTimeStep(ts, PetscRealConstant(0.001))); // temporary until TSTao is fixed to reset the initial time-step
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode evaluateInitialCondition(TS ts, Vec U)
{
  User         user;
  PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &user));
  PetscCall(VecGetArray(U, &u));
  u[0] = 2.0; // initial conditions of ODE
  u[1] = -2.0 / 3.0 + 10.0 / (81.0 * user->p) - 292.0 / (2187.0 * user->p * user->p);
  PetscCall(VecRestoreArray(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode evaluateICJacobianP(TS ts, Mat Jp, void *ctx)
{
  User        user = (User)ctx;
  PetscScalar J[2];
  PetscInt    rowcol[] = {0, 1};

  PetscFunctionBeginUser;
  J[0] = 0;
  J[1] = -10.0 / (81.0 * user->p * user->p) + 2.0 * 292.0 / (2187.0 * user->p * user->p * user->p);
  PetscCall(MatSetValues(Jp, 2, rowcol, 1, rowcol, &J[0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Jp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode evaluateObjectiveAndGradients(Tao tao, TS ts, Vec U, Vec P, PetscReal *obj, Vec Lambda, Vec Mu, void *ctx)
{
  User               user = (User)ctx;
  const PetscScalar *u;
  PetscScalar       *lambda, *mu;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));

  /* objective function obj */
  *obj = (u[0] - user->ob[0]) * (u[0] - user->ob[0]) + (u[1] - user->ob[1]) * (u[1] - user->ob[1]);

  /* gradient of objective w.r.t. U */
  PetscCall(VecGetArray(Lambda, &lambda));
  lambda[0] = 2. * (u[0] - user->ob[0]);
  lambda[1] = 2. * (u[1] - user->ob[1]);
  PetscCall(VecRestoreArray(Lambda, &lambda));
  PetscCall(VecRestoreArrayRead(U, &u));

  /* gradient of objective with respect to P is 0 */
  PetscCall(VecGetArray(Mu, &mu));
  mu[0] = 0.0;
  PetscCall(VecRestoreArray(Mu, &mu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscScalar   *u, *p;
  PetscMPIInt    size;
  struct _n_User user;
  Tao            tao;
  Vec            U, P;
  Mat            J, Jp, icJp;
  TS             ts;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBQNLS));

  /* Create necessary matrix and vectors */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));
  PetscCall(MatCreateVecs(J, &U, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &Jp));
  PetscCall(MatSetSizes(Jp, PETSC_DECIDE, PETSC_DECIDE, 2, 1));
  PetscCall(MatSetFromOptions(Jp));
  PetscCall(MatSetUp(Jp));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &icJp));
  PetscCall(MatSetSizes(icJp, PETSC_DECIDE, PETSC_DECIDE, 2, 1));
  PetscCall(MatSetFromOptions(icJp));
  PetscCall(MatSetUp(icJp));

  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetEquationType(ts, TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  PetscCall(TSSetRHSFunction(ts, NULL, evaluateRHSFunction, &user));
  PetscCall(TSSetRHSJacobian(ts, J, J, evaluateRHSJacobian, &user));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetRHSJacobianP(ts, Jp, evaluateRHSJacobianP, &user));
  PetscCall(TSSetICJacobianP(ts, icJp, evaluateICJacobianP, &user));
  PetscCall(TSSetComputeInitialCondition(ts, evaluateInitialCondition));
  PetscCall(TSSetMaxTime(ts, .5));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, PetscRealConstant(0.001)));
  PetscCall(TSSetInputParameters(ts, inputParameters, &user));
  PetscCall(TSSetApplicationContext(ts, &user));
  PetscCall(TSSetFromOptions(ts));

  /* Solve the ODE for the give parameter to get the final solution that will go into the cost/objective function */
  user.p = PetscRealConstant(1.0e3);
  PetscCall(TSComputeInitialCondition(ts, U));
  PetscCall(TSSolve(ts, U));
  PetscCall(VecGetArray(U, &u));
  user.ob[0] = u[0];
  user.ob[1] = u[1];
  PetscCall(VecRestoreArray(U, &u));

  /* Set initial guess for the Tao parameter */
  PetscCall(MatCreateVecs(Jp, &P, NULL));
  PetscCall(VecGetArray(P, &p));
  p[0] = PetscRealConstant(1.2);
  PetscCall(VecRestoreArray(P, &p));
  PetscCall(TaoSetSolution(tao, P));

  /* Set routine for objective and gradient evaluation */
  PetscCall(TaoTSSetObjectiveAndGradients(tao, ts, evaluateObjectiveAndGradients, &user));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&U));
  PetscCall(MatDestroy(&Jp));
  PetscCall(MatDestroy(&icJp));
  PetscCall(TSDestroy(&ts));
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
