#include <petsctao.h>
#include <petsctaoregularizer.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates various ways to use TaoRegularizer. \n";

typedef struct {
  PetscInt  n;       /* dimension */
  PetscInt  problem; /* Types of problems to solve. */
  PetscReal stepsize;
  PetscReal mu1; /* Parameter for soft-threshold */
  Mat       A;
  Vec       b, workvec;
} AppCtx;

/* Objective
 *
 * f(x) = 0.5 x.T A x - b.T x */
PetscErrorCode UserObj(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecScale(user->workvec, 0.5));
  PetscCall(VecAXPY(user->workvec, -1., user->b));
  PetscCall(VecTDot(user->workvec, X, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Gradient: grad f = Ax - b */
PetscErrorCode UserGrad(Tao tao, Vec X, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, G));
  PetscCall(VecAXPY(G, -1., user->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 x.T A x - b.T x
 * grad f = A x - b                         */
PetscErrorCode UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecWAXPY(G, -1., user->b, user->workvec));
  PetscCall(VecTDot(user->workvec, X, f));
  *f *= 0.5;

  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Full Obj and Grad for iterative refinement.
 *
 * f(x) = 0.5 x.T A x - b.T x + 0.5 \|x\|_2^2
 * grad f = A x + x  - b                        */
PetscErrorCode FullUserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp;

  PetscFunctionBegin;
  /* Ax + x */
  PetscCall(MatMult(user->A, X, G));
  PetscCall(VecAXPY(G, 1., X));
  /* 0.5 * x^T (Ax + x) */
  PetscCall(VecTDot(G, X, f));
  *f *= 0.5;
  /* Full grad f(x) */
  PetscCall(VecAXPY(G, -1., user->b));
  /* b^T x */
  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* L2 Metric. \|X\|_2^2 */
PetscErrorCode L2_ObjGrad(TaoRegularizer reg, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscReal temp, step;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetScale(reg, &step));
  PetscCall(VecCopy(X, G));
  PetscCall(VecScale(G, step * 2));
  PetscCall(VecNorm(X, NORM_2, &temp));
  temp = PetscPowReal(temp, 2);
  *f   = step * temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao            tao, tao_full;
  TaoRegularizer reg;
  Vec            x, x_full;
  Mat            temp_mat;
  PetscMPIInt    size;
  AppCtx         user;
  PetscBool      flg;
  PetscRandom    rctx;
  PetscReal      vec_dist;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.problem  = 0;
  user.n        = 10;
  user.stepsize = 1;
  user.mu1      = 1;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  /* Types of problems to solve */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-problem", &user.problem, &flg));
  /* If stepsize ==1, default case. Else, adaptive version */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user.stepsize, &flg));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x_full));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.workvec));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetInterval(rctx, 10, 100));
  PetscCall(PetscRandomSetFromOptions(rctx));
  //  PetscCall(PetscRandomSetInterval(rctx, -10, 10));
  /* y : Random vec, from -10 to 10 */
  /* x: zero vec */
  PetscCall(VecZeroEntries(x));
  PetscCall(PetscRandomDestroy(&rctx));

  /* A,b data */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, user.n, user.n, NULL, &temp_mat));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  //  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(MatSetRandom(temp_mat, rctx));
  PetscCall(MatAssemblyBegin(temp_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(temp_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatTransposeMatMult(temp_mat, temp_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user.A));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.b));
  PetscCall(VecSetRandom(user.b, rctx));
  PetscCall(MatDestroy(&temp_mat));
  PetscCall(PetscRandomDestroy(&rctx));

  /* f(x) = 0.5 x^T A x - b^T x */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao_full));
  PetscCall(TaoSetType(tao, TAOCG));
  PetscCall(TaoSetType(tao_full, TAOCG));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetSolution(tao_full, x_full));
  PetscCall(TaoSetOptionsPrefix(tao, "reg_"));
  PetscCall(TaoSetOptionsPrefix(tao_full, "normal_"));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetFromOptions(tao_full));

  /* Create Regularizer, g(x) = 0.5 \|x\|_2^2
   * Cases:
   * 0: f(x) + L2 Metric with y vector
   * 1: f(x) + built-in L2 Metric  */
  PetscCall(TaoRegularizerCreate(PetscObjectComm((PetscObject)tao), &reg));
  if (user.problem == 0 || user.problem == 1) {
    PetscCall(TaoRegularizerSetType(reg, TAOREGULARIZERL2));
  } else if (user.problem == 2) {
    PetscCall(TaoRegularizerSetObjectiveAndGradient(reg, L2_ObjGrad, (void *)&user));
  }
  /* SetScale needs to be called after SetType */
  PetscCall(TaoRegularizerSetScale(reg, 0.5));

  if (user.problem == 0 || user.problem == 2) {
    PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *)&user));
  } else if (user.problem == 1) {
    PetscCall(TaoSetObjective(tao, UserObj, (void *)&user));
    PetscCall(TaoSetGradient(tao, NULL, UserGrad, (void *)&user));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem type.");
  PetscCall(TaoSetObjectiveAndGradient(tao_full, NULL, FullUserObjGrad, (void *)&user));

  /* Solve full version */
  /* Solve Regularizer version */
  PetscCall(TaoSetRegularizer(tao, reg));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoSolve(tao_full));

  /* Testing Regularizer version vs Full version */
  PetscCall(VecAXPY(x, -1., x_full));
  PetscCall(VecNorm(x, NORM_2, &vec_dist));
  if (vec_dist < 1.e-11) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Full TaoSolve: < 1.e-11\n"));
  } else if (vec_dist < 1.e-6) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Ful TaoSolve: < 1.e-6\n"));
  } else {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Full TaoSolve: %e\n", (double)vec_dist));
  }

  PetscCall(TaoRegularizerDestroy(&reg));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoDestroy(&tao_full));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x_full));
  PetscCall(VecDestroy(&user.b));
  PetscCall(VecDestroy(&user.workvec));
  PetscCall(MatDestroy(&user.A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 0
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 0
      requires: !single

   test:
      suffix: 1
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 1
      requires: !single

   test:
      suffix: 2
      args: -reg_tao_gatol 1.e-4 -normal_tao_gatol 1.e-4 -problem 2
      requires: !single

TEST*/
