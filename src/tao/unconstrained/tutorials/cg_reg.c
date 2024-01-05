#include <petsctao.h>
#include <petscdm.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates various ways to use TaoPD. \n";

typedef struct {
  PetscBool sketch;
  PetscBool y_bool;
  PetscInt  n;       /* dimension */
  PetscInt  problem; /* Types of problems to solve. */
  PetscInt  g_type;  /* How to formulate regularizer */
  PetscReal stepsize;
  PetscReal mu1; /* Parameter for soft-threshold */
  Mat       A;
  Vec       b, workvec, y;
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

/* Objective
 *
 * f(x) = 0.5 x.T A x - b.T x */
PetscErrorCode UserObj_DM(DM dm, Vec X, PetscReal *f, void *ptr)
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
PetscErrorCode UserGrad_DM(DM dm, Vec X, Vec G, void *ptr)
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
PetscErrorCode UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
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
 * f(x) = 0.5 x.T A x - b.T x + step * \|x-y\|_2^2
 * grad f = A x - b + x - y                       */
PetscErrorCode FullUserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp, reg_val, stepsize;

  PetscFunctionBegin;
  stepsize = user->stepsize;
  /* workvec :  x-y */
  if (user->y_bool) {
    PetscCall(VecWAXPY(user->workvec, -1, user->y, X));
  } else {
    PetscCall(VecCopy(X, user->workvec));
  }
  PetscCall(VecTDot(user->workvec, user->workvec, &reg_val));
  reg_val *= stepsize;

  /* f = 0.5 x^T A x */
  PetscCall(MatMult(user->A, X, G));
  PetscCall(VecTDot(G, X, f));
  *f *= 0.5;

  PetscCall(VecAXPY(G, -1., user->b));
  PetscCall(VecAXPY(G, 1., user->workvec));
  /* f -= b^T x */
  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  /* Add reg term */
  *f += reg_val;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* L2 Metric. \|X-Y\|_2^2
 * This is to set Tao routines for TaoPD */
PetscErrorCode L2_ObjGrad_Tao(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(VecCopy(X, G));
  /* Note: PDScale part will be done internally */
  if (user->y_bool) {
    Vec y;
    DM  reg;
    PetscCall(TaoGetRegularizer(tao, &reg));
    PetscCall(DMTaoGetCentralVector(reg, &y));
    PetscCall(VecAXPY(G, -1., y));
  }
  PetscCall(VecTDot(G, G, f));
  PetscCall(VecScale(G, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Obj_Tao(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  if (user->y_bool) {
    Vec y;
    DM  reg;
    PetscCall(TaoGetRegularizer(tao, &reg));
    PetscCall(DMTaoGetCentralVector(reg, &y));
    PetscCall(VecAXPBYPCZ(user->workvec, 1, -1, 0, X, y));
    PetscCall(VecTDot(user->workvec, user->workvec, f));
  } else {
    PetscCall(VecTDot(X, X, f));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Grad_Tao(Tao tao, Vec X, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(VecCopy(X, G));
  if (user->y_bool) {
    Vec y;
    DM  reg;
    PetscCall(TaoGetRegularizer(tao, &reg));
    PetscCall(DMTaoGetCentralVector(reg, &y));
    PetscCall(VecAXPY(G, -1., y));
  }
  PetscCall(VecScale(G, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* L2 Metric. \|X-Y\|_2^2
 *
 * Note: stepsize is not included in this routine.
 *       Stepsize computation is handled internally */

///TODO theese are prob wrong. tao here is dm_subtao.... it self is reg, shouldn't have reg? internally do getparenttao?
PetscErrorCode L2_ObjGrad(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(VecCopy(X, G));
  if (user->y_bool) {
    Vec y;
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecAXPY(G, -1., y));
  }
  PetscCall(VecTDot(G, G, f));
  PetscCall(VecScale(G, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Obj(DM dm, Vec X, PetscReal *f, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  if (user->y_bool) {
    Vec y;
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecWAXPY(user->workvec, -1., y, X));
    PetscCall(VecTDot(user->workvec, user->workvec, f));
  } else {
    PetscCall(VecTDot(X, X, f));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Grad(DM dm, Vec X, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  if (user->y_bool) {
    Vec y;
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecWAXPY(G, -1., y, X));
  } else {
    PetscCall(VecCopy(X, G));
  }
  PetscCall(VecScale(G, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao         tao, tao_full, dm_tao;
  DM          dm, dm_master;
  Vec         x, x_full;
  Mat         temp_mat;
  PetscMPIInt size;
  AppCtx      user;
  PetscBool   flg;
  PetscRandom rctx;
  PetscReal   vec_dist;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.problem  = 0;
  user.g_type   = 0;
  user.n        = 10;
  user.stepsize = 0.5;
  user.mu1      = 1;
  user.y_bool   = PETSC_FALSE;
  user.sketch   = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  /* Types of problems to solve */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-problem", &user.problem, &flg));
  /* How to formulate obj and grad for regularizer */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-g_type", &user.g_type, &flg));
  /* Determines whether y is zeros, or random */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-central_vec", &user.y_bool, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user.stepsize, &flg));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x_full));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.workvec));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.y));
  /* x: zero vec */
  PetscCall(VecZeroEntries(x));

  /* A,b data */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, user.n, user.n, NULL, &temp_mat));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetSeed(rctx, 1234));
  PetscCall(MatSetRandom(temp_mat, rctx));
  PetscCall(MatAssemblyBegin(temp_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(temp_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatTransposeMatMult(temp_mat, temp_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user.A));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &user.b));
  PetscCall(PetscRandomSetSeed(rctx, 5678));
  PetscCall(VecSetRandom(user.b, rctx));
  PetscCall(MatDestroy(&temp_mat));
  /* y: random vec */
  PetscCall(PetscRandomSetSeed(rctx, 9012));
  PetscCall(VecSetRandom(user.y, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  /* tao      = 0.5 x^T A x - b^T x
   * tao_full = 0.5 x^T A x - b^T x + 0.5 \|x-y\|_2^2 */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao_full));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &dm_tao));
  PetscCall(TaoSetFromOptions(dm_tao)); //im not even setting solution vec, nor type... will this work, or do i need to set something?

  PetscCall(TaoSetType(tao, TAOCG));
  PetscCall(TaoSetType(tao_full, TAOCG));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetSolution(tao_full, x_full));
  PetscCall(TaoSetOptionsPrefix(tao, "added_"));
  PetscCall(TaoSetOptionsPrefix(tao_full, "normal_"));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetFromOptions(tao_full));
  PetscCall(TaoSetObjectiveAndGradient(tao_full, NULL, FullUserObjGrad, (void *)&user));

  /* Sketch: try to se DM for main objective */

  PetscCall(DMCreate(PETSC_COMM_SELF, &dm_master));
  /* problem:
   *
   * 0: f: ObjGrad
   * 1: f: Obj and Grad
   * 2: Set f via TaoSetDM */
  switch (user.problem) {
  case 0:
    if (user.sketch) {
      PetscCall(DMTaoSetObjectiveAndGradient(dm_master, UserObjGrad_DM, (void *)&user));
    } else {
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *)&user));
    }
    break;
  case 1:
    if (user.sketch) {
      PetscCall(DMTaoSetObjective(dm_master, UserObj_DM, (void *)&user));
      PetscCall(DMTaoSetGradient(dm_master, UserGrad_DM, (void *)&user));
    } else {
      PetscCall(TaoSetObjective(tao, UserObj, (void *)&user));
      PetscCall(TaoSetGradient(tao, NULL, UserGrad, (void *)&user));
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
  }

  if (user.sketch) {
    PetscCall(TaoSetDMSize(tao, 1));
    PetscCall(TaoSetDM(tao, dm_master, 0));
  }

  /* Create DM, g(x,y) = 0.5 \|x-y\|_2^2
   *
   * g_type:
   *
   * 0: ObjGrad
   * 1: Obj and Grad
   * 2: Built-in type
   * 3: Tao              */
  PetscCall(DMCreate(PetscObjectComm((PetscObject)tao), &dm));

  /* Set PD */
  switch (user.g_type) {
  case 0:
    PetscCall(DMTaoSetObjectiveAndGradient(dm, L2_ObjGrad, (void *)&user));
    break;
  case 1:
    PetscCall(DMTaoSetObjective(dm, L2_Obj, (void *)&user));
    PetscCall(DMTaoSetGradient(dm, L2_Grad, (void *)&user));
    break;
  case 2:
    PetscCall(DMTaoSetType(dm, DMTAOL2));
    break;
  case 3:
    PetscCall(TaoSetObjectiveAndGradient(dm_tao, NULL, L2_ObjGrad_Tao, (void *)&user));
    PetscCall(DMTaoUseTaoRoutines(dm, dm_tao));
    break;
  case 4:
    PetscCall(TaoSetObjective(dm_tao, L2_Obj_Tao, (void *)&user));
    PetscCall(TaoSetGradient(dm_tao, NULL, L2_Grad_Tao, (void *)&user));
    PetscCall(DMTaoUseTaoRoutines(dm, dm_tao));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid DM formulation type.");
  }
  // TODO  SCALE?

  /* Solve full version */
  /* Solve Regularizer version */
  PetscCall(DMTaoSetScale(dm, user.stepsize));
  if (user.y_bool) { PetscCall(DMTaoSetCentralVector(dm, user.y)); }
  PetscCall(TaoSetRegularizer(tao, dm));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoSolve(tao_full));

  /* Testing Regularizer version vs Full version */
  PetscCall(VecAXPY(x, -1., x_full));
  PetscCall(VecNorm(x, NORM_2, &vec_dist));
  if (vec_dist < 1.e-6) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Ful TaoSolve: < 1.e-6\n"));
  } else {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Full TaoSolve: %e\n", (double)vec_dist));
  }

  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dm_master));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoDestroy(&tao_full));
  PetscCall(TaoDestroy(&dm_tao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x_full));
  PetscCall(VecDestroy(&user.y));
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
      suffix: 000
      args: -problem 0 -g_type 0 -central_vec 0
      requires: !single

   test:
      suffix: 010
      args: -problem 0 -g_type 1 -central_vec 0
      requires: !single

   test:
      suffix: 020
      args: -problem 0 -g_type 2 -central_vec 0
      requires: !single

   test:
      suffix: 030
      args: -problem 0 -g_type 3 -central_vec 0
      requires: !single

   test:
      suffix: 040
      args: -problem 0 -g_type 4 -central_vec 0
      requires: !single

   test:
      suffix: 100
      args: -problem 1 -g_type 0 -central_vec 0
      requires: !single

   test:
      suffix: 110
      args: -problem 1 -g_type 1 -central_vec 0
      requires: !single

   test:
      suffix: 120
      args: -problem 1 -g_type 2 -central_vec 0
      requires: !single

   test:
      suffix: 130
      args: -problem 1 -g_type 3 -central_vec 0
      requires: !single

   test:
      suffix: 140
      args: -problem 1 -g_type 4 -central_vec 0
      requires: !single

   test:
      suffix: 001
      args: -problem 0 -g_type 0 -central_vec 1
      requires: !single

   test:
      suffix: 011
      args: -problem 0 -g_type 1 -central_vec 1
      requires: !single

   test:
      suffix: 021
      args: -problem 0 -g_type 2 -central_vec 1
      requires: !single

   test:
      suffix: 031
      args: -problem 0 -g_type 3 -central_vec 1
      requires: !single

   test:
      suffix: 041
      args: -problem 0 -g_type 4 -central_vec 1
      requires: !single

   test:
      suffix: 101
      args: -problem 1 -g_type 0 -central_vec 1
      requires: !single

   test:
      suffix: 111
      args: -problem 1 -g_type 1 -central_vec 1
      requires: !single

   test:
      suffix: 121
      args: -problem 1 -g_type 2 -central_vec 1
      requires: !single

   test:
      suffix: 131
      args: -problem 1 -g_type 3 -central_vec 1
      requires: !single

   test:
      suffix: 141
      args: -problem 1 -g_type 4 -central_vec 1
      requires: !single
TEST*/
