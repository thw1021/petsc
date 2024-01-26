/* TAOPROX example */

#include <petsctao.h>
#include <petscdm.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates various ways to use DM to solve proximal algorithms.. \n";

typedef struct {
  PetscScalar lb, ub;
  PetscInt    n;       /* dimension */
  PetscInt    problem; /* Types of problems to solve. */
  PetscInt    create;
  PetscInt    solve;
  PetscReal   stepsize, simp; /* simp: size of simplex */
  PetscReal   mu1;            /* Parameter for soft-threshold */
  PetscReal   tol;
} AppCtx;

int main(int argc, char **argv)
{
  Tao         tao;
  DM          dm0, dm1;
  Vec         x, x_test, y;
  PetscMPIInt size;
  PetscReal   vec_dist;
  AppCtx      user;
  PetscBool   flg;
  PetscRandom rctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.problem  = 0;
  user.create   = 0;
  user.solve    = 0;
  user.n        = 10;
  user.stepsize = 1;
  user.mu1      = 1;
  user.lb       = -1;
  user.ub       = 1;
  user.simp     = 1.;
  user.tol      = 1.e-12;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  /* Types of problems to solve */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-problem", &user.problem, &flg));
  /* Determine how to create PD objects */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-create", &user.create, &flg));
  /* Determine how to solve proximal problem */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-solve", &user.solve, &flg));
  /* Simplex problem size */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-simplex", &user.simp, &flg));
  /* Simplex problem tolerance */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-simp_tol", &user.tol, &flg));

  /* If stepsize ==1, default case. Else, adaptive version */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.n, &y));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(PetscRandomSetInterval(rctx, -10, 10));
  /* x : Random vec, from -10 to 10 */
  PetscCall(PetscRandomSetSeed(rctx, 1234));
  PetscCall(VecSetRandom(x, rctx));
  PetscCall(VecDuplicate(x, &x_test));
  PetscCall(VecCopy(x, x_test));
  /* y : Random vec, from -10 to 10 */
  PetscCall(PetscRandomSetSeed(rctx, 5678));
  PetscCall(VecSetRandom(y, rctx));
  PetscCall(VecZeroEntries(y));
  PetscCall(PetscRandomDestroy(&rctx));

  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOPROX));
  PetscCall(TaoSetSolution(tao, x));

  switch (user.problem) {
  case 0:
    //TODO below... context and stuff
    //TODO should VM be Mat or Vec? Really it should be Mat, but Vec is easier to work with..
    switch (user.create) {
    case 0:
      PetscCall(DMTaoCreate_L1(PETSC_COMM_SELF, &dm0, NULL, NULL, user.lb, user.ub));
      PetscCall(DMTaoCreate_L2(PETSC_COMM_SELF, &dm1, NULL, y));
      break;
    case 1:
      PetscCall(DMCreate(PETSC_COMM_SELF, &dm0));
      PetscCall(DMCreate(PETSC_COMM_SELF, &dm1));
      PetscCall(DMTaoSetType(dm0, DMTAOL1));
      PetscCall(DMTaoSetType(dm1, DMTAOL2));
      PetscCall(DMTaoSetCentralVector(dm1, y));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Unsupported create type!");
    }
    /* Try built-in Soft-Threshold */
    PetscCall(TaoSoftThreshold(y, user.lb, user.ub, x_test));
    break;
  case 1:
    switch (user.create) {
    case 0:
      PetscCall(DMTaoCreate_Simplex(PETSC_COMM_SELF, &dm0, NULL, NULL, user.simp, user.tol));
      PetscCall(DMTaoCreate_L2(PETSC_COMM_SELF, &dm1, NULL, y));
      break;
    case 1:
      PetscCall(DMCreate(PETSC_COMM_SELF, &dm0));
      PetscCall(DMCreate(PETSC_COMM_SELF, &dm1));
      PetscCall(DMTaoSetType(dm0, DMTAOSIMPLEX));
      PetscCall(DMTaoSetType(dm1, DMTAOL2));
      PetscCall(DMTaoSetCentralVector(dm1, y));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Unsupported create type!");
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Unsupported problem type!");
  }

  /* TODO. for TaoSolve of TAOPROX, setting which DM to be first is important */
  PetscCall(TaoSetDM(tao, dm0, 0, 1.));
  PetscCall(TaoAddDM(tao, dm1, user.stepsize));
  PetscCall(TaoSetFromOptions(tao));

  switch (user.solve) {
  case 0:
    PetscCall(TaoSolve(tao));
    break;
  case 1:
    PetscCall(DMTaoApplyProximalMap(dm0, dm1, user.stepsize, y, x, NULL));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Unsupported solve type!");
  }

  switch (user.problem) {
  case 0:
    /* Testing Regularizer version vs Full version */
    PetscCall(VecAXPY(x, -1., x_test));
    PetscCall(VecNorm(x, NORM_2, &vec_dist));
    if (vec_dist < 1.e-11) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with TAOPROX and SoftThreshold: < 1.e-11\n"));
    } else if (vec_dist < 1.e-6) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with TAOPROX and SoftThreshold: < 1.e-6\n"));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with TAOPROX and SoftThreshold: %e\n", (double)vec_dist));
    }
    break;
  case 1: {
    PetscReal sum, min;
    PetscCall(VecSum(x, &sum));
    PetscCall(VecMin(x, NULL, &min));
    vec_dist = PetscAbsReal(sum - user.simp);
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Smallest element of solution: %e\n", (double)min));
    if (vec_dist < 1.e-11) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "distance between VecSum and Simplex Size: < 1.e-11\n"));
    } else if (vec_dist < 1.e-6) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "distance between VecSum and Simplex Size: < 1.e-6\n"));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "distance between VecSum and Simplex Size: %e\n", (double)vec_dist));
    }
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Unsupported problem type!");
  }

  PetscCall(DMDestroy(&dm0));
  PetscCall(DMDestroy(&dm1));
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x_test));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: soft00
      args: -tao_gatol 1.e-4 -problem 0 -create 0 -solve 0
      requires: !single

   test:
      suffix: soft01
      args: -tao_gatol 1.e-4 -problem 0 -create 0 -solve 1
      requires: !single

   test:
      suffix: soft10
      args: -tao_gatol 1.e-4 -problem 0 -create 1 -solve 0
      requires: !single

   test:
      suffix: soft11
      args: -tao_gatol 1.e-4 -problem 0 -create 1 -solve 1
      requires: !single

   test:
      suffix: simplex00
      args: -tao_gatol 1.e-4 -problem 1 -create 0 -solve 0
      requires: !single

   test:
      suffix: simplex01
      args: -tao_gatol 1.e-4 -problem 1 -create 0 -solve 1
      requires: !single

   test:
      suffix: simplex10
      args: -tao_gatol 1.e-4 -problem 1 -create 1 -solve 0
      requires: !single

   test:
      suffix: simplex11
      args: -tao_gatol 1.e-4 -problem 1 -create 1 -solve 1
      requires: !single
TEST*/
