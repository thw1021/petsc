/* TAOPROX example */

#include <petsctao.h>
#include <petsctaoregularizer.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates various ways to use TAOPROX, and TaoRegularizer. \n";

typedef struct {
  PetscScalar lb, ub;
  PetscInt    n;              /* dimension */
  PetscInt    problem;        /* Types of problems to solve. */
  PetscReal   stepsize, simp; /* simp: size of simplex */
  PetscReal   mu1;            /* Parameter for soft-threshold */
} AppCtx;

int main(int argc, char **argv)
{
  Tao            tao;
  TaoRegularizer reg;
  Vec            x, x_test, y;
  PetscMPIInt    size;
  PetscReal      vec_dist;
  AppCtx         user;
  PetscBool      flg;
  PetscRandom    rctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  user.problem  = 0;
  user.n        = 10;
  user.stepsize = 1;
  user.mu1      = 1;
  user.lb       = -1;
  user.ub       = 1;
  user.simp     = 1.;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user.n, &flg));
  /* Types of problems to solve */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-problem", &user.problem, &flg));
  /* Simplex problem size */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-simplex", &user.simp, &flg));

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
  PetscCall(TaoRegularizerCreate(PETSC_COMM_SELF, &reg));
  PetscCall(TaoSetType(tao, TAOPROX));
  PetscCall(TaoSetSolution(tao, x));

  /* Cases that we want to try:
   *
   *  0: TAOPROX_L1 vs Soft-Threshold
   *  1: TAOPROX_SIMPLEX */

  if (user.problem == 0) {
    PetscCall(TaoProxSetType(tao, TAOPROX_L1));
    /* TODO should it be SetSTContext, or SetL1Context ? */
    /* Stepsize of 1 */
    PetscCall(TaoProxL1SetContext(tao, user.lb, user.ub));
    /* Try built-in Soft-Threshold */
    PetscCall(TaoSoftThreshold(y, user.lb, user.ub, x_test));
  } else if (user.problem == 1) {
    PetscCall(TaoProxSetType(tao, TAOPROX_SIMPLEX));
    PetscCall(TaoProxSimplexSetContext(tao, user.simp));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Unsupported problem type!");

  PetscCall(TaoRegularizerSetType(reg, TAOREGULARIZERL2));
  PetscCall(TaoRegularizerSetCentralVector(reg, y));
  PetscCall(TaoSetRegularizer(tao, reg));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  if (user.problem == 0) {
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
  } else if (user.problem == 1) {
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
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Unsupported problem type!");

  PetscCall(TaoRegularizerDestroy(&reg));
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
      suffix: soft
      args: -tao_gatol 1.e-4 -problem 0
      requires: !single

   test:
      suffix: simplex
      args: -tao_gatol 1.e-4 -problem 1
      requires: !single
TEST*/
