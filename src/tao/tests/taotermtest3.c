#include <petsctao.h>

static char help[] = "Regression test for the TAOTERMSUM matrix-free Hessian cache.\n\n\
Builds a two-term TAOTERMSUM whose summands provide an assembled Hessian but no\n\
Hessian-vector-product callback, so TaoTermComputeHessianMult() must use the\n\
per-summand cache that memoizes each summand's assembled Hessian.  The cache\n\
shares each summand's internal Hessian matrix with the assembled-Hessian path\n\
(TaoTermComputeHessian()); this test checks that an assembled Hessian evaluated\n\
at a different point does not corrupt a previously cached matrix-free product.\n\n";

static PetscErrorCode FormObjectiveAndGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode CreateDiagonalHessianTerm(MPI_Comm, PetscInt, TaoTerm *);

int main(int argc, char **argv)
{
  TaoTerm      t0, t1, sum;
  Mat          H = NULL, Hpre = NULL;
  Vec          x0, x1, v, Hv0, Hv1;
  MPI_Comm     comm;
  PetscScalar *a;
  PetscReal    diff;
  PetscMPIInt  size;
  PetscInt     n = 4;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* Two unmapped summands, each providing only an assembled (point-dependent)
     Hessian H_i(x) = diag(x) and NO TaoTermShellSetHessianMult().  The missing
     hessianmult op forces TaoTermComputeHessianMult_Sum() down the assembled
     fallback that caches each summand's Hessian. */
  PetscCall(CreateDiagonalHessianTerm(comm, n, &t0));
  PetscCall(CreateDiagonalHessianTerm(comm, n, &t1));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumSetNumberTerms(sum, 2));
  PetscCall(TaoTermSumSetTerm(sum, 0, "t0_", 1.0, t0, NULL));
  PetscCall(TaoTermSumSetTerm(sum, 1, "t1_", 1.0, t1, NULL));
  PetscCall(TaoTermSetUp(sum));

  /* Allocating the sum's assembled Hessian matrices lazily creates each
     summand's internal _unmapped_H, which the matrix-free cache then aliases. */
  PetscCall(TaoTermCreateHessianMatrices(sum, &H, &Hpre));

  PetscCall(TaoTermCreateSolutionVec(sum, &x0));
  PetscCall(VecDuplicate(x0, &x1));
  PetscCall(VecDuplicate(x0, &v));
  PetscCall(VecDuplicate(x0, &Hv0));
  PetscCall(VecDuplicate(x0, &Hv1));

  /* Two distinct evaluation points, so H(x0) != H(x1), and a probe direction. */
  PetscCall(VecGetArrayWrite(x0, &a));
  for (PetscInt i = 0; i < n; i++) a[i] = (PetscScalar)(i + 1);
  PetscCall(VecRestoreArrayWrite(x0, &a));
  PetscCall(VecGetArrayWrite(x1, &a));
  for (PetscInt i = 0; i < n; i++) a[i] = (PetscScalar)(i + 1 + n);
  PetscCall(VecRestoreArrayWrite(x1, &a));
  PetscCall(VecSet(v, 1.0));

  /* (1) Matrix-free Hessian-vector product at x0; this primes the cache. */
  PetscCall(TaoTermComputeHessianMult(sum, x0, NULL, v, Hv0));

  /* (2) Assembled Hessian at the DIFFERENT point x1.  This rewrites each
     summand's internal Hessian in place without invalidating the cache. */
  PetscCall(TaoTermComputeHessian(sum, x1, NULL, H, Hpre));

  /* (3) The identical query to (1): H(x0) v.  It must reproduce Hv0. */
  PetscCall(TaoTermComputeHessianMult(sum, x0, NULL, v, Hv1));

  PetscCall(VecAXPY(Hv1, -1.0, Hv0));
  PetscCall(VecNorm(Hv1, NORM_2, &diff));
  PetscCheck(diff <= 1.e-10, comm, PETSC_ERR_PLIB, "Stale TAOTERMSUM matrix-free Hessian cache: repeated TaoTermComputeHessianMult() at x0 disagreed after an assembled TaoTermComputeHessian() at x1 (||delta|| = %g)", (double)diff);
  PetscCall(PetscPrintf(comm, "TAOTERMSUM matrix-free Hessian-vector product is consistent across an interleaved assembled Hessian\n"));

  PetscCall(VecDestroy(&x0));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&Hv0));
  PetscCall(VecDestroy(&Hv1));
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Hpre));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(TaoTermDestroy(&t0));
  PetscCall(TaoTermDestroy(&t1));
  PetscCall(PetscFinalize());
  return 0;
}

/*
  CreateDiagonalHessianTerm - Build a TAOTERMSHELL whose Hessian depends on the
  evaluation point, with an assembled Hessian but no Hessian-vector callback.

  Input Parameters:
+ comm - the communicator
- n    - the solution size

  Output Parameter:
. term_out - the new `TaoTerm`
*/
static PetscErrorCode CreateDiagonalHessianTerm(MPI_Comm comm, PetscInt n, TaoTerm *term_out)
{
  TaoTerm term;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term, PETSC_TRUE /* H == Hpre */, MATAIJ, NULL));
  PetscCall(TaoTermShellSetHessian(term, FormHessian));
  /* Deliberately no TaoTermShellSetHessianMult(): see the header comment. */
  PetscCall(TaoTermSetUp(term));
  *term_out = term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormObjectiveAndGradient - Evaluates f(x) = sum_i x_i^3 / 6 and its gradient
  g_i = x_i^2 / 2, consistent with the Hessian diag(x).  Only the Hessian is
  exercised by this test; this is provided so the term is fully configured.
*/
static PetscErrorCode FormObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  const PetscScalar *xa;
  PetscScalar       *ga;
  PetscReal          f = 0.0;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArrayWrite(g, &ga));
  for (PetscInt i = 0; i < n; i++) {
    ga[i] = 0.5 * xa[i] * xa[i];
    f += PetscRealPart(xa[i] * xa[i] * xa[i]) / 6.0;
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(VecRestoreArrayWrite(g, &ga));
  *value = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormHessian - Evaluates the point-dependent Hessian H(x) = diag(x).
*/
static PetscErrorCode FormHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  const PetscScalar *xa;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayRead(x, &xa));
  if (H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(H, i, i, xa[i], INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  if (Hpre && Hpre != H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(Hpre, i, i, xa[i], INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex

   test:
     args: -n 4

TEST*/
