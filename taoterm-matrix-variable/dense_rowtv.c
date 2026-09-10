/*
  min_X  0.5 || A X - B ||_F^2 + lambda * rowTV(X)

  A: m x M sparse (MATAIJ), X: M x N dense unknown, B: m x N dense data.

  The point of this example is the plumbing between a matrix-valued unknown and Tao,
  which only accepts a Vec. X is never a separate object: it is the Tao solution Vec of
  local length Mloc*N. Whenever a matrix operation is needed, the Vec's storage is lent to
  a persistent MATDENSE placeholder (rows distributed like the columns of A, lda = Mloc,
  column-major within the rank) with MatDensePlaceArray(). Nothing is copied.

  The row-TV term (TAOTERMTV1D) does not exist yet. Until it lands, the regularizer falls
  back to TAOTERML1 so that everything else in this file can be compiled and run today.
  Build with -DTAOTERM_HAVE_TV1D once the term exists.

  Layout consequence for the future TV term: row i of X on this rank is the strided set of
  entries xa[i + Mloc*j], j = 0..N-1. Every row lives entirely on one rank, so a row-wise
  1D TV prox (Condat) needs no communication. A column-wise TV would need the neighbor
  rank's boundary row.
*/

#include <petsctao.h>

static char help[] = "Least squares in a dense matrix unknown, with X stored as a Vec and lent to a MATDENSE placeholder.\n";

typedef struct {
  Mat      A, B; /* data */
  Mat      X, G; /* placeholders whose storage is swapped for Tao's vectors on every call */
  Mat      R;    /* persistent product A X - B; keeps the symbolic product data */
  PetscInt M, N, Mloc;
} AppCtx;

/* f(x) = 0.5 || A X - B ||_F^2,  g = A^T (A X - B),  with X = reshape(x, M, N) */
static PetscErrorCode ObjGrad_Data(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  AppCtx            *user;
  const PetscScalar *xa;
  PetscScalar       *ga;
  PetscReal          nrm;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArrayWrite(g, &ga));
  PetscCall(MatDensePlaceArray(user->X, xa)); /* X = reshape(x, M, N), no copy */
  PetscCall(MatDensePlaceArray(user->G, ga)); /* G writes straight into g */
  PetscCall(MatMatMult(user->A, user->X, MAT_REUSE_MATRIX, PETSC_DETERMINE, &user->R)); /* R = A X */
  PetscCall(MatAXPY(user->R, -1.0, user->B, SAME_NONZERO_PATTERN));                    /* R = A X - B */
  PetscCall(MatNorm(user->R, NORM_FROBENIUS, &nrm));
  *f = 0.5 * nrm * nrm;
  PetscCall(MatTransposeMatMult(user->A, user->R, MAT_REUSE_MATRIX, PETSC_DETERMINE, &user->G)); /* G = A^T R */
  PetscCall(MatDenseResetArray(user->G));
  PetscCall(MatDenseResetArray(user->X));
  PetscCall(VecRestoreArrayWrite(g, &ga));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Central-difference check of the shell term's gradient along a random direction; run with -check_gradient */
static PetscErrorCode CheckGradient(TaoTerm term, Vec x)
{
  Vec         x0, d, g;
  PetscRandom rctx;
  PetscScalar gd;
  PetscReal   f0, fp, fm, fd, h = 1.e-5;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &x0));
  PetscCall(VecDuplicate(x, &d));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)x), &rctx));
  PetscCall(PetscRandomSetInterval(rctx, -1.0, 1.0));
  PetscCall(VecSetRandom(x0, rctx));
  PetscCall(VecSetRandom(d, rctx));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, x0, NULL, &f0, g));
  PetscCall(VecDot(g, d, &gd));
  PetscCall(VecAXPY(x0, h, d));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, x0, NULL, &fp, g));
  PetscCall(VecAXPY(x0, -2.0 * h, d));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, x0, NULL, &fm, g));
  fd = (fp - fm) / (2.0 * h);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)x), "gradient check: g.d = %g, central difference = %g, relative difference = %g\n", (double)PetscRealPart(gd), (double)fd, (double)(PetscAbsReal(PetscRealPart(gd) - fd) / PetscMax(PetscAbsReal(fd), 1.e-12))));
  PetscCall(VecDestroy(&x0));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx             user;
  Tao                tao;
  TaoTerm            fterm, gterm;
  Vec                x;
  Mat                Xtrue;
  const PetscScalar *xa;
  PetscScalar       *xt;
  PetscReal          lambda = 1.e-3, anorm, f, err, xtnorm;
  PetscInt           m, its, rstart, rend;
  TaoConvergedReason reason;
  PetscBool          check = PETSC_FALSE, no_reg = PETSC_FALSE;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm   = PETSC_COMM_WORLD;
  user.M = 16;
  user.N = 8;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &user.M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &user.N, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &lambda, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_gradient", &check, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_reg", &no_reg, NULL)); /* smooth term only, for use with classical Tao solvers */
  PetscCheck(user.M > 2 && user.N > 1, comm, PETSC_ERR_USER, "Need M > 2 and N > 1");
  m = user.M - 2;

  /* A: banded m x M, three entries per row */
  PetscCall(MatCreate(comm, &user.A));
  PetscCall(MatSetSizes(user.A, PETSC_DECIDE, PETSC_DECIDE, m, user.M));
  PetscCall(MatSetFromOptions(user.A));
  PetscCall(MatSeqAIJSetPreallocation(user.A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(user.A, 3, NULL, 3, NULL));
  PetscCall(MatGetOwnershipRange(user.A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscInt    cols[3] = {i, i + 1, i + 2};
    PetscScalar vals[3] = {2.0, -1.0, 0.5};

    PetscCall(MatSetValues(user.A, 1, &i, 3, cols, vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(user.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetLocalSize(user.A, NULL, &user.Mloc)); /* X's row ownership is A's column ownership */
  PetscCall(MatNorm(user.A, NORM_FROBENIUS, &anorm));

  /* placeholders and persistent products, built once */
  PetscCall(MatCreateDense(comm, user.Mloc, PETSC_DECIDE, user.M, user.N, NULL, &user.X));
  PetscCall(MatMatMult(user.A, user.X, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &user.R));
  PetscCall(MatTransposeMatMult(user.A, user.R, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &user.G));

  /* X_true: piecewise constant along each row (in j), varying in i; B = A X_true */
  PetscCall(MatGetOwnershipRange(user.X, &rstart, &rend));
  PetscCall(MatDenseGetArrayWrite(user.X, &xt));
  for (PetscInt j = 0; j < user.N; j++) {
    for (PetscInt i = rstart; i < rend; i++) xt[(i - rstart) + user.Mloc * j] = (PetscScalar)((i % 3) + 1) * (j < user.N / 2 ? 1.0 : -1.0);
  }
  PetscCall(MatDenseRestoreArrayWrite(user.X, &xt));
  PetscCall(MatDuplicate(user.X, MAT_COPY_VALUES, &Xtrue));
  PetscCall(MatNorm(Xtrue, NORM_FROBENIUS, &xtnorm));
  PetscCall(MatMatMult(user.A, user.X, MAT_REUSE_MATRIX, PETSC_DETERMINE, &user.R));
  PetscCall(MatDuplicate(user.R, MAT_COPY_VALUES, &user.B));

  /* the unknown as Tao sees it: one Vec of global length M*N, column-major within each rank */
  PetscCall(VecCreate(comm, &x));
  PetscCall(VecSetSizes(x, user.Mloc * user.N, user.M * user.N));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSet(x, 0.0));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetType(tao, TAOFB));

  PetscCall(TaoTermCreate(comm, &fterm));
  PetscCall(TaoTermSetType(fterm, TAOTERMSHELL));
  PetscCall(TaoTermSetParametersMode(fterm, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermShellSetContext(fterm, &user));
  PetscCall(TaoTermShellSetObjectiveAndGradient(fterm, ObjGrad_Data));
  PetscCall(TaoTermSetLipschitz(fterm, anorm * anorm)); /* ||A||_2^2 <= ||A||_F^2 */
  PetscCall(TaoTermSetSolutionTemplate(fterm, x));

  PetscCall(TaoTermCreate(comm, &gterm));
#if defined(TAOTERM_HAVE_TV1D)
  PetscCall(TaoTermSetType(gterm, "tv1d"));                    /* not in the tree yet */
  PetscCall(TaoTermTV1DSetShape(gterm, user.M, user.N, 1));    /* not in the tree yet: X is M x N, TV along axis 1 (along each row) */
#else
  PetscCall(TaoTermSetType(gterm, TAOTERML1)); /* stand-in so the plumbing runs today */
#endif
  PetscCall(TaoTermSetSolutionTemplate(gterm, x));

  PetscCall(TaoAddTerm(tao, "f_", 1.0, fterm, NULL, NULL));
  if (!no_reg) PetscCall(TaoAddTerm(tao, "g_", lambda, gterm, NULL, NULL));
  PetscCall(TaoSetFromOptions(tao));
  if (check) {
    PetscCall(TaoSetUp(tao));
    PetscCall(CheckGradient(fterm, x));
  }
  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetSolutionStatus(tao, &its, &f, NULL, NULL, NULL, &reason));

  /* read the answer back as a matrix, still no copy */
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(MatDensePlaceArray(user.X, xa));
  PetscCall(MatAXPY(Xtrue, -1.0, user.X, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(Xtrue, NORM_FROBENIUS, &err));
  PetscCall(MatDenseResetArray(user.X));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(PetscPrintf(comm, "iterations %" PetscInt_FMT ", reason %s, objective %g, ||X - Xtrue||_F / ||Xtrue||_F = %g\n", its, TaoConvergedReasons[reason], (double)f, (double)(err / xtnorm)));

  PetscCall(TaoTermDestroy(&fterm));
  PetscCall(TaoTermDestroy(&gterm));
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Xtrue));
  PetscCall(MatDestroy(&user.A));
  PetscCall(MatDestroy(&user.B));
  PetscCall(MatDestroy(&user.X));
  PetscCall(MatDestroy(&user.G));
  PetscCall(MatDestroy(&user.R));
  PetscCall(PetscFinalize());
  return 0;
}
