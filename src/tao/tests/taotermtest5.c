#include <petsctao.h>

static char help[] = "Production-setting test of the TAOTERMSUM matrix-free Hessian cache.\n\n\
A TAOSHELL solver's solve routine drives the genuine Tao Hessian operator the way\n\
a real solver does -- TaoComputeHessian() to set the evaluation point, then MatMult()\n\
to apply it -- through many permutations of (point, vector) calls, including the\n\
HessianMult / Hessian / HessianMult sequence at the same and at differing points.\n\
Each application is checked against the analytic Hessian-vector product, so any\n\
stale-cache result is caught.\n\n";

typedef struct {
  TaoTerm   sum;
  Vec       x0, x1, v0, v1, hw, ew;
  PetscReal max_err;
} TestCtx;

static PetscErrorCode FormObjectiveAndGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode CreateDiagonalHessianTerm(MPI_Comm, PetscInt, TaoTerm *);
static PetscErrorCode ShellSolve(Tao);

/* Apply the operator at its current snapshot point and compare H(point) v to the
   analytic product 2 * (point .* v) (two summands, each Hessian diag(point), scale 1). */
static PetscErrorCode CheckProduct(TestCtx *ctx, Mat H, Vec point, Vec v)
{
  PetscReal err;

  PetscFunctionBeginUser;
  PetscCall(MatMult(H, v, ctx->hw));
  PetscCall(VecPointwiseMult(ctx->ew, point, v));
  PetscCall(VecScale(ctx->ew, 2.0));
  PetscCall(VecAXPY(ctx->hw, -1.0, ctx->ew));
  PetscCall(VecNorm(ctx->hw, NORM_2, &err));
  ctx->max_err = PetscMax(ctx->max_err, err);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set the evaluation point with TaoComputeHessian(), then apply and check. */
static PetscErrorCode ApplyAndCheck(Tao tao, TestCtx *ctx, Mat H, Mat Hpre, Vec point, Vec v)
{
  PetscFunctionBeginUser;
  PetscCall(TaoComputeHessian(tao, point, H, Hpre));
  PetscCall(CheckProduct(ctx, H, point, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TaoTerm      t0, t1, sum;
  Tao          tao;
  TestCtx      ctx;
  Vec          x0;
  MPI_Comm     comm;
  PetscScalar *a;
  PetscMPIInt  size;
  PetscInt     n             = 4;
  PetscBool    separate_hpre = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-separate_hpre", &separate_hpre, NULL));

  PetscCall(CreateDiagonalHessianTerm(comm, n, &t0));
  PetscCall(CreateDiagonalHessianTerm(comm, n, &t1));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAOSHELL));
  /* Two terms -> Tao promotes the objective to a TAOTERMSUM internally. */
  PetscCall(TaoAddTerm(tao, NULL, 1.0, t0, NULL, NULL));
  PetscCall(TaoAddTerm(tao, NULL, 1.0, t1, NULL, NULL));
  PetscCall(TaoGetTerm(tao, NULL, &sum, NULL, NULL));
  /* Make the SUM's Hessian matrix-free, programmatically, so nothing leaks into
     the summands (they keep their assembled MATAIJ Hessians). */
  if (separate_hpre) PetscCall(TaoTermSetCreateHessianMode(sum, PETSC_FALSE, MATSHELL, MATAIJ));
  else PetscCall(TaoTermSetCreateHessianMode(sum, PETSC_TRUE, MATSHELL, NULL));

  PetscCall(TaoTermCreateSolutionVec(sum, &x0));
  PetscCall(VecDuplicate(x0, &ctx.x1));
  PetscCall(VecDuplicate(x0, &ctx.v0));
  PetscCall(VecDuplicate(x0, &ctx.v1));
  PetscCall(VecDuplicate(x0, &ctx.hw));
  PetscCall(VecDuplicate(x0, &ctx.ew));
  PetscCall(VecGetArrayWrite(x0, &a));
  for (PetscInt i = 0; i < n; i++) a[i] = (PetscScalar)(i + 1);
  PetscCall(VecRestoreArrayWrite(x0, &a));
  PetscCall(VecGetArrayWrite(ctx.x1, &a));
  for (PetscInt i = 0; i < n; i++) a[i] = (PetscScalar)(2 * (i + 1) + 1);
  PetscCall(VecRestoreArrayWrite(ctx.x1, &a));
  PetscCall(VecSet(ctx.v0, 1.0));
  PetscCall(VecGetArrayWrite(ctx.v1, &a));
  for (PetscInt i = 0; i < n; i++) a[i] = (PetscScalar)(n - i);
  PetscCall(VecRestoreArrayWrite(ctx.v1, &a));
  ctx.x0      = x0;
  ctx.sum     = sum;
  ctx.max_err = 0.0;

  PetscCall(TaoSetSolution(tao, x0));
  PetscCall(TaoShellSetContext(tao, &ctx));
  PetscCall(TaoShellSetSolve(tao, ShellSolve));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCheck(ctx.max_err <= 1.e-10, comm, PETSC_ERR_PLIB, "Matrix-free Hessian cache returned a wrong/stale product (max error %g)", (double)ctx.max_err);
  PetscCall(PetscPrintf(comm, "All matrix-free Hessian call sequences are correct (cache is consistent in the Tao operator path)\n"));

  PetscCall(VecDestroy(&x0));
  PetscCall(VecDestroy(&ctx.x1));
  PetscCall(VecDestroy(&ctx.v0));
  PetscCall(VecDestroy(&ctx.v1));
  PetscCall(VecDestroy(&ctx.hw));
  PetscCall(VecDestroy(&ctx.ew));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&t0));
  PetscCall(TaoTermDestroy(&t1));
  PetscCall(PetscFinalize());
  return 0;
}

/*
  ShellSolve - the TAOSHELL "solve" that drives the genuine Tao Hessian operator
  through many call-sequence permutations, the way real solvers do.
*/
static PetscErrorCode ShellSolve(Tao tao)
{
  TestCtx *ctx;
  Mat      H, Hpre;
  Vec      x0, x1, v0, v1;

  PetscFunctionBeginUser;
  PetscCall(TaoShellGetContext(tao, &ctx));
  x0 = ctx->x0;
  x1 = ctx->x1;
  v0 = ctx->v0;
  v1 = ctx->v1;

  /* Create the genuine matrix-free Tao Hessian operator from the objective sum. */
  PetscCall(TaoTermCreateHessianMatrices(ctx->sum, &H, &Hpre));

  /* S1-S3: baseline, repeat (cache hit), same point different vector. */
  PetscCall(ApplyAndCheck(tao, ctx, H, Hpre, x0, v0));
  PetscCall(ApplyAndCheck(tao, ctx, H, Hpre, x0, v0));
  PetscCall(ApplyAndCheck(tao, ctx, H, Hpre, x0, v1));

  /* S4-S5: move to x1, then revisit x0 (key change and back). */
  PetscCall(ApplyAndCheck(tao, ctx, H, Hpre, x1, v0));
  PetscCall(ApplyAndCheck(tao, ctx, H, Hpre, x0, v0));

  /* S6: HessianMult / Hessian / HessianMult at the same point x0. */
  PetscCall(TaoComputeHessian(tao, x0, H, Hpre));
  PetscCall(CheckProduct(ctx, H, x0, v0));
  PetscCall(TaoComputeHessian(tao, x0, H, Hpre));
  PetscCall(CheckProduct(ctx, H, x0, v0));

  /* S7: HessianMult at x0, Hessian at x1, then re-establish x0 and HessianMult --
     the closest the Tao operator API allows to the stale-cache pattern. */
  PetscCall(ApplyAndCheck(tao, ctx, H, Hpre, x0, v0));
  PetscCall(TaoComputeHessian(tao, x1, H, Hpre));
  PetscCall(ApplyAndCheck(tao, ctx, H, Hpre, x0, v0));

  /* S8: alternate x0/x1 repeatedly. */
  for (PetscInt k = 0; k < 6; k++) PetscCall(ApplyAndCheck(tao, ctx, H, Hpre, (k % 2) ? x1 : x0, v0));

  /* S9: many vectors at one point without re-computing the Hessian (Krylov-like). */
  PetscCall(TaoComputeHessian(tao, x1, H, Hpre));
  PetscCall(CheckProduct(ctx, H, x1, v0));
  PetscCall(CheckProduct(ctx, H, x1, v1));
  PetscCall(CheckProduct(ctx, H, x1, v0));

  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Hpre));
  PetscCall(TaoSetConvergedReason(tao, TAO_CONVERGED_USER));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateDiagonalHessianTerm - TAOTERMSHELL with point-dependent Hessian H(x) = diag(x),
  an assembled Hessian but no Hessian-vector callback (forces the sum's assemble-and-
  multiply cache path).
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
  PetscCall(TaoTermSetCreateHessianMode(term, PETSC_TRUE, MATAIJ, NULL));
  PetscCall(TaoTermShellSetHessian(term, FormHessian));
  PetscCall(TaoTermSetUp(term));
  *term_out = term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
     suffix: shell
     args: -n 4

   test:
     suffix: separate_hpre
     args: -n 4 -separate_hpre

TEST*/
