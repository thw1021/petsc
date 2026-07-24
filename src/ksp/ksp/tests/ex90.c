static char help[] = "Tests KSP right diagonal scaling.\n\n";

#include <petscsnes.h>
#include <petscdmda.h>
#include <petsc/private/petscimpl.h>
#include <math.h>

typedef struct {
  Mat              A, P;
  PetscObjectState Astate, Pstate;
  PetscInt         setup_count;
  PetscBool        check_residual;
} TestCtx;

static PetscErrorCode MatMult_Shell(Mat shell, Vec x, Vec y)
{
  const PetscScalar *xa;
  PetscScalar       *ya;

  PetscFunctionBeginUser;
  (void)shell;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(y, &ya));
  ya[0] = 4.0 * xa[0] + xa[1];
  ya[1] = 2.0 * xa[0] + 3.0 * xa[1];
  PetscCall(VecRestoreArray(y, &ya));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormFunction_Linear(void *ctx, Vec x, Vec f)
{
  const PetscScalar *xa;
  PetscScalar       *fa;

  PetscFunctionBeginUser;
  (void)ctx;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(f, &fa));
  fa[0] = 4.0 * xa[0] + xa[1];
  fa[1] = 2.0 * xa[0] + 3.0 * xa[1];
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPhysicalState(KSP ksp, Vec b, Vec x, void *vctx)
{
  TestCtx         *ctx = (TestCtx *)vctx;
  Mat              A, P;
  Vec              r;
  PetscReal        norm;
  PetscObjectState state;

  PetscFunctionBeginUser;
  PetscCall(KSPGetOperators(ksp, &A, &P));
  PetscCheck(A == ctx->A && P == ctx->P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSP callbacks did not observe the physical operators");
  PetscCall(PetscObjectStateGet((PetscObject)A, &state));
  PetscCheck(state == ctx->Astate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Operator state changed from %" PetscInt64_FMT " to %" PetscInt64_FMT, (PetscInt64)ctx->Astate, (PetscInt64)state);
  PetscCall(PetscObjectStateGet((PetscObject)P, &state));
  PetscCheck(state == ctx->Pstate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Preconditioning matrix state changed from %" PetscInt64_FMT " to %" PetscInt64_FMT, (PetscInt64)ctx->Pstate, (PetscInt64)state);
  if (x && ctx->check_residual) {
    PetscCall(VecDuplicate(b, &r));
    PetscCall(MatMult(A, x, r));
    PetscCall(VecAXPY(r, -1.0, b));
    PetscCall(VecNorm(r, NORM_2, &norm));
    PetscCheck(norm < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Physical residual norm %g is too large", (double)norm);
    PetscCall(VecDestroy(&r));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPhysicalPreSolve(KSP ksp, Vec b, Vec x, void *vctx)
{
  PetscFunctionBeginUser;
  (void)x;
  PetscCall(CheckPhysicalState(ksp, b, NULL, vctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_Identity(PC pc, Vec x, Vec y)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Count(PC pc)
{
  TestCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  ++ctx->setup_count;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetScale(Vec d, PetscScalar d0, PetscScalar d1)
{
  const PetscInt    rows[]   = {0, 1};
  const PetscScalar values[] = {d0, d1};

  PetscFunctionBeginUser;
  PetscCall(VecSetValues(d, 2, rows, values, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(d));
  PetscCall(VecAssemblyEnd(d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSolution(Vec x)
{
  Vec               expected;
  const PetscInt     rows[]   = {0, 1};
  const PetscScalar  values[] = {1.0, 2.0};
  PetscReal          norm;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &expected));
  PetscCall(VecSetValues(expected, 2, rows, values, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(expected));
  PetscCall(VecAssemblyEnd(expected));
  PetscCall(VecAXPY(expected, -1.0, x));
  PetscCall(VecNorm(expected, NORM_2, &norm));
  PetscCheck(norm < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Solution error norm %g is too large", (double)norm);
  PetscCall(VecDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormDeferredOperators(KSP ksp, Mat A, Mat P, void *ctx)
{
  const PetscInt    rows[]   = {0, 1};
  const PetscScalar values[] = {4.0, 1.0, 2.0, 3.0};
  PetscInt         *count    = (PetscInt *)ctx;

  PetscFunctionBeginUser;
  (void)ksp;
  ++*count;
  PetscCall(MatSetValues(P, 2, rows, 2, rows, values, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (A != P) PetscCall(MatCopy(P, A, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDeferredOperators(void)
{
  const PetscInt    rows[] = {0, 1};
  const PetscScalar rhs[]  = {6.0, 8.0};
  DM                dm;
  KSP               ksp;
  Vec               b, x, d;
  PetscInt          operator_count = 0;

  PetscFunctionBeginUser;
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 2, 1, 1, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &d));
  PetscCall(VecSetValues(b, 2, rows, rhs, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecSet(x, 0.0));
  PetscCall(SetScale(d, 2.0, 0.5));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetComputeOperators(ksp, FormDeferredOperators, &operator_count));
  PetscCall(KSPSetRightDiagonalScale(ksp, d));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCheck(operator_count == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected one operator callback, got %" PetscInt_FMT, operator_count);
  PetscCall(CheckSolution(x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormSNESFunction(SNES snes, Vec x, Vec f, void *ctx)
{
  const PetscScalar *xa;
  PetscScalar       *fa;

  PetscFunctionBeginUser;
  (void)snes;
  (void)ctx;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(f, &fa));
  fa[0] = xa[0] * xa[0] - 4.0;
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormSNESJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx)
{
  const PetscScalar *xa;

  PetscFunctionBeginUser;
  (void)snes;
  (void)ctx;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(MatSetValue(P, 0, 0, 2.0 * xa[0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatCopy(P, J, SAME_NONZERO_PATTERN));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSNESRightScale(void)
{
  SNES        snes;
  KSP         ksp;
  Mat         J;
  Vec         x, f, d;
  PetscInt    row = 0;
  PetscScalar value;
  PetscReal   error;

  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, 1, &x));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(VecDuplicate(x, &d));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 1, 1, 1, NULL, &J));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, f, FormSNESFunction, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, FormSNESJacobian, NULL));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(VecSet(d, 3.0));
  PetscCall(KSPSetRightDiagonalScale(ksp, d));
  PetscCall(VecSet(x, 3.0));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(VecGetValues(x, 1, &row, &value));
  error = PetscAbsScalar(value - 2.0);
  PetscCheck(error < 1e-10, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNES solution error %g is too large", (double)error);
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat               assembled, A, P, B, X;
  Vec               b, x, d, got, mffd_base = NULL, mffd_fbase = NULL;
  KSP               ksp;
  PC                pc;
  TestCtx           ctx;
  PetscBool         distinct = PETSC_FALSE, shell = PETSC_FALSE, mffd = PETSC_FALSE, nonzero = PETSC_FALSE, test_reuse = PETSC_FALSE, test_errors = PETSC_FALSE, test_snes = PETSC_FALSE, test_deferred_operators = PETSC_FALSE;
  PetscMPIInt       size;
  PetscErrorCode    ierr;
  PetscObjectState  state;
  const PetscInt    rows[] = {0, 1};
  const PetscScalar matrix[] = {4.0, 1.0, 2.0, 3.0}, rhs[] = {6.0, 8.0};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This test requires one MPI rank");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-distinct_pmat", &distinct, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mat_shell", &shell, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mat_mffd", &mffd, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nonzero_guess", &nonzero, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_reuse", &test_reuse, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_errors", &test_errors, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_snes", &test_snes, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_deferred_operators", &test_deferred_operators, NULL));

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 2, 2, 2, NULL, &assembled));
  PetscCall(MatSetValues(assembled, 2, rows, 2, rows, matrix, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(assembled, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(assembled, MAT_FINAL_ASSEMBLY));
  PetscCheck(!shell || !mffd, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Use only one matrix-free option");
  if (mffd) {
    PetscCall(MatCreateMFFD(PETSC_COMM_WORLD, 2, 2, 2, 2, &A));
    PetscCall(MatMFFDSetFunction(A, FormFunction_Linear, NULL));
    PetscCall(MatCreateVecs(assembled, &mffd_base, &mffd_fbase));
    PetscCall(VecSet(mffd_base, 0.0));
    PetscCall(VecSet(mffd_fbase, 0.0));
    PetscCall(MatMFFDSetBase(A, mffd_base, mffd_fbase));
    P = assembled;
    PetscCall(PetscObjectReference((PetscObject)P));
  } else if (shell) {
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, 2, 2, 2, 2, NULL, &A));
    PetscCall(MatShellSetOperation(A, MATOP_MULT, (void (*)(void))MatMult_Shell));
    P = assembled;
    PetscCall(PetscObjectReference((PetscObject)P));
  } else {
    A = assembled;
    PetscCall(PetscObjectReference((PetscObject)A));
    if (distinct) PetscCall(MatDuplicate(assembled, MAT_COPY_VALUES, &P));
    else {
      P = A;
      PetscCall(PetscObjectReference((PetscObject)P));
    }
  }
  PetscCall(MatCreateVecs(assembled, &x, &b));
  PetscCall(VecSetValues(b, 2, rows, rhs, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecDuplicate(x, &d));
  PetscCall(SetScale(d, 2.0, 0.5));
  if (nonzero) PetscCall(VecSet(x, 0.25));
  else PetscCall(VecSet(x, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, P));
  PetscCall(KSPSetInitialGuessNonzero(ksp, nonzero));
  PetscCall(KSPSetRightDiagonalScale(ksp, d));
  PetscCall(KSPGetRightDiagonalScale(ksp, &got));
  PetscCheck(got == d, PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSPGetRightDiagonalScale() returned the wrong vector");
  PetscCall(KSPSetFromOptions(ksp));
  if (test_reuse) {
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetContext(pc, &ctx));
    PetscCall(PCShellSetApply(pc, PCApply_Identity));
    PetscCall(PCShellSetSetUp(pc, PCSetUp_Count));
  }

  ctx.A = A;
  ctx.P = P;
  ctx.setup_count = 0;
  ctx.check_residual = PETSC_TRUE;
  PetscCall(PetscObjectStateGet((PetscObject)A, &ctx.Astate));
  PetscCall(PetscObjectStateGet((PetscObject)P, &ctx.Pstate));
  PetscCall(KSPSetPreSolve(ksp, CheckPhysicalPreSolve, &ctx));
  PetscCall(KSPSetPostSolve(ksp, CheckPhysicalState, &ctx));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(CheckSolution(x));

  if (test_reuse) {
    PetscCheck(ctx.setup_count == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected one PC setup, got %" PetscInt_FMT, ctx.setup_count);
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCheck(ctx.setup_count == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unchanged scaling rebuilt the PC");
    PetscCall(SetScale(d, 0.5, 4.0));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCheck(ctx.setup_count == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Changed scaling did not rebuild the PC");
    PetscCall(CheckSolution(x));
    PetscCall(MatShift(P, 0.25));
    PetscCall(PetscObjectStateGet((PetscObject)P, &ctx.Pstate));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCheck(ctx.setup_count == 3, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Changed matrix did not rebuild the PC");
    PetscCall(CheckSolution(x));
  }

  PetscCall(PetscObjectStateGet((PetscObject)A, &state));
  PetscCheck(state == ctx.Astate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Physical operator state was not restored");
  PetscCall(PetscObjectStateGet((PetscObject)P, &state));
  PetscCheck(state == ctx.Pstate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Physical preconditioning matrix state was not restored");

  if (test_errors) {
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    PetscCall(SetScale(d, 0.0, 1.0));
    ierr = KSPSolve(ksp, b, x);
    PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Zero scale returned error %d", (int)ierr);
    PetscCall(SetScale(d, (PetscScalar)NAN, 1.0));
    ierr = KSPSolve(ksp, b, x);
    PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nonfinite scale returned error %d", (int)ierr);
    PetscCall(SetScale(d, 2.0, 0.5));
    ierr = KSPSolveTranspose(ksp, b, x);
    PetscCheck(ierr == PETSC_ERR_SUP, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Transpose solve returned error %d", (int)ierr);
    PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD, 2, 1, NULL, &B));
    PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));
    ierr = KSPMatSolve(ksp, B, X);
    PetscCheck(ierr == PETSC_ERR_SUP, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Multiple-RHS solve returned error %d", (int)ierr);
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&B));
    PetscCall(PetscPopErrorHandler());
  }

  PetscCall(KSPSetRightDiagonalScale(ksp, NULL));
  PetscCall(KSPGetRightDiagonalScale(ksp, &got));
  PetscCheck(!got, PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSP right diagonal scale was not cleared");
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&mffd_fbase));
  PetscCall(VecDestroy(&mffd_base));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&assembled));
  if (test_snes) PetscCall(TestSNESRightScale());
  if (test_deferred_operators) PetscCall(TestDeferredOperators());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: direct_alias
    output_file: output/empty.out
    args: -ksp_type preonly -pc_type lu

  test:
    suffix: direct_distinct_nonzero
    output_file: output/empty.out
    args: -distinct_pmat -nonzero_guess -ksp_type richardson -pc_type lu -ksp_max_it 1

  test:
    suffix: iterative
    output_file: output/empty.out
    args: -ksp_type gmres -pc_type jacobi -ksp_rtol 1e-12

  test:
    suffix: shell
    output_file: output/empty.out
    args: -mat_shell -ksp_type gmres -pc_type jacobi -ksp_rtol 1e-12

  test:
    suffix: mffd
    output_file: output/empty.out
    args: -mat_mffd -ksp_type gmres -pc_type jacobi -ksp_rtol 1e-12

  test:
    suffix: reuse
    output_file: output/empty.out
    args: -distinct_pmat -test_reuse -ksp_type gmres -ksp_rtol 1e-12

  test:
    suffix: errors
    output_file: output/empty.out
    args: -test_errors -ksp_type preonly -pc_type lu

  test:
    suffix: snes
    output_file: output/empty.out
    args: -test_snes -snes_linesearch_type bt -snes_ksp_ew -snes_rtol 1e-12

  test:
    suffix: deferred_operators
    output_file: output/empty.out
    args: -test_deferred_operators -ksp_type preonly -pc_type lu

TEST*/
