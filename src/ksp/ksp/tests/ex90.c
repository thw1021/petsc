static char help[] = "Tests explicit KSP left and right diagonal scaling.\n\n";

#include <petscsnes.h>
#include <petscdmda.h>
#include <petsc/private/petscimpl.h>
#include <math.h>

typedef struct {
  Mat              A, P, Aphysical, Pphysical;
  Vec              rhs;
  Vec              left, right;
  PetscObjectState Astate, Pstate, rhs_state;
  PetscInt         setup_count;
  PetscBool        check_residual, check_properties;
} TestCtx;

typedef struct {
  PetscReal expected_rtol;
  PetscInt  presolve_count, setup_count;
} SNESCallbackCtx;

static PetscErrorCode MatMult_Shell(Mat shell, Vec x, Vec y)
{
  const PetscScalar *xa;
  PetscScalar       *ya;

  PetscFunctionBeginUser;
  (void)shell;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(y, &ya));
  ya[0] = 4.0 * xa[0] + xa[1];
  ya[1] = xa[0] + 3.0 * xa[1];
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
  fa[1] = xa[0] + 3.0 * xa[1];
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPhysicalState(KSP ksp, Vec b, Vec x, void *vctx)
{
  TestCtx         *ctx = (TestCtx *)vctx;
  Mat              A, P;
  Vec              r;
  PetscReal        norm, bnorm, rtol, atol, tolerance;
  PetscObjectState state;
  PetscBool        set, flag;

  PetscFunctionBeginUser;
  PetscCall(KSPGetOperators(ksp, &A, &P));
  PetscCheck(A == ctx->A && P == ctx->P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSP callbacks did not observe the physical operators");
  PetscCall(PetscObjectStateGet((PetscObject)A, &state));
  PetscCheck(state == ctx->Astate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Operator state changed from %" PetscInt64_FMT " to %" PetscInt64_FMT, (PetscInt64)ctx->Astate, (PetscInt64)state);
  PetscCall(PetscObjectStateGet((PetscObject)P, &state));
  PetscCheck(state == ctx->Pstate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Preconditioning matrix state changed from %" PetscInt64_FMT " to %" PetscInt64_FMT, (PetscInt64)ctx->Pstate, (PetscInt64)state);
  PetscCheck(b == ctx->rhs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSP callbacks did not observe the physical right-hand side");
  PetscCall(PetscObjectStateGet((PetscObject)b, &state));
  PetscCheck(state == ctx->rhs_state, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Right-hand side state changed from %" PetscInt64_FMT " to %" PetscInt64_FMT, (PetscInt64)ctx->rhs_state, (PetscInt64)state);
  if (ctx->check_properties) {
    PetscCall(MatIsSymmetricKnown(P, &set, &flag));
    PetscCheck(set && flag, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Preconditioning matrix lost its symmetric property");
    PetscCall(MatIsHermitianKnown(P, &set, &flag));
    PetscCheck(set && flag, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Preconditioning matrix lost its Hermitian property");
    PetscCall(MatIsSPDKnown(P, &set, &flag));
    PetscCheck(set && flag, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Preconditioning matrix lost its SPD property");
  }
  if (x && ctx->check_residual) {
    PetscCall(VecDuplicate(b, &r));
    PetscCall(MatMult(A, x, r));
    PetscCall(VecAXPY(r, -1.0, b));
    PetscCall(VecNorm(r, NORM_2, &norm));
    PetscCall(VecNorm(b, NORM_2, &bnorm));
    PetscCall(KSPGetTolerances(ksp, &rtol, &atol, NULL, NULL));
    tolerance = 10.0 * PetscMax(atol, PetscMax(rtol * bnorm, PETSC_MACHINE_EPSILON * bnorm));
    PetscCall(VecDestroy(&r));
    PetscCheck(norm <= tolerance, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Physical residual norm %g exceeds tolerance %g", (double)norm, (double)tolerance);
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

static PetscErrorCode CheckScaledMatrix(Mat scaled, Mat physical, Vec left, Vec right, const char *name)
{
  Mat       expected;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(MatDuplicate(physical, MAT_COPY_VALUES, &expected));
  PetscCall(MatDiagonalScale(expected, left, right));
  PetscCall(MatAXPY(expected, -1.0, scaled, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(expected, NORM_FROBENIUS, &norm));
  PetscCheck(norm < 100 * PETSC_MACHINE_EPSILON, PetscObjectComm((PetscObject)scaled), PETSC_ERR_PLIB, "Scaled %s error norm %g is too large", name, (double)norm);
  PetscCall(MatDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Count(PC pc)
{
  TestCtx *ctx;
  Mat      A, P;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(PCGetOperators(pc, &A, &P));
  PetscCall(CheckScaledMatrix(A, ctx->Aphysical, ctx->left, ctx->right, "operator"));
  PetscCall(CheckScaledMatrix(P, ctx->Pphysical, ctx->left, ctx->right, "preconditioning matrix"));
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

static PetscErrorCode CheckSolution(KSP ksp, Vec x)
{
  Vec               expected;
  const PetscInt    rows[]   = {0, 1};
  const PetscScalar values[] = {1.0, 2.0};
  PetscReal         norm, expected_norm, rtol, atol, tolerance;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &expected));
  PetscCall(VecSetValues(expected, 2, rows, values, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(expected));
  PetscCall(VecAssemblyEnd(expected));
  PetscCall(VecNorm(expected, NORM_2, &expected_norm));
  PetscCall(VecAXPY(expected, -1.0, x));
  PetscCall(VecNorm(expected, NORM_2, &norm));
  PetscCall(KSPGetTolerances(ksp, &rtol, &atol, NULL, NULL));
  tolerance = 10.0 * PetscMax(atol, PetscMax(rtol * expected_norm, PETSC_MACHINE_EPSILON * expected_norm));
  PetscCall(VecDestroy(&expected));
  PetscCheck(norm <= tolerance, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Solution error norm %g exceeds tolerance %g", (double)norm, (double)tolerance);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormDeferredOperators(KSP ksp, Mat A, Mat P, void *ctx)
{
  const PetscInt    rows[]   = {0, 1};
  const PetscScalar values[] = {4.0, 1.0, 1.0, 3.0};
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
  const PetscScalar rhs[]  = {6.0, 7.0};
  DM                dm;
  KSP               ksp;
  Vec               b, x, left, right;
  PetscInt          operator_count = 0;

  PetscFunctionBeginUser;
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 2, 1, 1, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &left));
  PetscCall(VecDuplicate(x, &right));
  PetscCall(VecSetValues(b, 2, rows, rhs, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecSet(x, 0.0));
  PetscCall(SetScale(left, 0.25, 3.0));
  PetscCall(SetScale(right, 2.0, 0.5));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetComputeOperators(ksp, FormDeferredOperators, &operator_count));
  PetscCall(KSPSetLeftDiagonalScale(ksp, left));
  PetscCall(KSPSetRightDiagonalScale(ksp, right));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCheck(operator_count == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected one operator callback, got %" PetscInt_FMT, operator_count);
  PetscCall(CheckSolution(ksp, x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&right));
  PetscCall(VecDestroy(&left));
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

static PetscErrorCode CheckSNESPreSolve(KSP ksp, Vec b, Vec x, void *vctx)
{
  SNESCallbackCtx *ctx = (SNESCallbackCtx *)vctx;
  PetscReal        rtol;

  PetscFunctionBeginUser;
  (void)b;
  (void)x;
  PetscCall(KSPGetTolerances(ksp, &rtol, NULL, NULL, NULL));
  if (!ctx->presolve_count) PetscCheck(PetscAbsReal(rtol - ctx->expected_rtol) <= PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "User pre-solve callback ran before Eisenstat-Walker updated KSP rtol: expected %g, got %g", (double)ctx->expected_rtol, (double)rtol);
  ++ctx->presolve_count;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_CheckSNESPreSolve(PC pc)
{
  SNESCallbackCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx->presolve_count, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PC setup ran before the user pre-solve callback");
  ++ctx->setup_count;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSNESScale(void)
{
  SNES            snes;
  KSP             ksp;
  PC              pc;
  Mat             J;
  Vec             x, f, left, right;
  SNESCallbackCtx ctx = {.expected_rtol = 0.25, .presolve_count = 0, .setup_count = 0};
  PetscInt        row = 0;
  PetscScalar     value;
  PetscReal       error, rtol, atol, tolerance;

  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, 1, &x));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(VecDuplicate(x, &left));
  PetscCall(VecDuplicate(x, &right));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 1, 1, 1, NULL, &J));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, f, FormSNESFunction, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, FormSNESJacobian, NULL));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, &ctx));
  PetscCall(PCShellSetApply(pc, PCApply_Identity));
  PetscCall(PCShellSetSetUp(pc, PCSetUp_CheckSNESPreSolve));
  PetscCall(KSPSetPreSolve(ksp, CheckSNESPreSolve, &ctx));
  PetscCall(VecSet(left, 2.0));
  PetscCall(VecSet(right, 3.0));
  PetscCall(KSPSetLeftDiagonalScale(ksp, left));
  PetscCall(KSPSetRightDiagonalScale(ksp, right));
  PetscCall(VecSet(x, 3.0));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(VecGetValues(x, 1, &row, &value));
  PetscCall(SNESGetTolerances(snes, &atol, &rtol, NULL, NULL, NULL));
  PetscCheck(ctx.presolve_count, PETSC_COMM_SELF, PETSC_ERR_PLIB, "User pre-solve callback was not called");
  PetscCheck(ctx.setup_count, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PC setup callback was not called");
  error     = PetscAbsScalar(value - (PetscScalar)2.0);
  tolerance = 100.0 * PetscMax(PETSC_MACHINE_EPSILON, PetscMax(atol, rtol));
  PetscCheck(error <= tolerance, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNES solution error %g exceeds tolerance %g", (double)error, (double)tolerance);
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&right));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSBAIJError(PetscBool left)
{
  Mat               A, physical;
  Vec               scale;
  KSP               ksp;
  PetscErrorCode    ierr;
  PetscObjectState  state, restored_state;
  PetscReal         norm;
  PetscBool         set, flag;
  const PetscInt    rows[]   = {0, 1};
  const PetscScalar values[] = {4.0, 1.0, 1.0, 3.0};

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqSBAIJ(PETSC_COMM_WORLD, 1, 2, 2, 2, NULL, &A));
  PetscCall(MatSetOption(A, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
  PetscCall(MatSetValues(A, 2, rows, 2, rows, values, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &physical));
  PetscCall(MatCreateVecs(A, &scale, NULL));
  PetscCall(SetScale(scale, 2.0, 0.5));
  PetscCall(PetscObjectStateGet((PetscObject)A, &state));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  if (left) PetscCall(KSPSetLeftDiagonalScale(ksp, scale));
  else PetscCall(KSPSetRightDiagonalScale(ksp, scale));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = KSPSetUp(ksp);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s scaling of MATSBAIJ returned error %d", left ? "Left" : "Right", (int)ierr);

  PetscCall(PetscObjectStateGet((PetscObject)A, &restored_state));
  PetscCheck(restored_state == state, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MATSBAIJ object state was not restored");
  PetscCall(MatAXPY(physical, -1.0, A, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(physical, NORM_FROBENIUS, &norm));
  PetscCheck(norm < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MATSBAIJ restoration error norm %g is too large", (double)norm);
  PetscCall(MatIsSymmetricKnown(A, &set, &flag));
  PetscCheck(set && flag, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MATSBAIJ lost its symmetric property");

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&scale));
  PetscCall(MatDestroy(&physical));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat     assembled, A, P, B, X;
  Vec     b, x, left, right, got, bad, error_scale, mffd_base = NULL, mffd_fbase = NULL;
  KSP     ksp, legacy_ksp;
  PC      pc;
  TestCtx ctx;
  PetscBool scale_left = PETSC_FALSE, scale_right = PETSC_TRUE, distinct = PETSC_FALSE, shell = PETSC_FALSE, mffd = PETSC_FALSE, nonzero = PETSC_FALSE, test_reuse = PETSC_FALSE, test_errors = PETSC_FALSE, test_snes = PETSC_FALSE, test_deferred_operators = PETSC_FALSE, test_explicit_setup = PETSC_FALSE;
  PetscMPIInt       size;
  PetscErrorCode    ierr;
  PetscObjectState  state;
  PetscInt          expected_setup_count;
  const PetscInt    rows[]   = {0, 1};
  const PetscScalar matrix[] = {4.0, 1.0, 1.0, 3.0}, rhs[] = {6.0, 7.0};

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
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_explicit_setup", &test_explicit_setup, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-scale_left", &scale_left, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-scale_right", &scale_right, NULL));
  PetscCheck(scale_left || scale_right, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Enable at least one explicit diagonal scale");

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 2, 2, 2, NULL, &assembled));
  PetscCall(MatSetValues(assembled, 2, rows, 2, rows, matrix, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(assembled, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(assembled, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(assembled, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(assembled, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatSetOption(assembled, MAT_SPD, PETSC_TRUE));
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
  PetscCall(VecDuplicate(x, &left));
  PetscCall(VecDuplicate(x, &right));
  PetscCall(SetScale(left, 0.25, 3.0));
  PetscCall(SetScale(right, 2.0, 0.5));
  if (nonzero) PetscCall(VecSet(x, 0.25));
  else PetscCall(VecSet(x, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, P));
  PetscCall(KSPSetInitialGuessNonzero(ksp, nonzero));
  if (scale_left) PetscCall(KSPSetLeftDiagonalScale(ksp, left));
  if (scale_right) PetscCall(KSPSetRightDiagonalScale(ksp, right));
  PetscCall(KSPGetLeftDiagonalScale(ksp, &got));
  PetscCheck(got == (scale_left ? left : NULL), PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSPGetLeftDiagonalScale() returned the wrong vector");
  PetscCall(KSPGetRightDiagonalScale(ksp, &got));
  PetscCheck(got == (scale_right ? right : NULL), PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSPGetRightDiagonalScale() returned the wrong vector");
  PetscCall(KSPSetFromOptions(ksp));
  ctx.Aphysical = NULL;
  ctx.Pphysical = NULL;
  ctx.left      = scale_left ? left : NULL;
  ctx.right     = scale_right ? right : NULL;
  if (test_reuse) {
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &ctx.Aphysical));
    PetscCall(MatDuplicate(P, MAT_COPY_VALUES, &ctx.Pphysical));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetContext(pc, &ctx));
    PetscCall(PCShellSetApply(pc, PCApply_Identity));
    PetscCall(PCShellSetSetUp(pc, PCSetUp_Count));
  }

  ctx.A                = A;
  ctx.P                = P;
  ctx.rhs              = b;
  ctx.setup_count      = 0;
  ctx.check_residual   = PETSC_TRUE;
  ctx.check_properties = PETSC_TRUE;
  PetscCall(PetscObjectStateGet((PetscObject)A, &ctx.Astate));
  PetscCall(PetscObjectStateGet((PetscObject)P, &ctx.Pstate));
  PetscCall(PetscObjectStateGet((PetscObject)b, &ctx.rhs_state));
  PetscCall(KSPSetPreSolve(ksp, CheckPhysicalPreSolve, &ctx));
  PetscCall(KSPSetPostSolve(ksp, CheckPhysicalState, &ctx));
  if (test_explicit_setup) PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(CheckSolution(ksp, x));

  if (test_reuse) {
    expected_setup_count = 1;
    PetscCheck(ctx.setup_count == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected one PC setup, got %" PetscInt_FMT, ctx.setup_count);
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCheck(ctx.setup_count == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unchanged scaling rebuilt the PC");
    if (scale_right) {
      PetscCall(SetScale(right, 0.5, 4.0));
      PetscCall(VecSet(x, 0.0));
      PetscCall(KSPSolve(ksp, b, x));
      ++expected_setup_count;
      PetscCheck(ctx.setup_count == expected_setup_count, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Changed right scaling did not rebuild the PC");
      PetscCall(CheckSolution(ksp, x));
    }
    if (scale_left) {
      PetscCall(SetScale(left, 4.0, 0.125));
      PetscCall(VecSet(x, 0.0));
      PetscCall(KSPSolve(ksp, b, x));
      ++expected_setup_count;
      PetscCheck(ctx.setup_count == expected_setup_count, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Changed left scaling did not rebuild the PC");
      PetscCall(CheckSolution(ksp, x));
    }
    PetscCall(MatShift(P, 0.25));
    PetscCall(MatShift(ctx.Pphysical, 0.25));
    PetscCall(PetscObjectStateGet((PetscObject)P, &ctx.Pstate));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x));
    ++expected_setup_count;
    PetscCheck(ctx.setup_count == expected_setup_count, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Changed matrix did not rebuild the PC");
    PetscCall(CheckSolution(ksp, x));
  }

  PetscCall(PetscObjectStateGet((PetscObject)A, &state));
  PetscCheck(state == ctx.Astate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Physical operator state was not restored");
  PetscCall(PetscObjectStateGet((PetscObject)P, &state));
  PetscCheck(state == ctx.Pstate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Physical preconditioning matrix state was not restored");

  if (test_errors) {
    error_scale = scale_right ? right : left;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    PetscCall(SetScale(error_scale, 0.0, 1.0));
    ierr = KSPSolve(ksp, b, x);
    PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Zero scale returned error %d", (int)ierr);
    PetscCall(SetScale(error_scale, (PetscScalar)NAN, 1.0));
    ierr = KSPSolve(ksp, b, x);
    PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nonfinite scale returned error %d", (int)ierr);
    PetscCall(SetScale(error_scale, 2.0, 0.5));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, 1, &bad));
    if (scale_right) PetscCall(KSPSetRightDiagonalScale(ksp, bad));
    else PetscCall(KSPSetLeftDiagonalScale(ksp, bad));
    ierr = KSPSolve(ksp, b, x);
    PetscCheck(ierr == PETSC_ERR_ARG_SIZ, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong-sized scale returned error %d", (int)ierr);
    if (scale_right) PetscCall(KSPSetRightDiagonalScale(ksp, right));
    else PetscCall(KSPSetLeftDiagonalScale(ksp, left));
    PetscCall(VecDestroy(&bad));
    ierr = KSPSetDiagonalScale(ksp, PETSC_TRUE);
    PetscCheck(ierr == PETSC_ERR_SUP, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Combining explicit and legacy scaling returned error %d", (int)ierr);
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &legacy_ksp));
    PetscCall(KSPSetDiagonalScale(legacy_ksp, PETSC_TRUE));
    if (scale_right) ierr = KSPSetRightDiagonalScale(legacy_ksp, right);
    else ierr = KSPSetLeftDiagonalScale(legacy_ksp, left);
    PetscCheck(ierr == PETSC_ERR_SUP, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Setting explicit scaling after legacy scaling returned error %d", (int)ierr);
    PetscCall(KSPDestroy(&legacy_ksp));
    ierr = KSPSolveTranspose(ksp, b, x);
    PetscCheck(ierr == PETSC_ERR_SUP, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Transpose solve returned error %d", (int)ierr);
    PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD, 2, 1, NULL, &B));
    PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));
    ierr = KSPMatSolve(ksp, B, X);
    PetscCheck(ierr == PETSC_ERR_SUP, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Multiple-RHS solve returned error %d", (int)ierr);
    ierr = KSPMatSolveTranspose(ksp, B, X);
    PetscCheck(ierr == PETSC_ERR_SUP, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Transpose multiple-RHS solve returned error %d", (int)ierr);
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&B));
    PetscCall(PetscPopErrorHandler());
    PetscCall(CheckPhysicalState(ksp, b, NULL, &ctx));
    PetscCall(TestSBAIJError(PETSC_FALSE));
    PetscCall(TestSBAIJError(PETSC_TRUE));
  }

  PetscCall(KSPSetLeftDiagonalScale(ksp, NULL));
  PetscCall(KSPSetRightDiagonalScale(ksp, NULL));
  PetscCall(KSPGetLeftDiagonalScale(ksp, &got));
  PetscCheck(!got, PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSP left diagonal scale was not cleared");
  PetscCall(KSPGetRightDiagonalScale(ksp, &got));
  PetscCheck(!got, PETSC_COMM_SELF, PETSC_ERR_PLIB, "KSP right diagonal scale was not cleared");
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&ctx.Pphysical));
  PetscCall(MatDestroy(&ctx.Aphysical));
  PetscCall(VecDestroy(&right));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&mffd_fbase));
  PetscCall(VecDestroy(&mffd_base));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&assembled));
  if (test_snes) PetscCall(TestSNESScale());
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
    suffix: left
    output_file: output/empty.out
    args: -scale_left -scale_right false -ksp_type preonly -pc_type lu

  test:
    suffix: left_right
    output_file: output/empty.out
    args: -scale_left -distinct_pmat -nonzero_guess -ksp_type richardson -pc_type lu -ksp_max_it 1

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
    args: -mat_shell -scale_left -ksp_type gmres -pc_type jacobi -ksp_rtol 1e-12

  test:
    suffix: mffd
    output_file: output/empty.out
    args: -mat_mffd -ksp_type gmres -pc_type jacobi -ksp_rtol 1e-12

  test:
    suffix: reuse
    output_file: output/empty.out
    args: -scale_left -distinct_pmat -test_reuse -ksp_type gmres -ksp_rtol 1e-12

  test:
    suffix: errors
    output_file: output/empty.out
    args: -test_errors -ksp_type preonly -pc_type lu

  test:
    suffix: errors_left
    output_file: output/empty.out
    args: -scale_left -scale_right false -test_errors -ksp_type preonly -pc_type lu

  test:
    suffix: snes
    output_file: output/empty.out
    args: -test_snes -snes_linesearch_type bt -snes_ksp_ew -snes_ksp_ew_rtol0 0.25 -snes_rtol 1e-12

  test:
    suffix: deferred_operators
    output_file: output/empty.out
    args: -test_deferred_operators -ksp_type preonly -pc_type lu

  test:
    suffix: explicit_setup
    output_file: output/empty.out
    args: -scale_left -distinct_pmat -test_reuse -test_explicit_setup -ksp_type gmres -ksp_rtol 1e-12

TEST*/
