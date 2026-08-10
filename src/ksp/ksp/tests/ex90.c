static char help[] = "Tests KSP pre-solve callbacks with SNES Eisenstat-Walker and the pre-solve matrix modification check.\n\n";

#include <petscsnes.h>
typedef struct {
  PetscReal expected_rtol;
  PetscInt  presolve_count;
  PetscBool modify_matrix;
} CallbackCtx;

static PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, PetscCtx ctx)
{
  const PetscScalar *xa;
  PetscScalar       *fa;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(f, &fa));
  fa[0] = xa[0] * xa[0] - 4.0;
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat P, PetscCtx ctx)
{
  const PetscScalar *xa;

  PetscFunctionBeginUser;
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

static PetscErrorCode PCApply_Identity(PC pc, Vec x, Vec y)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPreSolve(KSP ksp, Vec b, Vec x, PetscCtx vctx)
{
  CallbackCtx *ctx = (CallbackCtx *)vctx;
  PetscReal    rtol;

  PetscFunctionBeginUser;
  PetscCall(KSPGetTolerances(ksp, &rtol, NULL, NULL, NULL));
  if (!ctx->presolve_count)
    PetscCheck(PetscAbsReal(rtol - ctx->expected_rtol) <= PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "User pre-solve callback ran before Eisenstat-Walker updated KSP rtol: expected %g, got %g", (double)ctx->expected_rtol, (double)rtol);
  ++ctx->presolve_count;
  if (ctx->modify_matrix) {
    Mat P;

    PetscCall(KSPGetOperators(ksp, NULL, &P));
    PetscCall(MatScale(P, 2.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES        snes;
  KSP         ksp;
  PC          pc;
  Mat         J;
  Vec         x, f;
  CallbackCtx ctx = {.expected_rtol = 0.25, .presolve_count = 0, .modify_matrix = PETSC_FALSE};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-modify_matrix", &ctx.modify_matrix, NULL));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, 1, &x));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 1, 1, 1, NULL, &J));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, f, FormFunction, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, NULL));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, &ctx));
  PetscCall(PCShellSetApply(pc, PCApply_Identity));
  PetscCall(KSPSetPreSolve(ksp, CheckPreSolve, &ctx));
  PetscCall(VecSet(x, 3.0));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCheck(ctx.presolve_count, PETSC_COMM_SELF, PETSC_ERR_PLIB, "User pre-solve callback was not called");
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    output_file: output/empty.out
    args: -snes_linesearch_type bt -snes_ksp_ew -snes_ksp_ew_rtol0 0.25 -snes_rtol 1e-12

  test:
    # Testing errors so only look for errors
    suffix: 2
    requires: !defined(PETSCTEST_VALGRIND) !defined(PETSC_HAVE_SANITIZER)
    args: -snes_linesearch_type bt -snes_ksp_ew -snes_ksp_ew_rtol0 0.25 -snes_rtol 1e-12 -modify_matrix -petsc_ci_portable_error_output -error_output_stdout
    filter: grep -E "(PETSC ERROR)" | grep -E "(modified the KSP|KSPPreSolve)"

TEST*/
