static char help[] = "Tests SNES Eisenstat-Walker with KSP pre-solve callbacks and the pre-solve matrix modification check.\n\n";

#include <petscsnes.h>

typedef struct {
  PetscReal expected_rtol;
  PetscInt  presolve_count;
  PetscBool modify_amat, modify_pmat, replace_amat;
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
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatSetValue(J, 0, 0, 2.0 * xa[0], INSERT_VALUES));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
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
  PetscCheck(ctx->presolve_count || PetscAbsReal(rtol - ctx->expected_rtol) <= PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "User pre-solve callback ran before Eisenstat-Walker updated KSP rtol: expected %g, got %g", (double)ctx->expected_rtol, (double)rtol);
  ++ctx->presolve_count;
  if (ctx->modify_amat || ctx->modify_pmat) {
    Mat Amat, Pmat;

    PetscCall(KSPGetOperators(ksp, &Amat, &Pmat));
    if (ctx->modify_amat) PetscCall(MatScale(Amat, 2.0));
    if (ctx->modify_pmat) PetscCall(MatScale(Pmat, 2.0));
  }
  if (ctx->replace_amat) {
    Mat Amat, Pmat, Amat_new;

    PetscCall(KSPGetOperators(ksp, &Amat, &Pmat));
    PetscCall(MatDuplicate(Amat, MAT_COPY_VALUES, &Amat_new));
    PetscCall(KSPSetOperators(ksp, Amat_new, Pmat));
    PetscCall(MatDestroy(&Amat_new));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES                snes;
  KSP                 ksp;
  PC                  pc;
  Mat                 J, P;
  Vec                 x, f;
  SNESConvergedReason reason;
  CallbackCtx         ctx = {.expected_rtol = 0.25, .presolve_count = 0, .modify_amat = PETSC_FALSE, .modify_pmat = PETSC_FALSE, .replace_amat = PETSC_FALSE};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-modify_amat", &ctx.modify_amat, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-modify_pmat", &ctx.modify_pmat, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-replace_amat", &ctx.replace_amat, NULL));
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, 1, &x));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 1, 1, 1, NULL, &J));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 1, 1, 1, NULL, &P));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, f, FormFunction, NULL));
  PetscCall(SNESSetJacobian(snes, J, P, FormJacobian, NULL));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetApply(pc, PCApply_Identity));
  PetscCall(KSPSetPreSolve(ksp, CheckPreSolve, &ctx));
  PetscCall(VecSet(x, 3.0));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  PetscCheck(reason > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNESSolve() did not converge, reason %s", SNESConvergedReasons[reason]);
  PetscCheck(ctx.presolve_count > 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "User pre-solve callback ran %" PetscInt_FMT " time(s); expected more than one SNES iteration to exercise the Eisenstat-Walker rtol update", ctx.presolve_count);
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&P));
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

  testset:
    # Testing errors so only look for errors
    requires: !defined(PETSCTEST_VALGRIND) !defined(PETSC_HAVE_SANITIZER)
    args: -snes_linesearch_type bt -snes_ksp_ew -snes_ksp_ew_rtol0 0.25 -snes_rtol 1e-12 -petsc_ci_portable_error_output -error_output_stdout
    filter: grep -E "(PETSC ERROR)" | grep -E "(modified the KSP|KSPPreSolve)"
    test:
      suffix: 2
      args: -modify_amat
    test:
      suffix: 3
      args: -modify_pmat
    test:
      suffix: 4
      args: -replace_amat

TEST*/
