#include <petsctao.h>

static char help[] = "Tests callback-term promotion with separate Hpre and selection between assembled Hessian and HessianMult callbacks.\n";

typedef struct {
  PetscInt hessian_calls;
  PetscInt mult_calls;
} TermCtx;

static PetscErrorCode FormObjectiveAndGradientTao(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  PetscFunctionBeginUser;
  *f = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormSquareObjectiveAndGradientTao(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecDotRealPart(x, x, f));
  PetscCall(VecCopy(x, g));
  PetscCall(VecScale(g, 2.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetDiagonal(Mat H, PetscScalar value)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(H));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(H, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianTao(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(SetDiagonal(H, 2.0));
  if (Hpre != H) PetscCall(SetDiagonal(Hpre, 3.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormObjectiveAndGradientTerm(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  PetscFunctionBeginUser;
  *f = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianTerm(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  ctx->hessian_calls++;
  if (H) PetscCall(SetDiagonal(H, 5.0));
  if (Hpre && Hpre != H) PetscCall(SetDiagonal(Hpre, 7.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianSelection(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  ctx->hessian_calls++;
  if (H) PetscCall(SetDiagonal(H, 2.0));
  if (Hpre && Hpre != H) PetscCall(SetDiagonal(Hpre, 4.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianMultSelection(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  ctx->mult_calls++;
  PetscCall(VecCopy(v, Hv));
  PetscCall(VecScale(Hv, 3.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateAIJ(MPI_Comm comm, PetscInt n, Mat *H)
{
  PetscFunctionBeginUser;
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 1, NULL, H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckActionTolerance(Mat H, Vec v, Vec work, PetscReal expected, PetscReal tolerance)
{
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(MatMult(H, v, work));
  PetscCall(VecAXPY(work, -expected, v));
  PetscCall(VecNorm(work, NORM_2, &norm));
  PetscCheck(norm <= tolerance, PetscObjectComm((PetscObject)H), PETSC_ERR_PLIB, "Hessian action has error norm %g", (double)norm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckAction(Mat H, Vec v, Vec work, PetscReal expected)
{
  PetscFunctionBeginUser;
  PetscCall(CheckActionTolerance(H, v, work, expected, 1.e-10));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCallbackPromotion(MPI_Comm comm)
{
  const PetscInt n = 4;
  TermCtx       ctx = {0};
  Tao           tao;
  TaoTerm       term;
  Mat           callback_H, callback_Hpre, H, Hpre;
  Vec           x, v, work;

  PetscFunctionBeginUser;
  PetscCall(CreateAIJ(comm, n, &callback_H));
  PetscCall(CreateAIJ(comm, n, &callback_Hpre));
  PetscCall(MatCreateVecs(callback_H, &x, NULL));
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &work));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecSet(v, 1.0));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormObjectiveAndGradientTao, NULL));
  PetscCall(TaoSetHessian(tao, callback_H, callback_Hpre, FormHessianTao, NULL));
  PetscCall(TaoTermCreateShell(comm, &ctx, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradientTerm));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term, PETSC_FALSE, MATAIJ, MATAIJ));
  PetscCall(TaoTermShellSetHessian(term, FormHessianTerm));
  PetscCall(TaoAddTerm(tao, "added_", 1.0, term, NULL, NULL));
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCheck(H != Hpre, comm, PETSC_ERR_PLIB, "Callback promotion lost the separate Hpre configuration");
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(CheckAction(H, v, work, 7.0));
  PetscCall(CheckAction(Hpre, v, work, 10.0));
  PetscCheck(ctx.hessian_calls == 1, comm, PETSC_ERR_PLIB, "Added term Hessian callback was called %" PetscInt_FMT " times", ctx.hessian_calls);
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(MatDestroy(&callback_H));
  PetscCall(MatDestroy(&callback_Hpre));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCallbackSelection(MPI_Comm comm, PetscBool use_shell)
{
  const PetscInt n = 4;
  TermCtx       ctx = {0};
  Tao           tao;
  TaoTerm       term;
  Mat           map, H, Hpre;
  Vec           x, v, work;
  PetscReal     norm;
  PetscInt      rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(CreateAIJ(comm, n, &map));
  PetscCall(MatGetOwnershipRange(map, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(MatSetValue(map, i, i, i + 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(map, &x, NULL));
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &work));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecSet(v, 1.0));
  PetscCall(TaoTermCreateShell(comm, &ctx, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradientTerm));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term, use_shell, use_shell ? MATSHELL : MATAIJ, use_shell ? NULL : MATAIJ));
  PetscCall(TaoTermShellSetHessian(term, FormHessianSelection));
  PetscCall(TaoTermShellSetHessianMult(term, FormHessianMultSelection));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoAddTerm(tao, NULL, 1.0, term, NULL, map));
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(MatMult(H, v, work));
  {
    PetscScalar *a;

    PetscCall(VecGetArray(work, &a));
    for (PetscInt i = rstart; i < rend; i++) a[i - rstart] -= (use_shell ? 3.0 : 2.0) * (i + 1.0) * (i + 1.0);
    PetscCall(VecRestoreArray(work, &a));
  }
  PetscCall(VecNorm(work, NORM_2, &norm));
  PetscCheck(norm <= 1.e-10, comm, PETSC_ERR_PLIB, "Mapped selected Hessian action has error norm %g", (double)norm);
  if (use_shell) PetscCheck(ctx.mult_calls > 0 && ctx.hessian_calls == 0, comm, PETSC_ERR_PLIB, "MATSHELL did not select HessianMult exclusively");
  else {
    PetscCheck(H != Hpre, comm, PETSC_ERR_PLIB, "Mapped assembled term lost its separate Hpre");
    PetscCall(MatMult(Hpre, v, work));
    {
      PetscScalar *a;

      PetscCall(VecGetArray(work, &a));
      for (PetscInt i = rstart; i < rend; i++) a[i - rstart] -= 4.0 * (i + 1.0) * (i + 1.0);
      PetscCall(VecRestoreArray(work, &a));
    }
    PetscCall(VecNorm(work, NORM_2, &norm));
    PetscCheck(norm <= 1.e-10, comm, PETSC_ERR_PLIB, "Mapped separate Hpre action has error norm %g", (double)norm);
    PetscCheck(ctx.hessian_calls == 1 && ctx.mult_calls == 0, comm, PETSC_ERR_PLIB, "Assembled Hessian did not select the Hessian callback exclusively");
  }
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(MatDestroy(&map));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCallbackMFFD(MPI_Comm comm)
{
  const PetscInt n = 4;
  Tao           tao;
  Mat           H, Hpre;
  Vec           x, v, work;

  PetscFunctionBeginUser;
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, n, &x));
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &work));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecSet(v, 1.0));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormSquareObjectiveAndGradientTao, NULL));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(CheckActionTolerance(H, v, work, 2.0, 1.e-7));
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscBool callback_promotion = PETSC_FALSE, callback_mffd = PETSC_FALSE, assembled = PETSC_FALSE, shell = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-callback_promotion", &callback_promotion, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-callback_mffd", &callback_mffd, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-assembled", &assembled, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-shell", &shell, NULL));
  PetscCheck((PetscInt)callback_promotion + (PetscInt)callback_mffd + (PetscInt)assembled + (PetscInt)shell == 1, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Select exactly one test case");
  if (callback_promotion) PetscCall(TestCallbackPromotion(PETSC_COMM_WORLD));
  else if (callback_mffd) PetscCall(TestCallbackMFFD(PETSC_COMM_WORLD));
  else PetscCall(TestCallbackSelection(PETSC_COMM_WORLD, shell));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "TaoTerm Hessian callback dispatch is correct\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/taotermtest11.out
    requires: !complex

    test:
      suffix: callback_promotion
      args: -callback_promotion

    test:
      suffix: assembled_selection
      args: -assembled

    test:
      suffix: callback_mffd
      args: -callback_mffd -tao_mf_hessian

    test:
      suffix: shell_selection
      args: -shell

TEST*/
