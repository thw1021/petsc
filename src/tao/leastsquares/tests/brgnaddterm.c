const char help[] = "Tests -tao_add_terms with a residual-callback TAOBRGN data model.\n";

#include <petsctao.h>

typedef struct {
  PetscBool nonlinear;
} AppCtx;

static PetscErrorCode MatMult_Jacobian(Mat J, Vec x, Vec y)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Residual(Tao tao, Vec x, Vec r, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, r));
  if (user->nonlinear) PetscCall(VecExp(r));
  PetscCall(VecShift(r, -2.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Jacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
  AppCtx     *user  = (AppCtx *)ctx;
  PetscScalar value = 1.0;
  PetscInt    index = 0;
  PetscBool   is_shell;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)J, MATSHELL, &is_shell));
  if (is_shell) PetscFunctionReturn(PETSC_SUCCESS);
  if (user->nonlinear) {
    PetscCall(VecGetValues(x, 1, &index, &value));
    value = PetscExpScalar(value);
  }
  PetscCall(MatSetValue(J, 0, 0, value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao         tao, subsolver;
  Vec         x, r;
  Mat         H, J;
  PetscScalar value;
  PetscInt    index          = 0;
  PetscBool   shell_jacobian = PETSC_FALSE, test_nonlinear_hessian = PETSC_FALSE, is_shell;
  AppCtx      user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-shell_jacobian", &shell_jacobian, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_nonlinear_hessian", &test_nonlinear_hessian, NULL));
  user.nonlinear = test_nonlinear_hessian;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &x));
  PetscCall(VecDuplicate(x, &r));
  if (shell_jacobian) {
    PetscCall(MatCreateShell(PETSC_COMM_SELF, 1, 1, 1, 1, NULL, &J));
    PetscCall(MatShellSetOperation(J, MATOP_MULT, (PetscErrorCodeFn *)MatMult_Jacobian));
    PetscCall(MatShellSetOperation(J, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMult_Jacobian));
  } else PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 1, 1, 1, NULL, &J));
  PetscCall(VecSet(x, 0.0));
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOBRGN));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetResidual(tao, r, Residual, &user));
  PetscCall(TaoSetJacobianResidual(tao, J, J, Jacobian, &user));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoBRGNGetSubsolver(tao, &subsolver));
  PetscCall(TaoGetHessianMatrices(subsolver, &H, NULL));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)H, MATSHELL, &is_shell));
  PetscCheck(is_shell == shell_jacobian, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected %s Hessian for %s residual Jacobian", shell_jacobian ? "a shell" : "an assembled", shell_jacobian ? "a shell" : "an assembled");
  if (test_nonlinear_hessian) {
    PetscCall(TaoComputeHessian(subsolver, x, H, H));
    PetscCall(MatGetValue(H, 0, 0, &value));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Gauss-Newton Hessian at x=0: %g\n", (double)PetscRealPart(value)));
    PetscCall(VecSet(x, PetscLogReal(2.0)));
    PetscCall(TaoComputeHessian(subsolver, x, H, H));
    PetscCall(MatGetValue(H, 0, 0, &value));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Gauss-Newton Hessian at x=log(2): %g\n", (double)PetscRealPart(value)));
  } else {
    PetscCall(TaoSolve(tao));
    PetscCall(VecGetValues(x, 1, &index, &value));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "solution after adding term: %g\n", (double)PetscRealPart(value)));
  }
  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    args: -tao_add_terms reg_ -reg_tao_term_type halfl2squared
    output_file: output/brgnaddterm.out

  test:
    suffix: shell
    args: -tao_add_terms reg_ -reg_tao_term_type halfl2squared -shell_jacobian
    output_file: output/brgnaddterm.out

  test:
    suffix: residual_count
    args: -tao_add_terms reg_ -reg_tao_term_type halfl2squared -tao_view
    filter: grep "total number of residual evaluations"
    output_file: output/brgnaddterm_residual_count.out

  test:
    suffix: nonlinear_hessian
    args: -test_nonlinear_hessian -tao_brgn_regularizer_weight 0

TEST*/
