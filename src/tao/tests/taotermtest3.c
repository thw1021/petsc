#include <petsctao.h>
#include "taotermtestclassic.h"

static char help[] = "Resolve a regularized least-squares problem after updating its observation operator.\n";

static PetscErrorCode FormObjectiveGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode SetMap(Mat, PetscBool, PetscBool);
static PetscErrorCode SetTarget(Vec, PetscReal);
static PetscErrorCode CheckHessianAction(Tao, Mat, PetscReal, Vec);
static PetscErrorCode CheckSolution(Mat, Vec, PetscReal, Vec);

int main(int argc, char **argv)
{
  const PetscInt    n      = 8;
  const PetscReal   lambda = 0.2;
  Tao               tao, ctao;
  TaoTerm           data, regularizer;
  ExampleClassicCtx cctx = {0};
  Mat               A, cH;
  Vec               b, x, cx;
  PetscBool         change_structure = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-change_structure", &change_structure, NULL));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, 2, NULL, 2, NULL, &A));
  PetscCall(SetMap(A, PETSC_FALSE, PETSC_FALSE));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(SetTarget(b, 1.0));
  PetscCall(VecZeroEntries(x));

  PetscCall(TaoTermCreateShell(PETSC_COMM_WORLD, NULL, NULL, &data));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data, "data_"));
  PetscCall(TaoTermSetSolutionSizes(data, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetParametersSizes(data, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(data, FormObjectiveGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(data, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(data, PETSC_TRUE, MATAIJ, NULL));
  PetscCall(TaoTermShellSetHessian(data, FormHessian));
  PetscCall(TaoTermSetFromOptions(data));

  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &regularizer));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)regularizer, "reg_"));
  PetscCall(TaoTermSetSolutionSizes(regularizer, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetType(regularizer, TAOTERMHALFL2SQUARED));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoAddTerm(tao, "data_", 1.0, data, b, A));
  PetscCall(TaoAddTerm(tao, "reg_", lambda, regularizer, NULL, NULL));
  PetscCall(TaoSetFromOptions(tao));
  cctx.nleaves              = 2;
  cctx.leaves[0].type       = EXAMPLE_CLASSIC_LEAST_SQUARES;
  cctx.leaves[0].map        = A;
  cctx.leaves[0].parameters = b;
  cctx.leaves[0].scale      = 1.0;
  cctx.leaves[1].type       = EXAMPLE_CLASSIC_HALF_L2;
  cctx.leaves[1].scale      = lambda;
  PetscCall(VecDuplicate(x, &cx));
  PetscCall(VecCopy(x, cx));
  PetscCall(ExampleClassicCreateTao(PETSC_COMM_WORLD, tao, cx, &cctx, &ctao, &cH));

  PetscCall(TaoSolve(tao));
  PetscCall(TaoSolve(ctao));
  PetscCall(ExampleClassicCompareResults(tao, x, ctao, cx));
  PetscCall(CheckSolution(A, b, lambda, x));
  PetscCall(CheckHessianAction(tao, A, lambda, x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial least-squares solution check passed\n"));

  PetscCall(SetMap(A, PETSC_TRUE, change_structure));
  PetscCall(SetTarget(b, 1.5));
  PetscCall(CheckHessianAction(tao, A, lambda, x));
  PetscCall(VecZeroEntries(x));
  PetscCall(VecZeroEntries(cx));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoSolve(ctao));
  PetscCall(ExampleClassicCompareResults(tao, x, ctao, cx));
  PetscCall(CheckSolution(A, b, lambda, x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Updated least-squares solution check passed\n"));
  PetscCall(TaoViewFromOptions(tao, NULL, "-tao_view"));

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoDestroy(&ctao));
  PetscCall(TaoTermDestroy(&data));
  PetscCall(TaoTermDestroy(&regularizer));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&cx));
  PetscCall(MatDestroy(&cH));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode FormObjectiveGradient(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecWAXPY(g, -1.0, params, x));
  PetscCall(VecDot(g, g, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(H));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(H, 1.0));
  if (Hpre && Hpre != H) PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetMap(Mat A, PetscBool update, PetscBool change_structure)
{
  PetscInt rstart, rend, n;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatGetSize(A, &n, NULL));
  if (change_structure) PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatZeroEntries(A));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscCall(MatSetValue(A, i, i, (update ? 1.5 : 1.2) + 0.1 * (i + 1), INSERT_VALUES));
    if (change_structure) PetscCall(MatSetValue(A, i, (i + 1) % n, 0.15, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetTarget(Vec b, PetscReal scale)
{
  PetscInt     rstart, rend;
  PetscScalar *a;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(b, &rstart, &rend));
  PetscCall(VecGetArrayWrite(b, &a));
  for (PetscInt i = rstart; i < rend; i++) a[i - rstart] = scale * (1.0 + 0.05 * (i + 1));
  PetscCall(VecRestoreArrayWrite(b, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckHessianAction(Tao tao, Mat A, PetscReal lambda, Vec x)
{
  Mat       H, Hpre;
  Vec       v, Av, actual, expected;
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, &v, &Av));
  PetscCall(VecDuplicate(v, &actual));
  PetscCall(VecDuplicate(v, &expected));
  PetscCall(VecSet(v, 1.0));
  PetscCall(MatMult(A, v, Av));
  PetscCall(MatMultTranspose(A, Av, expected));
  PetscCall(VecAXPY(expected, lambda, v));
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(MatMult(H, v, actual));
  PetscCall(VecAXPY(actual, -1.0, expected));
  PetscCall(VecNorm(actual, NORM_2, &error));
  PetscCheck(error <= 1.e-9, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Hessian action after updating the map differs from the current mapped operator by %g", (double)error);
  PetscCall(VecDestroy(&expected));
  PetscCall(VecDestroy(&actual));
  PetscCall(VecDestroy(&Av));
  PetscCall(VecDestroy(&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSolution(Mat A, Vec b, PetscReal lambda, Vec x)
{
  KSP       ksp;
  PC        pc;
  Mat       normal;
  Vec       rhs, expected;
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(MatTransposeMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &normal));
  PetscCall(MatShift(normal, lambda));
  PetscCall(MatCreateVecs(A, &rhs, NULL));
  PetscCall(VecDuplicate(rhs, &expected));
  PetscCall(MatMultTranspose(A, b, rhs));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)A), &ksp));
  PetscCall(KSPSetOperators(ksp, normal, normal));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, expected));
  PetscCall(VecAXPY(expected, -1.0, x));
  PetscCall(VecNorm(expected, NORM_2, &error));
  PetscCheck(error <= 1.e-7, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Reused Tao solution differs from freshly assembled normal equations by %g", (double)error);
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&normal));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: map_values
    args: -tao_type nls -tao_view ::ascii_info_detail
    filter: grep -E "solution check passed|Hessian (preconditioning )?MatType|rows=.*cols=|Solution converged"

  test:
    suffix: map_structure
    args: -change_structure -tao_type nls -tao_view ::ascii_info_detail
    filter: grep -E "solution check passed|Hessian (preconditioning )?MatType|rows=.*cols=|Solution converged"
    output_file: output/taotermtest3_map_values.out

  test:
    suffix: map_values_shell
    args: -tao_type nls -tao_term_hessian_mat_type shell
    filter: grep -E "solution check passed"
    output_file: output/taotermtest3_map_values_shell.out

TEST*/
