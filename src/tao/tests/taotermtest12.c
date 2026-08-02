#include <petsctao.h>

static char help[] = "Tests TaoTerm masks, zero scales, parameter-mode derivation, finite-difference gradients, and unmapped built-in terms.\n";

typedef struct {
  PetscReal value;
} TermCtx;

static PetscErrorCode Objective(TaoTerm term, Vec x, Vec params, PetscReal *f)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  *f = ctx->value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Gradient(TaoTerm term, Vec x, Vec params, Vec g)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  PetscCall(VecSet(g, ctx->value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Hessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(H, ctx->value));
  }
  if (Hpre && Hpre != H) PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SquareObjective(TaoTerm term, Vec x, Vec params, PetscReal *f)
{
  PetscFunctionBeginUser;
  PetscCall(VecDotRealPart(x, x, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateShell(MPI_Comm comm, PetscInt n, TermCtx *ctx, TaoTermParametersMode mode, TaoTerm *term)
{
  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, ctx, NULL, term));
  PetscCall(TaoTermSetParametersMode(*term, mode));
  PetscCall(TaoTermSetSolutionSizes(*term, PETSC_DECIDE, n, 1));
  if (mode != TAOTERM_PARAMETERS_NONE) PetscCall(TaoTermSetParametersSizes(*term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjective(*term, Objective));
  PetscCall(TaoTermShellSetGradient(*term, Gradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(*term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(*term, PETSC_TRUE, MATAIJ, NULL));
  PetscCall(TaoTermShellSetHessian(*term, Hessian));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckConstantVec(Vec v, PetscReal expected)
{
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(VecShift(v, -expected));
  PetscCall(VecNorm(v, NORM_2, &norm));
  PetscCheck(norm <= 1.e-10, PetscObjectComm((PetscObject)v), PETSC_ERR_PLIB, "Vector has error norm %g", (double)norm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMasksAndZeroScale(MPI_Comm comm)
{
  const TaoTermMask masks[] = {TAOTERM_MASK_OBJECTIVE, TAOTERM_MASK_GRADIENT, TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_GRADIENT, TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_HESSIAN, TAOTERM_MASK_GRADIENT | TAOTERM_MASK_HESSIAN, TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_GRADIENT | TAOTERM_MASK_HESSIAN};
  const PetscInt    n       = 3;
  TermCtx           ctx[2]  = {{1.0}, {2.0}};
  TaoTerm           term[2];

  PetscFunctionBeginUser;
  PetscCall(CreateShell(comm, n, &ctx[0], TAOTERM_PARAMETERS_NONE, &term[0]));
  PetscCall(CreateShell(comm, n, &ctx[1], TAOTERM_PARAMETERS_NONE, &term[1]));
  for (PetscInt k = 0; k < PETSC_STATIC_ARRAY_LENGTH(masks) + 1; k++) {
    TaoTerm  sum;
    Mat      H, Hpre;
    Vec      x, g, v;
    PetscReal f;
    PetscReal first_scale = k == PETSC_STATIC_ARRAY_LENGTH(masks) ? 0.0 : 1.0;
    PetscBool obj_masked = k < PETSC_STATIC_ARRAY_LENGTH(masks) && (masks[k] & TAOTERM_MASK_OBJECTIVE);
    PetscBool grad_masked = k < PETSC_STATIC_ARRAY_LENGTH(masks) && (masks[k] & TAOTERM_MASK_GRADIENT);
    PetscBool hess_masked = k < PETSC_STATIC_ARRAY_LENGTH(masks) && (masks[k] & TAOTERM_MASK_HESSIAN);

    PetscCall(TaoTermCreate(comm, &sum));
    PetscCall(TaoTermSetType(sum, TAOTERMSUM));
    PetscCall(TaoTermSumAddTerm(sum, "first_", first_scale, term[0], NULL, NULL));
    PetscCall(TaoTermSumAddTerm(sum, "second_", 1.0, term[1], NULL, NULL));
    if (k < PETSC_STATIC_ARRAY_LENGTH(masks)) PetscCall(TaoTermSumSetTermMask(sum, 0, masks[k]));
    PetscCall(TaoTermSetUp(sum));
    PetscCall(TaoTermCreateSolutionVec(sum, &x));
    PetscCall(VecDuplicate(x, &g));
    PetscCall(VecDuplicate(x, &v));
    PetscCall(VecSet(x, 1.0));
    PetscCall(VecSet(v, 1.0));
    PetscCall(TaoTermComputeObjectiveAndGradient(sum, x, NULL, &f, g));
    PetscCheck(PetscAbsReal(f - (2.0 + (obj_masked ? 0.0 : first_scale))) <= 1.e-10, comm, PETSC_ERR_PLIB, "Masked objective is incorrect");
    PetscCall(CheckConstantVec(g, 2.0 + (grad_masked ? 0.0 : first_scale)));
    PetscCall(TaoTermCreateHessianMatrices(sum, &H, &Hpre));
    PetscCall(TaoTermComputeHessian(sum, x, NULL, H, Hpre));
    PetscCall(MatMult(H, v, g));
    PetscCall(CheckConstantVec(g, 2.0 + (hess_masked ? 0.0 : first_scale)));
    PetscCall(MatDestroy(&H));
    PetscCall(MatDestroy(&Hpre));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&g));
    PetscCall(VecDestroy(&v));
    PetscCall(TaoTermDestroy(&sum));
  }
  PetscCall(TaoTermDestroy(&term[0]));
  PetscCall(TaoTermDestroy(&term[1]));
  PetscCall(PetscPrintf(comm, "TaoTerm masks and zero-scale contributions are correct\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSumMode(MPI_Comm comm, TaoTerm a, TaoTerm b, TaoTermParametersMode expected)
{
  TaoTerm               sum;
  TaoTermParametersMode mode;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(sum, NULL, 1.0, a, NULL, NULL));
  PetscCall(TaoTermSumAddTerm(sum, NULL, 1.0, b, NULL, NULL));
  PetscCall(TaoTermSetUp(sum));
  PetscCall(TaoTermGetParametersMode(sum, &mode));
  PetscCheck(mode == expected, comm, PETSC_ERR_PLIB, "TAOTERMSUM derived parameter mode %d instead of %d", (int)mode, (int)expected);
  PetscCall(TaoTermDestroy(&sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestParameterModesAndFD(MPI_Comm comm)
{
  const PetscInt n = 3;
  TermCtx       ctx = {1.0};
  TaoTerm       none[2], optional[2], fd;
  Vec           x, g;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < 2; i++) {
    PetscCall(CreateShell(comm, n, &ctx, TAOTERM_PARAMETERS_NONE, &none[i]));
    PetscCall(CreateShell(comm, n, &ctx, TAOTERM_PARAMETERS_OPTIONAL, &optional[i]));
  }
  PetscCall(CheckSumMode(comm, none[0], none[1], TAOTERM_PARAMETERS_NONE));
  PetscCall(CheckSumMode(comm, optional[0], optional[1], TAOTERM_PARAMETERS_OPTIONAL));
  PetscCall(CheckSumMode(comm, none[0], optional[0], TAOTERM_PARAMETERS_OPTIONAL));
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &fd));
  PetscCall(TaoTermSetParametersMode(fd, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(fd, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjective(fd, SquareObjective));
  PetscCall(TaoTermComputeGradientSetUseFD(fd, PETSC_TRUE));
  PetscCall(TaoTermSetUp(fd));
  PetscCall(TaoTermCreateSolutionVec(fd, &x));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoTermComputeGradient(fd, x, NULL, g));
  PetscCall(CheckConstantVec(g, 2.0));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&g));
  PetscCall(TaoTermDestroy(&fd));
  for (PetscInt i = 0; i < 2; i++) {
    PetscCall(TaoTermDestroy(&none[i]));
    PetscCall(TaoTermDestroy(&optional[i]));
  }
  PetscCall(PetscPrintf(comm, "TAOTERMSUM parameter modes and finite-difference gradients are correct\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CompareTranslatedTerm(TaoTerm term, Vec x, Vec p, Vec shifted)
{
  Mat       H0, Hpre0, H1, Hpre1;
  Vec       g0, g1, d0, d1;
  PetscReal f0, f1, norm;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &g0));
  PetscCall(VecDuplicate(x, &g1));
  PetscCall(VecDuplicate(x, &d0));
  PetscCall(VecDuplicate(x, &d1));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, x, p, &f0, g0));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, shifted, NULL, &f1, g1));
  PetscCheck(PetscAbsReal(f0 - f1) <= 1.e-10, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "Translated objective values differ");
  PetscCall(VecAXPY(g0, -1.0, g1));
  PetscCall(VecNorm(g0, NORM_2, &norm));
  PetscCheck(norm <= 1.e-10, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "Translated gradients differ");
  PetscCall(TaoTermCreateHessianMatrices(term, &H0, &Hpre0));
  PetscCall(TaoTermCreateHessianMatrices(term, &H1, &Hpre1));
  PetscCall(TaoTermComputeHessian(term, x, p, H0, Hpre0));
  PetscCall(TaoTermComputeHessian(term, shifted, NULL, H1, Hpre1));
  PetscCall(MatGetDiagonal(H0, d0));
  PetscCall(MatGetDiagonal(H1, d1));
  PetscCall(VecAXPY(d0, -1.0, d1));
  PetscCall(VecNorm(d0, NORM_2, &norm));
  PetscCheck(norm <= 1.e-10, PetscObjectComm((PetscObject)term), PETSC_ERR_PLIB, "Translated Hessians differ");
  PetscCall(MatDestroy(&H0));
  PetscCall(MatDestroy(&Hpre0));
  PetscCall(MatDestroy(&H1));
  PetscCall(MatDestroy(&Hpre1));
  PetscCall(VecDestroy(&g0));
  PetscCall(VecDestroy(&g1));
  PetscCall(VecDestroy(&d0));
  PetscCall(VecDestroy(&d1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestBuiltins(MPI_Comm comm)
{
  const PetscInt n = 3;
  TaoTerm       l1, quadratic;
  Mat           A;
  Vec           x, p, shifted;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateL1(comm, PETSC_DECIDE, n, 0.5, &l1));
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 1, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(A, 2.0));
  PetscCall(TaoTermCreateQuadratic(A, &quadratic));
  PetscCall(TaoTermSetUp(l1));
  PetscCall(TaoTermSetUp(quadratic));
  PetscCall(TaoTermCreateSolutionVec(l1, &x));
  PetscCall(VecDuplicate(x, &p));
  PetscCall(VecDuplicate(x, &shifted));
  PetscCall(VecSet(x, 2.0));
  PetscCall(VecSet(p, 0.5));
  PetscCall(VecWAXPY(shifted, -1.0, p, x));
  PetscCall(CompareTranslatedTerm(l1, x, p, shifted));
  PetscCall(CompareTranslatedTerm(quadratic, x, p, shifted));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&shifted));
  PetscCall(TaoTermDestroy(&l1));
  PetscCall(TaoTermDestroy(&quadratic));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscPrintf(comm, "Unmapped smooth-L1 and quadratic terms handle absent and present parameters correctly\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(TestMasksAndZeroScale(PETSC_COMM_WORLD));
  PetscCall(TestParameterModesAndFD(PETSC_COMM_WORLD));
  PetscCall(TestBuiltins(PETSC_COMM_WORLD));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: !complex !single

TEST*/
