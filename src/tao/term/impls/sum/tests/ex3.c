const char help[] = "Equivalence test: hand-coded TaoSetObjectiveAndGradient + TaoSetHessian vs.\n"
                    "TaoAddTerm-based sum, on a bound-constrained quadratic + halfl2squared problem.\n"
                    "Both paths must produce the same final solution and the same iteration count.\n";

#include <petsctao.h>
#include <petsctaoterm.h>

typedef struct {
  Mat       A;       /* I or other SPD */
  Vec       p;       /* halfl2squared parameter */
  PetscReal lambda;  /* halfl2squared scale */
  Vec       work;    /* x - p */
} Baseline;

static PetscErrorCode BaselineObjGrad(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  Baseline *b = (Baseline *)ctx;
  PetscReal fa, fr, dot;

  PetscFunctionBeginUser;
  /* quadratic: 1/2 x^T A x, gradient A x */
  PetscCall(MatMult(b->A, x, g));
  PetscCall(VecDot(g, x, &dot));
  fa = 0.5 * dot;

  /* halfl2squared: lambda/2 ||x - p||^2, gradient lambda * (x - p) */
  PetscCall(VecWAXPY(b->work, -1.0, b->p, x));
  PetscCall(VecDot(b->work, b->work, &fr));
  fr *= 0.5 * b->lambda;
  PetscCall(VecAXPY(g, b->lambda, b->work));

  *f = fa + fr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BaselineHess(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  Baseline *b = (Baseline *)ctx;

  PetscFunctionBeginUser;
  /* H = A + lambda * I */
  PetscCall(MatCopy(b->A, H, SAME_NONZERO_PATTERN));
  PetscCall(MatShift(H, b->lambda));
  if (Hpre != H) PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RunBaseline(MPI_Comm comm, TaoType type, PetscInt n, PetscReal lambda, Vec x_out, PetscInt *iters)
{
  Tao       tao;
  Baseline  b;
  Mat       A, H;
  Vec       x, xl, xu;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(A, 1.0));

  PetscCall(MatCreateVecs(A, &x, NULL));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecDuplicate(x, &b.p));
  PetscCall(VecDuplicate(x, &b.work));
  PetscCall(VecSet(xl, -1.0));
  PetscCall(VecSet(xu, 1.0));
  PetscCall(VecSet(b.p, 0.5));
  PetscCall(VecSet(x, 0.8));
  b.A      = A;
  b.lambda = lambda;

  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &H));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, type));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetVariableBounds(tao, xl, xu));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, BaselineObjGrad, &b));
  PetscCall(TaoSetHessian(tao, H, H, BaselineHess, &b));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetIterationNumber(tao, iters));
  PetscCall(VecCopy(x, x_out));

  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&b.p));
  PetscCall(VecDestroy(&b.work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RunSum(MPI_Comm comm, TaoType type, PetscInt n, PetscReal lambda, Vec x_out, PetscInt *iters)
{
  Tao     tao;
  TaoTerm t_quad, t_reg;
  Mat     A;
  Vec     x, xl, xu, p;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(A, 1.0));

  PetscCall(MatCreateVecs(A, &x, NULL));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecDuplicate(x, &p));
  PetscCall(VecSet(xl, -1.0));
  PetscCall(VecSet(xu, 1.0));
  PetscCall(VecSet(p, 0.5));
  PetscCall(VecSet(x, 0.8));

  PetscCall(TaoTermCreateQuadratic(A, &t_quad));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &t_reg));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, type));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetVariableBounds(tao, xl, xu));
  PetscCall(TaoAddTerm(tao, "quad_", 1.0, t_quad, NULL, NULL));
  PetscCall(TaoAddTerm(tao, "reg_", lambda, t_reg, p, NULL));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetIterationNumber(tao, iters));
  PetscCall(VecCopy(x, x_out));

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&t_quad));
  PetscCall(TaoTermDestroy(&t_reg));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm  comm;
  PetscInt  n              = 12;
  PetscReal lambda         = 0.1;
  PetscReal sol_tol        = 1.e-8;
  PetscBool require_iter_match = PETSC_TRUE;
  char      tao_type[64]   = TAOBNLS;
  Vec       x_baseline, x_sum, diff;
  PetscInt  iters_baseline, iters_sum;
  PetscReal err;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, NULL, "ex3 options", NULL);
  PetscCall(PetscOptionsInt("-n", "Problem size", "ex3", n, &n, NULL));
  PetscCall(PetscOptionsReal("-lambda", "Regulariser scale", "ex3", lambda, &lambda, NULL));
  PetscCall(PetscOptionsReal("-sol_tol", "Tolerance on ||x_baseline - x_sum||_inf", "ex3", sol_tol, &sol_tol, NULL));
  PetscCall(PetscOptionsString("-solver_type", "TaoType to test", "ex3", tao_type, tao_type, sizeof(tao_type), NULL));
  PetscCall(PetscOptionsBool("-require_iter_match", "Require equal iteration counts in baseline vs sum runs", "ex3", require_iter_match, &require_iter_match, NULL));
  PetscOptionsEnd();

  PetscCall(VecCreate(comm, &x_baseline));
  PetscCall(VecSetSizes(x_baseline, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x_baseline));
  PetscCall(VecDuplicate(x_baseline, &x_sum));

  PetscCall(RunBaseline(comm, tao_type, n, lambda, x_baseline, &iters_baseline));
  PetscCall(RunSum(comm, tao_type, n, lambda, x_sum, &iters_sum));

  PetscCall(VecDuplicate(x_baseline, &diff));
  PetscCall(VecWAXPY(diff, -1.0, x_baseline, x_sum));
  PetscCall(VecNorm(diff, NORM_INFINITY, &err));

  if (err > sol_tol) {
    PetscCall(PetscPrintf(comm, "equiv FAIL: ||x_baseline - x_sum||_inf = %g > tol = %g\n", (double)err, (double)sol_tol));
  } else if (require_iter_match && iters_baseline != iters_sum) {
    PetscCall(PetscPrintf(comm, "equiv FAIL: iteration count mismatch (baseline %" PetscInt_FMT ", sum %" PetscInt_FMT ")\n", iters_baseline, iters_sum));
  } else PetscCall(PetscPrintf(comm, "equiv ok\n"));

  PetscCall(VecDestroy(&x_baseline));
  PetscCall(VecDestroy(&x_sum));
  PetscCall(VecDestroy(&diff));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: equiv_bnls
    args: -solver_type bnls -tao_gatol 1.e-9
    output_file: output/ex3_equiv.out
  test:
    suffix: equiv_bntr
    args: -solver_type bntr -tao_gatol 1.e-9
    output_file: output/ex3_equiv.out
  test:
    suffix: equiv_tron
    args: -solver_type tron -tao_gatol 1.e-9
    output_file: output/ex3_equiv.out

TEST*/
