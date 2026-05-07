const char help[] = "Test TaoAddTerm() on bounded solvers (BNLS/BNTR/BNTL/BNK/TRON/BLMVM).\n\n"
                    "Minimize 1/2 x^T x + lambda/2 ||x - p||_2^2 over the box [-1, 1]^n.\n"
                    "All TaoTerm summands provide assembled Hessians, so Newton-family\n"
                    "bounded solvers must be able to extract sub-matrices from the assembled sum.\n\n"
                    "Modes:\n"
                    "  (default)            : programmatic two-term run for one solver type\n"
                    "  -test_dup            : two identical halfl2squared terms vs one halfl2squared with twice the scale (must match)\n"
                    "  -test_cli            : -tao_add_terms-driven path on a bounded solver\n"
                    "  -test_map_diag       : second summand wrapped in a MATDIAGONAL map\n"
                    "  -test_mask_hess      : mask the Hessian of one summand; solver must still converge using the other summand's Hessian\n"
                    "  -test_shell_no_hess  : negative test: shell summand with no Hessian; expect PETSC_ERR_SUP\n"
                    "  -test_bqnk           : negative test: BQNK family rejects TaoAddTerm; expect PETSC_ERR_SUP\n";

#include <petsctao.h>
#include <petsctaoterm.h>

static PetscErrorCode RunBoundedSumTest(MPI_Comm comm, TaoType type, PetscInt n, PetscReal lambda, PetscBool *converged)
{
  Tao       tao;
  TaoTerm   t_quad, t_reg;
  Mat       A;
  Vec       x, xl, xu, p;
  PetscReal gnorm;

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

  PetscCall(TaoGetSolutionStatus(tao, NULL, NULL, &gnorm, NULL, NULL, NULL));
  *converged = (PetscBool)(gnorm < 1e-4);

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

/*
  Solve "1/2 x^T x + lambda/2 ||x - p||^2" over [-1, 1]^n two ways:
    (a) two identical halfl2squared regulariser terms with parameter p, each scale = scale/2
    (b) one halfl2squared regulariser term with parameter p, scale = scale
  The final objective values must match.
*/
static PetscErrorCode RunDuplicateTermTest(MPI_Comm comm, TaoType type, PetscInt n, PetscReal scale)
{
  Tao       tao_dup, tao_single;
  TaoTerm   t_quad, t_reg_a, t_reg_b, t_reg_single, t_quad2;
  Mat       A, A2;
  Vec       x_dup, x_single, xl, xu, p;
  PetscReal f_dup, f_single;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(A, 1.0));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A2));

  PetscCall(MatCreateVecs(A, &x_dup, NULL));
  PetscCall(VecDuplicate(x_dup, &x_single));
  PetscCall(VecDuplicate(x_dup, &xl));
  PetscCall(VecDuplicate(x_dup, &xu));
  PetscCall(VecDuplicate(x_dup, &p));
  PetscCall(VecSet(xl, -1.0));
  PetscCall(VecSet(xu, 1.0));
  PetscCall(VecSet(p, 0.5));
  PetscCall(VecSet(x_dup, 0.8));
  PetscCall(VecSet(x_single, 0.8));

  PetscCall(TaoTermCreateQuadratic(A, &t_quad));
  PetscCall(TaoTermCreateQuadratic(A2, &t_quad2));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &t_reg_a));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &t_reg_b));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &t_reg_single));

  /* (a) duplicate-term solve: two halfl2squared with scale = scale/2 each */
  PetscCall(TaoCreate(comm, &tao_dup));
  PetscCall(TaoSetType(tao_dup, type));
  PetscCall(TaoSetSolution(tao_dup, x_dup));
  PetscCall(TaoSetVariableBounds(tao_dup, xl, xu));
  PetscCall(TaoAddTerm(tao_dup, "quad_", 1.0, t_quad, NULL, NULL));
  PetscCall(TaoAddTerm(tao_dup, "rega_", scale * 0.5, t_reg_a, p, NULL));
  PetscCall(TaoAddTerm(tao_dup, "regb_", scale * 0.5, t_reg_b, p, NULL));
  PetscCall(TaoSetFromOptions(tao_dup));
  PetscCall(TaoSolve(tao_dup));
  PetscCall(TaoGetSolutionStatus(tao_dup, NULL, &f_dup, NULL, NULL, NULL, NULL));

  /* (b) single-term solve: one halfl2squared with scale = scale */
  PetscCall(TaoCreate(comm, &tao_single));
  PetscCall(TaoSetType(tao_single, type));
  PetscCall(TaoSetSolution(tao_single, x_single));
  PetscCall(TaoSetVariableBounds(tao_single, xl, xu));
  PetscCall(TaoAddTerm(tao_single, "quad_", 1.0, t_quad2, NULL, NULL));
  PetscCall(TaoAddTerm(tao_single, "reg_", scale, t_reg_single, p, NULL));
  PetscCall(TaoSetFromOptions(tao_single));
  PetscCall(TaoSolve(tao_single));
  PetscCall(TaoGetSolutionStatus(tao_single, NULL, &f_single, NULL, NULL, NULL, NULL));

  if (PetscAbsReal(f_dup - f_single) < 1e-8 * (1.0 + PetscAbsReal(f_single))) PetscCall(PetscPrintf(comm, "dup vs single: equal\n"));
  else PetscCall(PetscPrintf(comm, "dup vs single: MISMATCH (f_dup = %g, f_single = %g)\n", (double)f_dup, (double)f_single));

  PetscCall(TaoDestroy(&tao_dup));
  PetscCall(TaoDestroy(&tao_single));
  PetscCall(TaoTermDestroy(&t_quad));
  PetscCall(TaoTermDestroy(&t_quad2));
  PetscCall(TaoTermDestroy(&t_reg_a));
  PetscCall(TaoTermDestroy(&t_reg_b));
  PetscCall(TaoTermDestroy(&t_reg_single));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&A2));
  PetscCall(VecDestroy(&x_dup));
  PetscCall(VecDestroy(&x_single));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CLI-driven path: do not call TaoAddTerm in code.  Pre-create the objective term
  via -tao_add_terms reg1_,reg2_ -reg1_tao_term_type halfl2squared ... and
  rely on TaoSetFromOptions to assemble the sum.  Each term's parameter is left
  as the default zero vector.
*/
static PetscErrorCode RunCLITest(MPI_Comm comm, TaoType type, PetscInt n, PetscBool *converged)
{
  Tao       tao;
  Vec       x, xl, xu;
  PetscReal gnorm;

  PetscFunctionBeginUser;
  PetscCall(VecCreate(comm, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecSet(xl, -1.0));
  PetscCall(VecSet(xu, 1.0));
  PetscCall(VecSet(x, 0.7));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, type));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetVariableBounds(tao, xl, xu));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(TaoGetSolutionStatus(tao, NULL, NULL, &gnorm, NULL, NULL, NULL));
  *converged = (PetscBool)(gnorm < 1e-4);

  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Wrap the regulariser in a MATDIAGONAL map M = diag(d).
  Objective becomes 1/2 x^T x + lambda/2 ||M x - p||^2 over [-1, 1]^n.
*/
static PetscErrorCode RunMappedDiagTest(MPI_Comm comm, TaoType type, PetscInt n, PetscReal lambda, PetscBool *converged)
{
  Tao       tao;
  TaoTerm   t_quad, t_reg;
  Mat       A, M;
  Vec       d, x, xl, xu, p;
  PetscReal gnorm;

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
  PetscCall(VecDuplicate(x, &d));
  PetscCall(VecSet(xl, -1.0));
  PetscCall(VecSet(xu, 1.0));
  PetscCall(VecSet(p, 0.0));
  PetscCall(VecSet(x, 0.8));
  PetscCall(VecSet(d, 0.5));
  PetscCall(MatCreateDiagonal(d, &M));

  PetscCall(TaoTermCreateQuadratic(A, &t_quad));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &t_reg));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, type));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetVariableBounds(tao, xl, xu));
  PetscCall(TaoAddTerm(tao, "quad_", 1.0, t_quad, NULL, NULL));
  PetscCall(TaoAddTerm(tao, "reg_", lambda, t_reg, p, M));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(TaoGetSolutionStatus(tao, NULL, NULL, &gnorm, NULL, NULL, NULL));
  *converged = (PetscBool)(gnorm < 1e-4);

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&t_quad));
  PetscCall(TaoTermDestroy(&t_reg));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&M));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Two-summand sum where the regulariser's Hessian is masked off.
  The quadratic summand still provides an assembled Hessian, so the
  bounded Newton solver should converge.  Verifies that the new
  TaoTermSumCheckHessianAssembleable_Private helper short-circuits
  Hessian-masked summands.
*/
static PetscErrorCode RunMaskedHessianTest(MPI_Comm comm, TaoType type, PetscInt n, PetscReal lambda, PetscBool *converged)
{
  Tao       tao;
  TaoTerm   sum_term, t_quad, t_reg;
  Mat       A;
  Vec       x, xl, xu, p;
  PetscReal gnorm;

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
  /* Mask the regulariser's Hessian; quadratic still contributes one. */
  PetscCall(TaoGetTerm(tao, NULL, &sum_term, NULL, NULL));
  PetscCall(TaoTermSumSetTermMask(sum_term, 1, TAOTERM_MASK_HESSIAN));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(TaoGetSolutionStatus(tao, NULL, NULL, &gnorm, NULL, NULL, NULL));
  *converged = (PetscBool)(gnorm < 1e-4);

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

static PetscErrorCode RunShellNoHessianNegativeTest(MPI_Comm comm, PetscInt n)
{
  Tao            tao;
  TaoTerm        t_reg, t_shell;
  Vec            x, xl, xu, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &t_reg));
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &t_shell));
  PetscCall(TaoTermSetParametersMode(t_shell, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(t_shell, PETSC_DECIDE, n, 1));

  PetscCall(VecCreate(comm, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecDuplicate(x, &p));
  PetscCall(VecSet(xl, -1.0));
  PetscCall(VecSet(xu, 1.0));
  PetscCall(VecSet(p, 0.0));
  PetscCall(VecSet(x, 0.5));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAOBNLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetVariableBounds(tao, xl, xu));
  PetscCall(TaoAddTerm(tao, "reg_", 1.0, t_reg, p, NULL));
  PetscCall(TaoAddTerm(tao, "shell_", 1.0, t_shell, NULL, NULL));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoSetUp(tao);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_SUP, comm, PETSC_ERR_PLIB, "Expected PETSC_ERR_SUP from TAOBNLS setup with non-assembleable summand, got %d", (int)ierr);
  PetscCall(PetscPrintf(comm, "shell-no-hessian negative test: got expected PETSC_ERR_SUP\n"));

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&t_reg));
  PetscCall(TaoTermDestroy(&t_shell));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RunBQNKNegativeTest(MPI_Comm comm, TaoType bqnk_type, PetscInt n)
{
  Tao            tao;
  TaoTerm        t_a, t_b;
  Vec            x, xl, xu, pa, pb;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &t_a));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &t_b));

  PetscCall(VecCreate(comm, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecDuplicate(x, &pa));
  PetscCall(VecDuplicate(x, &pb));
  PetscCall(VecSet(xl, -1.0));
  PetscCall(VecSet(xu, 1.0));
  PetscCall(VecSet(pa, 0.0));
  PetscCall(VecSet(pb, 0.5));
  PetscCall(VecSet(x, 0.3));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, bqnk_type));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetVariableBounds(tao, xl, xu));
  PetscCall(TaoAddTerm(tao, "a_", 1.0, t_a, pa, NULL));
  PetscCall(TaoAddTerm(tao, "b_", 1.0, t_b, pb, NULL));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoSetUp(tao);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_SUP, comm, PETSC_ERR_PLIB, "Expected PETSC_ERR_SUP from %s setup with TaoAddTerm, got %d", bqnk_type, (int)ierr);
  PetscCall(PetscPrintf(comm, "BQNK negative test: got expected PETSC_ERR_SUP\n"));

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&t_a));
  PetscCall(TaoTermDestroy(&t_b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&pa));
  PetscCall(VecDestroy(&pb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm  comm;
  PetscInt  n               = 8;
  PetscReal lambda          = 0.1;
  PetscReal dup_scale       = 0.55;
  PetscBool test_dup        = PETSC_FALSE;
  PetscBool test_cli        = PETSC_FALSE;
  PetscBool test_map_diag   = PETSC_FALSE;
  PetscBool test_mask_hess  = PETSC_FALSE;
  PetscBool test_shell      = PETSC_FALSE;
  PetscBool test_bqnk       = PETSC_FALSE;
  char      tao_type[64]    = TAOBNLS;
  char      bqnk_type[64]   = TAOBQNLS;
  PetscBool converged;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, NULL, "ex2 options", NULL);
  PetscCall(PetscOptionsInt("-n", "Problem size", "ex2", n, &n, NULL));
  PetscCall(PetscOptionsReal("-lambda", "Regulariser scale", "ex2", lambda, &lambda, NULL));
  PetscCall(PetscOptionsReal("-dup_scale", "Total regulariser scale for -test_dup", "ex2", dup_scale, &dup_scale, NULL));
  PetscCall(PetscOptionsString("-solver_type", "TaoType to test", "ex2", tao_type, tao_type, sizeof(tao_type), NULL));
  PetscCall(PetscOptionsString("-bqnk_solver", "BQNK family TaoType for negative test", "ex2", bqnk_type, bqnk_type, sizeof(bqnk_type), NULL));
  PetscCall(PetscOptionsBool("-test_dup", "Same-type/same-scale duplication equivalence", "ex2", test_dup, &test_dup, NULL));
  PetscCall(PetscOptionsBool("-test_cli", "Run sum built via -tao_add_terms", "ex2", test_cli, &test_cli, NULL));
  PetscCall(PetscOptionsBool("-test_map_diag", "Wrap regulariser in a MATDIAGONAL map", "ex2", test_map_diag, &test_map_diag, NULL));
  PetscCall(PetscOptionsBool("-test_mask_hess", "Mask one summand's Hessian", "ex2", test_mask_hess, &test_mask_hess, NULL));
  PetscCall(PetscOptionsBool("-test_shell_no_hess", "Run negative shell-no-Hessian test", "ex2", test_shell, &test_shell, NULL));
  PetscCall(PetscOptionsBool("-test_bqnk", "Run negative BQNK test", "ex2", test_bqnk, &test_bqnk, NULL));
  PetscOptionsEnd();
  if (test_shell) {
    PetscCall(RunShellNoHessianNegativeTest(comm, n));
  } else if (test_bqnk) {
    PetscCall(RunBQNKNegativeTest(comm, bqnk_type, n));
  } else if (test_dup) {
    PetscCall(RunDuplicateTermTest(comm, tao_type, n, dup_scale));
  } else if (test_cli) {
    PetscCall(RunCLITest(comm, tao_type, n, &converged));
    PetscCall(PetscPrintf(comm, "%s\n", converged ? "converged" : "DID NOT CONVERGE"));
  } else if (test_map_diag) {
    PetscCall(RunMappedDiagTest(comm, tao_type, n, lambda, &converged));
    PetscCall(PetscPrintf(comm, "%s\n", converged ? "converged" : "DID NOT CONVERGE"));
  } else if (test_mask_hess) {
    PetscCall(RunMaskedHessianTest(comm, tao_type, n, lambda, &converged));
    PetscCall(PetscPrintf(comm, "%s\n", converged ? "converged" : "DID NOT CONVERGE"));
  } else {
    PetscCall(RunBoundedSumTest(comm, tao_type, n, lambda, &converged));
    PetscCall(PetscPrintf(comm, "%s\n", converged ? "converged" : "DID NOT CONVERGE"));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: bnls
    args: -solver_type bnls -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: bntr
    args: -solver_type bntr -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: bntl
    args: -solver_type bntl -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: tron
    args: -solver_type tron -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: blmvm
    args: -solver_type blmvm -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: bnls_par
    nsize: 2
    args: -solver_type bnls -tao_gatol 1.e-8 -n 16
    output_file: output/ex2_converged.out
  test:
    suffix: tron_par
    nsize: 2
    args: -solver_type tron -tao_gatol 1.e-8 -n 16
    output_file: output/ex2_converged.out
  test:
    suffix: dup_bnls
    args: -solver_type bnls -test_dup -tao_gatol 1.e-10
    output_file: output/ex2_dup.out
  test:
    suffix: dup_bntr
    args: -solver_type bntr -test_dup -tao_gatol 1.e-10
    output_file: output/ex2_dup.out
  test:
    suffix: dup_tron
    args: -solver_type tron -test_dup -tao_gatol 1.e-10
    output_file: output/ex2_dup.out
  test:
    suffix: cli_bnls
    args: -solver_type bnls -test_cli -tao_add_terms a_,b_ -a_tao_term_type halfl2squared -tao_term_sum_a_scale 1.1 -b_tao_term_type halfl2squared -tao_term_sum_b_scale 1.1 -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: map_diag_bnls
    args: -solver_type bnls -test_map_diag -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: map_diag_tron
    args: -solver_type tron -test_map_diag -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: mask_hess_bnls
    args: -solver_type bnls -test_mask_hess -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: mask_hess_tron
    args: -solver_type tron -test_mask_hess -tao_gatol 1.e-8
    output_file: output/ex2_converged.out
  test:
    suffix: shell_no_hessian
    args: -test_shell_no_hess
    output_file: output/ex2_shell_no_hessian.out
  test:
    suffix: bqnk_reject
    args: -test_bqnk
    output_file: output/ex2_bqnk_reject.out
  test:
    suffix: bqnktr_reject
    args: -test_bqnk -bqnk_solver bqnktr
    output_file: output/ex2_bqnk_reject.out

TEST*/
