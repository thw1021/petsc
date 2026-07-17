#include <petsctao.h>

static char help[] = "Regression tests for TAOTERMSUM Hessian masking and mapped separate-Hpre assembly.\n\n\
Cell 'masked': a Hessian-masked summand that defines NO Hessian at all must be\n\
skipped by TaoTermComputeHessianMult(); an unguarded implementation tries to\n\
assemble the masked summand's Hessian for its matrix-free cache and fails.\n\
Cell 'all_masked': when every summand is Hessian-masked the sum's assembled\n\
Hessian must be zeroed and the matrix-free product must return zero.\n\
Cell 'mapped_separate_hpre': a mapped summand whose term builds different H and\n\
Hpre matrices, summed with a MATSHELL Hessian and a separate assembled Hpre;\n\
the assembled Hpre must be P^T Hpre_term P, not P^T H_term P.\n\n";

static PetscErrorCode FormObjectiveAndGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormDiagHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode FormConstHessian(TaoTerm, Vec, Vec, Mat, Mat);

/* TAOTERMSHELL with point-dependent Hessian H(x) = diag(x) (H == Hpre). */
static PetscErrorCode CreateDiagonalHessianTerm(MPI_Comm comm, PetscInt n, TaoTerm *term_out)
{
  TaoTerm term;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term, PETSC_TRUE, MATAIJ, NULL));
  PetscCall(TaoTermShellSetHessian(term, FormDiagHessian));
  PetscCall(TaoTermSetUp(term));
  *term_out = term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TAOTERMSHELL with objective and gradient but NO Hessian of any kind. */
static PetscErrorCode CreateHessianlessTerm(MPI_Comm comm, PetscInt n, TaoTerm *term_out)
{
  TaoTerm term;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradient));
  PetscCall(TaoTermSetUp(term));
  *term_out = term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TAOTERMSHELL on an m-dimensional space with constant, DIFFERENT H = I and
   Hpre = 2 I, so a sum that assembles Hpre from the wrong source is detected. */
static PetscErrorCode CreateSplitHessianTerm(MPI_Comm comm, PetscInt m, TaoTerm *term_out)
{
  TaoTerm term;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, m, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term, PETSC_FALSE, MATAIJ, MATAIJ));
  PetscCall(TaoTermShellSetHessian(term, FormConstHessian));
  PetscCall(TaoTermSetUp(term));
  *term_out = term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMasked(MPI_Comm comm, PetscInt n, PetscBool mask_all)
{
  TaoTerm      t0, t1, sum;
  Mat          H = NULL, Hpre = NULL;
  Vec          x, v, Hv, Hv_assembled;
  PetscScalar *a;
  PetscReal    diff, norm;

  PetscFunctionBeginUser;
  PetscCall(CreateDiagonalHessianTerm(comm, n, &t0));
  PetscCall(CreateHessianlessTerm(comm, n, &t1));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumSetNumberTerms(sum, 2));
  PetscCall(TaoTermSumSetTerm(sum, 0, "t0_", 1.0, t0, NULL));
  PetscCall(TaoTermSumSetTerm(sum, 1, "t1_", 1.0, t1, NULL));
  /* The Hessian-less summand is always masked; otherwise the sum could not
     provide a Hessian at all. */
  PetscCall(TaoTermSumSetTermMask(sum, 1, TAOTERM_MASK_HESSIAN));
  if (mask_all) PetscCall(TaoTermSumSetTermMask(sum, 0, TAOTERM_MASK_HESSIAN));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(TaoTermCreateHessianMatrices(sum, &H, &Hpre));
  PetscCall(TaoTermCreateSolutionVec(sum, &x));
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &Hv));
  PetscCall(VecDuplicate(x, &Hv_assembled));
  PetscCall(VecGetArrayWrite(x, &a));
  for (PetscInt i = 0; i < n; i++) a[i] = (PetscScalar)(i + 1);
  PetscCall(VecRestoreArrayWrite(x, &a));
  PetscCall(VecSet(v, 1.0));

  /* Matrix-free product: must skip the masked summands entirely. */
  PetscCall(TaoTermComputeHessianMult(sum, x, NULL, v, Hv));

  /* Assembled path for cross-validation. */
  PetscCall(TaoTermComputeHessian(sum, x, NULL, H, Hpre));
  PetscCall(MatMult(H, v, Hv_assembled));

  PetscCall(VecAXPY(Hv_assembled, -1.0, Hv));
  PetscCall(VecNorm(Hv_assembled, NORM_2, &diff));
  PetscCheck(diff <= 1.e-10, comm, PETSC_ERR_PLIB, "Matrix-free and assembled Hessian products of the masked sum disagree (||delta|| = %g)", (double)diff);

  PetscCall(VecNorm(Hv, NORM_2, &norm));
  if (mask_all) {
    PetscCheck(norm <= 1.e-10, comm, PETSC_ERR_PLIB, "All summands are Hessian-masked but the product is nonzero (||Hv|| = %g)", (double)norm);
    PetscCall(PetscPrintf(comm, "All-masked TAOTERMSUM Hessian is zero in both assembled and matrix-free paths\n"));
  } else {
    /* Only the diagonal term contributes: H(x) v = x .* v. */
    PetscCall(VecPointwiseMult(Hv_assembled, x, v));
    PetscCall(VecAXPY(Hv_assembled, -1.0, Hv));
    PetscCall(VecNorm(Hv_assembled, NORM_2, &diff));
    PetscCheck(diff <= 1.e-10, comm, PETSC_ERR_PLIB, "Masked summand leaked into the Hessian product (||delta|| = %g)", (double)diff);
    PetscCall(PetscPrintf(comm, "Hessian-masked summand without Hessian callbacks is correctly skipped\n"));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&Hv));
  PetscCall(VecDestroy(&Hv_assembled));
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Hpre));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(TaoTermDestroy(&t0));
  PetscCall(TaoTermDestroy(&t1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMappedSeparateHpre(MPI_Comm comm)
{
  TaoTerm            term, sum;
  Mat                P, H = NULL, Hpre = NULL;
  Vec                x, v, Hv, diag;
  const PetscScalar *da;
  PetscReal          diff = 0.0;
  const PetscInt     m = 2, n = 3;
  const PetscScalar  Pvals[6]   = {1., 1., 1., 1., 2., 3.};
  const PetscReal    Hpre_ex[3] = {4., 10., 20.}; /* diag(P^T (2 I) P) */
  const PetscReal    Hv_ex[3]   = {9., 15., 21.}; /* P^T (I) P [1,1,1] */

  PetscFunctionBeginUser;
  /* P = [1 1 1; 1 2 3] maps the 3-dimensional sum space to the term's 2-dimensional space. */
  PetscCall(MatCreateSeqDense(comm, m, n, NULL, &P));
  for (PetscInt i = 0; i < m; i++)
    for (PetscInt j = 0; j < n; j++) PetscCall(MatSetValue(P, i, j, Pvals[i * n + j], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  PetscCall(CreateSplitHessianTerm(comm, m, &term));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSumSetNumberTerms(sum, 1));
  PetscCall(TaoTermSumSetTerm(sum, 0, "mapped_", 1.0, term, P));
  /* MATSHELL outer Hessian with a separate assembled preconditioning matrix. */
  PetscCall(TaoTermSetCreateHessianMode(sum, PETSC_FALSE, MATSHELL, MATAIJ));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(TaoTermCreateHessianMatrices(sum, &H, &Hpre));
  PetscCall(TaoTermCreateSolutionVec(sum, &x));
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &Hv));
  PetscCall(VecDuplicate(x, &diag));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecSet(v, 1.0));

  PetscCall(TaoTermComputeHessian(sum, x, NULL, H, Hpre));

  /* The assembled Hpre must come from the term's Hpre = 2 I, not from its H = I. */
  PetscCall(MatGetDiagonal(Hpre, diag));
  PetscCall(VecGetArrayRead(diag, &da));
  for (PetscInt i = 0; i < n; i++) diff = PetscMax(diff, PetscAbsReal(PetscRealPart(da[i]) - Hpre_ex[i]));
  PetscCall(VecRestoreArrayRead(diag, &da));
  PetscCheck(diff <= 1.e-10, comm, PETSC_ERR_PLIB, "Assembled Hpre of the mapped shell sum was built from the wrong per-term matrix (max diagonal error %g)", (double)diff);

  /* The shell H must still apply P^T H_term P. */
  PetscCall(MatMult(H, v, Hv));
  PetscCall(VecGetArrayRead(Hv, &da));
  diff = 0.0;
  for (PetscInt i = 0; i < n; i++) diff = PetscMax(diff, PetscAbsReal(PetscRealPart(da[i]) - Hv_ex[i]));
  PetscCall(VecRestoreArrayRead(Hv, &da));
  PetscCheck(diff <= 1.e-10, comm, PETSC_ERR_PLIB, "MATSHELL Hessian of the mapped sum applied the wrong operator (max error %g)", (double)diff);

  PetscCall(PetscPrintf(comm, "Mapped summand with MATSHELL Hessian and separate assembled Hpre is consistent\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&Hv));
  PetscCall(VecDestroy(&diag));
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Hpre));
  PetscCall(MatDestroy(&P));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(TaoTermDestroy(&term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscMPIInt size;
  PetscInt    n           = 4;
  PetscBool   mask_all    = PETSC_FALSE;
  PetscBool   test_mapped = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mask_all", &mask_all, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_mapped", &test_mapped, NULL));

  if (test_mapped) PetscCall(TestMappedSeparateHpre(comm));
  else PetscCall(TestMasked(comm, n, mask_all));

  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode FormObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  const PetscScalar *xa;
  PetscScalar       *ga;
  PetscReal          f = 0.0;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArrayWrite(g, &ga));
  for (PetscInt i = 0; i < n; i++) {
    ga[i] = 0.5 * xa[i] * xa[i];
    f += PetscRealPart(xa[i] * xa[i] * xa[i]) / 6.0;
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(VecRestoreArrayWrite(g, &ga));
  *value = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormDiagHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  const PetscScalar *xa;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayRead(x, &xa));
  if (H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(H, i, i, xa[i], INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  if (Hpre && Hpre != H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(Hpre, i, i, xa[i], INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* H = I and Hpre = 2 I: deliberately different so the consumer of the wrong
   matrix is detectable. */
static PetscErrorCode FormConstHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscInt n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  if (H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(H, i, i, 1.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  if (Hpre && Hpre != H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(Hpre, i, i, 2.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex

   test:
     suffix: masked
     args: -n 4

   test:
     suffix: all_masked
     args: -n 4 -mask_all

   test:
     suffix: mapped_separate_hpre
     args: -test_mapped

TEST*/
