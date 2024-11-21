const char help[] = "Coverage tests for TAOTERMSHELL";

#include <petsctaoterm.h>

static PetscErrorCode TaoTermCreateVecs_Test(TaoTerm term, Vec *solution, Vec *params)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, (void *)&A));
  PetscCall(MatCreateVecs(A, params, solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Test(TaoTerm term, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "TaoTermView_Test()\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_Test(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  Mat A;
  Vec r;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, (void *)&A));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(MatMult(A, params, r));
  PetscCall(VecDotRealPart(x, r, value));
  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_Test(TaoTerm term, Vec x, Vec params, Vec g)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, (void *)&A));
  PetscCall(MatMult(A, params, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_Test(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, (void *)&A));
  PetscCall(MatMult(A, params, g));
  PetscCall(VecDotRealPart(x, g, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testShell(MPI_Comm comm, PetscBool separate)
{
  PetscRandom rand;
  Mat         A;
  TaoTerm     term;
  PetscInt    m = 23, n = 11;
  Vec         x, params, g;
  PetscInt    test_m, test_n;
  PetscReal   value, g_norm;

  PetscFunctionBegin;
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &A));
  PetscCall(MatSetRandom(A, rand));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(TaoTermCreateShell(comm, (void *)A, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_REQUIRED));

  if (separate) {
    PetscCall(MatCreateVecs(A, &params, &x));
    PetscCall(TaoTermSetSolutionTemplate(term, x));
    PetscCall(TaoTermSetParametersTemplate(term, params));
    PetscCall(VecDestroy(&params));
    PetscCall(VecDestroy(&x));
  } else {
    PetscCall(TaoTermShellSetCreateVecs(term, TaoTermCreateVecs_Test));
  }

  PetscCall(TaoTermSetUp(term));

  PetscCall(TaoTermGetSolutionSizes(term, NULL, &test_m, NULL));
  PetscCall(TaoTermGetParametersSizes(term, NULL, &test_n, NULL));
  PetscCheck(test_m == m, comm, PETSC_ERR_PLIB, "Inconsistent solution size");
  PetscCheck(test_n == n, comm, PETSC_ERR_PLIB, "Inconsistent parameters size");

  if (separate) {
    PetscCall(TaoTermShellSetObjective(term, TaoTermObjective_Test));
    PetscCall(TaoTermShellSetGradient(term, TaoTermGradient_Test));
  } else {
    PetscCall(TaoTermShellSetObjectiveAndGradient(term, TaoTermObjectiveAndGradient_Test));
    PetscCall(TaoTermShellSetObjectiveAndGradient(term, TaoTermObjectiveAndGradient_Test));
    PetscCall(TaoTermShellSetView(term, TaoTermView_Test));
  }

  PetscCall(TaoTermView(term, PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(TaoTermCreateVecs(term, &x, &params));

  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecSetRandom(params, rand));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(TaoTermObjectiveAndGradient(term, x, params, &value, g));
  PetscCall(VecNorm(g, NORM_2, &g_norm));
  PetscCall(PetscPrintf(comm, "objective: %g, gradient norm %g\n", (double)value, (double)g_norm));

  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&params));
  PetscCall(VecDestroy(&x));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(testShell(PETSC_COMM_WORLD, PETSC_TRUE));
  PetscCall(testShell(PETSC_COMM_WORLD, PETSC_FALSE));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 0

TEST*/
