static char help[] = "Tests that structural TaoTerm configuration is immutable after setup.\n";

#include <petsctao.h>

static PetscErrorCode Objective(TaoTerm term, Vec x, Vec params, PetscReal *f)
{
  PetscFunctionBeginUser;
  PetscCall(VecDotRealPart(x, x, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckWrongState(MPI_Comm comm, PetscErrorCode ierr, const char function[])
{
  PetscFunctionBeginUser;
  PetscCheck(ierr == PETSC_ERR_ARG_WRONGSTATE, comm, PETSC_ERR_PLIB, "%s() returned error code %d instead of PETSC_ERR_ARG_WRONGSTATE", function, (int)ierr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestTermGuards(MPI_Comm comm)
{
  PetscErrorCode ierr;
  TaoTerm        term, shell, sum, quadratic, subterm;
  Mat            A;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, 3, &term));
  PetscCall(TaoTermSetUp(term));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoTermSetSolutionSizes(term, PETSC_DECIDE, 4, 1);
  PetscCall(PetscPopErrorHandler());
  PetscCall(CheckWrongState(comm, ierr, "TaoTermSetSolutionSizes"));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoTermSetCreateHessianMode(term, PETSC_TRUE, MATAIJ, NULL);
  PetscCall(PetscPopErrorHandler());
  PetscCall(CheckWrongState(comm, ierr, "TaoTermSetCreateHessianMode"));
  PetscCall(TaoTermSetFDDelta(term, 1e-5));

  PetscCall(TaoTermCreateL1(comm, PETSC_DECIDE, 3, 0.0, &shell));
  PetscCall(TaoTermSetUp(shell));
  PetscCall(TaoTermL1SetEpsilon(shell, 1e-3));
  PetscCall(TaoTermDestroy(&shell));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &shell));
  PetscCall(TaoTermSetParametersMode(shell, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(shell, PETSC_DECIDE, 3, 1));
  PetscCall(TaoTermShellSetObjective(shell, Objective));
  PetscCall(TaoTermSetUp(shell));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoTermShellSetObjective(shell, Objective);
  PetscCall(PetscPopErrorHandler());
  PetscCall(CheckWrongState(comm, ierr, "TaoTermShellSetObjective"));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(sum, NULL, 1.0, term, NULL, NULL));
  PetscCall(TaoTermSetUp(sum));
  PetscCall(TaoTermSumGetTerm(sum, 0, NULL, NULL, &subterm, NULL));
  PetscCheck(subterm == term, comm, PETSC_ERR_PLIB, "TaoTermSumGetTerm() returned the wrong term after setup");
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoTermSumSetTermMask(sum, 0, TAOTERM_MASK_HESSIAN);
  PetscCall(PetscPopErrorHandler());
  PetscCall(CheckWrongState(comm, ierr, "TaoTermSumSetTermMask"));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoTermSumSetTerm(sum, 0, NULL, 1.0, shell, NULL);
  PetscCall(PetscPopErrorHandler());
  PetscCall(CheckWrongState(comm, ierr, "TaoTermSumSetTerm"));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoTermSumAddTerm(sum, NULL, 1.0, shell, NULL, NULL);
  PetscCall(PetscPopErrorHandler());
  PetscCall(CheckWrongState(comm, ierr, "TaoTermSumAddTerm"));

  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, 3, 3, 1, NULL, 0, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(A, 1.0));
  PetscCall(TaoTermCreateQuadratic(A, &quadratic));
  PetscCall(TaoTermSetUp(quadratic));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoTermQuadraticSetMat(quadratic, A);
  PetscCall(PetscPopErrorHandler());
  PetscCall(CheckWrongState(comm, ierr, "TaoTermQuadraticSetMat"));

  PetscCall(TaoTermDestroy(&quadratic));
  PetscCall(MatDestroy(&A));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(TaoTermDestroy(&shell));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(PetscPrintf(comm, "TaoTerm structural mutations rejected after setup; numerical updates accepted\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestTaoAddTermGuard(MPI_Comm comm)
{
  PetscErrorCode ierr;
  Tao            tao;
  TaoTerm        term, other;
  Vec            x;

  PetscFunctionBeginUser;
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, 3, &term));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, 3, &other));
  PetscCall(TaoAddTerm(tao, NULL, 1.0, term, NULL, NULL));
  PetscCall(VecCreate(comm, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 3));
  PetscCall(VecSetFromOptions(x));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetUp(tao));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = TaoAddTerm(tao, NULL, 1.0, other, NULL, NULL);
  PetscCall(PetscPopErrorHandler());
  PetscCall(CheckWrongState(comm, ierr, "TaoAddTerm"));
  PetscCall(VecDestroy(&x));
  PetscCall(TaoTermDestroy(&other));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscPrintf(comm, "TaoAddTerm rejected after Tao setup\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(TestTermGuards(PETSC_COMM_WORLD));
  PetscCall(TestTaoAddTermGuard(PETSC_COMM_WORLD));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
