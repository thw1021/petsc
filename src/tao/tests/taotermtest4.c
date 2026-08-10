static char help[] = "Test TAOTERMSUM Hessian storage and assembled-output fallbacks.\n\n";

#include <petsctao.h>

static PetscErrorCode Hess_T0(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscInt n;

  PetscFunctionBeginUser;
  PetscCall(VecGetSize(x, &n));
  if (H) {
    for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(H, i, i, 2.0, INSERT_VALUES));
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

static PetscErrorCode TestHpreFallback(MPI_Comm comm)
{
  TaoTerm     sum, t0;
  Mat         H0, Hout, Hpre_out;
  Vec         x;
  PetscReal   norm;
  PetscInt    n = 4;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(comm, n, n, 3, NULL, &H0));
  PetscCall(MatCreateSeqAIJ(comm, n, n, 3, NULL, &Hout));
  PetscCall(MatCreateSeqAIJ(comm, n, n, 3, NULL, &Hpre_out));
  /* The outer matrices are MatCopy()/MatAXPY() destinations for the summand contributions
     and must be assembled, matching what TaoTermCreateHessianMatrices_Sum() produces. */
  PetscCall(MatAssemblyBegin(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Hpre_out, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hpre_out, MAT_FINAL_ASSEMBLY));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &t0));
  PetscCall(TaoTermSetParametersMode(t0, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(t0, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetHessian(t0, Hess_T0));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumSetNumberTerms(sum, 1));
  PetscCall(TaoTermSumSetTerm(sum, 0, "t0_", 1.0, t0, NULL));
  /* H-only designated storage: the summand has no Hessian preconditioning matrix anywhere */
  PetscCall(TaoTermSumSetTermHessianMatrices(sum, 0, H0, NULL, H0, NULL));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(VecCreateSeq(comm, n, &x));
  PetscCall(VecSet(x, 1.0));

  /* Assembled outer Hessian with a distinct preconditioning matrix: the summand's H-only
     storage must serve both, with the Hessian standing in for the missing Hpre */
  PetscCall(TaoTermComputeHessian(sum, x, NULL, Hout, Hpre_out));
  PetscCall(MatAXPY(Hpre_out, -1.0, Hout, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(Hpre_out, NORM_FROBENIUS, &norm));
  PetscCheck(norm <= 1.e-9, comm, PETSC_ERR_PLIB, "outer Hpre differs from outer Hessian by %g", (double)norm);
  PetscCall(MatNorm(Hout, NORM_FROBENIUS, &norm));
  PetscCheck(PetscAbsReal(norm - 2.0 * PetscSqrtReal((PetscReal)n)) <= 1.e-9, comm, PETSC_ERR_PLIB, "outer Hessian norm %g does not match the summand Hessian", (double)norm);
  PetscCall(PetscPrintf(comm, "H-only summand storage fills a distinct outer Hpre\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Hpre_out));
  PetscCall(MatDestroy(&Hout));
  PetscCall(MatDestroy(&H0));
  PetscCall(TaoTermDestroy(&t0));
  PetscCall(TaoTermDestroy(&sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMappedOnlyStorage(MPI_Comm comm)
{
  TaoTerm   sum, t0;
  Mat       map, mapped_H, Hout;
  Vec       x;
  PetscReal norm;
  PetscInt  index;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &t0));
  PetscCall(TaoTermSetParametersMode(t0, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(t0, PETSC_DECIDE, 2, 1));
  PetscCall(TaoTermShellSetHessian(t0, Hess_T0));
  PetscCall(TaoTermShellSetCreateHessianMatrices(t0, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(t0, PETSC_TRUE, MATAIJ, NULL));

  PetscCall(MatCreateSeqAIJ(comm, 2, 3, 1, NULL, &map));
  PetscCall(MatSetValue(map, 0, 0, 1.0, INSERT_VALUES));
  PetscCall(MatSetValue(map, 1, 1, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(map, MAT_FINAL_ASSEMBLY));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(sum, "t0_", 1.0, t0, map, &index));
  PetscCall(MatCreateSeqAIJ(comm, 3, 3, 1, NULL, &mapped_H));
  PetscCall(MatAssemblyBegin(mapped_H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mapped_H, MAT_FINAL_ASSEMBLY));
  PetscCall(TaoTermSumSetTermHessianMatrices(sum, index, NULL, NULL, mapped_H, NULL));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, 3, 1));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(MatDuplicate(mapped_H, MAT_DO_NOT_COPY_VALUES, &Hout));
  PetscCall(MatAssemblyBegin(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateSeq(comm, 3, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoTermComputeHessian(sum, x, NULL, Hout, Hout));
  PetscCall(MatNorm(Hout, NORM_FROBENIUS, &norm));
  PetscCheck(PetscAbsReal(norm - 2.0 * PetscSqrtReal(2.0)) <= 1.e-9, comm, PETSC_ERR_PLIB, "mapped Hessian norm %g is incorrect", (double)norm);
  PetscCall(PetscPrintf(comm, "Mapped-only storage uses separately created raw storage\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Hout));
  PetscCall(MatDestroy(&mapped_H));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(MatDestroy(&map));
  PetscCall(TaoTermDestroy(&t0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestNoStorageAssembledOutput(MPI_Comm comm)
{
  TaoTerm   sum, t0;
  Mat       Hout;
  Vec       x;
  PetscReal norm;
  PetscInt  n = 4;

  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &t0));
  PetscCall(TaoTermSetParametersMode(t0, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(t0, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetHessian(t0, Hess_T0));

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(sum, "t0_", 1.0, t0, NULL, NULL));
  PetscCall(TaoTermSetSolutionSizes(sum, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSetUp(sum));

  PetscCall(MatCreateSeqAIJ(comm, n, n, 1, NULL, &Hout));
  for (PetscInt i = 0; i < n; i++) PetscCall(MatSetValue(Hout, i, i, 0.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hout, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateSeq(comm, n, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(TaoTermComputeHessian(sum, x, NULL, Hout, Hout));
  PetscCall(MatNorm(Hout, NORM_FROBENIUS, &norm));
  PetscCheck(PetscAbsReal(norm - 2.0 * PetscSqrtReal((PetscReal)n)) <= 1.e-9, comm, PETSC_ERR_PLIB, "direct-output Hessian norm %g is incorrect", (double)norm);
  PetscCall(PetscPrintf(comm, "No-storage summand evaluates into assembled caller output\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&Hout));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(TaoTermDestroy(&t0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");
  PetscCall(TestHpreFallback(comm));
  PetscCall(TestMappedOnlyStorage(comm));
  PetscCall(TestNoStorageAssembledOutput(comm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: hpre_fallback

TEST*/
