#include <petsctao.h>

static char help[] = "Tests mapped finite-difference TaoTerm Hessians with aliased and separate Hpre matrices.\n";

static PetscErrorCode FormObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  PetscScalar       *ga;
  const PetscScalar *xa;
  PetscReal          value = 0.0;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArrayWrite(g, &ga));
  for (PetscInt i = 0; i < n; i++) {
    ga[i] = 2.0 * xa[i];
    value += PetscRealPart(xa[i] * xa[i]);
  }
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(VecRestoreArrayWrite(g, &ga));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &value, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)x)));
  *f = value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckMappedFD(Mat H)
{
  Vec               diag;
  const PetscScalar *a;
  PetscReal          error = 0.0;
  PetscInt           rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(H, &diag, NULL));
  PetscCall(MatGetDiagonal(H, diag));
  PetscCall(VecGetOwnershipRange(diag, &rstart, &rend));
  PetscCall(VecGetArrayRead(diag, &a));
  for (PetscInt i = rstart; i < rend; i++) error = PetscMax(error, PetscAbsReal(PetscRealPart(a[i - rstart]) - 2.0 * (i + 1.0) * (i + 1.0)));
  PetscCall(VecRestoreArrayRead(diag, &a));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &error, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)H)));
  PetscCheck(error <= 1.e-5, PetscObjectComm((PetscObject)H), PETSC_ERR_PLIB, "Mapped FD Hessian has maximum diagonal error %g", (double)error);
  PetscCall(VecDestroy(&diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  const PetscInt n = 4;
  Tao            tao;
  TaoTerm        term;
  Mat            H, Hpre, map;
  Vec            x;
  PetscBool      separate = PETSC_FALSE;
  PetscInt       rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-separate", &separate, NULL));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 1, NULL, &map));
  PetscCall(MatGetOwnershipRange(map, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(MatSetValue(map, i, i, i + 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(map, MAT_FINAL_ASSEMBLY));

  PetscCall(TaoTermCreateShell(PETSC_COMM_WORLD, NULL, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(term, FormObjectiveAndGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(term, separate ? PETSC_FALSE : PETSC_TRUE, MATAIJ, separate ? MATAIJ : NULL));
  PetscCall(TaoTermComputeHessianSetUseFD(term, PETSC_TRUE));
  PetscCall(MatCreateVecs(map, &x, NULL));
  PetscCall(VecSet(x, 0.5));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoAddTerm(tao, NULL, 1.0, term, NULL, map));
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));
  PetscCall(CheckMappedFD(H));
  PetscCall(CheckMappedFD(Hpre));
  PetscCheck(separate == (PetscBool)(H != Hpre), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected H/Hpre aliasing");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Mapped finite-difference Hessian is correct\n"));

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(MatDestroy(&map));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/taotermtest10.out
    requires: !complex !single

    test:
      suffix: aliased

    test:
      suffix: separate
      args: -separate

TEST*/
