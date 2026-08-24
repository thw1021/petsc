static char help[] = "Tests SNES explicit left and right diagonal scaling.\n\n";

#include <petscsnes.h>

static PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *ctx)
{
  const PetscScalar *xa;
  PetscScalar       *fa;

  PetscFunctionBeginUser;
  (void)snes;
  (void)ctx;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(VecGetArray(f, &fa));
  fa[0] = 1.0e8 * (xa[0] - 2.0);
  fa[1] = xa[1] * xa[1] - 4.0;
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx)
{
  const PetscScalar *xa;

  PetscFunctionBeginUser;
  (void)snes;
  (void)ctx;
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(MatSetValue(P, 0, 0, 1.0e8, INSERT_VALUES));
  PetscCall(MatSetValue(P, 1, 1, 2.0 * xa[1], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatCopy(P, J, SAME_NONZERO_PATTERN));
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetScale(Vec d, PetscScalar d0, PetscScalar d1)
{
  const PetscInt    rows[]   = {0, 1};
  const PetscScalar values[] = {d0, d1};

  PetscFunctionBeginUser;
  PetscCall(VecSetValues(d, 2, rows, values, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(d));
  PetscCall(VecAssemblyEnd(d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSolution(SNES snes, Vec x)
{
  Vec               expected;
  const PetscInt    rows[]   = {0, 1};
  const PetscScalar values[] = {2.0, 2.0};
  PetscReal         norm, expected_norm, rtol, atol, tolerance;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &expected));
  PetscCall(VecSetValues(expected, 2, rows, values, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(expected));
  PetscCall(VecAssemblyEnd(expected));
  PetscCall(VecNorm(expected, NORM_2, &expected_norm));
  PetscCall(VecAXPY(expected, -1.0, x));
  PetscCall(VecNorm(expected, NORM_2, &norm));
  PetscCall(SNESGetTolerances(snes, &atol, &rtol, NULL, NULL, NULL));
  tolerance = 1000.0 * PetscMax(atol, PetscMax(rtol * expected_norm, PETSC_MACHINE_EPSILON * expected_norm));
  PetscCall(VecDestroy(&expected));
  PetscCheck(norm <= tolerance, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNES solution error %g exceeds tolerance %g", (double)norm, (double)tolerance);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestWrongType(void)
{
  SNES           snes;
  Mat            J;
  Vec            x, f, right;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 2, 2, 1, NULL, &J));
  PetscCall(MatCreateVecs(J, &x, &f));
  PetscCall(VecDuplicate(x, &right));
  PetscCall(VecSet(right, 1.0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, f, FormFunction, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, NULL));
  PetscCall(SNESSetRightDiagonalScale(snes, right));
  PetscCall(SNESSetType(snes, SNESNRICHARDSON));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = SNESSetUp(snes);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_WRONGSTATE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scaling a non-Newton SNES type returned error %d", (int)ierr);
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&right));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES        snes;
  KSP         ksp;
  Mat         J;
  Vec         x, f, left, right, got;
  PetscMPIInt size;
  PetscBool   use_scale = PETSC_FALSE, test_wrongtype = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This test requires one MPI rank");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_scale", &use_scale, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_wrongtype", &test_wrongtype, NULL));

  if (test_wrongtype) {
    PetscCall(TestWrongType());
  } else {
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 2, 2, 1, NULL, &J));
    PetscCall(MatCreateVecs(J, &x, &f));
    PetscCall(VecDuplicate(x, &left));
    PetscCall(VecDuplicate(x, &right));
    PetscCall(SetScale(left, 1.0e-8, 1.0));
    PetscCall(SetScale(right, 1.0e8, 1.0));

    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
    PetscCall(SNESSetFunction(snes, f, FormFunction, NULL));
    PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, NULL));
    if (use_scale) {
      PetscCall(SNESSetLeftDiagonalScale(snes, left));
      PetscCall(SNESSetRightDiagonalScale(snes, right));
    }
    PetscCall(SNESGetLeftDiagonalScale(snes, &got));
    PetscCheck(got == (use_scale ? left : NULL), PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNESGetLeftDiagonalScale() returned the wrong vector");
    PetscCall(SNESGetRightDiagonalScale(snes, &got));
    PetscCheck(got == (use_scale ? right : NULL), PETSC_COMM_SELF, PETSC_ERR_PLIB, "SNESGetRightDiagonalScale() returned the wrong vector");
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(VecSet(x, 1.0));
    PetscCall(SNESSolve(snes, NULL, x));
    PetscCall(CheckSolution(snes, x));

    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetLeftDiagonalScale(ksp, &got));
    PetscCheck(got == (use_scale ? left : NULL), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scaling was not visible through SNESGetKSP()+KSPGetLeftDiagonalScale()");
    PetscCall(KSPGetRightDiagonalScale(ksp, &got));
    PetscCheck(got == (use_scale ? right : NULL), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scaling was not visible through SNESGetKSP()+KSPGetRightDiagonalScale()");

    PetscCall(SNESDestroy(&snes));
    PetscCall(VecDestroy(&right));
    PetscCall(VecDestroy(&left));
    PetscCall(VecDestroy(&f));
    PetscCall(VecDestroy(&x));
    PetscCall(MatDestroy(&J));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: newtonls
    output_file: output/empty.out
    args: -snes_type newtonls -ksp_type preonly -pc_type lu -snes_rtol 0 -snes_atol 1e-11 -snes_stol 0 -snes_max_it 20

  test:
    suffix: newtonls_scaled
    output_file: output/empty.out
    args: -use_scale -snes_type newtonls -ksp_type preonly -pc_type lu -snes_rtol 0 -snes_atol 1e-11 -snes_stol 0 -snes_max_it 20

  test:
    suffix: newtontr
    output_file: output/empty.out
    args: -snes_type newtontr -ksp_type preonly -pc_type lu -snes_rtol 0 -snes_atol 1e-11 -snes_stol 0 -snes_max_it 20

  test:
    suffix: newtontr_scaled
    output_file: output/empty.out
    args: -use_scale -snes_type newtontr -ksp_type preonly -pc_type lu -snes_rtol 0 -snes_atol 1e-11 -snes_stol 0 -snes_max_it 20

  test:
    suffix: wrongtype
    output_file: output/empty.out
    args: -test_wrongtype

TEST*/
