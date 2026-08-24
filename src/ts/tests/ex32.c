static char help[] = "Tests TS explicit left and right diagonal scaling.\n\n";

#include <petscts.h>

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  const PetscScalar *ua;
  PetscScalar       *fa;

  PetscFunctionBeginUser;
  (void)ts;
  (void)t;
  (void)ctx;
  PetscCall(VecGetArrayRead(U, &ua));
  PetscCall(VecGetArray(F, &fa));
  fa[0] = -1.0e8 * (ua[0] - 2.0);
  fa[1] = -(ua[1] * ua[1] - 4.0);
  PetscCall(VecRestoreArray(F, &fa));
  PetscCall(VecRestoreArrayRead(U, &ua));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat A, Mat P, void *ctx)
{
  const PetscScalar *ua;

  PetscFunctionBeginUser;
  (void)ts;
  (void)t;
  (void)ctx;
  PetscCall(VecGetArrayRead(U, &ua));
  PetscCall(MatSetValue(P, 0, 0, -1.0e8, INSERT_VALUES));
  PetscCall(MatSetValue(P, 1, 1, -2.0 * ua[1], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &ua));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (A != P) {
    PetscCall(MatCopy(P, A, SAME_NONZERO_PATTERN));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
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

/* Confirms the left/right scaling set on a TS is visible through TSGetSNES()+SNESGetKSP(), the same
   propagation path exercised at the SNES level by src/snes/tests/ex74.c. */
static PetscErrorCode TestImplicitPropagation(void)
{
  TS   ts;
  SNES snes;
  KSP  ksp;
  Mat  J;
  Vec  u, left, right, got;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 2, 2, 1, NULL, &J));
  PetscCall(MatCreateVecs(J, &u, NULL));
  PetscCall(VecDuplicate(u, &left));
  PetscCall(VecDuplicate(u, &right));
  PetscCall(SetScale(left, 1.0e-8, 1.0));
  PetscCall(SetScale(right, 1.0e8, 1.0));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, NULL));
  PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, NULL));
  PetscCall(TSSetLeftDiagonalScale(ts, left));
  PetscCall(TSSetRightDiagonalScale(ts, right));
  PetscCall(TSGetLeftDiagonalScale(ts, &got));
  PetscCheck(got == left, PETSC_COMM_SELF, PETSC_ERR_PLIB, "TSGetLeftDiagonalScale() returned the wrong vector");
  PetscCall(TSGetRightDiagonalScale(ts, &got));
  PetscCheck(got == right, PETSC_COMM_SELF, PETSC_ERR_PLIB, "TSGetRightDiagonalScale() returned the wrong vector");

  PetscCall(SetScale(u, 1.0, 1.0));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetMaxSteps(ts, 2));
  PetscCall(TSSetMaxTime(ts, 0.02));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSolve(ts, u));

  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetLeftDiagonalScale(ksp, &got));
  PetscCheck(got == left, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scaling was not visible through TSGetSNES()+SNESGetKSP()+KSPGetLeftDiagonalScale()");
  PetscCall(KSPGetRightDiagonalScale(ksp, &got));
  PetscCheck(got == right, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scaling was not visible through TSGetSNES()+SNESGetKSP()+KSPGetRightDiagonalScale()");

  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&right));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Confirms TSSetRightDiagonalScale()'s vatol = atol*scale derivation, and that it defers to an
   explicitly-set vatol rather than overwriting it. */
static PetscErrorCode TestVatolDerivation(void)
{
  TS                 ts1, ts2;
  Vec                right, myvatol, vatol;
  PetscReal          atol;
  const PetscScalar *va, *ra;

  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 2, &right));
  PetscCall(SetScale(right, 1.0e8, 1.0));

  PetscCall(TSCreate(PETSC_COMM_SELF, &ts1));
  PetscCall(TSSetTolerances(ts1, 1.0e-6, NULL, 1.0e-6, NULL));
  PetscCall(TSSetRightDiagonalScale(ts1, right));
  PetscCall(TSGetTolerances(ts1, &atol, &vatol, NULL, NULL));
  PetscCheck(vatol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "TSSetRightDiagonalScale() did not derive a vatol");
  PetscCall(VecGetArrayRead(vatol, &va));
  PetscCall(VecGetArrayRead(right, &ra));
  PetscCheck(PetscAbsScalar(va[0] - atol * ra[0]) <= PETSC_SMALL && PetscAbsScalar(va[1] - atol * ra[1]) <= PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Derived vatol does not equal atol*scale");
  PetscCall(VecRestoreArrayRead(right, &ra));
  PetscCall(VecRestoreArrayRead(vatol, &va));

  /* an explicit vatol takes precedence over, and is not clobbered by, a derived one */
  PetscCall(VecDuplicate(right, &myvatol));
  PetscCall(SetScale(myvatol, 1.0, 1.0));
  PetscCall(TSSetTolerances(ts1, PETSC_CURRENT, myvatol, PETSC_CURRENT, NULL));
  PetscCall(TSGetTolerances(ts1, NULL, &vatol, NULL, NULL));
  PetscCheck(vatol == myvatol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "An explicitly-set vatol did not override the derived one");
  PetscCall(TSSetRightDiagonalScale(ts1, right));
  PetscCall(TSGetTolerances(ts1, NULL, &vatol, NULL, NULL));
  PetscCheck(vatol == myvatol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "TSSetRightDiagonalScale() clobbered an explicitly-set vatol");
  PetscCall(TSSetRightDiagonalScale(ts1, NULL));
  PetscCall(TSGetTolerances(ts1, NULL, &vatol, NULL, NULL));
  PetscCheck(vatol == myvatol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Clearing the right diagonal scale discarded an explicitly-set vatol");

  /* clearing the right diagonal scale discards a derived vatol */
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts2));
  PetscCall(TSSetTolerances(ts2, 1.0e-6, NULL, 1.0e-6, NULL));
  PetscCall(TSSetRightDiagonalScale(ts2, right));
  PetscCall(TSGetTolerances(ts2, NULL, &vatol, NULL, NULL));
  PetscCheck(vatol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "TSSetRightDiagonalScale() did not derive a vatol");
  PetscCall(TSSetRightDiagonalScale(ts2, NULL));
  PetscCall(TSGetTolerances(ts2, NULL, &vatol, NULL, NULL));
  PetscCheck(!vatol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Clearing the right diagonal scale did not discard the derived vatol");

  PetscCall(TSDestroy(&ts2));
  PetscCall(TSDestroy(&ts1));
  PetscCall(VecDestroy(&myvatol));
  PetscCall(VecDestroy(&right));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This test requires one MPI rank");
  PetscCall(TestImplicitPropagation());
  PetscCall(TestVatolDerivation());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: !complex
    output_file: output/empty.out

TEST*/
