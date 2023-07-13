#include <petsc.h>

static char help[] = "Tests for MatEliminateZeros().\n\n";

int main(int argc, char **args)
{
  Mat       A, B, C, D;
  PetscInt  M = 40, bs = 2;
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, M, NULL, &D));
  PetscCall(MatSetRandom(D, NULL));
  PetscCall(MatTranspose(D, MAT_INITIAL_MATRIX, &A));
  PetscCall(MatAXPY(D, 1.0, A, SAME_NONZERO_PATTERN));
  PetscCall(MatDestroy(&A));
  PetscCall(MatSetBlockSize(D, bs));
  PetscCall(MatConvert(D, MATAIJ, MAT_INITIAL_MATRIX, &A));
  PetscCall(MatConvert(D, MATBAIJ, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatConvert(D, MATSBAIJ, MAT_INITIAL_MATRIX, &C));
  PetscCall(MatViewFromOptions(D, NULL, "-input_dense"));
  PetscCall(MatViewFromOptions(A, NULL, "-input_aij"));
  PetscCall(MatViewFromOptions(B, NULL, "-input_baij"));
  PetscCall(MatViewFromOptions(C, NULL, "-input_sbaij"));
  PetscCall(MatChop(D, 1.2));
  PetscCall(MatChop(A, 1.2));
  PetscCall(MatEliminateZeros(A));
  PetscCall(MatChop(B, 1.2));
  PetscCall(MatEliminateZeros(B));
  PetscCall(MatChop(C, 1.2));
  PetscCall(MatEliminateZeros(C));
  PetscCall(MatViewFromOptions(D, NULL, "-output_dense"));
  PetscCall(MatViewFromOptions(A, NULL, "-output_aij"));
  PetscCall(MatViewFromOptions(B, NULL, "-output_baij"));
  PetscCall(MatViewFromOptions(C, NULL, "-output_sbaij"));
  PetscCall(MatMultEqual(D, A, 10, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "A != D");
  PetscCall(MatMultEqual(D, B, 10, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "B != D");
  PetscCall(MatMultEqual(D, C, 10, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "C != D");
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&D));
  PetscCall(PetscFinalize());
  return 0;
}
