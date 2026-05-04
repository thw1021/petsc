static const char help[] = "Test: CUSPARSE AtB reuse with changed values in left operand.\n";

/*
  Demonstrates that MatProductNumeric for MATPRODUCT_AtB on CUSPARSE
  uses a stale cached transpose when the left operand's values change
  but the sparsity pattern stays the same.

  Run: ./ex_cusparse_atb_stale -mat_type aijcusparse
  Expected: C_reuse should equal C_fresh, but it doesn't because
  the cached transpose of P is not refreshed.
*/

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         P, C, C_fresh;
  PetscInt    i, N = 4;
  PetscScalar v;
  PetscReal   norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Create P as a sparse matrix with some off-diagonal structure */
  PetscCall(MatCreate(PETSC_COMM_SELF, &P));
  PetscCall(MatSetSizes(P, N, N, N, N));
  PetscCall(MatSetFromOptions(P));
  PetscCall(MatSetUp(P));
  for (i = 0; i < N; i++) {
    v = (PetscScalar)(i + 1);
    PetscCall(MatSetValue(P, i, i, v, INSERT_VALUES));
    if (i + 1 < N) {
      PetscCall(MatSetValue(P, i, i + 1, 0.5, INSERT_VALUES));
      PetscCall(MatSetValue(P, i + 1, i, 0.5, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  /* Set up C = P^T * P (symbolic + first numeric) */
  PetscCall(MatProductCreate(P, P, NULL, &C));
  PetscCall(MatProductSetType(C, MATPRODUCT_AtB));
  PetscCall(MatProductSetFromOptions(C));
  PetscCall(MatProductSymbolic(C));
  PetscCall(MatProductNumeric(C));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "=== C after first numeric (original P) ===\n"));
  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_SELF));

  /* Change P's VALUES only (same sparsity pattern) */
  for (i = 0; i < N; i++) {
    v = (PetscScalar)(10 * (i + 1));
    PetscCall(MatSetValue(P, i, i, v, INSERT_VALUES));
    if (i + 1 < N) {
      PetscCall(MatSetValue(P, i, i + 1, 5.0, INSERT_VALUES));
      PetscCall(MatSetValue(P, i + 1, i, 5.0, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  /* Reuse: call only MatProductNumeric (no re-symbolic) */
  PetscCall(MatProductNumeric(C));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n=== C_reuse after second numeric (changed P, reused product) ===\n"));
  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_SELF));

  /* Fresh: build a brand-new product with the updated P for comparison */
  PetscCall(MatProductCreate(P, P, NULL, &C_fresh));
  PetscCall(MatProductSetType(C_fresh, MATPRODUCT_AtB));
  PetscCall(MatProductSetFromOptions(C_fresh));
  PetscCall(MatProductSymbolic(C_fresh));
  PetscCall(MatProductNumeric(C_fresh));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n=== C_fresh (new product with changed P) ===\n"));
  PetscCall(MatView(C_fresh, PETSC_VIEWER_STDOUT_SELF));

  /* Compare: C_reuse - C_fresh should be zero */
  PetscCall(MatAXPY(C, -1.0, C_fresh, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(C, NORM_FROBENIUS, &norm));
  if (norm > PETSC_SMALL) PetscCall(PetscPrintf(PETSC_COMM_SELF, "\nBUG: ||C_reuse - C_fresh|| = %g (should be 0)\n", (double)norm));
  else PetscCall(PetscPrintf(PETSC_COMM_SELF, "\nOK: ||C_reuse - C_fresh|| = %g\n", (double)norm));

  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&C_fresh));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: cuda

  test:
    suffix: cusparse
    requires: cuda
    args: -mat_type aijcusparse
    output_file: output/empty.out
    filter: grep -E "^BUG"

TEST*/
