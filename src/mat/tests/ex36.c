static char help[] = "Tests assembly of a matrix from another matrix's hash table.\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         A, B;
  PetscInt    n, i, j, rstart, rend;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* ------- Set values in A --------- */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatGetSize(A, NULL, &n));
  for (i = rstart; i < rend; i++) {
    for (j = 0; j < n; j++) {
      v = 10.0 * i + j + 1.0;
      PetscCall(MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }

  /* Create B */
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
  PetscCall(MatCopyHashToXAIJ(A, B));
  PetscCall(MatView(B, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: seq
      args: -mat_type seqaij
      filter: grep -v "Mat Object"

TEST*/
