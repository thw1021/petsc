static char help[] = "Tests MATMPISEQAIJ, building a sequential matrix on MPI rank 0 from COO entries provided in parallel.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat          A, S;
  PetscInt     M = 8, rstart, rend, ncoo = 0;
  PetscInt    *coo_i, *coo_j;
  PetscScalar *v;
  PetscMPIInt  rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, M));
  PetscCall(MatSetType(A, MATMPISEQAIJ));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

  /* COO entries for this rank's rows of a 1d Laplacian; the diagonal is given as two duplicate entries
     and one entry with negative indices is included to test that it is ignored */
  PetscCall(PetscMalloc3(4 * (rend - rstart) + 1, &coo_i, 4 * (rend - rstart) + 1, &coo_j, 4 * (rend - rstart) + 1, &v));
  for (PetscInt row = rstart; row < rend; row++) {
    coo_i[ncoo] = row;
    coo_j[ncoo] = row;
    v[ncoo++]   = 1.0;
    coo_i[ncoo] = row;
    coo_j[ncoo] = row;
    v[ncoo++]   = 1.0;
    if (row > 0) {
      coo_i[ncoo] = row;
      coo_j[ncoo] = row - 1;
      v[ncoo++]   = -1.0;
    }
    if (row < M - 1) {
      coo_i[ncoo] = row;
      coo_j[ncoo] = row + 1;
      v[ncoo++]   = -1.0;
    }
  }
  coo_i[ncoo] = -1;
  coo_j[ncoo] = -1;
  v[ncoo++]   = 100.0;

  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetPreallocationCOO(A, ncoo, coo_i, coo_j));
  PetscCall(MatSetValuesCOO(A, v, INSERT_VALUES));
  PetscCall(MatMPISeqAIJGetMat(A, &S));
  if (rank == 0) PetscCall(MatView(S, PETSC_VIEWER_STDOUT_SELF));

  /* Add the same values again, doubling the matrix */
  PetscCall(MatSetValuesCOO(A, v, ADD_VALUES));
  if (rank == 0) PetscCall(MatView(S, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscFree3(coo_i, coo_j, v));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 1
     requires: defined(PETSC_HAVE_SHMGET)
     nsize: {{1 2 3}}

   test:
     suffix: 2
     output_file: output/ex312_1.out
     args: -mat_mpiseqaij_use_shmget false
     nsize: {{1 2 3}}

TEST*/
