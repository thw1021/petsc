static char help[] = "Tests MATMPISEQBAIJ format.\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat          A;
  PetscInt     m = 2, bs = 2, M, row, col, start, end, i, j, k;
  PetscMPIInt  rank, size;
  PetscScalar  data = 100;
  PetscScalar *bval;
  Mat         *subs;
  IS           is;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Test MatSetValues() and MatGetValues() */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_block_size", &bs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_size", &m, NULL));

  M = m * bs * size;
  PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD, bs, PETSC_DECIDE, PETSC_DECIDE, M, M, M / bs, NULL, M / bs, NULL, &A));
  PetscCall(MatGetOwnershipRange(A, &start, &end));

  for (row = start; row < end; row++) {
    for (col = start; col < end; col++, data += 1) PetscCall(MatSetValues(A, 1, &row, 1, &col, &data, ADD_VALUES));
  }
  row /= bs;
  col = start / bs;
  PetscCall(PetscMalloc1(bs * bs, &bval));
  k = 1;
  /* row-oriented - default */
  for (i = 0; i < bs; i++) {
    for (j = 0; j < bs; j++) {
      bval[i * bs + j] = (PetscScalar)k;
      k++;
    }
  }
  for (row = start / bs; row < end / bs; row++) {
    for (col = start / bs; col < end / bs; col++) PetscCall(MatSetValuesBlocked(A, 1, &row, 1, &col, bval, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(bval));

  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  if (rank == 0) PetscCall(ISCreateStride(PETSC_COMM_SELF, M, 0, 1, &is));
  PetscCall(MatCreateSubMatrices(A, rank == 0 ? 1 : 0, &is, &is, MAT_INITIAL_MATRIX, &subs));
  if (rank == 0) PetscCall(ISDestroy(&is));
  if (rank == 0) PetscCall(MatView(subs[0], PETSC_VIEWER_STDOUT_SELF));
#if PetscDefined(HAVE_CUDA)
  {
    Mat seqaijcusparse;

    if (rank == 0) {
      PetscCall(MatConvert(subs[0], MATSEQAIJCUSPARSE, MAT_INITIAL_MATRIX, &seqaijcusparse));
      PetscCall(MatDestroy(&seqaijcusparse));
    }
  }
#endif
  PetscCall(MatDestroyMatrices(1, &subs));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 2

   test:
      suffix: 3
      requires: defined(PETSC_HAVE_SHMGET)
      nsize: 1
      args: -mat_mpibaij_use_mpiseqbaij

   test:
      suffix: 4
      nsize: 2
      requires: defined(PETSC_HAVE_SHMGET)
      args: -mat_mpibaij_use_mpiseqbaij

TEST*/
