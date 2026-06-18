static char help[] = "Test MATMPIBAIJKOKKOS <-> MPIAIJ conversion round-trip (F10.1).\n\
Builds a block-structured MPIAIJ matrix, converts to MPIBAIJKOKKOS and back,\n\
and checks the Frobenius norm of the difference is ~0 (square and rectangular blocks).\n\n";

#include <petscmat.h>

/*
  TestConvertRoundTrip - build a block-tridiagonal MPIAIJ matrix with the given block sizes,
  convert MPIAIJ -> MPIBAIJKOKKOS -> MPIAIJ, and verify the round-trip is exact.

  mbs    - global block rows
  nbs    - global block columns
  row_bs - block size in the row direction
  col_bs - block size in the column direction
*/
static PetscErrorCode TestConvertRoundTrip(PetscInt mbs, PetscInt nbs, PetscInt row_bs, PetscInt col_bs, const char *case_name)
{
  Mat         A_aij, A_kok, A_back, A_diff;
  PetscInt    m = mbs * row_bs, n = nbs * col_bs;
  PetscInt    iblock, jblock, ii, jj, jlo, jhi;
  PetscReal   norm_ref, norm_diff, rel_err;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A_aij));
  PetscCall(MatSetSizes(A_aij, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(A_aij, MATMPIAIJ));
  PetscCall(MatSetBlockSizes(A_aij, row_bs, col_bs));
  PetscCall(MatMPIAIJSetPreallocation(A_aij, 3 * col_bs, NULL, 3 * col_bs, NULL));

  /* Block-tridiagonal: each block-row touches block-cols {iblock-1, iblock, iblock+1}. */
  for (iblock = 0; iblock < mbs; iblock++) {
    jlo = PetscMax(iblock - 1, 0);
    jhi = PetscMin(iblock + 1, nbs - 1);
    for (ii = 0; ii < row_bs; ii++) {
      PetscInt row = iblock * row_bs + ii;
      for (jblock = jlo; jblock <= jhi; jblock++) {
        for (jj = 0; jj < col_bs; jj++) {
          PetscInt    col = jblock * col_bs + jj;
          PetscScalar val = PetscSinReal((PetscReal)(iblock + jblock + ii + jj + 1));
          PetscCall(MatSetValues(A_aij, 1, &row, 1, &col, &val, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_aij, MAT_FINAL_ASSEMBLY));

  /* Round trip: MPIAIJ -> MPIBAIJKOKKOS -> MPIAIJ. */
  PetscCall(MatConvert(A_aij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &A_kok));
  PetscCall(MatConvert(A_kok, MATMPIAIJ, MAT_INITIAL_MATRIX, &A_back));

  PetscCall(MatDuplicate(A_back, MAT_COPY_VALUES, &A_diff));
  PetscCall(MatAXPY(A_diff, -1.0, A_aij, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(A_aij, NORM_FROBENIUS, &norm_ref));
  PetscCall(MatNorm(A_diff, NORM_FROBENIUS, &norm_diff));
  rel_err = norm_ref > 0.0 ? norm_diff / norm_ref : norm_diff;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-28s row_bs=%" PetscInt_FMT " col_bs=%" PetscInt_FMT " rel_err=%g %s\n", case_name, row_bs, col_bs, (double)rel_err, rel_err < 1e-12 ? "PASS" : "FAIL"));

  PetscCall(MatDestroy(&A_aij));
  PetscCall(MatDestroy(&A_kok));
  PetscCall(MatDestroy(&A_back));
  PetscCall(MatDestroy(&A_diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(TestConvertRoundTrip(8, 8, 1, 1, "square_bs1"));
  PetscCall(TestConvertRoundTrip(8, 8, 3, 3, "square_bs3"));
  PetscCall(TestConvertRoundTrip(8, 8, 6, 6, "square_bs6"));
  PetscCall(TestConvertRoundTrip(8, 16, 3, 6, "rect_3x6"));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      requires: kokkos_kernels
      output_file: output/ex308_1.out
      nsize: {{1 2 4}}
      test:
        suffix: 1

TEST*/
