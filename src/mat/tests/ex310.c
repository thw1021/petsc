static char help[] = "Test blocked COO assembly for MATSEQBAIJKOKKOS (MatCOOUseBlockIndices).\n\
Builds a block matrix by hand with MatSetValuesBlocked and again with blocked COO\n\
(block indices + row-major rbs*cbs value blocks), and checks the two are identical.\n\
COO duplicates (each block emitted twice, half value) and a negative index are exercised.\n\n";

#include <petscmat.h>

/* Deterministic value of element (ii,jj) of block (ib,jb). */
static inline PetscScalar BlockVal(PetscInt ib, PetscInt jb, PetscInt ii, PetscInt jj, PetscInt col_bs)
{
  return (PetscScalar)(1 + ib * 100 + jb * 10 + ii * col_bs + jj);
}

/*
  TestBlockCOO - build a block-tridiagonal SEQBAIJKOKKOS by hand and via blocked COO, compare.

  mbs/nbs - block rows/columns; row_bs/col_bs - block sizes.
*/
static PetscErrorCode TestBlockCOO(PetscInt mbs, PetscInt nbs, PetscInt row_bs, PetscInt col_bs, const char *case_name)
{
  Mat          A_ref, A_coo, R_aij, C_aij, D;
  PetscInt     m = mbs * row_bs, n = nbs * col_bs, bs2 = row_bs * col_bs;
  PetscInt     ib, jb, ii, jj, jlo, jhi, nblk = 0, e, c;
  PetscInt    *coo_i, *coo_j;
  PetscScalar *coo_v, *blk;
  PetscReal    norm_ref, norm_diff, rel_err;

  PetscFunctionBeginUser;
  /* Reference: hand assembly with MatSetValuesBlocked. */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A_ref));
  PetscCall(MatSetSizes(A_ref, m, n, m, n));
  PetscCall(MatSetBlockSizes(A_ref, row_bs, col_bs));
  PetscCall(MatSetType(A_ref, MATSEQBAIJKOKKOS));
  PetscCall(MatSeqBAIJSetPreallocation(A_ref, row_bs, 3, NULL));
  PetscCall(PetscMalloc1(bs2, &blk));
  for (ib = 0; ib < mbs; ib++) {
    jlo = PetscMax(ib - 1, 0);
    jhi = PetscMin(ib + 1, nbs - 1);
    for (jb = jlo; jb <= jhi; jb++) {
      for (ii = 0; ii < row_bs; ii++)
        for (jj = 0; jj < col_bs; jj++) blk[ii * col_bs + jj] = BlockVal(ib, jb, ii, jj, col_bs);
      PetscCall(MatSetValuesBlocked(A_ref, 1, &ib, 1, &jb, blk, INSERT_VALUES));
      nblk++;
    }
  }
  PetscCall(MatAssemblyBegin(A_ref, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_ref, MAT_FINAL_ASSEMBLY));

  /* COO: each block emitted twice (half value each) to exercise duplicate summing, plus one ignored
     entry with a negative block-row index. */
  PetscCall(PetscMalloc2(2 * nblk + 1, &coo_i, 2 * nblk + 1, &coo_j));
  PetscCall(PetscCalloc1((size_t)(2 * nblk + 1) * bs2, &coo_v));
  c = 0;
  for (ib = 0; ib < mbs; ib++) {
    jlo = PetscMax(ib - 1, 0);
    jhi = PetscMin(ib + 1, nbs - 1);
    for (jb = jlo; jb <= jhi; jb++) {
      for (e = 0; e < 2; e++) {
        coo_i[c] = ib;
        coo_j[c] = jb;
        for (ii = 0; ii < row_bs; ii++)
          for (jj = 0; jj < col_bs; jj++) coo_v[(size_t)c * bs2 + ii * col_bs + jj] = 0.5 * BlockVal(ib, jb, ii, jj, col_bs);
        c++;
      }
    }
  }
  coo_i[c] = -1; /* ignored entry */
  coo_j[c] = 0;
  c++;

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A_coo));
  PetscCall(MatSetSizes(A_coo, m, n, m, n));
  PetscCall(MatSetBlockSizes(A_coo, row_bs, col_bs));
  PetscCall(MatSetType(A_coo, MATSEQBAIJKOKKOS));
  PetscCall(MatCOOUseBlockIndices(A_coo, PETSC_TRUE));
  PetscCall(MatSetPreallocationCOO(A_coo, c, coo_i, coo_j));
  PetscCall(MatSetValuesCOO(A_coo, coo_v, INSERT_VALUES));

  /* Compare via SeqAIJ. */
  PetscCall(MatConvert(A_ref, MATSEQAIJ, MAT_INITIAL_MATRIX, &R_aij));
  PetscCall(MatConvert(A_coo, MATSEQAIJ, MAT_INITIAL_MATRIX, &C_aij));
  PetscCall(MatDuplicate(R_aij, MAT_COPY_VALUES, &D));
  PetscCall(MatAXPY(D, -1.0, C_aij, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(R_aij, NORM_FROBENIUS, &norm_ref));
  PetscCall(MatNorm(D, NORM_FROBENIUS, &norm_diff));
  rel_err = norm_diff / (norm_ref > 0 ? norm_ref : 1.0);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-18s rbs=%" PetscInt_FMT " cbs=%" PetscInt_FMT ": %s\n", case_name, row_bs, col_bs, rel_err < 1e-12 ? "PASS" : "FAIL"));

  PetscCall(PetscFree(blk));
  PetscCall(PetscFree2(coo_i, coo_j));
  PetscCall(PetscFree(coo_v));
  PetscCall(MatDestroy(&A_ref));
  PetscCall(MatDestroy(&A_coo));
  PetscCall(MatDestroy(&R_aij));
  PetscCall(MatDestroy(&C_aij));
  PetscCall(MatDestroy(&D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(TestBlockCOO(5, 5, 1, 1, "scalar 1x1"));
  PetscCall(TestBlockCOO(5, 5, 3, 3, "square 3x3"));
  PetscCall(TestBlockCOO(4, 4, 6, 6, "square 6x6"));
  PetscCall(TestBlockCOO(5, 5, 3, 6, "rect 3x6"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    requires: kokkos_kernels
    nsize: 1
    output_file: output/ex310_1.out

TEST*/
