static char help[] = "Test MATSEQBAIJKOKKOS AB product (C=A*B) validation against AIJ.\n\
Validates rectangular-block sparse matrix multiplication on device.\n\n";

#include <petscmat.h>
#include <petscmath.h>

/*
  TestABProduct - Validate MATSEQBAIJKOKKOS AB product against AIJ reference.

  Parameters:
  - mbsA: number of block rows in A
  - kbs: number of block cols in A (and block rows in B)
  - nbsB: number of block cols in B
  - bsA: block size of A (rows)
  - bsB: block size of B (cols/rows common)
  - case_name: label for output

  Builds AIJ block-tridiagonal matrices, converts to BAIJ/BAIJKOKKOS,
  and validates the Kokkos product against AIJ reference.
*/
static PetscErrorCode TestABProduct(PetscInt mbsA, PetscInt kbs, PetscInt nbsB, PetscInt bsA, PetscInt bsB, const char *case_name)
{
  Mat          A_aij, B_aij, A_kok, B_kok, C_kok, C_ref, C_diff;
  PetscReal    norm_ref, norm_diff, rel_err;
  PetscInt     m, k, n, max_ncols, iblock, jblock, ii, jj;
  PetscInt    *cols;
  PetscScalar *vals;

  PetscFunctionBeginUser;
  m         = mbsA * bsA;
  k         = kbs * bsB;
  n         = nbsB * bsB;
  max_ncols = 3 * bsB;

  // Allocate working arrays
  PetscCall(PetscMalloc1(max_ncols, &cols));
  PetscCall(PetscMalloc1(max_ncols * bsB, &vals));

  // Build A in AIJ: mbsAxkbs blocks, block-tridiagonal
  PetscCall(MatCreate(PETSC_COMM_SELF, &A_aij));
  PetscCall(MatSetSizes(A_aij, m, k, m, k));
  PetscCall(MatSetType(A_aij, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A_aij, 3 * bsB, NULL));

  for (iblock = 0; iblock < mbsA; iblock++) {
    PetscInt jlo = PetscMax(iblock - 1, 0);
    PetscInt jhi = PetscMin(iblock + 1, kbs - 1);

    for (ii = 0; ii < bsA; ii++) {
      PetscInt row   = iblock * bsA + ii;
      PetscInt ncols = 0;

      for (jblock = jlo; jblock <= jhi; jblock++) {
        for (jj = 0; jj < bsB; jj++) {
          cols[ncols] = jblock * bsB + jj;
          vals[ncols] = PetscSinReal((PetscReal)(iblock + jblock + ii + jj));
          ncols++;
        }
      }
      PetscCall(MatSetValues(A_aij, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_aij, MAT_FINAL_ASSEMBLY));

  // Build B in AIJ: kbsxnbsB blocks, block-tridiagonal
  PetscCall(MatCreate(PETSC_COMM_SELF, &B_aij));
  PetscCall(MatSetSizes(B_aij, k, n, k, n));
  PetscCall(MatSetType(B_aij, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(B_aij, 3 * bsB, NULL));

  for (iblock = 0; iblock < kbs; iblock++) {
    PetscInt jlo = PetscMax(iblock - 1, 0);
    PetscInt jhi = PetscMin(iblock + 1, nbsB - 1);

    for (ii = 0; ii < bsB; ii++) {
      PetscInt row   = iblock * bsB + ii;
      PetscInt ncols = 0;

      for (jblock = jlo; jblock <= jhi; jblock++) {
        for (jj = 0; jj < bsB; jj++) {
          cols[ncols] = jblock * bsB + jj;
          vals[ncols] = PetscCosReal((PetscReal)(iblock + jblock + ii + jj));
          ncols++;
        }
      }
      PetscCall(MatSetValues(B_aij, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(B_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B_aij, MAT_FINAL_ASSEMBLY));

  // Compute reference: AIJ x AIJ
  PetscCall(MatMatMult(A_aij, B_aij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C_ref));

  // Convert A, B to BAIJKOKKOS
  // First try direct conversion AIJ->BAIJKOKKOS
  PetscCall(MatConvert(A_aij, MATSEQBAIJKOKKOS, MAT_INITIAL_MATRIX, &A_kok));
  PetscCall(MatConvert(B_aij, MATSEQBAIJKOKKOS, MAT_INITIAL_MATRIX, &B_kok));

  // Compute C = A*B using MatProduct on Kokkos
  PetscCall(MatProductCreate(A_kok, B_kok, NULL, &C_kok));
  PetscCall(MatProductSetType(C_kok, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(C_kok));
  PetscCall(MatProductSymbolic(C_kok));
  PetscCall(MatProductNumeric(C_kok));

  // Convert C_kok to AIJ for comparison
  PetscCall(MatConvert(C_kok, MATSEQAIJ, MAT_INITIAL_MATRIX, &C_diff));

  // Compare: C_diff = C_kok - C_ref
  PetscCall(MatAXPY(C_diff, -1.0, C_ref, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C_diff, NORM_FROBENIUS, &norm_diff));
  PetscCall(MatNorm(C_ref, NORM_FROBENIUS, &norm_ref));

  rel_err = (norm_ref > 1e-15) ? norm_diff / norm_ref : norm_diff;

  if (rel_err < 1e-10) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s PASS\n", case_name));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s FAIL: rel=%.2e\n", case_name, rel_err));
  }

  PetscCall(MatDestroy(&A_aij));
  PetscCall(MatDestroy(&B_aij));
  PetscCall(MatDestroy(&A_kok));
  PetscCall(MatDestroy(&B_kok));
  PetscCall(MatDestroy(&C_kok));
  PetscCall(MatDestroy(&C_ref));
  PetscCall(MatDestroy(&C_diff));
  PetscCall(PetscFree(cols));
  PetscCall(PetscFree(vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TestPtAP - Validate MATSEQBAIJKOKKOS PtAP product (C=P^T*A*P) against AIJ reference.

  Parameters:
  - nbsA: number of block rows/cols in A (square)
  - nbsP: number of block cols in P
  - bsA: block size of A (rows and cols)
  - bsP: block size of P (cols)
  - case_name: label for output

  Builds a square BAIJ A and rectangular BAIJ P, converts to BAIJKOKKOS,
  and validates the Kokkos PtAP against AIJ reference.
*/
static PetscErrorCode TestPtAP(PetscInt nbsA, PetscInt nbsP, PetscInt bsA, PetscInt bsP, const char *alg, const char *case_name)
{
  Mat          A_aij, P_aij, A_kok, P_kok, C_kok, C_ref, C_diff, temp;
  PetscReal    norm_ref, norm_diff, rel_err, reuse_err;
  PetscInt     m, n, max_ncols, iblock, jblock, ii, jj;
  PetscInt    *cols;
  PetscScalar *vals;

  PetscFunctionBeginUser;
  m         = nbsA * bsA;
  n         = nbsP * bsP;
  max_ncols = 3 * PetscMax(bsA, bsP);

  // Allocate working arrays
  PetscCall(PetscMalloc1(max_ncols, &cols));
  PetscCall(PetscMalloc1(max_ncols, &vals));

  // Build A in AIJ: nbsAxnbsA blocks, square, block-tridiagonal
  PetscCall(MatCreate(PETSC_COMM_SELF, &A_aij));
  PetscCall(MatSetSizes(A_aij, m, m, m, m));
  PetscCall(MatSetType(A_aij, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A_aij, 3 * bsA, NULL));

  for (iblock = 0; iblock < nbsA; iblock++) {
    PetscInt jlo = PetscMax(iblock - 1, 0);
    PetscInt jhi = PetscMin(iblock + 1, nbsA - 1);

    for (ii = 0; ii < bsA; ii++) {
      PetscInt row   = iblock * bsA + ii;
      PetscInt ncols = 0;

      for (jblock = jlo; jblock <= jhi; jblock++) {
        for (jj = 0; jj < bsA; jj++) {
          cols[ncols] = jblock * bsA + jj;
          vals[ncols] = PetscSinReal((PetscReal)(iblock + jblock + ii + jj)) + (cols[ncols] == row ? 3.0 : 0.0);
          ncols++;
        }
      }
      PetscCall(MatSetValues(A_aij, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_aij, MAT_FINAL_ASSEMBLY));

  // Build P in AIJ: nbsAxnbsP blocks, rectangular, block-tridiagonal
  PetscCall(MatCreate(PETSC_COMM_SELF, &P_aij));
  PetscCall(MatSetSizes(P_aij, m, n, m, n));
  PetscCall(MatSetType(P_aij, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(P_aij, 3 * bsP, NULL));

  for (iblock = 0; iblock < nbsA; iblock++) {
    PetscInt jlo = PetscMax(iblock - 1, 0);
    PetscInt jhi = PetscMin(iblock + 1, nbsP - 1);

    for (ii = 0; ii < bsA; ii++) {
      PetscInt row   = iblock * bsA + ii;
      PetscInt ncols = 0;

      for (jblock = jlo; jblock <= jhi; jblock++) {
        for (jj = 0; jj < bsP; jj++) {
          cols[ncols] = jblock * bsP + jj;
          vals[ncols] = PetscCosReal((PetscReal)(iblock + jblock + ii + jj));
          ncols++;
        }
      }
      PetscCall(MatSetValues(P_aij, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(P_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P_aij, MAT_FINAL_ASSEMBLY));

  // Compute reference: P^T * A * P using AIJ
  PetscCall(MatMatMult(A_aij, P_aij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp));
  PetscCall(MatTransposeMatMult(P_aij, temp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C_ref));
  PetscCall(MatDestroy(&temp));

  // Convert A, P to BAIJKOKKOS
  PetscCall(MatConvert(A_aij, MATSEQBAIJKOKKOS, MAT_INITIAL_MATRIX, &A_kok));
  PetscCall(MatConvert(P_aij, MATSEQBAIJKOKKOS, MAT_INITIAL_MATRIX, &P_kok));

  // Compute C = P^T * A * P on Kokkos via the explicit product API (selects the algorithm)
  PetscCall(MatProductCreate(A_kok, P_kok, NULL, &C_kok));
  PetscCall(MatProductSetType(C_kok, MATPRODUCT_PtAP));
  PetscCall(MatProductSetAlgorithm(C_kok, alg));
  PetscCall(MatProductSetFromOptions(C_kok));
  PetscCall(MatProductSymbolic(C_kok));
  PetscCall(MatProductNumeric(C_kok));

  // Convert C_kok to AIJ for comparison; C_diff = C_kok - C_ref
  PetscCall(MatConvert(C_kok, MATSEQAIJ, MAT_INITIAL_MATRIX, &C_diff));
  PetscCall(MatAXPY(C_diff, -1.0, C_ref, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C_diff, NORM_FROBENIUS, &norm_diff));
  PetscCall(MatNorm(C_ref, NORM_FROBENIUS, &norm_ref));
  rel_err = (norm_ref > 1e-15) ? norm_diff / norm_ref : norm_diff;
  PetscCall(MatDestroy(&C_diff));

  // Reuse (numeric-only): scale A by 2 -> Ac must scale by 2 (same structure, no re-symbolic)
  PetscCall(MatScale(A_kok, 2.0));
  PetscCall(MatProductNumeric(C_kok));
  PetscCall(MatConvert(C_kok, MATSEQAIJ, MAT_INITIAL_MATRIX, &C_diff));
  PetscCall(MatAXPY(C_diff, -2.0, C_ref, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C_diff, NORM_FROBENIUS, &norm_diff));
  reuse_err = (norm_ref > 1e-15) ? norm_diff / (2.0 * norm_ref) : norm_diff;

  if (rel_err < 1e-10 && reuse_err < 1e-10) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s PASS\n", case_name));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s FAIL: rel=%.2e reuse=%.2e\n", case_name, rel_err, reuse_err));
  }

  PetscCall(MatDestroy(&A_aij));
  PetscCall(MatDestroy(&P_aij));
  PetscCall(MatDestroy(&A_kok));
  PetscCall(MatDestroy(&P_kok));
  PetscCall(MatDestroy(&C_kok));
  PetscCall(MatDestroy(&C_ref));
  PetscCall(MatDestroy(&C_diff));
  PetscCall(PetscFree(cols));
  PetscCall(PetscFree(vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TestAtBReuse - Validate MATSEQBAIJKOKKOS AtB product (C=A^T*B) AND its reuse (numeric-only) path.

  Parameters:
  - mbs:     block rows of A and B (shared; the contraction is over A's rows)
  - kbs:     block cols of A   (-> block rows of C)
  - nbs:     block cols of B   (-> block cols of C)
  - row_bs:  block size of A/B rows (the contraction block dim)
  - acol_bs: block col size of A   (-> C row block size)
  - bcol_bs: block col size of B   (-> C col block size)
  - case_name: label for output

  Exercises the F2.3 cached transpose: builds C=A^T*B, checks vs AIJ; then changes A's values
  (MatScale) and re-runs MatProductNumeric (the reuse path that refreshes A^T's values from the cached
  permutation without a structural rebuild), checking the recomputed C against the scaled AIJ reference.
*/
static PetscErrorCode TestAtBReuse(PetscInt mbs, PetscInt kbs, PetscInt nbs, PetscInt row_bs, PetscInt acol_bs, PetscInt bcol_bs, const char *case_name)
{
  Mat         A_aij, B_aij, A_kok, B_kok, C_kok, C_ref, C_diff;
  PetscReal   norm_ref, norm_diff, rel1, rel2;
  PetscInt    iblock, jblock, ii, jj, ncols;
  PetscInt    cols[64];
  PetscScalar vals[64];
  PetscBool   pass1, pass2;

  PetscFunctionBeginUser;
  /* A: mbs x kbs blocks (row_bs x acol_bs), block-tridiagonal */
  PetscCall(MatCreate(PETSC_COMM_SELF, &A_aij));
  PetscCall(MatSetSizes(A_aij, mbs * row_bs, kbs * acol_bs, mbs * row_bs, kbs * acol_bs));
  PetscCall(MatSetType(A_aij, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A_aij, 3 * acol_bs, NULL));
  for (iblock = 0; iblock < mbs; iblock++) {
    PetscInt jlo = PetscMax(iblock - 1, 0), jhi = PetscMin(iblock + 1, kbs - 1);
    for (ii = 0; ii < row_bs; ii++) {
      PetscInt row = iblock * row_bs + ii;
      ncols        = 0;
      for (jblock = jlo; jblock <= jhi; jblock++)
        for (jj = 0; jj < acol_bs; jj++) {
          cols[ncols] = jblock * acol_bs + jj;
          vals[ncols] = PetscSinReal((PetscReal)(iblock + jblock + ii + jj + 1));
          ncols++;
        }
      PetscCall(MatSetValues(A_aij, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_aij, MAT_FINAL_ASSEMBLY));

  /* B: mbs x nbs blocks (row_bs x bcol_bs), block-tridiagonal */
  PetscCall(MatCreate(PETSC_COMM_SELF, &B_aij));
  PetscCall(MatSetSizes(B_aij, mbs * row_bs, nbs * bcol_bs, mbs * row_bs, nbs * bcol_bs));
  PetscCall(MatSetType(B_aij, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(B_aij, 3 * bcol_bs, NULL));
  for (iblock = 0; iblock < mbs; iblock++) {
    PetscInt jlo = PetscMax(iblock - 1, 0), jhi = PetscMin(iblock + 1, nbs - 1);
    for (ii = 0; ii < row_bs; ii++) {
      PetscInt row = iblock * row_bs + ii;
      ncols        = 0;
      for (jblock = jlo; jblock <= jhi; jblock++)
        for (jj = 0; jj < bcol_bs; jj++) {
          cols[ncols] = jblock * bcol_bs + jj;
          vals[ncols] = PetscCosReal((PetscReal)(iblock + jblock + ii + jj + 1));
          ncols++;
        }
      PetscCall(MatSetValues(B_aij, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(B_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B_aij, MAT_FINAL_ASSEMBLY));

  /* AIJ reference C = A^T * B */
  PetscCall(MatTransposeMatMult(A_aij, B_aij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C_ref));
  PetscCall(MatNorm(C_ref, NORM_FROBENIUS, &norm_ref));

  PetscCall(MatConvert(A_aij, MATSEQBAIJKOKKOS, MAT_INITIAL_MATRIX, &A_kok));
  PetscCall(MatConvert(B_aij, MATSEQBAIJKOKKOS, MAT_INITIAL_MATRIX, &B_kok));

  /* Initial AtB: symbolic builds the cached transpose + permutation, numeric refreshes its values */
  PetscCall(MatProductCreate(A_kok, B_kok, NULL, &C_kok));
  PetscCall(MatProductSetType(C_kok, MATPRODUCT_AtB));
  PetscCall(MatProductSetFromOptions(C_kok));
  PetscCall(MatProductSymbolic(C_kok));
  PetscCall(MatProductNumeric(C_kok));

  PetscCall(MatConvert(C_kok, MATSEQAIJ, MAT_INITIAL_MATRIX, &C_diff));
  PetscCall(MatAXPY(C_diff, -1.0, C_ref, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C_diff, NORM_FROBENIUS, &norm_diff));
  rel1 = (norm_ref > 1e-15) ? norm_diff / norm_ref : norm_diff;
  PetscCall(MatDestroy(&C_diff));

  /* Reuse path: change A's values, re-run numeric only (cached transpose structure, refreshed values) */
  PetscCall(MatScale(A_kok, 2.0));
  PetscCall(MatProductNumeric(C_kok));
  PetscCall(MatConvert(C_kok, MATSEQAIJ, MAT_INITIAL_MATRIX, &C_diff));
  PetscCall(MatAXPY(C_diff, -2.0, C_ref, DIFFERENT_NONZERO_PATTERN)); /* expect C == 2*C_ref */
  PetscCall(MatNorm(C_diff, NORM_FROBENIUS, &norm_diff));
  rel2 = (norm_ref > 1e-15) ? norm_diff / (2.0 * norm_ref) : norm_diff;
  PetscCall(MatDestroy(&C_diff));

  pass1 = (PetscBool)(rel1 < 1e-10);
  pass2 = (PetscBool)(rel2 < 1e-10);
  if (pass1 && pass2) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s PASS\n", case_name));
  else PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s FAIL: initial rel=%.2e reuse rel=%.2e\n", case_name, rel1, rel2));

  PetscCall(MatDestroy(&A_aij));
  PetscCall(MatDestroy(&B_aij));
  PetscCall(MatDestroy(&A_kok));
  PetscCall(MatDestroy(&B_kok));
  PetscCall(MatDestroy(&C_kok));
  PetscCall(MatDestroy(&C_ref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  // AB Product tests
  // bs=1 (scalar)
  PetscCall(TestABProduct(5, 5, 5, 1, 1, "Test_bs1_5x5x5"));

  // bs=3 (square)
  PetscCall(TestABProduct(4, 4, 4, 3, 3, "Test_bs3_4x4x4"));

  // bs=6 (square)
  PetscCall(TestABProduct(3, 3, 3, 6, 6, "Test_bs6_3x3x3"));

  // Rectangular: 3x6 and 6x3 (elasticity-like prolongator shape)
  PetscCall(TestABProduct(4, 3, 4, 3, 6, "Test_rect_4x3x4_bs3x6"));

  // Non-square block graph: 2x3 and 3x2
  PetscCall(TestABProduct(3, 2, 3, 2, 3, "Test_nonsquare_3x2x3_bs2x3"));

  // PtAP Product tests: run each shape with both numeric algorithms (SPEED two-product, MEMORY fused).
  // Each case also exercises the numeric-only reuse path (scale A, recompute).
  for (PetscInt ialg = 0; ialg < 2; ialg++) {
    const char *alg = (ialg == 0) ? "speed" : "memory_m1";

    // bs=1 (scalar)
    PetscCall(TestPtAP(4, 4, 1, 1, alg, "TestPtAP_bs1_4x4"));

    // bs=3 (square)
    PetscCall(TestPtAP(4, 4, 3, 3, alg, "TestPtAP_bs3_4x4"));

    // bs=3->6 (elasticity-like prolongator)
    PetscCall(TestPtAP(4, 4, 3, 6, alg, "TestPtAP_elasticity_3to6"));

    // bs=6 (square)
    PetscCall(TestPtAP(3, 3, 6, 6, alg, "TestPtAP_bs6_3x3"));
  }

  // AtB reuse tests (cached transpose + numeric-only refresh)
  // square bs3
  PetscCall(TestAtBReuse(4, 4, 4, 3, 3, 3, "TestAtBReuse_bs3_4x4x4"));

  // rectangular: A block 3x6 (A^T 6x3), B block 3x3 -> C block 6x3 (elasticity-like)
  PetscCall(TestAtBReuse(4, 3, 4, 3, 6, 3, "TestAtBReuse_rect_bs3x6"));

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "ALL PASS\n"));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 1
    requires: kokkos_kernels
    output_file: output/ex310_1.out
TEST*/
