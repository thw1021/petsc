static char help[] = "Test fully-blocked parallel PtAP and AB for MATMPIBAIJKOKKOS (F10.2/F10.4).\n\
Computes C = P^T A P and C = A P with block operands and verifies (a) C is returned as MATMPIBAIJKOKKOS\n\
and (b) C matches the scalar MPIAIJ reference (Frobenius norm of the difference ~0).\n\n";

#include <petscmat.h>

/*
  TestPtAPBlock - build a block-tridiagonal square A (bsA x bsA blocks) and a banded rectangular
  prolongator P (bsA x bsP blocks), then compare the MATMPIBAIJKOKKOS PtAP against the MPIAIJ reference.

  mbs - global block rows of A (and block rows of P)
  cbs - global block columns of P (coarse block rows)
  bsA - block size of A (fine dof per node)
  bsP - column block size of P (coarse near-null-space dimension)
*/
static PetscErrorCode TestPtAPBlock(PetscInt mbs, PetscInt cbs, PetscInt bsA, PetscInt bsP, const char *case_name)
{
  Mat       A_aij, P_aij, C_ref, A_kok, P_kok, C_kok, C_back, C_diff;
  PetscInt  n = mbs * bsA, nc = cbs * bsP;
  PetscInt  ib, jb, ii, jj, jlo, jhi;
  PetscReal norm_ref, norm_diff, rel_err;
  PetscBool is_block;

  PetscFunctionBeginUser;
  /* A: square, block-tridiagonal (bsA x bsA blocks). */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A_aij));
  PetscCall(MatSetSizes(A_aij, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A_aij, MATMPIAIJ));
  PetscCall(MatSetBlockSizes(A_aij, bsA, bsA));
  PetscCall(MatMPIAIJSetPreallocation(A_aij, 3 * bsA, NULL, 3 * bsA, NULL));
  for (ib = 0; ib < mbs; ib++) {
    jlo = PetscMax(ib - 1, 0);
    jhi = PetscMin(ib + 1, mbs - 1);
    for (ii = 0; ii < bsA; ii++) {
      PetscInt row = ib * bsA + ii;
      for (jb = jlo; jb <= jhi; jb++)
        for (jj = 0; jj < bsA; jj++) {
          PetscInt    col = jb * bsA + jj;
          PetscScalar v   = PetscSinReal((PetscReal)(ib + jb + ii + jj + 1)) + (row == col ? 4.0 : 0.0);
          PetscCall(MatSetValues(A_aij, 1, &row, 1, &col, &v, INSERT_VALUES));
        }
    }
  }
  PetscCall(MatAssemblyBegin(A_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_aij, MAT_FINAL_ASSEMBLY));

  /* P: rectangular n x nc, each fine block-row ib maps to coarse block-cols {ib/ratio +/- 0}. */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &P_aij));
  PetscCall(MatSetSizes(P_aij, PETSC_DECIDE, PETSC_DECIDE, n, nc));
  PetscCall(MatSetType(P_aij, MATMPIAIJ));
  PetscCall(MatSetBlockSizes(P_aij, bsA, bsP));
  PetscCall(MatMPIAIJSetPreallocation(P_aij, 2 * bsP, NULL, 2 * bsP, NULL));
  for (ib = 0; ib < mbs; ib++) {
    PetscInt cb0 = PetscMin((ib * cbs) / mbs, cbs - 1);
    for (ii = 0; ii < bsA; ii++) {
      PetscInt row = ib * bsA + ii;
      for (jj = 0; jj < bsP; jj++) {
        PetscInt    col = cb0 * bsP + jj;
        PetscScalar v   = PetscCosReal((PetscReal)(ib + ii + jj + 1));
        PetscCall(MatSetValues(P_aij, 1, &row, 1, &col, &v, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(P_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P_aij, MAT_FINAL_ASSEMBLY));

  /* Reference: scalar MPIAIJ PtAP. */
  PetscCall(MatPtAP(A_aij, P_aij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C_ref));

  /* Block path: convert operands, PtAP via MatProduct. */
  PetscCall(MatConvert(A_aij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &A_kok));
  PetscCall(MatConvert(P_aij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &P_kok));
  PetscCall(MatProductCreate(A_kok, P_kok, NULL, &C_kok));
  PetscCall(MatProductSetType(C_kok, MATPRODUCT_PtAP));
  PetscCall(MatProductSetFromOptions(C_kok));
  PetscCall(MatProductSymbolic(C_kok));
  PetscCall(MatProductNumeric(C_kok));

  PetscCall(PetscObjectTypeCompare((PetscObject)C_kok, MATMPIBAIJKOKKOS, &is_block));

  /* Compare C_kok (as AIJ) against the reference. */
  PetscCall(MatConvert(C_kok, MATMPIAIJ, MAT_INITIAL_MATRIX, &C_back));
  PetscCall(MatDuplicate(C_back, MAT_COPY_VALUES, &C_diff));
  PetscCall(MatAXPY(C_diff, -1.0, C_ref, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C_ref, NORM_FROBENIUS, &norm_ref));
  PetscCall(MatNorm(C_diff, NORM_FROBENIUS, &norm_diff));
  rel_err = norm_ref > 0.0 ? norm_diff / norm_ref : norm_diff;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-26s bsA=%" PetscInt_FMT " bsP=%" PetscInt_FMT " block=%s rel_err=%g %s\n", case_name, bsA, bsP, is_block ? "yes" : "no", (double)rel_err, (is_block && rel_err < 1e-10) ? "PASS" : "FAIL"));

  PetscCall(MatDestroy(&A_aij));
  PetscCall(MatDestroy(&P_aij));
  PetscCall(MatDestroy(&C_ref));
  PetscCall(MatDestroy(&A_kok));
  PetscCall(MatDestroy(&P_kok));
  PetscCall(MatDestroy(&C_kok));
  PetscCall(MatDestroy(&C_back));
  PetscCall(MatDestroy(&C_diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TestABBlock - build the same block-tridiagonal square A and banded rectangular prolongator P as
  TestPtAPBlock(), then compare the MATMPIBAIJKOKKOS product C = A P (MATPRODUCT_AB, rectangular
  bsA x bsP blocks) against the MPIAIJ MatMatMult reference. This is the GAMG prolongator-smoothing path.
*/
static PetscErrorCode TestABBlock(PetscInt mbs, PetscInt cbs, PetscInt bsA, PetscInt bsP, const char *case_name)
{
  Mat       A_aij, P_aij, C_ref, A_kok, P_kok, C_kok, C_back, C_diff;
  PetscInt  n = mbs * bsA, nc = cbs * bsP;
  PetscInt  ib, jb, ii, jj, jlo, jhi;
  PetscReal norm_ref, norm_diff, rel_err;
  PetscBool is_block;

  PetscFunctionBeginUser;
  /* A: square, block-tridiagonal (bsA x bsA blocks). */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A_aij));
  PetscCall(MatSetSizes(A_aij, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A_aij, MATMPIAIJ));
  PetscCall(MatSetBlockSizes(A_aij, bsA, bsA));
  PetscCall(MatMPIAIJSetPreallocation(A_aij, 3 * bsA, NULL, 3 * bsA, NULL));
  for (ib = 0; ib < mbs; ib++) {
    jlo = PetscMax(ib - 1, 0);
    jhi = PetscMin(ib + 1, mbs - 1);
    for (ii = 0; ii < bsA; ii++) {
      PetscInt row = ib * bsA + ii;
      for (jb = jlo; jb <= jhi; jb++)
        for (jj = 0; jj < bsA; jj++) {
          PetscInt    col = jb * bsA + jj;
          PetscScalar v   = PetscSinReal((PetscReal)(ib + jb + ii + jj + 1)) + (row == col ? 4.0 : 0.0);
          PetscCall(MatSetValues(A_aij, 1, &row, 1, &col, &v, INSERT_VALUES));
        }
    }
  }
  PetscCall(MatAssemblyBegin(A_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_aij, MAT_FINAL_ASSEMBLY));

  /* P: rectangular n x nc, each fine block-row ib maps to coarse block-cols {ib/ratio +/- 0}. */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &P_aij));
  PetscCall(MatSetSizes(P_aij, PETSC_DECIDE, PETSC_DECIDE, n, nc));
  PetscCall(MatSetType(P_aij, MATMPIAIJ));
  PetscCall(MatSetBlockSizes(P_aij, bsA, bsP));
  PetscCall(MatMPIAIJSetPreallocation(P_aij, 2 * bsP, NULL, 2 * bsP, NULL));
  for (ib = 0; ib < mbs; ib++) {
    PetscInt cb0 = PetscMin((ib * cbs) / mbs, cbs - 1);
    for (ii = 0; ii < bsA; ii++) {
      PetscInt row = ib * bsA + ii;
      for (jj = 0; jj < bsP; jj++) {
        PetscInt    col = cb0 * bsP + jj;
        PetscScalar v   = PetscCosReal((PetscReal)(ib + ii + jj + 1));
        PetscCall(MatSetValues(P_aij, 1, &row, 1, &col, &v, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(P_aij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P_aij, MAT_FINAL_ASSEMBLY));

  /* Reference: scalar MPIAIJ A*P. */
  PetscCall(MatMatMult(A_aij, P_aij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C_ref));

  /* Block path: convert operands, AB via MatProduct. */
  PetscCall(MatConvert(A_aij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &A_kok));
  PetscCall(MatConvert(P_aij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &P_kok));
  PetscCall(MatProductCreate(A_kok, P_kok, NULL, &C_kok));
  PetscCall(MatProductSetType(C_kok, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(C_kok));
  PetscCall(MatProductSymbolic(C_kok));
  PetscCall(MatProductNumeric(C_kok));

  PetscCall(PetscObjectTypeCompare((PetscObject)C_kok, MATMPIBAIJKOKKOS, &is_block));

  /* Compare C_kok (as AIJ) against the reference. */
  PetscCall(MatConvert(C_kok, MATMPIAIJ, MAT_INITIAL_MATRIX, &C_back));
  PetscCall(MatDuplicate(C_back, MAT_COPY_VALUES, &C_diff));
  PetscCall(MatAXPY(C_diff, -1.0, C_ref, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C_ref, NORM_FROBENIUS, &norm_ref));
  PetscCall(MatNorm(C_diff, NORM_FROBENIUS, &norm_diff));
  rel_err = norm_ref > 0.0 ? norm_diff / norm_ref : norm_diff;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-26s bsA=%" PetscInt_FMT " bsP=%" PetscInt_FMT " block=%s rel_err=%g %s\n", case_name, bsA, bsP, is_block ? "yes" : "no", (double)rel_err, (is_block && rel_err < 1e-10) ? "PASS" : "FAIL"));

  PetscCall(MatDestroy(&A_aij));
  PetscCall(MatDestroy(&P_aij));
  PetscCall(MatDestroy(&C_ref));
  PetscCall(MatDestroy(&A_kok));
  PetscCall(MatDestroy(&P_kok));
  PetscCall(MatDestroy(&C_kok));
  PetscCall(MatDestroy(&C_back));
  PetscCall(MatDestroy(&C_diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(TestPtAPBlock(12, 6, 1, 1, "ptap_bs1"));
  PetscCall(TestPtAPBlock(12, 6, 3, 3, "ptap_bs3"));
  PetscCall(TestPtAPBlock(12, 4, 3, 6, "ptap_elasticity_3to6"));
  PetscCall(TestPtAPBlock(12, 4, 6, 6, "ptap_bs6"));

  PetscCall(TestABBlock(12, 6, 1, 1, "ab_bs1"));
  PetscCall(TestABBlock(12, 6, 3, 3, "ab_bs3"));
  PetscCall(TestABBlock(12, 4, 3, 6, "ab_elasticity_3to6"));
  PetscCall(TestABBlock(12, 4, 6, 6, "ab_bs6"));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      requires: kokkos_kernels
      output_file: output/ex309_1.out
      nsize: {{1 2 4}}
      test:
        suffix: 1

TEST*/
