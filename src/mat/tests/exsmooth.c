static char help[] = "Isolate the GAMG prolongator-smoothing ops for MATMPIBAIJKOKKOS.\n\
Replicates the agg.c smoothing sequence  P := P0 - (1.4/emax) D^{-1} A P0  on a rectangular block\n\
prolongator and compares the block path against the scalar MPIAIJ reference after each stage:\n\
  (1) tMat = A*P0           (MATPRODUCT_AB)\n\
  (2) MatDiagonalScale(tMat, D^{-1}, NULL)\n\
  (3) MatAYPX(tMat, -1.4/emax, P0, SUBSET_NONZERO_PATTERN)\n\
This pinpoints which op introduces the block-vs-AIJ divergence seen in ex56 GAMG (31 vs 17 iters).\n\n";

#include <petscmat.h>

/* Frobenius rel_err between a block matrix (converted to MPIAIJ) and an MPIAIJ reference. */
static PetscErrorCode CompareToRef(Mat C_kok, Mat C_ref, const char *stage)
{
  Mat       C_back, C_diff;
  PetscReal norm_ref, norm_diff, rel_err;

  PetscFunctionBeginUser;
  PetscCall(MatConvert(C_kok, MATMPIAIJ, MAT_INITIAL_MATRIX, &C_back));
  PetscCall(MatDuplicate(C_back, MAT_COPY_VALUES, &C_diff));
  PetscCall(MatAXPY(C_diff, -1.0, C_ref, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C_ref, NORM_FROBENIUS, &norm_ref));
  PetscCall(MatNorm(C_diff, NORM_FROBENIUS, &norm_diff));
  rel_err = norm_ref > 0.0 ? norm_diff / norm_ref : norm_diff;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  stage %-16s %s\n", stage, rel_err < 1e-10 ? "PASS" : "FAIL"));
  PetscCall(MatDestroy(&C_back));
  PetscCall(MatDestroy(&C_diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build the block-tridiagonal square A (bsA x bsA) and banded rectangular P0 (bsA x bsP) used by ex309. */
static PetscErrorCode BuildAP(PetscInt mbs, PetscInt cbs, PetscInt bsA, PetscInt bsP, Mat *A_out, Mat *P_out)
{
  Mat      A, P;
  PetscInt n = mbs * bsA, nc = cbs * bsP;
  PetscInt ib, jb, ii, jj, jlo, jhi;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatSetBlockSizes(A, bsA, bsA));
  PetscCall(MatMPIAIJSetPreallocation(A, 3 * bsA, NULL, 3 * bsA, NULL));
  for (ib = 0; ib < mbs; ib++) {
    jlo = PetscMax(ib - 1, 0);
    jhi = PetscMin(ib + 1, mbs - 1);
    for (ii = 0; ii < bsA; ii++) {
      PetscInt row = ib * bsA + ii;
      for (jb = jlo; jb <= jhi; jb++)
        for (jj = 0; jj < bsA; jj++) {
          PetscInt    col = jb * bsA + jj;
          PetscScalar v   = PetscSinReal((PetscReal)(ib + jb + ii + jj + 1)) + (row == col ? 4.0 : 0.0);
          PetscCall(MatSetValues(A, 1, &row, 1, &col, &v, INSERT_VALUES));
        }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &P));
  PetscCall(MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, n, nc));
  PetscCall(MatSetType(P, MATMPIAIJ));
  PetscCall(MatSetBlockSizes(P, bsA, bsP));
  PetscCall(MatMPIAIJSetPreallocation(P, 2 * bsP, NULL, 2 * bsP, NULL));
  for (ib = 0; ib < mbs; ib++) {
    PetscInt cb0 = PetscMin((ib * cbs) / mbs, cbs - 1);
    for (ii = 0; ii < bsA; ii++) {
      PetscInt row = ib * bsA + ii;
      for (jj = 0; jj < bsP; jj++) {
        PetscInt    col = cb0 * bsP + jj;
        PetscScalar v   = PetscCosReal((PetscReal)(ib + ii + jj + 1));
        PetscCall(MatSetValues(P, 1, &row, 1, &col, &v, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  *A_out = A;
  *P_out = P;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Run the agg.c smoothing sequence on AIJ (reference) and block, compare after each stage. */
static PetscErrorCode TestSmooth(PetscInt mbs, PetscInt cbs, PetscInt bsA, PetscInt bsP, const char *case_name)
{
  Mat         A_aij, P_aij, A_kok, P_kok;
  Mat         t_aij, t_kok;
  Vec         d_aij, d_kok;
  PetscScalar alpha = -1.4 / 3.0; /* mimic -1.4/emax with a fixed emax so AIJ and block use the same factor */

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s (bsA=%" PetscInt_FMT " bsP=%" PetscInt_FMT "):\n", case_name, bsA, bsP));
  PetscCall(BuildAP(mbs, cbs, bsA, bsP, &A_aij, &P_aij));
  PetscCall(MatConvert(A_aij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &A_kok));
  PetscCall(MatConvert(P_aij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &P_kok));

  /* D^{-1} from A. */
  PetscCall(MatCreateVecs(A_aij, &d_aij, NULL));
  PetscCall(MatGetDiagonal(A_aij, d_aij));
  PetscCall(VecReciprocal(d_aij));
  PetscCall(MatCreateVecs(A_kok, &d_kok, NULL));
  PetscCall(MatGetDiagonal(A_kok, d_kok));
  PetscCall(VecReciprocal(d_kok));

  /* Stage 1: tMat = A*P0. */
  PetscCall(MatMatMult(A_aij, P_aij, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &t_aij));
  PetscCall(MatProductCreate(A_kok, P_kok, NULL, &t_kok));
  PetscCall(MatProductSetType(t_kok, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(t_kok));
  PetscCall(MatProductSymbolic(t_kok));
  PetscCall(MatProductNumeric(t_kok));
  PetscCall(CompareToRef(t_kok, t_aij, "AB"));

  /* Stage 1b: MatScale on the product result (uniform scale; does not read i/j). Harmless to the
     downstream comparison since both sides get the same factor. Isolates whether plain MatScale works
     on an AB-product result, vs. MatDiagonalScale which reads the device i/j block structure. */
  PetscCall(MatScale(t_aij, 0.5));
  PetscCall(MatScale(t_kok, 0.5));
  PetscCall(CompareToRef(t_kok, t_aij, "Scale(prod)"));

  /* Stage 1c: MatDiagonalScale on a fresh, assembly-built block matrix (P0), which has device i/j from
     MatAssemblyEnd. If this PASSes but Stage 2 FAILs, the bug is device i/j on the AB-product result. */
  {
    Mat Pa, Pk;
    PetscCall(MatDuplicate(P_aij, MAT_COPY_VALUES, &Pa));
    PetscCall(MatConvert(P_aij, MATMPIBAIJKOKKOS, MAT_INITIAL_MATRIX, &Pk));
    PetscCall(MatDiagonalScale(Pa, d_aij, NULL));
    PetscCall(MatDiagonalScale(Pk, d_kok, NULL));
    PetscCall(CompareToRef(Pk, Pa, "DiagScale(fresh)"));
    PetscCall(MatDestroy(&Pa));
    PetscCall(MatDestroy(&Pk));
  }

  /* Stage 2: MatDiagonalScale(tMat, D^{-1}, NULL) on the AB-product result. */
  PetscCall(MatDiagonalScale(t_aij, d_aij, NULL));
  PetscCall(MatDiagonalScale(t_kok, d_kok, NULL));
  PetscCall(CompareToRef(t_kok, t_aij, "DiagScale(prod)"));

  /* Stage 3: MatAYPX(tMat, alpha, P0, SUBSET_NONZERO_PATTERN). */
  PetscCall(MatAYPX(t_aij, alpha, P_aij, SUBSET_NONZERO_PATTERN));
  PetscCall(MatAYPX(t_kok, alpha, P_kok, SUBSET_NONZERO_PATTERN));
  PetscCall(CompareToRef(t_kok, t_aij, "AYPX"));

  PetscCall(VecDestroy(&d_aij));
  PetscCall(VecDestroy(&d_kok));
  PetscCall(MatDestroy(&A_aij));
  PetscCall(MatDestroy(&P_aij));
  PetscCall(MatDestroy(&A_kok));
  PetscCall(MatDestroy(&P_kok));
  PetscCall(MatDestroy(&t_aij));
  PetscCall(MatDestroy(&t_kok));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(TestSmooth(12, 4, 3, 6, "smooth_elasticity_3to6"));
  PetscCall(TestSmooth(12, 6, 3, 3, "smooth_bs3"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      requires: kokkos_kernels
      nsize: {{1 2 4}}
      test:
        suffix: 1

TEST*/
