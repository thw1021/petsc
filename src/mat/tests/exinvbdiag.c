static char help[] = "Compare MatInvertBlockDiagonal: SEQBAIJKOKKOS vs SEQAIJ on a non-symmetric block matrix.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat                Aaij, Abk;
  PetscInt           n = 12, bs = 3, mbs, i, j, rstart, rend, bs2;
  PetscReal          nrm = 0.0, ref = 0.0;
  PetscScalar        v;
  const PetscScalar *idiag_aij, *idiag_bk;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  /* Build a NON-symmetric scalar SEQAIJ matrix with block structure (block tridiagonal, bs=3).
     Non-symmetry exercises the row-major -> column-major block transpose in the Kokkos path. */
  PetscCall(MatCreate(PETSC_COMM_SELF, &Aaij));
  PetscCall(MatSetSizes(Aaij, n, n, n, n));
  PetscCall(MatSetType(Aaij, MATSEQAIJ));
  PetscCall(MatSetBlockSizes(Aaij, bs, bs));
  PetscCall(MatSeqAIJSetPreallocation(Aaij, 3 * bs, NULL));
  PetscCall(MatGetOwnershipRange(Aaij, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    for (j = PetscMax(0, i - bs); j < PetscMin(n, i + bs + 1); j++) {
      if (i == j) v = 4.0 * bs;
      else v = -1.0 - 0.25 * (i - j); /* asymmetric: (i,j) != (j,i) */
      PetscCall(MatSetValue(Aaij, i, j, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Aaij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Aaij, MAT_FINAL_ASSEMBLY));

  /* Block Kokkos copy of the same matrix. */
  PetscCall(MatConvert(Aaij, MATSEQBAIJKOKKOS, MAT_INITIAL_MATRIX, &Abk));

  /* Invert block diagonals through both paths and compare the returned (column-major) arrays. */
  PetscCall(MatInvertBlockDiagonal(Aaij, &idiag_aij));
  PetscCall(MatInvertBlockDiagonal(Abk, &idiag_bk));

  bs2 = bs * bs;
  mbs = n / bs;
  for (i = 0; i < bs2 * mbs; i++) {
    v = idiag_bk[i] - idiag_aij[i];
    nrm += PetscRealPart(v * PetscConj(v));
    ref += PetscRealPart(idiag_aij[i] * PetscConj(idiag_aij[i]));
  }
  nrm = PetscSqrtReal(nrm);
  ref = PetscSqrtReal(ref);
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "SEQBAIJKOKKOS vs SEQAIJ block-diagonal inverse: %s\n", (nrm / ref) < 1e-10 ? "PASS" : "FAIL"));

  PetscCall(MatDestroy(&Aaij));
  PetscCall(MatDestroy(&Abk));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      requires: kokkos_kernels
      nsize: 1

TEST*/
