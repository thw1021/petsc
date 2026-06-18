static char help[] = "Compare MatGetFactor LU solve: SEQBAIJKOKKOS (via AIJ-copy wrapper) vs SEQAIJ.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         Aaij, Abk, Fbk, Faij;
  Vec         b, xbk, xaij;
  IS          r, c;
  PetscInt    n = 12, bs = 3, i, j, rstart, rend;
  PetscReal   nrm;
  PetscScalar v;
  PetscRandom rnd;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  /* Build an SPD-ish scalar SEQAIJ matrix with block structure (block tridiagonal, bs=3). */
  PetscCall(MatCreate(PETSC_COMM_SELF, &Aaij));
  PetscCall(MatSetSizes(Aaij, n, n, n, n));
  PetscCall(MatSetType(Aaij, MATSEQAIJ));
  PetscCall(MatSetBlockSizes(Aaij, bs, bs));
  PetscCall(MatSeqAIJSetPreallocation(Aaij, 3 * bs, NULL));
  PetscCall(MatGetOwnershipRange(Aaij, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    for (j = PetscMax(0, i - bs); j < PetscMin(n, i + bs + 1); j++) {
      v = (i == j) ? 4.0 * bs : -1.0;
      PetscCall(MatSetValue(Aaij, i, j, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Aaij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Aaij, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(Aaij, MAT_SYMMETRIC, PETSC_TRUE));

  /* Block Kokkos copy of the same matrix. */
  PetscCall(MatConvert(Aaij, MATSEQBAIJKOKKOS, MAT_INITIAL_MATRIX, &Abk));

  /* RHS */
  PetscCall(MatCreateVecs(Aaij, &b, NULL));
  PetscCall(VecDuplicate(b, &xaij));
  PetscCall(MatCreateVecs(Abk, &xbk, NULL));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rnd));
  PetscCall(VecSetRandom(b, rnd));

  /* SEQAIJ factor+solve (reference). Use the same ND ordering the block wrapper computes internally. */
  PetscCall(MatGetOrdering(Aaij, MATORDERINGND, &r, &c));
  PetscCall(MatGetFactor(Aaij, MATSOLVERPETSC, MAT_FACTOR_LU, &Faij));
  PetscCall(MatLUFactorSymbolic(Faij, Aaij, r, c, NULL));
  PetscCall(MatLUFactorNumeric(Faij, Aaij, NULL));
  PetscCall(MatSolve(Faij, b, xaij));
  PetscCall(ISDestroy(&r));
  PetscCall(ISDestroy(&c));

  /* SEQBAIJKOKKOS factor+solve (via the AIJ-copy wrapper under test). The wrapper ignores the passed
     orderings and computes its own ND ordering on the converted copy, so NULL is fine here. */
  PetscCall(MatGetFactor(Abk, MATSOLVERPETSC, MAT_FACTOR_LU, &Fbk));
  PetscCall(MatLUFactorSymbolic(Fbk, Abk, NULL, NULL, NULL));
  PetscCall(MatLUFactorNumeric(Fbk, Abk, NULL));
  PetscCall(MatSolve(Fbk, b, xbk));

  /* Compare: two LU solves of the same matrix must agree to round-off. */
  PetscCall(VecAXPY(xbk, -1.0, xaij));
  PetscCall(VecNorm(xbk, NORM_2, &nrm));
  PetscCall(VecNorm(xaij, NORM_2, &v));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "SEQBAIJKOKKOS-factor vs SEQAIJ-factor solve: %s\n", (nrm / PetscRealPart(v)) < 1e-10 ? "PASS" : "FAIL"));

  PetscCall(PetscRandomDestroy(&rnd));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&xaij));
  PetscCall(VecDestroy(&xbk));
  PetscCall(MatDestroy(&Aaij));
  PetscCall(MatDestroy(&Abk));
  PetscCall(MatDestroy(&Faij));
  PetscCall(MatDestroy(&Fbk));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      requires: kokkos_kernels
      nsize: 1

TEST*/
