static char help[] = "Tests MatCreateKAIJAB() with a general second operand: K = (A x T) + (B x S).\n\n";

#include <petscmat.h>

/* Verify K2 = (A \otimes T) + (B \otimes S) built with MatCreateKAIJAB() against the sum of two
   single-operand KAIJ matrices: KAIJ(A, NULL, T) = A \otimes T and KAIJ(B, NULL, S) = B \otimes S.
   B is a structural copy of A so it shares the parallel layout and ghost columns required by MatCreateKAIJAB(). */
int main(int argc, char **argv)
{
  Mat          A, B, K2, KAT, KBS;
  Vec          x, y2, yr, ytmp;
  PetscScalar *S, *T;
  PetscReal    nrm;
  PetscInt     n = 30, i, j, p = 3, q = 3, rstart, rend, cols[3];
  PetscScalar  vals[3];
  PetscMPIInt  size;
  PetscBool    bnull = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-p", &p, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-q", &q, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-b_null", &bnull, NULL));

  /* Build a parallel tridiagonal AIJ matrix A (nonzero off-diagonal coupling exercises the MPI ghost path) */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 1, NULL));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscInt nc = 0;
    if (i > 0) {
      cols[nc]   = i - 1;
      vals[nc++] = -1.0;
    }
    cols[nc]   = i;
    vals[nc++] = 2.0 + (PetscScalar)i;
    if (i < n - 1) {
      cols[nc]   = i + 1;
      vals[nc++] = -1.0 - 0.5 * (PetscScalar)i;
    }
    PetscCall(MatSetValues(A, 1, &i, nc, cols, vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* B shares A's nonzero structure but has distinct values */
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatScale(B, 0.37));

  /* Dense p x q blocks S and T (column-major) */
  PetscCall(PetscMalloc2(p * q, &S, p * q, &T));
  for (i = 0; i < p; i++) {
    for (j = 0; j < q; j++) {
      S[i + p * j] = ((PetscReal)((i + 1) * (j + 2))) / ((PetscReal)(p + q));
      T[i + p * j] = ((PetscReal)((p - i) + 2 * j + 1)) / ((PetscReal)(p * q));
    }
  }

  /* K2 = (A \otimes T) + (B \otimes S); with -b_null, B is treated as the identity */
  PetscCall(MatCreateKAIJAB(A, bnull ? NULL : B, p, q, S, T, &K2));

  /* Reference operators */
  PetscCall(MatCreateKAIJ(A, p, q, NULL, T, &KAT));            /* A \otimes T */
  if (bnull) PetscCall(MatCreateKAIJ(B, p, q, S, NULL, &KBS)); /* I \otimes S (matches identity B) */
  else PetscCall(MatCreateKAIJ(B, p, q, NULL, S, &KBS));       /* B \otimes S */

  PetscCall(MatCreateVecs(K2, &x, &y2));
  PetscCall(VecDuplicate(y2, &yr));
  PetscCall(VecDuplicate(y2, &ytmp));

  for (i = 0; i < 10; i++) {
    PetscCall(VecSetRandom(x, NULL));
    PetscCall(MatMult(K2, x, y2));
    PetscCall(MatMult(KAT, x, yr));
    PetscCall(MatMult(KBS, x, ytmp));
    PetscCall(VecAXPY(yr, 1.0, ytmp)); /* yr = (A x T)x + (B x S)x */
    PetscCall(VecAXPY(yr, -1.0, y2));
    PetscCall(VecNorm(yr, NORM_2, &nrm));
    PetscCheck(nrm <= 1.e-10, PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED, "MatCreateKAIJAB() MatMult mismatch: ||K2*x - ((AxT)+(BxS))*x|| = %g", (double)nrm);
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatCreateKAIJAB() MatMult matches (A x T) + (B x S)\n"));

  PetscCall(PetscFree2(S, T));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y2));
  PetscCall(VecDestroy(&yr));
  PetscCall(VecDestroy(&ytmp));
  PetscCall(MatDestroy(&K2));
  PetscCall(MatDestroy(&KAT));
  PetscCall(MatDestroy(&KBS));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    nsize: {{1 2 4}}
    args: -p {{2 3}} -q {{2 3}}

  test:
    suffix: b_null
    nsize: {{1 3}}
    args: -b_null -p 3 -q 3

TEST*/
