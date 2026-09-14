static const char help[] = "Tests MatDenseGetColumnVec() and friends on dense matrices created from a VecType, and on the Mats a MatProduct and MatDenseGetSubMatrix() derive from them\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat                A, C, P, S;
  Vec                v, w;
  char               vtype[64] = VECSTANDARD;
  VecType            avtype, cvtype, pvtype;
  PetscBool          same;
  PetscInt           M = 9, N = 3, lda, rstart, rend, i, j;
  PetscReal          norm;
  PetscScalar        sum;
  const PetscScalar *array;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-vec_type", vtype, sizeof(vtype), NULL));
  /* A VECKOKKOS type gives a MATDENSECUDA/MATDENSEHIP with VECKOKKOS vectors when the Kokkos backend runs on that device */
  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, PETSC_DECIDE, PETSC_DECIDE, M, N, PETSC_DECIDE, NULL, &A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

  /* Write column j through its column vector, then scale it through a read/write column vector: A(:, j) = 2 (j + 1) */
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVecWrite(A, j, &v));
    PetscCall(VecSet(v, (PetscScalar)(j + 1)));
    PetscCall(MatDenseRestoreColumnVecWrite(A, j, &v));
  }
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVec(A, j, &v));
    PetscCall(VecScale(v, 2.0));
    PetscCall(MatDenseRestoreColumnVec(A, j, &v));
  }

  /* Check through read-only column vectors, and independently through the matrix itself */
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVecRead(A, j, &v));
    PetscCall(VecSum(v, &sum));
    PetscCheck(PetscAbsScalar(sum - 2.0 * (j + 1) * M) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " read back through MatDenseGetColumnVecRead() sums to %g, expected %g", j, (double)PetscRealPart(sum), 2.0 * (j + 1) * M);
    PetscCall(MatDenseRestoreColumnVecRead(A, j, &v));
  }
  PetscCall(MatNorm(A, NORM_INFINITY, &norm));
  PetscCheck(PetscAbsReal(norm - N * (N + 1)) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatNorm() gives %g, expected %g", (double)norm, (double)(N * (N + 1)));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(MatDenseGetArrayRead(A, &array));
  for (j = 0; j < N; j++) {
    for (i = 0; i < rend - rstart; i++)
      PetscCheck(array[i + j * lda] == 2.0 * (j + 1), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Entry (%" PetscInt_FMT ", %" PetscInt_FMT ") is %g, expected %g", rstart + i, j, (double)PetscRealPart(array[i + j * lda]), 2.0 * (j + 1));
  }
  PetscCall(MatDenseRestoreArrayRead(A, &array));

  /* The Mat a MatProduct creates must have the VecType of the Mat it is built from */
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, rend - rstart, rend - rstart, M, M, 3.0, &S));
  PetscCall(MatMatMult(S, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatGetVecType(A, &avtype));
  PetscCall(MatGetVecType(C, &cvtype));
  PetscCall(PetscStrcmp(avtype, cvtype, &same));
  PetscCheck(same, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The Mat the product created has VecType %s, expected %s", cvtype, avtype);

  /* C is 3 A, so the columns cancel, which needs both column Vecs to be of the same type */
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVec(C, j, &v));
    PetscCall(MatDenseGetColumnVecRead(A, j, &w));
    PetscCall(VecAXPY(v, -3.0, w));
    PetscCall(VecNorm(v, NORM_INFINITY, &norm));
    PetscCall(MatDenseRestoreColumnVecRead(A, j, &w));
    PetscCall(MatDenseRestoreColumnVec(C, j, &v));
    PetscCheck(norm < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " of the product differs from three times the column it was built from by %g", j, (double)norm);
  }

  /* A submatrix must have the VecType of its parent Mat */
  PetscCall(MatDenseGetSubMatrix(C, PETSC_DECIDE, PETSC_DECIDE, 1, N, &P));
  PetscCall(MatGetVecType(P, &pvtype));
  PetscCall(PetscStrcmp(avtype, pvtype, &same));
  PetscCheck(same, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The submatrix has VecType %s, expected %s", pvtype, avtype);

  /* The columns of C are zero after the loop above, so adding back a column of A recovers it */
  for (j = 0; j < N - 1; j++) {
    PetscCall(MatDenseGetColumnVec(P, j, &v));
    PetscCall(MatDenseGetColumnVecRead(A, j + 1, &w));
    PetscCall(VecAXPY(v, 1.0, w));
    PetscCall(VecNorm(v, NORM_INFINITY, &norm));
    PetscCall(MatDenseRestoreColumnVecRead(A, j + 1, &w));
    PetscCall(MatDenseRestoreColumnVec(P, j, &v));
    PetscCheck(PetscAbsReal(norm - 2.0 * (j + 2)) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " of the submatrix gives %g, expected %g", j, (double)norm, 2.0 * (j + 2));
  }
  PetscCall(MatDenseRestoreSubMatrix(C, &P));

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}
    test:
      suffix: standard
      args: -vec_type standard
    test:
      suffix: cuda
      requires: cuda
      args: -vec_type cuda
    test:
      suffix: hip
      requires: hip
      args: -vec_type hip
    # On a CUDA or HIP build this is a MATDENSECUDA or MATDENSEHIP with VECKOKKOS column vectors
    test:
      suffix: kokkos
      requires: kokkos_kernels !sycl
      args: -vec_type kokkos

TEST*/
