static const char help[] = "Tests MatDenseGetColumnVec() and friends on dense matrices created from a VecType, including device dense matrices whose vectors are VECKOKKOS, and on the block a MatProduct builds from one\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat                A, C, S;
  Vec                v, w;
  char               vtype[64] = VECSTANDARD;
  VecType            avtype, cvtype;
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

  /* The block a MatProduct creates is given the MatType of the block it is built from, so it must be given its
     VecType too; a MATDENSECUDA/MATDENSEHIP block whose VecType is VECKOKKOS would otherwise hand out CUDA/HIP
     column Vecs, which cannot be combined with the VECKOKKOS Vecs of the rest of the run */
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, rend - rstart, rend - rstart, M, M, 3.0, &S));
  PetscCall(MatMatMult(S, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatGetVecType(A, &avtype));
  PetscCall(MatGetVecType(C, &cvtype));
  PetscCall(PetscStrcmp(avtype, cvtype, &same));
  PetscCheck(same, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The block of the product has VecType %s, expected %s", cvtype, avtype);

  /* C is 3 A, so subtracting the columns of the two blocks, which needs their column Vecs to be of the same type, gives zero */
  for (j = 0; j < N; j++) {
    PetscCall(MatDenseGetColumnVec(C, j, &v));
    PetscCall(MatDenseGetColumnVecRead(A, j, &w));
    PetscCall(VecAXPY(v, -3.0, w));
    PetscCall(VecNorm(v, NORM_INFINITY, &norm));
    PetscCall(MatDenseRestoreColumnVecRead(A, j, &w));
    PetscCall(MatDenseRestoreColumnVec(C, j, &v));
    PetscCheck(norm < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Column %" PetscInt_FMT " of the product differs from three times the column of the block it was built from by %g", j, (double)norm);
  }

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
