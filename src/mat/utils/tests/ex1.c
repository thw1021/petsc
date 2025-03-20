const char help[] = "Test MatDenseSkinnyQR() and MatDenseSkinnyQB()";

#include <petscmat.h>

static PetscErrorCode TestMatDenseSkinny(PetscInt M, PetscInt N, MatReuse reuse_Q, PetscBool build_R, PetscBool test_upper_trapezoidal, PetscErrorCode (*MatDenseSkinnyFunc)(Mat, MatReuse, Mat *, Mat, PetscInt *))
{
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscInt r;
  Mat      A, A_copy;
  Mat      Q = NULL, R = NULL;
  MatType  mat_type;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_copy));
  if (reuse_Q == MAT_INPLACE_MATRIX) {
    Q = A;
  } else if (reuse_Q == MAT_REUSE_MATRIX) {
    PetscCall(MatDuplicate(A, MAT_SHARE_NONZERO_PATTERN, &Q));
  }
  if (build_R) {
    Mat A_local;

    PetscCall(MatDenseGetLocalMatrix(A, &A_local));
    PetscCall(MatGetType(A_local, &mat_type));
    PetscCall(MatCreate(PETSC_COMM_SELF, &R));
    PetscCall(MatSetSizes(R, N, N, N, N));
    PetscCall(MatSetType(R, mat_type));
    PetscCall(MatSetUp(R));
  }
  PetscCall((*MatDenseSkinnyFunc)(A, reuse_Q, &Q, R, &r));

  // Test 1: Size tests
  if (Q != A) {
    PetscLayout A_row_layout, A_col_layout, Q_row_layout, Q_col_layout;
    PetscBool   same;

    PetscCall(MatGetLayouts(A_copy, &A_row_layout, &A_col_layout));
    PetscCall(MatGetLayouts(Q, &Q_row_layout, &Q_col_layout));
    PetscCall(PetscLayoutCompare(A_row_layout, Q_row_layout, &same));
    PetscCheck(same, comm, PETSC_ERR_PLIB, "Q does not have the same row layout as A");
    PetscCall(PetscLayoutCompare(A_col_layout, Q_col_layout, &same));
    PetscCheck(same, comm, PETSC_ERR_PLIB, "Q does not have the same column layout as A");
  }

  // Test 2: R has zeros in row r and below / is upper trapezoidal
  if (R) {
    const PetscScalar *_R;
    PetscInt           ldR;

    PetscCall(MatDenseGetLDA(R, &ldR));
    PetscCall(MatDenseGetArrayRead(R, &_R));
    for (PetscInt j = 0; j < N; j++) {
      PetscInt limit = (test_upper_trapezoidal) ? PetscMin(r, j + 1) : r;
      for (PetscInt i = limit; i < N; i++) {
        PetscScalar _r = _R[i + j * ldR];

        if (!PetscDefined(USE_COMPLEX)) PetscCheck(_r == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "R[%" PetscInt_FMT ", %" PetscInt_FMT "] = %g, expecting zero", i, j, (double)PetscRealPart(_r));
        else PetscCheck(_r == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "R[%" PetscInt_FMT ", %" PetscInt_FMT "] = %g + i %g, expecting zero", i, j, (double)PetscRealPart(_r), (double)PetscImaginaryPart(r));
      }
    }
    PetscCall(MatDenseRestoreArrayRead(R, &_R));
  }

  {
    Mat       Q_sub, Q_conj;
    Mat       QHQ, QHA, QQHA;
    PetscReal A_norm, err;

    PetscCall(MatDenseGetSubMatrix(Q, PETSC_DECIDE, PETSC_DECIDE, 0, r, &Q_sub));
    Q_conj = Q_sub;

    // Test 3: Q is orthonormal
    if (PetscDefined(USE_COMPLEX)) {
      PetscCall(MatDuplicate(Q_sub, MAT_COPY_VALUES, &Q_conj));
      PetscCall(MatConjugate(Q_conj));
    }
    PetscCall(MatTransposeMatMult(Q_conj, Q_sub, MAT_INITIAL_MATRIX, PETSC_DECIDE, &QHQ));
    PetscCall(MatShift(QHQ, -1.0));
    PetscCall(MatNorm(QHQ, NORM_FROBENIUS, &err));
    PetscCheck(err <= M * r * r * PETSC_MACHINE_EPSILON, comm, PETSC_ERR_PLIB, "|| Q'Q - I ||_F = %g", err);
    PetscCall(MatDestroy(&QHQ));

    // Test 4: projection error
    PetscCall(MatTransposeMatMult(Q_conj, A_copy, MAT_INITIAL_MATRIX, PETSC_DECIDE, &QHA));
    PetscCall(MatMatMult(Q_sub, QHA, MAT_INITIAL_MATRIX, PETSC_DECIDE, &QQHA));

    PetscCall(MatAXPY(QQHA, -1.0, A_copy, UNKNOWN_NONZERO_PATTERN));

    PetscCall(MatNorm(A_copy, NORM_FROBENIUS, &A_norm));
    PetscCall(MatNorm(QQHA, NORM_FROBENIUS, &err));
    PetscCheck(err <= M * r * r * A_norm * PETSC_MACHINE_EPSILON, comm, PETSC_ERR_PLIB, "|| (I - QQ')A ||_F = %g", err);

    PetscCall(MatDestroy(&QQHA));
    PetscCall(MatDestroy(&QHA));

    if (PetscDefined(USE_COMPLEX)) PetscCall(MatDestroy(&Q_conj));
    PetscCall(MatDenseRestoreSubMatrix(Q, &Q_sub));
  }

  // Test 5: Reconstruction error
  if (R) {
    Mat       A_local, A_local2;
    Mat       Q_local;
    PetscReal err, R_norm, Q_norm;

    PetscCall(MatDenseGetLocalMatrix(A_copy, &A_local));
    PetscCall(MatDenseGetLocalMatrix(Q, &Q_local));
    PetscCall(MatMatMult(Q_local, R, MAT_INITIAL_MATRIX, PETSC_DECIDE, &A_local2));
    PetscCall(MatAXPY(A_local2, -1.0, A_local, UNKNOWN_NONZERO_PATTERN));
    PetscCall(MatNorm(R, NORM_FROBENIUS, &R_norm));
    PetscCall(MatNorm(Q_local, NORM_FROBENIUS, &Q_norm));
    PetscCall(MatNorm(A_local2, NORM_FROBENIUS, &err));
    PetscCheck(err <= M * r * N * PETSC_MACHINE_EPSILON * PetscSqrtReal(R_norm * Q_norm), PETSC_COMM_SELF, PETSC_ERR_PLIB, "|| A - QR ||_F = %g", err);
    PetscCall(MatDestroy(&A_local2));
  }

  PetscCall(MatDestroy(&R));
  if (Q != A) PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&A_copy));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt M = 100, N = 6;
  MPI_Comm comm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, NULL, help, NULL);
  PetscCall(PetscOptionsInt("-m", "Number of columns", NULL, M, &M, NULL));
  PetscCall(PetscOptionsInt("-n", "Number of rows", NULL, N, &N, NULL));
  PetscOptionsEnd();

  for (PetscInt reuse = 0; reuse < 3; reuse++) {
    MatReuse reuse_Q = (reuse == 0) ? MAT_INPLACE_MATRIX : (reuse == 1) ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;

    for (PetscInt build = 0; build < 2; build++) {
      PetscBool build_R = build ? PETSC_TRUE : PETSC_FALSE;

      PetscCall(TestMatDenseSkinny(M, N, reuse_Q, build_R, PETSC_TRUE, MatDenseSkinnyQR));
      PetscCall(TestMatDenseSkinny(M, N, reuse_Q, build_R, PETSC_FALSE, MatDenseSkinnyQB));
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/ex1.out
    nsize: {{1 2 3 4 5}}

  test:
    suffix: 0_cuda
    requires: cuda
    output_file: output/ex1.out
    nsize: {{1 2}}
    args: -mat_type densecuda

  test:
    suffix: 0_hip
    requires: hip
    output_file: output/ex1.out
    nsize: {{1 2}}
    args: -mat_type hipcuda

  test:
    suffix: 1
    output_file: output/ex1.out
    nsize: {{1 2 3 4 5}}
    args: -m 4

  test:
    suffix: 1_cuda
    requires: cuda
    output_file: output/ex1.out
    nsize: {{1 2}}
    args: -m 4 -mat_type densecuda

  test:
    suffix: 1_hip
    requires: hip
    output_file: output/ex1.out
    nsize: {{1 2}}
    args: -m 4 -mat_type hipcuda

TEST*/
