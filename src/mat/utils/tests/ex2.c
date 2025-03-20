const char help[] = "Test MatDenseSkinnySVD()";

#include <petscmat.h>

static PetscErrorCode TestMatDenseSkinnySVD(PetscInt M, PetscInt N, MatReuse reuse_U, PetscBool build_VH)
{
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscInt r;
  Mat      A, A_local, A_copy;
  Mat      U = NULL, VH = NULL;
  MatType  mat_type;
  Vec      S;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_copy));
  if (reuse_U == MAT_INPLACE_MATRIX) {
    U = A;
  } else if (reuse_U == MAT_REUSE_MATRIX) {
    PetscCall(MatDuplicate(A, MAT_SHARE_NONZERO_PATTERN, &U));
  }
  if (build_VH) {
    Mat A_local;

    PetscCall(MatDenseGetLocalMatrix(A, &A_local));
    PetscCall(MatGetType(A_local, &mat_type));
    PetscCall(MatCreate(PETSC_COMM_SELF, &VH));
    PetscCall(MatSetSizes(VH, N, N, N, N));
    PetscCall(MatSetType(VH, mat_type));
    PetscCall(MatSetUp(VH));
  }
  PetscCall(MatDenseGetLocalMatrix(A, &A_local));
  PetscCall(MatCreateVecs(A_local, &S, NULL));
  PetscCall(MatDenseSkinnySVD(A, reuse_U, &U, S, VH, &r));

  // Test 1: Size tests
  if (U != A) {
    PetscLayout A_row_layout, A_col_layout, Q_row_layout, Q_col_layout;
    PetscBool   same;

    PetscCall(MatGetLayouts(A_copy, &A_row_layout, &A_col_layout));
    PetscCall(MatGetLayouts(U, &Q_row_layout, &Q_col_layout));
    PetscCall(PetscLayoutCompare(A_row_layout, Q_row_layout, &same));
    PetscCheck(same, comm, PETSC_ERR_PLIB, "Q does not have the same row layout as A");
    PetscCall(PetscLayoutCompare(A_col_layout, Q_col_layout, &same));
    PetscCheck(same, comm, PETSC_ERR_PLIB, "Q does not have the same column layout as A");
  }

  // Test 2: S sorted descending
  {
    const PetscScalar *_S;

    PetscCall(VecGetArrayRead(S, &_S));
    for (PetscInt i = 0; i < N; i++) {
      PetscScalar s  = _S[i];
      PetscReal   sr = PetscRealPart(s);

      PetscCheck(PetscImaginaryPart(s) == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Imag(S[%" PetscInt_FMT "]) = %g", i, (double)PetscImaginaryPart(s));
      PetscCheck(sr >= 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "S[%" PetscInt_FMT "] = %g < 0", i, (double)sr);
      if (i > 0) {
        PetscReal srprev = PetscRealPart(_S[i - 1]);

        PetscCheck(sr <= srprev, PETSC_COMM_SELF, PETSC_ERR_PLIB, "S[%" PetscInt_FMT "] = %g < S[%" PetscInt_FMT "] = %g", i - 1, (double)srprev, i, (double)sr);
      }
    }
    PetscCall(VecRestoreArrayRead(S, &_S));
  }

  if (VH) {
    Mat       VH_conj = VH, VHV, AV, AVVH;
    PetscReal err, err_local, A_norm;
    Mat       A_local2;

    if (PetscDefined(USE_COMPLEX)) {
      PetscCall(MatDuplicate(VH, MAT_COPY_VALUES, &VH_conj));
      PetscCall(MatConjugate(VH_conj));
    }

    // Test 3: VH orthogonality
    PetscCall(MatMatTransposeMult(VH, VH_conj, MAT_INITIAL_MATRIX, PETSC_DECIDE, &VHV));
    PetscCall(MatShift(VHV, -1.0));
    PetscCall(MatNorm(VHV, NORM_FROBENIUS, &err));
    PetscCheck(err <= N * N * N * PETSC_MACHINE_EPSILON, comm, PETSC_ERR_PLIB, "|| V'V - I ||_F = %g", err);
    PetscCall(MatDestroy(&VHV));

    // Test 4: V projection error
    PetscCall(MatDenseGetLocalMatrix(A_copy, &A_local2));
    PetscCall(MatMatTransposeMult(A_local2, VH_conj, MAT_INITIAL_MATRIX, PETSC_DECIDE, &AV));
    PetscCall(MatMatMult(AV, VH, MAT_INITIAL_MATRIX, PETSC_DECIDE, &AVVH));
    PetscCall(MatAXPY(AVVH, -1.0, A_local2, UNKNOWN_NONZERO_PATTERN));

    PetscCall(MatNorm(A_copy, NORM_FROBENIUS, &A_norm));
    PetscCall(MatNorm(AVVH, NORM_FROBENIUS, &err_local));
    err_local = err_local * err_local;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &err_local, 1, MPIU_REAL, MPI_SUM, comm));
    err = PetscSqrtReal(err_local);
    PetscCheck(err <= M * N * N * A_norm * PETSC_MACHINE_EPSILON, comm, PETSC_ERR_PLIB, "|| A(I - VV') ||_F = %g", err);

    PetscCall(MatDestroy(&AVVH));
    PetscCall(MatDestroy(&AV));

    if (PetscDefined(USE_COMPLEX)) PetscCall(MatDestroy(&VH_conj));
  }

  {
    Mat       U_sub, U_conj;
    Mat       UHU, UHA, UUHA;
    PetscReal A_norm, err;

    PetscCall(MatDenseGetSubMatrix(U, PETSC_DECIDE, PETSC_DECIDE, 0, r, &U_sub));
    U_conj = U_sub;

    // Test 5: U is orthonormal
    if (PetscDefined(USE_COMPLEX)) {
      PetscCall(MatDuplicate(U_sub, MAT_COPY_VALUES, &U_conj));
      PetscCall(MatConjugate(U_conj));
    }
    PetscCall(MatTransposeMatMult(U_conj, U_sub, MAT_INITIAL_MATRIX, PETSC_DECIDE, &UHU));
    PetscCall(MatShift(UHU, -1.0));
    PetscCall(MatNorm(UHU, NORM_FROBENIUS, &err));
    PetscCheck(err <= M * r * r * PETSC_MACHINE_EPSILON, comm, PETSC_ERR_PLIB, "|| U'U - I ||_F = %g", err);
    PetscCall(MatDestroy(&UHU));

    // Test 6: U projection error
    PetscCall(MatTransposeMatMult(U_conj, A_copy, MAT_INITIAL_MATRIX, PETSC_DECIDE, &UHA));
    PetscCall(MatMatMult(U_sub, UHA, MAT_INITIAL_MATRIX, PETSC_DECIDE, &UUHA));

    PetscCall(MatAXPY(UUHA, -1.0, A_copy, UNKNOWN_NONZERO_PATTERN));

    PetscCall(MatNorm(A_copy, NORM_FROBENIUS, &A_norm));
    PetscCall(MatNorm(UUHA, NORM_FROBENIUS, &err));
    PetscCheck(err <= M * r * r * A_norm * PETSC_MACHINE_EPSILON, comm, PETSC_ERR_PLIB, "|| (I - UU')A ||_F = %g", err);

    PetscCall(MatDestroy(&UUHA));
    PetscCall(MatDestroy(&UHA));

    if (PetscDefined(USE_COMPLEX)) PetscCall(MatDestroy(&U_conj));
    PetscCall(MatDenseRestoreSubMatrix(U, &U_sub));
  }

  // Test 7: Reconstruction error
  if (VH) {
    Mat       A_local, A_local2;
    Mat       U_local;
    PetscReal err, VH_norm, U_norm;

    PetscCall(MatDiagonalScale(VH, S, NULL));
    PetscCall(MatDenseGetLocalMatrix(A_copy, &A_local));
    PetscCall(MatDenseGetLocalMatrix(U, &U_local));
    PetscCall(MatMatMult(U_local, VH, MAT_INITIAL_MATRIX, PETSC_DECIDE, &A_local2));
    PetscCall(MatAXPY(A_local2, -1.0, A_local, UNKNOWN_NONZERO_PATTERN));
    PetscCall(MatNorm(VH, NORM_FROBENIUS, &VH_norm));
    PetscCall(MatNorm(U_local, NORM_FROBENIUS, &U_norm));
    PetscCall(MatNorm(A_local2, NORM_FROBENIUS, &err));
    PetscCheck(err <= M * r * N * PETSC_MACHINE_EPSILON * PetscSqrtReal(VH_norm * U_norm), PETSC_COMM_SELF, PETSC_ERR_PLIB, "|| A - USV' ||_F = %g", err);
    PetscCall(MatDestroy(&A_local2));
  }

  PetscCall(MatDestroy(&VH));
  PetscCall(VecDestroy(&S));
  if (U != A) PetscCall(MatDestroy(&U));
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

      PetscCall(TestMatDenseSkinnySVD(M, N, reuse_Q, build_R));
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/ex2.out
    args: -m {{4 100}}

  test:
    suffix: 1
    output_file: output/ex2.out
    nsize: {{2 3 4 5}}
    args: -m {{4 100}} -mat_dense_svd_algorithm {{svqb tsqr}}

TEST*/
