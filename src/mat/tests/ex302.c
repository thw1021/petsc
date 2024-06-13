const char help[] = "Test MatTallSkinnySVD";

#include <petscmat.h>
#include <petscsys.h>

static PetscErrorCode MatSVDTest(Mat X, Mat U, Vec S, Mat VH)
{
  Mat       UH, UHU, V, VHV, US, Xrecon, UpX, UHX, XVp, XV;
  PetscInt  M, N;
  MPI_Comm  comm;
  PetscReal u_ortho_err, v_ortho_err, recon_err, upx_err, xvp_err;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCall(MatGetSize(X, &M, &N));

  PetscCall(MatHermitianTranspose(U, MAT_INITIAL_MATRIX, &UH));
  PetscCall(MatMatMult(UH, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UHU));
  PetscCall(MatShift(UHU, -1.0));
  PetscCall(MatNorm(UHU, NORM_FROBENIUS, &u_ortho_err));
  PetscCall(MatDestroy(&UHU));

  PetscCall(MatHermitianTranspose(VH, MAT_INITIAL_MATRIX, &V));
  PetscCall(MatMatMult(VH, V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VHV));
  PetscCall(MatShift(VHV, -1.0));
  PetscCall(MatNorm(VHV, NORM_FROBENIUS, &v_ortho_err));
  PetscCall(MatDestroy(&VHV));

  PetscCall(MatDuplicate(U, MAT_COPY_VALUES, &US));
  PetscCall(MatDiagonalScale(US, NULL, S));
  PetscCall(MatMatMult(US, VH, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Xrecon));
  PetscCall(MatAXPY(Xrecon, -1.0, X, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(Xrecon, NORM_FROBENIUS, &recon_err));
  PetscCall(MatDestroy(&Xrecon));
  PetscCall(MatDestroy(&US));

  PetscCall(MatMatMult(UH, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UHX));
  PetscCall(MatMatMult(U, UHX, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UpX));
  PetscCall(MatAXPY(UpX, -1.0, X, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(UpX, NORM_FROBENIUS, &upx_err));
  PetscCall(MatDestroy(&UpX));
  PetscCall(MatDestroy(&UHX));

  PetscCall(MatMatMult(X, V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &XV));
  PetscCall(MatMatMult(XV, VH, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &XVp));
  PetscCall(MatAXPY(XVp, -1.0, X, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(XVp, NORM_FROBENIUS, &xvp_err));
  PetscCall(MatDestroy(&XVp));
  PetscCall(MatDestroy(&XV));

  PetscCall(MatDestroy(&UH));
  PetscCall(MatDestroy(&V));
  PetscCall(PetscPrintf(comm, "(%" PetscInt_FMT " x %" PetscInt_FMT "): ||U'U-I||_F = %e, ||V'V-I||_F = %e, ||USV'-X||_F = %e\n", M, N, (double)u_ortho_err, (double)v_ortho_err, (double)recon_err));
  PetscCall(PetscPrintf(comm, "(%" PetscInt_FMT " x %" PetscInt_FMT "): ||X - UU'X||_F = %e ||X - XVV'||_F = %e\n", M, N, (double)upx_err, (double)xvp_err));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscMPIInt rank;
  PetscRandom rand;
  PetscInt    n_min = 0, n_max = 100, n_stride = 10, n_zero_columns = 2;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, NULL, help, "Mat");
  PetscCall(PetscOptionsInt("-min", "Smallest number of columns to test", NULL, n_min, &n_min, NULL));
  PetscCall(PetscOptionsInt("-max", "Largest number of columns to test", NULL, n_max, &n_max, NULL));
  PetscCall(PetscOptionsInt("-stride", "Stride between numbers of columns to test", NULL, n_stride, &n_stride, NULL));
  PetscCall(PetscOptionsInt("-zero_column", "Number of zero columns to append to the test matrix", NULL, n_zero_columns, &n_zero_columns, NULL));
  PetscOptionsEnd();
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
#if PetscDefined(USE_COMPLEX)
  PetscCall(PetscRandomSetInterval(rand, -1.0 - PETSC_i, 1.0 + PETSC_i));
#else
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
#endif
  for (PetscInt n = n_min; n <= n_max; n += n_stride) {
    PetscInt     N       = n + n_zero_columns;
    PetscInt     m       = 2 * n + 1;
    PetscInt     n_local = (rank == 0) ? N : 0;
    Mat          X, U = NULL, VH = NULL;
    Vec          S = NULL;
    PetscLayout  rows, cols;
    PetscInt     i_min, i_max;
    PetscInt     lda;
    PetscScalar *_X;

    PetscCall(MatCreateDense(comm, PETSC_DECIDE, n_local, m, N, NULL, &X));
    PetscCall(MatSetFromOptions(X));

    // Make a random triangular matrix: the condition number gets worse as the matrix size increases
    PetscCall(MatSetRandom(X, rand));
    PetscCall(MatGetLayouts(X, &rows, &cols));
    PetscCall(MatDenseGetLDA(X, &lda));
    PetscCall(PetscLayoutGetRange(rows, &i_min, &i_max));
    PetscCall(MatDenseGetArray(X, &_X));
    for (PetscInt j = 0; j < n; j++) {
      for (PetscInt i = PetscMax(0, j + 1 - i_min); i < i_max - i_min; i++) _X[i + j * lda] = 0.0;
    }
    for (PetscInt j = n; j < N; j++) {
      for (PetscInt i = 0; i < i_max - i_min; i++) _X[i + j * lda] = 0.0;
    }
    PetscCall(MatDenseRestoreArray(X, &_X));

    for (PetscInt iter = 0; iter < 2; iter++) {
      MatReuse reuse = iter ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;
      PetscCall(MatDenseTallSkinnySVD(X, reuse, &U, &S, &VH));
      PetscCall(MatSVDTest(X, U, S, VH));
    }
    PetscCall(MatDestroy(&U));
    PetscCall(VecDestroy(&S));
    PetscCall(MatDestroy(&VH));
    PetscCall(MatDestroy(&X));
  }
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2 3 4}}

  # test viewing
  test:
    requires: double !complex
    suffix: 1
    nsize: 2
    args: -min 100 -max 100 -mat_dense_tall_skinny_svd_monitor

  # test viewing
  test:
    requires: double !complex
    suffix: 2
    nsize: 2
    args: -min 5 -max 5 -mat_dense_tall_skinny_svd_monitor ascii::ascii_info_detail

TEST*/
