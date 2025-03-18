const char help[] = "Test MatDenseTSQR";

#include <petscmat.h>
#include <petscsys.h>

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscMPIInt rank;
  PetscRandom rand;
  PetscInt    M = 20, N = 10, K;
  Mat         X, Q = NULL, R = NULL;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, NULL, help, "Mat");
  PetscCall(PetscOptionsInt("-m", "Number of rows", NULL, M, &M, NULL));
  PetscCall(PetscOptionsInt("-n", "Number of columns", NULL, N, &N, NULL));
  PetscOptionsEnd();
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
#if PetscDefined(USE_COMPLEX)
  PetscCall(PetscRandomSetInterval(rand, -1.0 - PETSC_i, 1.0 + PETSC_i));
#else
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
#endif
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &X));
  PetscCall(MatSetFromOptions(X));
  PetscCall(MatSetRandom(X, rand));

  K = PetscMin(M, N);

  for (PetscInt r = 0; r < 2; r++) {
    Mat                QH, QHQ, Xcopy, Xcopy_local, X_local, Q_local;
    PetscReal          ortho_err, recon_err, X_norm;
    const PetscScalar *_R;
    PetscScalar       *_Rcol;
    PetscMPIInt        r_size;
    PetscInt           q_m, q_n, r_m, r_n, ldR;

    MatReuse reuse = r ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;
    PetscCall(MatDenseTSQR(X, reuse, &Q, &R));
    PetscCall(MatGetSize(Q, &q_m, &q_n));
    PetscCall(MatGetSize(R, &r_m, &r_n));

    PetscCheck(q_m == M && q_n == K, comm, PETSC_ERR_PLIB, "Q matrix is wrong size");
    PetscCheck(r_m == K && r_n == N, comm, PETSC_ERR_PLIB, "R matrix is wrong size");

    PetscCall(MatHermitianTranspose(Q, MAT_INITIAL_MATRIX, &QH));
    PetscCall(MatMatMult(QH, Q, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &QHQ));
    PetscCall(MatShift(QHQ, -1.0));
    PetscCall(MatNorm(QHQ, NORM_FROBENIUS, &ortho_err));
    PetscCall(PetscPrintf(comm, "(%" PetscInt_FMT " x %" PetscInt_FMT "): ||Q'Q - I||_F = %e\n", M, N, (double)ortho_err));
    PetscCheck(ortho_err < PETSC_SMALL, comm, PETSC_ERR_PLIB, "Q matrix does not have orthonormal columns");
    PetscCall(MatDestroy(&QHQ));
    PetscCall(MatDestroy(&QH));

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)R), &r_size));
    PetscCheck(r_size == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "R does not have PETSC_COMM_SELF communicator");

    PetscCall(MatDenseGetLDA(R, &ldR));
    PetscCall(MatDenseGetArrayRead(R, &_R));

    PetscCall(PetscMalloc1(K, &_Rcol));
    for (PetscInt j = 0; j < N; j++) {
      PetscCall(PetscArraycpy(_Rcol, &_R[j * ldR], K));
      PetscCallMPI(MPI_Bcast(_Rcol, K, MPIU_SCALAR, 0, comm));
      for (PetscInt i = 0; i < K; i++) {
        PetscCheck(_R[i + j * ldR] == _Rcol[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "R matrix is not identical on all processses");
        if (i > j) PetscCheck(_R[i + j * ldR] == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "R matrix is not upper triangular");
      }
    }
    PetscCall(PetscFree(_Rcol));
    PetscCall(MatDenseRestoreArrayRead(R, &_R));

    PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Xcopy));
    PetscCall(MatDenseGetLocalMatrix(Xcopy, &Xcopy_local));
    PetscCall(MatDenseGetLocalMatrix(X, &X_local));
    PetscCall(MatDenseGetLocalMatrix(Q, &Q_local));
    PetscCall(MatMatMult(Q_local, R, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Xcopy_local));
    PetscCall(MatAXPY(Xcopy, -1.0, X, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(Xcopy, NORM_FROBENIUS, &recon_err));
    PetscCall(MatNorm(X, NORM_FROBENIUS, &X_norm));
    PetscCall(PetscPrintf(comm, "(%" PetscInt_FMT " x %" PetscInt_FMT "): ||X - QR||_F / ||X||_F = %e\n", M, N, (double)(recon_err / X_norm)));
    PetscCheck(recon_err < PETSC_SMALL, comm, PETSC_ERR_PLIB, "QR factorization does not reconstruct X");
    PetscCall(MatDestroy(&Xcopy));
  }
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&X));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2 8 9}}

TEST*/
