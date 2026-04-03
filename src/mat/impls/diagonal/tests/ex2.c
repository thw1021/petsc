static const char help[] = "Test MatPtAP with MATDIAGONAL\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         A, P, C;
  MPI_Comm    comm;
  PetscInt    m = 10, n = 8;
  PetscRandom rand;
  PetscBool   flg, pdiag;
  char        atype[64] = MATDIAGONAL, ptype[64] = MATDIAGONAL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(PetscOptionsGetString(NULL, NULL, "-atype", atype, sizeof(atype), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ptype", ptype, sizeof(ptype), NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(PetscRandomCreate(comm, &rand));

  /* Diagonal matrices are always square, so if P is diagonal, n must equal m */
  PetscCall(PetscStrcmp(ptype, MATDIAGONAL, &pdiag));
  if (pdiag) n = m;

  /* Create A (m x m) */
  PetscCall(PetscStrcmp(atype, MATDIAGONAL, &flg));
  if (flg) {
    Vec d;
    PetscCall(VecCreate(comm, &d));
    PetscCall(VecSetSizes(d, PETSC_DECIDE, m));
    PetscCall(VecSetFromOptions(d));
    PetscCall(VecSetRandom(d, rand));
    PetscCall(MatCreateDiagonal(d, &A));
    PetscCall(VecDestroy(&d));
  } else {
    PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &A));
    PetscCall(MatSetRandom(A, rand));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatConvert(A, atype, MAT_INPLACE_MATRIX, &A));
  }

  /* Create P (m x n) */
  if (pdiag) {
    Vec d;
    PetscCall(VecCreate(comm, &d));
    PetscCall(VecSetSizes(d, PETSC_DECIDE, m));
    PetscCall(VecSetFromOptions(d));
    PetscCall(VecSetRandom(d, rand));
    PetscCall(MatCreateDiagonal(d, &P));
    PetscCall(VecDestroy(&d));
  } else {
    PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &P));
    PetscCall(MatSetRandom(P, rand));
    PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatConvert(P, ptype, MAT_INPLACE_MATRIX, &P));
  }

  /* Initial PtAP */
  PetscCall(MatPtAP(A, P, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_INITIAL_MATRIX: MatPtAPMultEqual failed");

  /* First reuse */
  PetscCall(MatScale(A, 2.0));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (1st): MatPtAPMultEqual failed");

  /* Second reuse */
  PetscCall(MatScale(A, 0.5));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (2nd): MatPtAPMultEqual failed");

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  test:
    suffix: diag_diag
    args: -atype diagonal -ptype diagonal
    output_file: output/empty.out

  test:
    suffix: diag_seqaij
    args: -atype diagonal -ptype seqaij
    output_file: output/empty.out

  test:
    suffix: seqaij_diag
    args: -atype seqaij -ptype diagonal
    output_file: output/empty.out

  test:
    suffix: diag_seqdense
    args: -atype diagonal -ptype seqdense
    output_file: output/empty.out

  test:
    suffix: seqdense_diag
    args: -atype seqdense -ptype diagonal
    output_file: output/empty.out

  test:
    suffix: diag_diag_par
    nsize: 2
    args: -atype diagonal -ptype diagonal
    output_file: output/empty.out

  test:
    suffix: diag_mpiaij
    nsize: 2
    args: -atype diagonal -ptype mpiaij
    output_file: output/empty.out

  test:
    suffix: mpiaij_diag
    nsize: 2
    args: -atype mpiaij -ptype diagonal
    output_file: output/empty.out

  test:
    suffix: diag_mpidense
    nsize: 2
    args: -atype diagonal -ptype mpidense
    output_file: output/empty.out

  test:
    suffix: mpidense_diag
    nsize: 2
    args: -atype mpidense -ptype diagonal
    output_file: output/empty.out

  test:
    suffix: diag_diag_kokkos
    requires: kokkos_kernels
    args: -atype diagonal -ptype diagonal -vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_seqaijkokkos
    requires: kokkos_kernels
    args: -atype diagonal -ptype seqaijkokkos -vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: seqaijkokkos_diag
    requires: kokkos_kernels
    args: -atype seqaijkokkos -ptype diagonal -vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_diag_cuda
    requires: cuda
    args: -atype diagonal -ptype diagonal -vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_seqaijcusparse
    requires: cuda
    args: -atype diagonal -ptype seqaijcusparse -vec_type cuda
    output_file: output/empty.out

  test:
    suffix: seqaijcusparse_diag
    requires: cuda
    args: -atype seqaijcusparse -ptype diagonal -vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_diag_hip
    requires: hip
    args: -atype diagonal -ptype diagonal -vec_type hip
    output_file: output/empty.out

  test:
    suffix: diag_seqaijhipsparse
    requires: hip
    args: -atype diagonal -ptype seqaijhipsparse -vec_type hip
    output_file: output/empty.out

  test:
    suffix: seqaijhipsparse_diag
    requires: hip
    args: -atype seqaijhipsparse -ptype diagonal -vec_type hip
    output_file: output/empty.out

TEST*/
