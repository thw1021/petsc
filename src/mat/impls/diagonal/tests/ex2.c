static const char help[] = "Test MatPtAP with MATDIAGONAL\n";

#include <petscmat.h>

static PetscErrorCode CreateTestMatrix(MPI_Comm comm, const char type[], PetscInt m, PetscInt n, PetscRandom rand, Mat *M)
{
  PetscBool isdiag;
  char     *found;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcmp(type, MATDIAGONAL, &isdiag));
  if (isdiag) {
    Vec d;

    PetscCall(VecCreate(comm, &d));
    PetscCall(VecSetSizes(d, PETSC_DECIDE, m));
    PetscCall(VecSetFromOptions(d));
    PetscCall(VecSetRandom(d, rand));
    PetscCall(MatCreateDiagonal(d, M));
    PetscCall(VecDestroy(&d));
  } else {
    PetscCall(MatCreate(comm, M));
    PetscCall(MatSetSizes(*M, PETSC_DECIDE, PETSC_DECIDE, m, n));
    PetscCall(MatSetType(*M, type));
    PetscCall(PetscStrstr(type, "aij", &found));
    if (found) {
      PetscCall(MatSeqAIJSetPreallocation(*M, n, NULL));
      PetscCall(MatMPIAIJSetPreallocation(*M, n, NULL, n, NULL));
    } else {
      PetscCall(MatSetUp(*M));
    }
    PetscCall(MatSetRandom(*M, rand));
    PetscCall(MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat         A, P, C;
  MPI_Comm    comm;
  PetscInt    m = 10, n = 8;
  PetscRandom rand;
  PetscBool   flg, flg2, isdiag;
  const char *atype_default = MATDIAGONAL;
  const char *ptype_default = MATDIAGONAL;
  char        atype[256] = "";
  char        ptype[256] = "";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsFList("-amat_type", "A matrix type", "", MatList, atype_default, atype, 256, &flg));
  PetscCall(PetscOptionsFList("-pmat_type", "P matrix type", "", MatList, ptype_default, ptype, 256, &flg2));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscOptionsEnd();

  if (!flg) PetscCall(PetscStrcpy(atype, atype_default));
  if (!flg2) PetscCall(PetscStrcpy(ptype, ptype_default));

  PetscCall(PetscStrcmp(ptype, MATDIAGONAL, &isdiag));
  if (isdiag) n = m;

  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(CreateTestMatrix(comm, atype, m, m, rand, &A));
  PetscCall(CreateTestMatrix(comm, ptype, m, n, rand, &P));

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
    args: -amat_type diagonal -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_seqaij
    args: -amat_type diagonal -pmat_type seqaij
    output_file: output/empty.out

  test:
    suffix: seqaij_diag
    args: -amat_type seqaij -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_seqdense
    args: -amat_type diagonal -pmat_type seqdense
    output_file: output/empty.out

  test:
    suffix: seqdense_diag
    args: -amat_type seqdense -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_diag_par
    nsize: 2
    args: -amat_type diagonal -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_mpiaij
    nsize: 2
    args: -amat_type diagonal -pmat_type mpiaij
    output_file: output/empty.out

  test:
    suffix: mpiaij_diag
    nsize: 2
    args: -amat_type mpiaij -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_mpidense
    nsize: 2
    args: -amat_type diagonal -pmat_type mpidense
    output_file: output/empty.out

  test:
    suffix: mpidense_diag
    nsize: 2
    args: -amat_type mpidense -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_diag_kokkos
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type diagonal -vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_seqaijkokkos
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type seqaijkokkos -vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: seqaijkokkos_diag
    requires: kokkos_kernels
    args: -amat_type seqaijkokkos -pmat_type diagonal -vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_diag_cuda
    requires: cuda
    args: -amat_type diagonal -pmat_type diagonal -vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_seqaijcusparse
    requires: cuda
    args: -amat_type diagonal -pmat_type seqaijcusparse -vec_type cuda
    output_file: output/empty.out

  test:
    suffix: seqaijcusparse_diag
    requires: cuda
    args: -amat_type seqaijcusparse -pmat_type diagonal -vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_diag_hip
    requires: hip
    args: -amat_type diagonal -pmat_type diagonal -vec_type hip
    output_file: output/empty.out

  test:
    suffix: diag_seqaijhipsparse
    requires: hip
    args: -amat_type diagonal -pmat_type seqaijhipsparse -vec_type hip
    output_file: output/empty.out

  test:
    suffix: seqaijhipsparse_diag
    requires: hip
    args: -amat_type seqaijhipsparse -pmat_type diagonal -vec_type hip
    output_file: output/empty.out

TEST*/
