static const char help[] = "Test MatPtAP with MATDIAGONAL\n";

#include <petscmat.h>

/* KOKKOS: Following two cases will fail, as MatDiagonalScale_{Seq,MPI}AIJKOKKOS does *
 * not support CPU diagonal vector against AIJ KOKKOS.                                *
 * -amat_type diagonal -pmat_type aijkokkos -adiag_vec_type standard                  *
 * -amat_type aijkokkos -pmat_type diagonal -pdiag_vec_type standard                  */
static PetscErrorCode CreateTestMatrix(MPI_Comm comm, const char type[], const char prefix[], PetscInt m, PetscInt n, PetscRandom rand, Mat *M)
{
  PetscBool isdiag, isaij, isdense;
  PetscBool isaijkokkos, isaijcusparse, isaijhipsparse;
  PetscBool isdensecuda, isdensehip;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcmp(type, MATDIAGONAL, &isdiag));
  PetscCall(PetscStrcmp(type, MATAIJ, &isaij));
  PetscCall(PetscStrcmp(type, MATDENSE, &isdense));
  PetscCall(PetscStrcmp(type, MATAIJKOKKOS, &isaijkokkos));
  PetscCall(PetscStrcmp(type, MATAIJCUSPARSE, &isaijcusparse));
  PetscCall(PetscStrcmp(type, MATAIJHIPSPARSE, &isaijhipsparse));
  PetscCall(PetscStrcmp(type, MATDENSECUDA, &isdensecuda));
  PetscCall(PetscStrcmp(type, MATDENSEHIP, &isdensehip));
  if (isdiag) {
    Vec d;

    PetscCall(VecCreate(comm, &d));
    if (prefix) PetscCall(VecSetOptionsPrefix(d, prefix));
    PetscCall(VecSetSizes(d, PETSC_DECIDE, m));
    PetscCall(VecSetFromOptions(d));
    PetscCall(VecSetRandom(d, rand));
    PetscCall(MatCreateDiagonal(d, M));
    PetscCall(VecDestroy(&d));
  } else if (isaij || isaijkokkos || isaijcusparse || isaijhipsparse) {
    PetscCall(MatCreate(comm, M));
    PetscCall(MatSetSizes(*M, PETSC_DECIDE, PETSC_DECIDE, m, n));
    PetscCall(MatSetType(*M, MATAIJ));
    PetscCall(MatSeqAIJSetPreallocation(*M, n, NULL));
    PetscCall(MatMPIAIJSetPreallocation(*M, n, NULL, n, NULL));
    PetscCall(MatSetUp(*M));
    PetscCall(MatSetRandom(*M, rand));
    PetscCall(MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY));
    if (isaijkokkos) PetscCall(MatConvert(*M, MATAIJKOKKOS, MAT_INPLACE_MATRIX, M));
    else if (isaijcusparse) PetscCall(MatConvert(*M, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, M));
    else if (isaijhipsparse) PetscCall(MatConvert(*M, MATAIJHIPSPARSE, MAT_INPLACE_MATRIX, M));
  } else if (isdense || isdensecuda || isdensehip) {
    PetscCall(MatCreate(comm, M));
    PetscCall(MatSetSizes(*M, PETSC_DECIDE, PETSC_DECIDE, m, n));
    PetscCall(MatSetType(*M, MATDENSE));
    PetscCall(MatSetUp(*M));
    PetscCall(MatSetRandom(*M, rand));
    PetscCall(MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY));
    if (isdensecuda) PetscCall(MatConvert(*M, MATDENSECUDA, MAT_INPLACE_MATRIX, M));
    else if (isdensehip) PetscCall(MatConvert(*M, MATDENSEHIP, MAT_INPLACE_MATRIX, M));
  } else SETERRQ(comm, PETSC_ERR_USER, "Unsupported matrix type %s", type);
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
  char        atype[256]    = "";
  char        ptype[256]    = "";

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
  PetscCall(CreateTestMatrix(comm, atype, "adiag_", m, m, rand, &A));
  PetscCall(CreateTestMatrix(comm, ptype, "pdiag_", m, n, rand, &P));

  /* Initial PtAP */
  PetscCall(MatPtAP(A, P, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_INITIAL_MATRIX: MatPtAPMultEqual failed");

  /* Reuse with modified A */
  PetscCall(MatScale(A, 2.0));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified A): MatPtAPMultEqual failed");

  /* Reuse with modified P */
  PetscCall(MatScale(P, 0.5));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified P): MatPtAPMultEqual failed");

  /* Reuse with modified A, second time */
  PetscCall(MatScale(A, 1.1));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified A): MatPtAPMultEqual failed");

  /* Reuse with modified P */
  PetscCall(MatScale(P, 3.7));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified P): MatPtAPMultEqual failed");

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
    nsize: {{1 2}}
    args: -amat_type diagonal -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_aij
    nsize: {{1 2}}
    args: -amat_type diagonal -pmat_type aij
    output_file: output/empty.out

  test:
    suffix: aij_diag
    nsize: {{1 2}}
    args: -amat_type aij -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_dense
    nsize: {{1 2}}
    args: -amat_type diagonal -pmat_type dense
    output_file: output/empty.out

  test:
    suffix: dense_diag
    nsize: {{1 2}}
    args: -amat_type dense -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_diag_kokkos_both
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type kokkos -pdiag_vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_kokkos_diag_standard
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type kokkos -pdiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_kokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type standard -pdiag_vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_aijkokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type aijkokkos -adiag_vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: aijkokkos_diag
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type aijkokkos -pmat_type diagonal -pdiag_vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_diag_cuda_both
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type cuda -pdiag_vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_cuda_diag_standard
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type cuda -pdiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_cuda
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type standard -pdiag_vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_aijcusparse
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type aijcusparse -adiag_vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_standard_aijcusparse
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type aijcusparse -adiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: aijcusparse_diag
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type aijcusparse -pmat_type diagonal -pdiag_vec_type cuda
    output_file: output/empty.out

  test:
    suffix: aijcusparse_diag_standard
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type aijcusparse -pmat_type diagonal -pdiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: diag_densecuda
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type densecuda -adiag_vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_standard_densecuda
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type densecuda -adiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: densecuda_diag
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type densecuda -pmat_type diagonal -pdiag_vec_type cuda
    output_file: output/empty.out

  test:
    suffix: densecuda_diag_standard
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type densecuda -pmat_type diagonal -pdiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: diag_diag_hip_both
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type hip -pdiag_vec_type hip
    output_file: output/empty.out

  test:
    suffix: diag_hip_diag_standard
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type hip -pdiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_hip
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type standard -pdiag_vec_type hip
    output_file: output/empty.out

  test:
    suffix: diag_aijhipsparse
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type aijhipsparse -adiag_vec_type hip
    output_file: output/empty.out

  test:
    suffix: diag_standard_aijhipsparse
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type aijhipsparse -adiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: aijhipsparse_diag
    nsize: {{1 2}}
    requires: hip
    args: -amat_type aijhipsparse -pmat_type diagonal -pdiag_vec_type hip
    output_file: output/empty.out

  test:
    suffix: aijhipsparse_diag_standard
    nsize: {{1 2}}
    requires: hip
    args: -amat_type aijhipsparse -pmat_type diagonal -pdiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: diag_densehip
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type densehip -adiag_vec_type hip
    output_file: output/empty.out

  test:
    suffix: diag_standard_densehip
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type densehip -adiag_vec_type standard
    output_file: output/empty.out

  test:
    suffix: densehip_diag
    nsize: {{1 2}}
    requires: hip
    args: -amat_type densehip -pmat_type diagonal -pdiag_vec_type hip
    output_file: output/empty.out

  test:
    suffix: densehip_diag_standard
    nsize: {{1 2}}
    requires: hip
    args: -amat_type densehip -pmat_type diagonal -pdiag_vec_type standard
    output_file: output/empty.out

TEST*/
