static char help[] = "Test MatCreate() with MAT_STRUCTURE_ONLY .\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         mat;
  PetscInt    m = 7, n, nlocal, i, j, rstart, rend, bs;
  PetscMPIInt size;
  PetscScalar v;
  PetscBool   struct_only = PETSC_TRUE, explicit_preallocation = PETSC_FALSE, reassemble = PETSC_FALSE, blocked = PETSC_FALSE, ismpiaij, ismpibaij, ismpisbaij;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-struct_only", &struct_only, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-explicit_preallocation", &explicit_preallocation, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reassemble", &reassemble, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-blocked", &blocked, NULL));
  n      = m;
  nlocal = PETSC_DECIDE;
  PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &nlocal, &n));

  /* ------- Assemble matrix, test MatValid() --------- */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &mat));
  PetscCall(MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetFromOptions(mat));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATMPIAIJ, &ismpiaij));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATMPIBAIJ, &ismpibaij));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATMPISBAIJ, &ismpisbaij));
  if (struct_only) PetscCall(MatSetOption(mat, MAT_STRUCTURE_ONLY, PETSC_TRUE));
  if (explicit_preallocation) {
    PetscCall(MatGetBlockSize(mat, &bs));
    if (ismpiaij) PetscCall(MatMPIAIJSetPreallocation(mat, nlocal, NULL, n - nlocal, NULL));
    else if (ismpibaij) PetscCall(MatMPIBAIJSetPreallocation(mat, bs, nlocal / bs, NULL, (n - nlocal) / bs, NULL));
    else if (ismpisbaij) PetscCall(MatMPISBAIJSetPreallocation(mat, bs, nlocal / bs, NULL, (n - nlocal) / bs, NULL));
    else PetscCall(MatXAIJSetPreallocation(mat, bs, NULL, NULL, NULL, NULL));
  } else PetscCall(MatSetUp(mat));
  PetscCall(MatGetBlockSize(mat, &bs));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    for (j = 0; j < n - (reassemble ? bs : 0); j++) {
      v = 10.0 * i + j;
      PetscCall(MatSetValues(mat, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  if (reassemble) {
    PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    for (PetscInt pass = 0; pass < 2; pass++) {
      // Stash a new off-diagonal block column, then update the same entries.
      for (i = rstart; i < rend; i += bs) {
        PetscInt row = (i + rend - rstart) % m, col = n / bs - 1;

        if (blocked) {
          row /= bs;
          PetscCall(MatSetValuesBlocked(mat, 1, &row, 1, &col, NULL, ADD_VALUES));
        } else {
          for (PetscInt ii = row; ii < row + bs; ii++) {
            for (j = n - bs; j < n; j++) PetscCall(MatSetValues(mat, 1, &ii, 1, &j, NULL, ADD_VALUES));
          }
        }
      }
      PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
    }
  }
  if (struct_only && (ismpiaij || ismpibaij || ismpisbaij)) {
    Mat          Ad, Ao;
    PetscScalar *a;

    if (ismpiaij) PetscCall(MatMPIAIJGetSeqAIJ(mat, &Ad, &Ao, NULL));
    else PetscCall(MatMPIBAIJGetSeqBAIJ(mat, &Ad, &Ao, NULL));
    if (ismpiaij) PetscCall(MatSeqAIJGetArray(Ad, &a));
    else if (ismpisbaij) PetscCall(MatSeqSBAIJGetArray(Ad, &a));
    else PetscCall(MatSeqBAIJGetArray(Ad, &a));
    PetscCheck(!a, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Diagonal scalar array was allocated");
    if (ismpiaij) PetscCall(MatSeqAIJRestoreArray(Ad, &a));
    else if (ismpisbaij) PetscCall(MatSeqSBAIJRestoreArray(Ad, &a));
    else PetscCall(MatSeqBAIJRestoreArray(Ad, &a));
    if (ismpiaij) PetscCall(MatSeqAIJGetArray(Ao, &a));
    else PetscCall(MatSeqBAIJGetArray(Ao, &a));
    PetscCheck(!a, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Off-diagonal scalar array was allocated");
    if (ismpiaij) PetscCall(MatSeqAIJRestoreArray(Ao, &a));
    else PetscCall(MatSeqBAIJRestoreArray(Ao, &a));
  }
  if (size == 1) PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex107.out

   test:
      suffix: 2
      args: -mat_type {{baij sbaij}separate output} -mat_block_size 2 -m 10

   testset:
      nsize: 2
      output_file: output/empty.out
      test:
         suffix: mpi_aij
      test:
         suffix: mpi_aij_preallocation
         args: -explicit_preallocation
      test:
         suffix: mpi_baij
         args: -mat_type baij -mat_block_size 2 -m 8
      test:
         suffix: mpi_baij_preallocation
         args: -mat_type baij -mat_block_size 2 -m 8 -explicit_preallocation
      test:
         suffix: mpi_sbaij
         args: -mat_type sbaij -mat_block_size 2 -m 8
      test:
         suffix: mpi_sbaij_preallocation
         args: -mat_type sbaij -mat_block_size 2 -m 8 -explicit_preallocation

   test:
      suffix: mpi_sbaij_reassembly
      nsize: 2
      output_file: output/empty.out
      args: -mat_type sbaij -mat_block_size {{1 2 3}} -m 12 -reassemble -blocked {{0 1}} -explicit_preallocation {{0 1}} -malloc_debug

TEST*/
