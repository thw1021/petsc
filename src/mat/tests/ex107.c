static char help[] = "Test MatCreate() with MAT_STRUCTURE_ONLY.\n\n";

#include <petscmat.h>

static PetscErrorCode CheckStructureOnlyStorage(Mat mat)
{
  Mat          blocks[2] = {mat, NULL};
  PetscScalar *a;
  PetscBool    ismpiaij, ismpibaij, isaij, issbaij;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATMPIAIJ, &ismpiaij));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)mat, &ismpibaij, MATMPIBAIJ, MATMPISBAIJ, ""));
  if (ismpiaij) PetscCall(MatMPIAIJGetSeqAIJ(mat, &blocks[0], &blocks[1], NULL));
  else if (ismpibaij) PetscCall(MatMPIBAIJGetSeqBAIJ(mat, &blocks[0], &blocks[1], NULL));
  for (PetscInt i = 0; i < 2 && blocks[i]; i++) {
    PetscCall(PetscObjectTypeCompare((PetscObject)blocks[i], MATSEQAIJ, &isaij));
    PetscCall(PetscObjectTypeCompare((PetscObject)blocks[i], MATSEQSBAIJ, &issbaij));
    if (isaij) PetscCall(MatSeqAIJGetArray(blocks[i], &a));
    else if (issbaij) PetscCall(MatSeqSBAIJGetArray(blocks[i], &a));
    else PetscCall(MatSeqBAIJGetArray(blocks[i], &a));
    PetscCheck(!a, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scalar array was allocated in block %" PetscInt_FMT, i);
    if (isaij) PetscCall(MatSeqAIJRestoreArray(blocks[i], &a));
    else if (issbaij) PetscCall(MatSeqSBAIJRestoreArray(blocks[i], &a));
    else PetscCall(MatSeqBAIJRestoreArray(blocks[i], &a));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPattern(Mat mat, PetscBool missing_tail, PetscBool gap)
{
  PetscInt        rstart, rend, n, bs, nz, first, expected;
  const PetscInt *cols;
  PetscBool       issbaij;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(mat, NULL, &n));
  PetscCall(MatGetBlockSize(mat, &bs));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)mat, &issbaij, MATSEQSBAIJ, MATMPISBAIJ, ""));
  PetscCall(MatGetRowUpperTriangular(mat));
  for (PetscInt i = rstart; i < rend; i++) {
    first    = issbaij ? i / bs * bs : 0;
    expected = PetscMax(0, n - (missing_tail ? bs : 0) - first) - (gap && i < bs ? bs : 0);
    PetscCall(MatGetRow(mat, i, &nz, &cols, NULL));
    PetscCheck(nz == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Row %" PetscInt_FMT " has %" PetscInt_FMT " entries, expected %" PetscInt_FMT, i, nz, expected);
    for (PetscInt j = 0, col = first; j < nz; j++, col++) {
      if (gap && i < bs && col == n - 2 * bs) col += bs;
      PetscCheck(cols[j] == col, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Row %" PetscInt_FMT " column %" PetscInt_FMT " is %" PetscInt_FMT ", expected %" PetscInt_FMT, i, j, cols[j], col);
    }
    PetscCall(MatRestoreRow(mat, i, &nz, &cols, NULL));
  }
  PetscCall(MatRestoreRowUpperTriangular(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSubMatrices(Mat mat, PetscBool structure_only)
{
  MPI_Comm           comm;
  Mat                submat;
  IS                 isrow, iscol, iscol_all;
  const PetscInt    *rows, *cols, *subcols;
  const PetscScalar *vals;
  PetscInt           m, rstart, rend, nrow, ncol, start, nz, count;
  PetscMPIInt        rank;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatGetSize(mat, &m, NULL));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  for (PetscInt mode = 0; mode < 6; mode++) {
    // Exercise local, redistributed, permuted, gathered, contiguous, and empty index sets.
    nrow = (rend - rstart) / 2;
    if (mode == 3) nrow = rank ? 0 : m;
    if (mode == 5) nrow = 0;
    start = mode == 2 ? rend % m : rstart;
    if (mode == 3) start = 0;
    PetscCall(ISCreateStride(comm, nrow, start, mode < 3 ? 2 : 1, &isrow));
    if (mode == 1) PetscCall(ISCreateStride(comm, rank ? 0 : m / 2, 0, 2, &iscol));
    else if (mode == 2) PetscCall(ISCreateStride(comm, nrow, m - 1 - rstart, -2, &iscol));
    else PetscCall(ISDuplicate(isrow, &iscol));
    PetscCall(ISAllGather(iscol, &iscol_all));
    PetscCall(ISGetLocalSize(iscol_all, &ncol));
    PetscCall(ISGetIndices(isrow, &rows));
    PetscCall(ISGetIndices(iscol_all, &cols));
    for (PetscInt pass = 0; pass < 2; pass++) {
      if (pass && !structure_only) PetscCall(MatScale(mat, 2.0));
      PetscCall(MatCreateSubMatrix(mat, isrow, iscol, pass ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, &submat));
      if (structure_only) PetscCall(CheckStructureOnlyStorage(submat));
      PetscCall(MatGetOwnershipRange(submat, &start, NULL));
      for (PetscInt i = 0; i < nrow; i++) {
        PetscCall(MatGetRow(submat, start + i, &nz, &subcols, structure_only ? NULL : &vals));
        count = 0;
        for (PetscInt j = 0; j < ncol; j++) {
          if ((rows[i] + 2 * cols[j]) % 4 == 0 || rows[i] == m - 1 || cols[j] == m - 1) continue;
          PetscCheck(count < nz && subcols[count] == j, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect submatrix pattern for mode %" PetscInt_FMT " pass %" PetscInt_FMT, mode, pass);
          if (!structure_only) PetscCheck(vals[count] == (pass ? 2.0 : 1.0) * (10.0 * rows[i] + cols[j]), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect submatrix value");
          count++;
        }
        PetscCheck(count == nz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected submatrix entries");
        PetscCall(MatRestoreRow(submat, start + i, &nz, &subcols, structure_only ? NULL : &vals));
      }
      if (pass && !structure_only) PetscCall(MatScale(mat, 0.5));
    }
    PetscCall(MatDestroy(&submat));
    PetscCall(ISRestoreIndices(isrow, &rows));
    PetscCall(ISRestoreIndices(iscol_all, &cols));
    PetscCall(ISDestroy(&isrow));
    PetscCall(ISDestroy(&iscol));
    PetscCall(ISDestroy(&iscol_all));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat         mat;
  PetscInt    m = 7, n, nlocal, i, j, rstart, rend, bs;
  PetscMPIInt size;
  PetscScalar v;
  PetscBool   struct_only = PETSC_TRUE, explicit_preallocation = PETSC_FALSE, reassemble = PETSC_FALSE, blocked = PETSC_FALSE, product_test = PETSC_FALSE, submatrix_test = PETSC_FALSE, ismpiaij, ismpibaij, ismpisbaij;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-struct_only", &struct_only, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-explicit_preallocation", &explicit_preallocation, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reassemble", &reassemble, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-blocked", &blocked, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-product_test", &product_test, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-submatrix_test", &submatrix_test, NULL));
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
    else {
      PetscCall(MatSeqAIJSetPreallocation(mat, n, NULL));
      PetscCall(MatSeqBAIJSetPreallocation(mat, bs, n / bs, NULL));
      PetscCall(MatSeqSBAIJSetPreallocation(mat, bs, n / bs, NULL));
    }
  } else PetscCall(MatSetUp(mat));
  PetscCall(MatGetBlockSize(mat, &bs));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    for (j = 0; j < n - (reassemble ? bs : 0); j++) {
      if (reassemble && i < bs && j >= n - 2 * bs) continue;
      if (submatrix_test && ((i + 2 * j) % 4 == 0 || i == m - 1 || j == n - 1)) continue;
      v = 10.0 * i + j;
      PetscCall(MatSetValues(mat, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  if (!submatrix_test) PetscCall(CheckPattern(mat, reassemble, reassemble));
  if (struct_only) PetscCall(CheckStructureOnlyStorage(mat));
  if (reassemble) {
    PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    for (PetscInt pass = 0; pass < 3; pass++) {
      // Fill a gap in an existing off-diagonal column, grow the column map, then update existing entries.
      for (i = rstart; i < rend; i += bs) {
        PetscInt row = (i + rend - rstart) % m, col = n / bs - (pass ? 1 : 2);

        if (!pass && row) continue;

        if (blocked) {
          row /= bs;
          PetscCall(MatSetValuesBlocked(mat, 1, &row, 1, &col, NULL, ADD_VALUES));
        } else {
          for (PetscInt ii = row; ii < row + bs; ii++) {
            for (j = col * bs; j < (col + 1) * bs; j++) PetscCall(MatSetValues(mat, 1, &ii, 1, &j, NULL, ADD_VALUES));
          }
        }
      }
      PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
      PetscCall(CheckPattern(mat, (PetscBool)!pass, PETSC_FALSE));
      if (struct_only) PetscCall(CheckStructureOnlyStorage(mat));
    }
  }
  if (product_test) {
    for (PetscInt i = 0; i < 2; i++) {
      Mat product;

      PetscCall(MatProductCreate(mat, mat, NULL, &product));
      PetscCall(MatSetOption(product, MAT_STRUCTURE_ONLY, PETSC_TRUE));
      PetscCall(MatProductSetType(product, i ? MATPRODUCT_AtB : MATPRODUCT_AB));
      PetscCall(MatProductSetFromOptions(product));
      PetscCall(MatProductSymbolic(product));
      PetscCall(CheckStructureOnlyStorage(product));
      PetscCall(CheckPattern(product, PETSC_FALSE, PETSC_FALSE));
      PetscCall(MatDestroy(&product));
    }
  }
  if (submatrix_test) PetscCall(CheckSubMatrices(mat, struct_only));
  if (size == 1 && !reassemble && !product_test && !submatrix_test) PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatViewFromOptions(mat, NULL, "-mat_view"));

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
      suffix: mpi_reassembly
      nsize: {{1 2}}
      output_file: output/empty.out
      temporaries: ex107_view.txt
      args: -mat_view :ex107_view.txt -mat_type {{aij baij sbaij}shared output} -mat_block_size {{1 2 3}} -m 12 -reassemble -blocked {{0 1}} -explicit_preallocation {{0 1}}

   test:
      suffix: product
      nsize: {{1 2 3}}
      output_file: output/empty.out
      args: -mat_type aij -m 12 -struct_only false -product_test -explicit_preallocation {{0 1}}

   test:
      suffix: submatrix
      nsize: {{1 2 3}}
      output_file: output/empty.out
      temporaries: ex107_view.txt
      args: -mat_type aij -m 12 -submatrix_test -struct_only {{0 1}} -explicit_preallocation {{0 1}} -mat_view :ex107_view.txt

TEST*/
