static char help[] = "Tests that MATSELL honors MAT_NEW_NONZERO_LOCATIONS and MAT_IGNORE_ZERO_ENTRIES.\n\n";

/*
  MatSetValues() must silently ignore an entry that would create a new nonzero location once
  MAT_NEW_NONZERO_LOCATIONS is false, and must ignore a zero value once MAT_IGNORE_ZERO_ENTRIES is
  true. On one process this exercises MatSetValues_SeqSELL(); on two it exercises
  MatSetValues_MPISELL(), probing the diagonal block, an off-diagonal column that garray already
  knows, and an off-diagonal column it does not. A MATAIJ matrix receives the same calls and is
  used as the reference for the resulting values.

  MAT_IGNORE_ZERO_ENTRIES leaves nothing visible behind, so it is probed indirectly: the ignored
  zero is followed by a nonzero write to the same location with MAT_NEW_NONZERO_LOCATIONS false. If
  the zero had wrongly created the location, that second write would land and the values would differ.
*/

#include <petscmat.h>

static PetscErrorCode CheckEntry(Mat A, Mat B, PetscInt row, PetscInt col, const char *where)
{
  PetscScalar va, vb;

  PetscFunctionBeginUser;
  PetscCall(MatGetValues(A, 1, &row, 1, &col, &va));
  PetscCall(MatGetValues(B, 1, &row, 1, &col, &vb));
  PetscCheck(va == vb, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MATSELL has %g at (%" PetscInt_FMT ",%" PetscInt_FMT ") but MATAIJ has %g: %s", (double)PetscRealPart(va), row, col, (double)PetscRealPart(vb), where);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* set the same entry in both matrices, then reassemble both */
static PetscErrorCode SetAndAssemble(Mat A, Mat B, PetscInt row, PetscInt col, PetscScalar value)
{
  PetscFunctionBeginUser;
  PetscCall(MatSetValues(A, 1, &row, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatSetValues(B, 1, &row, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A, B;
  PetscInt    i, rstart, rend, col, known, unknown;
  PetscMPIInt rank, size;
  PetscScalar value = 1.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 1 || size == 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "This test requires 1 or 2 processes");

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 4, 4, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, MATSELL));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqSELLSetPreallocation(A, 4, NULL));
  PetscCall(MatMPISELLSetPreallocation(A, 4, NULL, 4, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, 4, 4, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(B, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(B, 4, NULL));
  PetscCall(MatMPIAIJSetPreallocation(B, 4, NULL, 4, NULL));

  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  /* known enters garray during the first assembly; unknown never does */
  known   = rank ? 0 : 4;
  unknown = rank ? 1 : 5;

  for (i = rstart; i < rend; i++) {
    PetscCall(MatSetValues(A, 1, &i, 1, &i, &value, INSERT_VALUES));
    PetscCall(MatSetValues(B, 1, &i, 1, &i, &value, INSERT_VALUES));
  }
  if (size > 1) {
    PetscCall(MatSetValues(A, 1, &rstart, 1, &known, &value, INSERT_VALUES));
    PetscCall(MatSetValues(B, 1, &rstart, 1, &known, &value, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  /* from here on, a new nonzero location must be ignored rather than created */
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));

  i = rstart + 1;
  PetscCall(SetAndAssemble(A, B, i, rstart + 2, 7.0)); /* diagonal block */
  PetscCall(CheckEntry(A, B, i, rstart + 2, "new location in the diagonal block"));
  if (size > 1) {
    PetscCall(SetAndAssemble(A, B, i, known, 7.0)); /* off-diagonal, column in garray */
    PetscCall(CheckEntry(A, B, i, known, "new location in an off-diagonal column garray knows"));
    PetscCall(SetAndAssemble(A, B, i, unknown, 7.0)); /* off-diagonal, column not in garray */
    PetscCall(CheckEntry(A, B, i, unknown, "new location in an off-diagonal column garray does not know"));
  }

  /* an existing location must still be writable */
  PetscCall(SetAndAssemble(A, B, i, i, 9.0));
  PetscCall(CheckEntry(A, B, i, i, "overwrite of an existing diagonal entry"));

  /* a zero value must not create a location either */
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  col = rstart + 3;
  PetscCall(SetAndAssemble(A, B, i, col, 0.0));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(SetAndAssemble(A, B, i, col, 5.0));
  PetscCall(CheckEntry(A, B, i, col, "location that an ignored zero must not have created"));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/empty.out

      test:
         suffix: 1
         nsize: {{1 2}}
         args: -mat_type sell

      test:
         suffix: cuda
         nsize: {{1 2}}
         requires: cuda
         args: -mat_type sellcuda

      test:
         suffix: hip
         nsize: {{1 2}}
         requires: hip
         args: -mat_type sellhip

TEST*/
