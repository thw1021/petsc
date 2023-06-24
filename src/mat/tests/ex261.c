static const char help[] = "Tests MatGetDiagonal().\n\n";

#include <petscmat.h>

static PetscErrorCode IsCloseAtTolScalar(PetscScalar lhs, PetscScalar rhs, PetscInt idx)
{
  const PetscReal lhs_r = PetscRealPart(lhs);
  const PetscReal lhs_i = PetscImaginaryPart(lhs);
  const PetscReal rhs_r = PetscRealPart(rhs);
  const PetscReal rhs_i = PetscImaginaryPart(rhs);

  PetscFunctionBegin;
  PetscCheck(PetscIsCloseAtTol(lhs_r, rhs_r, 1e-12, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Real component lhs[%" PetscInt_FMT "] %g != rhs[%" PetscInt_FMT "] %g", idx, (double)lhs_r, idx, (double)rhs_r);
  PetscCheck(PetscIsCloseAtTol(lhs_i, rhs_i, 1e-12, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Imaginary component lhs[%" PetscInt_FMT "] %g != rhs[%" PetscInt_FMT "] %g", idx, (double)lhs_i, idx, (double)rhs_i);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckDiagonal(Mat A, Vec diag, PetscScalar dval)
{
  static PetscBool   first_time = PETSC_TRUE;
  PetscInt           rstart, rend, n;
  const PetscScalar *arr;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  // If matrix is AIJ, MatSetRandom() will have randomly choosen the locations of nonzeros,
  // which may not be on the diagonal. So a reallocation is not necessarily a bad thing here.
  if (first_time) PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  for (PetscInt i = rstart; i < rend; ++i) PetscCall(MatSetValue(A, i, i, dval, INSERT_VALUES));
  if (first_time) {
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    first_time = PETSC_FALSE;
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(A, NULL, "-mat_view_assembled"));

  PetscCall(MatGetDiagonal(A, diag));
  PetscCall(VecViewFromOptions(diag, NULL, "-diag_vec_view"));

  PetscCall(VecGetLocalSize(diag, &n));
  PetscCall(VecGetArrayRead(diag, &arr));
  for (PetscInt i = 0; i < n; ++i) PetscCall(IsCloseAtTolScalar(arr[i], dval, i));
  PetscCall(VecRestoreArrayRead(diag, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeMatrix(Mat A)
{
  PetscInt  rows, cols, diag_nnz, offdiag_nnz;
  PetscInt *dnnz, *onnz;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(A, &rows, &cols));
  // at least 3 nonzeros in diagonal block
  diag_nnz = PetscMin(cols, 3);
  // leave at least 3 *zeros* per row
  offdiag_nnz = PetscMax(cols - diag_nnz - 3, 0);
  PetscCall(PetscMalloc2(rows, &dnnz, rows, &onnz));
  for (PetscInt i = 0; i < rows; ++i) {
    dnnz[i] = diag_nnz;
    onnz[i] = offdiag_nnz;
  }
  PetscCall(MatXAIJSetPreallocation(A, PETSC_DECIDE, dnnz, onnz, NULL, NULL));
  PetscCall(PetscFree2(dnnz, onnz));

  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatViewFromOptions(A, NULL, "-mat_view_setup"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat A;
  Vec diag;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 10, 10, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(InitializeMatrix(A));

  PetscCall(MatCreateVecs(A, &diag, NULL));

  PetscCall(CheckDiagonal(A, diag, 0.0));
  PetscCall(CheckDiagonal(A, diag, 1.0));
  PetscCall(CheckDiagonal(A, diag, 2.0));

  PetscCall(VecDestroy(&diag));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: ./output/empty.out
    nsize: {{1 2}}
    suffix: dense
    test:
      suffix: standard
      args: -mat_type dense
    test:
      suffix: cuda
      requires: cuda
      args: -mat_type densecuda
    test:
      suffix: hip
      requires: hip
      args: -mat_type densehip

  testset:
    output_file: ./output/empty.out
    nsize: {{1 2}}
    suffix: aij
    test:
      suffix: standard
      args: -mat_type aij
    test:
      suffix: viennacl
      requires: viennacl
      args: -mat_type aijviennacl
    test:
      suffix: cuda
      requires: cuda
      args: -mat_type aijcusparse
    test:
      suffix: hip
      requires: hip
      args: -mat_type aijhipsparse
    test:
      suffix: kokkos
      requires: kokkos, kokkos_kernels
      args: -mat_type aijkokkos

TEST*/
