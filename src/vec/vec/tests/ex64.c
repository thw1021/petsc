static char help[] = "Tests VecLog().\n\n";

#include <petscvec.h>

static PetscErrorCode IsCloseAtTolScalar(PetscScalar lhs, PetscScalar rhs, PetscInt idx)
{
  const PetscReal lhs_r = PetscRealPart(lhs);
  const PetscReal lhs_i = PetscImaginaryPart(lhs);
  const PetscReal rhs_r = PetscRealPart(rhs);
  const PetscReal rhs_i = PetscImaginaryPart(rhs);

  PetscFunctionBegin;
  PetscCheck(PetscIsCloseAtTol(lhs_r, rhs_r, 1e-9, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Real component lhs[%" PetscInt_FMT "] %g != rhs[%" PetscInt_FMT "] %g", idx, (double)lhs_r, idx, (double)rhs_r);
  PetscCheck(PetscIsCloseAtTol(lhs_i, rhs_i, 1e-9, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Imaginary component lhs[%" PetscInt_FMT "] %g != rhs[%" PetscInt_FMT "] %g", idx, (double)lhs_i, idx, (double)rhs_i);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckLog(Vec v, PetscInt n, PetscScalar *arr, PetscScalar value)
{
  const PetscScalar *varr;

  PetscFunctionBegin;
  PetscCall(VecSet(v, value));
  PetscCall(VecViewFromOptions(v, NULL, "-vec_view"));
  PetscCall(VecLog(v));
  PetscCall(VecViewFromOptions(v, NULL, "-vec_view"));

  for (PetscInt i = 0; i < n; ++i) arr[i] = PetscLogScalar(value);
  PetscCall(VecGetArrayRead(v, &varr));
  for (PetscInt i = 0; i < n; ++i) PetscCall(IsCloseAtTolScalar(varr[i], arr[i], i));
  PetscCall(VecRestoreArrayRead(v, &varr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec          v;
  PetscInt     n;
  PetscScalar *arr;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(VecSetSizes(v, 10, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(v));

  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(PetscMalloc1(n, &arr));

  PetscCall(CheckLog(v, n, arr, 1.0));
  PetscCall(CheckLog(v, n, arr, PetscExpScalar(1.0)));
  PetscCall(CheckLog(v, n, arr, 10.0));

  PetscCall(PetscFree(arr));
  PetscCall(VecDestroy(&v));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: ./output/empty.out
    nsize: {{1 2}}
    test:
      suffix: standard
      args: -vec_type standard
    test:
      suffix: viennacl
      requires: viennacl
      args: -vec_type viennacl
    test:
      suffix: cuda
      requires: cuda
      args: -vec_type cuda
    test:
      suffix: hip
      requires: hip
      args: -vec_type hip
    test:
      suffix: kokkos
      requires: kokkos, kokkos_kernels
      args: -vec_type kokkos

TEST*/
