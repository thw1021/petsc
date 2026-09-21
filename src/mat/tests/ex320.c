static char help[] = "Tests MatConvert() from MATMPIAIJ to a device AIJ type with MAT_INITIAL_MATRIX and MAT_REUSE_MATRIX.\n\n";

/* The converted matrix must be usable by the operations of its new type, which for the device
   types means its diagonal and off-diagonal sequential blocks carry the device representation.
   A product is the check: MatMult() alone is satisfied by the host blocks, while the symbolic
   phase of a product asks the blocks for their device matrix directly. */

#include <petscmat.h>

static PetscErrorCode CheckProduct(Mat B, Mat Cref, const char *label)
{
  Mat       C;
  PetscReal nrm, err;

  PetscFunctionBeginUser;
  PetscCall(MatMatMult(B, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatNorm(Cref, NORM_FROBENIUS, &nrm));
  PetscCall(MatAXPY(C, -1.0, Cref, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C, NORM_FROBENIUS, &err));
  PetscCheck(err <= PETSC_SMALL * nrm, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "%s: product differs from the host reference, ||E||_F = %g", label, (double)err);
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A, B, Cref;
  PetscInt    Istart, Iend, i, n = 20;
  PetscScalar v;
  char        convtype[256];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscStrncpy(convtype, MATMPIAIJ, sizeof(convtype)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-conv_mat_type", convtype, sizeof(convtype), NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    PetscInt j;

    v = 2.0;
    PetscCall(MatSetValues(A, 1, &i, 1, &i, &v, INSERT_VALUES));
    v = -1.0;
    if (i > 0) {
      j = i - 1;
      PetscCall(MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
    if (i < n - 1) {
      j = i + 1;
      PetscCall(MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cref));

  PetscCall(MatConvert(A, convtype, MAT_INITIAL_MATRIX, &B));
  PetscCall(CheckProduct(B, Cref, "MAT_INITIAL_MATRIX"));
  PetscCall(MatConvert(A, convtype, MAT_REUSE_MATRIX, &B));
  PetscCall(CheckProduct(B, Cref, "MAT_REUSE_MATRIX"));

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Cref));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: {{1 2}}
    output_file: output/empty.out

    test:
      suffix: aij
      args: -conv_mat_type aij

    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -conv_mat_type aijkokkos

    test:
      suffix: cusparse
      requires: cuda
      args: -conv_mat_type aijcusparse

    test:
      suffix: hipsparse
      requires: hip
      args: -conv_mat_type aijhipsparse

TEST*/
