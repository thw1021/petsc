static char help[] = "Tests MatMatMult() with a 2D Laplacian whose off-diagonal block has many scattered empty rows.\n\
  -M <M>, -N <N> : global grid size\n\n";

#include <petscdmda.h>

int main(int argc, char **argv)
{
  DM            da;
  Mat           A, C, Aaij, Caij;
  DMDALocalInfo info;
  PetscInt      M = 64, N = 64;
  PetscReal     norm, nrm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(DMDAGetLocalInfo(da, &info));
  for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
      MatStencil  row = {0}, col[5];
      PetscScalar v[5];
      PetscInt    n = 0;

      row.i = i;
      row.j = j;
      if (j > 0) {
        col[n]   = row;
        col[n].j = j - 1;
        v[n++]   = -1.0;
      }
      if (i > 0) {
        col[n]   = row;
        col[n].i = i - 1;
        v[n++]   = -1.0;
      }
      col[n] = row;
      v[n++] = 4.0;
      if (i < info.mx - 1) {
        col[n]   = row;
        col[n].i = i + 1;
        v[n++]   = -1.0;
      }
      if (j < info.my - 1) {
        col[n]   = row;
        col[n].j = j + 1;
        v[n++]   = -1.0;
      }
      PetscCall(MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // C = A*A, twice to also exercise the numeric-only (reuse) path, compared with the MATAIJ product
  PetscCall(MatMatMult(A, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatScale(A, 2.0));
  PetscCall(MatMatMult(A, A, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &Aaij));
  PetscCall(MatMatMult(Aaij, Aaij, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Caij));
  PetscCall(MatNorm(Caij, NORM_FROBENIUS, &nrm));
  PetscCall(MatConvert(C, MATAIJ, MAT_INPLACE_MATRIX, &C));
  PetscCall(MatAXPY(C, -1.0, Caij, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(C, NORM_FROBENIUS, &norm));
  PetscCheck(norm <= 100 * PETSC_MACHINE_EPSILON * nrm, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMatMult() differs from the MATAIJ product: relative error %g", (double)(norm / nrm));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&Aaij));
  PetscCall(MatDestroy(&Caij));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: kok
    nsize: 4
    requires: kokkos_kernels
    args: -dm_mat_type aijkokkos -M 512 -N 512
    output_file: output/empty.out

TEST*/
