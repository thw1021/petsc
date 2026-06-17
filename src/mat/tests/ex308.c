static char help[] = "Tests MatGetMultPetscSF() for the parallel AIJ, BAIJ, and SBAIJ matrix types.\n\n";

#include <petscmat.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  Mat       A;
  PetscSF   sf;
  PetscInt  N        = 12, rstart, rend, cstart, cend, nroots, nleaves, i;
  PetscInt *rootdata = NULL, *leafdata = NULL, ok = 1;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));

  /* Build a symmetric tridiagonal matrix; the type is selected with -mat_type. */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscScalar diag = 2.0, offd = -1.0;

    PetscCall(MatSetValue(A, i, i, diag, INSERT_VALUES));
    if (i + 1 < N) PetscCall(MatSetValue(A, i, i + 1, offd, INSERT_VALUES));
    if (i - 1 >= 0) PetscCall(MatSetValue(A, i, i - 1, offd, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetMultPetscSF(A, &sf));
  PetscCheck(sf, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatGetMultPetscSF() returned NULL on a parallel matrix");

  /* Broadcast each owned global column index to the off-process columns that couple
     to local rows; every gathered value must lie outside the local column ownership range. */
  PetscCall(MatGetOwnershipRangeColumn(A, &cstart, &cend));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL));
  PetscCall(PetscMalloc2(nroots, &rootdata, nleaves, &leafdata));
  for (i = 0; i < nroots; i++) rootdata[i] = cstart + i;
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, rootdata, leafdata, MPI_REPLACE));
  for (i = 0; i < nleaves; i++) {
    if (leafdata[i] < 0 || leafdata[i] >= N || (leafdata[i] >= cstart && leafdata[i] < cend)) ok = 0;
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &ok, 1, MPIU_INT, MPI_LAND, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatGetMultPetscSF() OK (PetscSF valid, all gathered columns off-process: %s)\n", ok ? "yes" : "no"));

  PetscCall(PetscFree2(rootdata, leafdata));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: aij
      nsize: {{2 3}}
      args: -mat_type mpiaij
      output_file: output/ex308.out

   test:
      suffix: baij
      nsize: {{2 3}}
      args: -mat_type mpibaij
      output_file: output/ex308.out

   test:
      suffix: sbaij
      nsize: {{2 3}}
      args: -mat_type mpisbaij
      output_file: output/ex308.out

   test:
      suffix: kokkos
      requires: kokkos_kernels
      nsize: {{2 3}}
      args: -mat_type mpiaijkokkos
      output_file: output/ex308.out

TEST*/
