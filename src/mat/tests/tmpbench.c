
static char help[] = "Benchmark dense matrix LU factorization (BLAS/LAPACK)\n\n";

#include <petscbm.h>
#include <petscmat.h>

int main(int argc, char **argv)
{
  PetscBM bm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(MatInitializePackage());
  PetscCall(PetscBMCreate(PETSC_COMM_SELF, &bm));
  PetscCall(PetscBMSetType(bm, PETSCBMHPL));
  PetscCall(PetscBMSetFromOptions(bm));
  PetscCall(PetscBMSetUp(bm));
  PetscCall(PetscBMRun(bm));
  PetscCall(PetscBMView(bm, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscBMSetSize(bm, 5000));
  PetscCall(PetscBMRun(bm));
  PetscCall(PetscBMView(bm, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscBMDestroy(&bm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: hpl

TEST*/
