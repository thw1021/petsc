static char help[] = "Tests PetscOptionsRealArray(), PetscOptionsScalarArray(), PetscOptionsIntArray()\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  PetscReal   array_r[10];
  PetscScalar array_s[10];
  PetscInt    array_i[10];
  PetscInt    nr = PETSC_STATIC_ARRAY_LENGTH(array_r);
  PetscInt    ns = PETSC_STATIC_ARRAY_LENGTH(array_s);
  PetscInt    ni = PETSC_STATIC_ARRAY_LENGTH(array_i);

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Test options", NULL);
  PetscCall(PetscOptionsRealArray("-rarray", "Input a real array", "ex14b.c", array_r, &nr, NULL));
  PetscCall(PetscOptionsScalarArray("-sarray", "Input a scalar array", "ex14b.c", array_s, &ns, NULL));
  PetscCall(PetscOptionsIntArray("-iarray", "Input an int array", "ex14b.c", array_i, &ni, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Real array of length %" PetscInt_FMT "\n", nr));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Scalar array of length %" PetscInt_FMT "\n", ns));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Int array of length %" PetscInt_FMT "\n", ni));
  PetscOptionsEnd();
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    args: -help
    filter: grep "array"
    filter_output: grep "array"

    test:
      suffix: 1

    test:
      suffix: 2
      requires: complex
      args: -sarray 1.0,-2-3i,4.5+6.2i,4.5,6.8+4i,i,-i,-1.2i

    test:
      suffix: 3
      args: -rarray 0,1.1,-2.2,3.3,4
      args: -sarray
      args: -iarray 2
TEST*/
