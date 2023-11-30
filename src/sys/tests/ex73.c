const char help[] = "Test dropping PetscLogEventEnd()";

#include <petsc.h>

int main(int argc, char **argv)
{
  FILE *file;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  file = fopen("test", "w");
  fclose(file);
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "Testing error handling with bad \n"));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: !defined(PETSCTEST_VALGRIND)
    args: -petsc_ci_portable_error_output -error_output_stdout

TEST*/
