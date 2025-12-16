static char help[] = "Demonstrates PetscRealGetNan().\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  PetscReal r;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscRealGetNaN(&r));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stdout, "NaN %g\n",(double)r));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
