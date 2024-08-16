static char help[] = "Error handling for external library call in void function.\n";

#include <petscsys.h>
#include <petscvec.h>

int ReturnAnError()
{
  return 1;
}

void MakeAnError()
{
  PetscCallExternalAbort(ReturnAnError);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  MakeAnError();
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: !defined(PETSCTEST_VALGRIND) defined(PETSC_USE_DEBUG) !defined(PETSC_HAVE_SANITIZER)
     args: -petsc_ci_portable_error_output -error_output_stdout

TEST*/
