static const char help[] = "Test -log_view inserted after PetscInitialize()\n\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Simulate a library inserting -log_view after initialization, e.g.
     via PetscOptionsInsertString() when reading its own config files.
     PetscLogDefaultBegin() must be called explicitly to start recording
     events because PetscInitialize() only starts logging when -log_view
     is already in the options database at init time. */
  PetscCall(PetscOptionsInsertString(NULL, "-log_view"));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: defined(PETSC_USE_LOG)
    suffix: 0
    filter: grep -c "PETSc Performance Summary"

TEST*/
