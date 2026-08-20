static char help[] = "Demonstrates restricting the -help output to specific man pages with -help <manpage>.\n\n";

#include <petscsys.h>
#include <petscoptions.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscReal r1 = 0., r2 = -1.;
  PetscReal s1 = 0., s2 = -1.;
  PetscInt  i1 = 0, i2 = -1;
  DM        da;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Scaled units options 1", "man1");
  PetscCall(PetscOptionsReal("-r1", "r1: real", "man1", r1, &r1, NULL));
  PetscCall(PetscOptionsReal("-s1", "s1: real", "man1", s1, &s1, NULL));
  PetscCall(PetscOptionsInt("-i1", "i1: int", "man1", i1, &i1, NULL));
  PetscOptionsEnd();

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Scaled units options 2", "man2");
  PetscCall(PetscOptionsReal("-r2", "r2: real", "man2", r2, &r2, NULL));
  PetscCall(PetscOptionsReal("-s2", "s2: real", "man2", s2, &s2, NULL));
  PetscCall(PetscOptionsInt("-i2", "i2: int", "man2", i2, &i2, NULL));
  PetscOptionsEnd();

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMView(da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   # "-help manpage" restricts the help output to the options registered with the listed man page(s),
   # each printed under its own block header; filter to the options blocks to avoid the version banner
   testset:
      filter: grep -E -e "^Scaled units options|\(man[12]\)"
      test:
         suffix: help_man1
         args: -r1 2 -help man1
      test:
         suffix: help_man1_man2
         args: -r1 2 -help man1,man2
      test:
          suffix: help
          args: -help

TEST*/
