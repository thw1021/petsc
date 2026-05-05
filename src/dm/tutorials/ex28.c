
static char help[] = "Test PetscSection with DMDA, created with DM_BOUNDARY_GHOSTED\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petsc.h>

// run with -da_use_section

int main(int argc, char *argv[])
{
  DM           da;
  PetscInt     Nx = 6;
  PetscSection lsection, gsection;
  PetscSF      sf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, Nx, Nx, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMGetLocalSection(da, &lsection));

  PetscCall(DMGetPointSF(da, &sf));
  PetscCall(PetscSectionCreateGlobalSection(lsection, sf, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &gsection));
  PetscCall(DMSetGlobalSection(da, gsection));

  PetscCall(PetscSectionDestroy(&gsection));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

// mpiexec -n 2 ./ex28 -da_use_section does not work
