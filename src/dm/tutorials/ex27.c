static char help[] = "Test PetscSection with DMDA\n";
#include <petscdm.h>
#include <petscdmda.h>
#include <petsc.h>


int main(int argc, char *argv[])
{
  DM              da;
  PetscInt       Nx = 6;
  PetscSection    lsection, gsection;
  PetscSF      sf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));

  /* Create 2D DMDA */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, Nx, Nx, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMGetLocalSection(da, &lsection));

  /* Create the global section */
  PetscCall(DMGetPointSF(da, &sf));
  PetscCall(PetscSectionCreateGlobalSection(lsection, sf, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &gsection));
  PetscCall(DMSetGlobalSection(da, gsection));
  /* View the global section */
  PetscSectionView(gsection, PETSC_VIEWER_STDOUT_WORLD);

  PetscCall(PetscFinalize());
  return 0;
}