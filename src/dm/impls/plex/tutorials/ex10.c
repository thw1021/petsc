static char help[] = "Create a Plex Schwarz P surface with quads\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscInt       extent[3] = {1,1,1}, refine = 0, three;
  DMBoundaryType periodic[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  DMPlexTPSType  tps_type = DMPLEX_TPS_SCHWARZ_P;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Schwarz P Example", NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-extent", "Number of replicas for each of three dimensions", NULL, extent, (three=3, &three), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-refine", "Number of refinements", NULL, refine, &refine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnumArray("-periodic", "Periodicity in each of three dimensions", NULL, DMBoundaryTypes, (PetscEnum*)periodic, (three=3, &three), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tps_type", "Type of triply-periodic surface", NULL, DMPlexTPSTypes, (PetscEnum)tps_type, (PetscEnum*)&tps_type, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = DMPlexCreateTPSMesh(PETSC_COMM_WORLD, tps_type, extent, periodic, 0, refine, &dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -extent 1,2,3 -refine 0
  test:
    suffix: 1
    args: -extent 2,3,1 -refine 1

  test:
    suffix: gyroid_0
    args: -extent 1,2,3 -refine 0 -tps_type gyroid
  test:
    suffix: gyroid_1
    args: -extent 2,3,1 -refine 1 -tps_type gyroid

TEST*/
