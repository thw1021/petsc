static char help[] = "Tests mesh reordering\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm, ndm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = DMCreate(PETSC_COMM_WORLD, &dm); CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  {
  }
  ierr = DMPlexDistribute(dm, 2, NULL, &ndm); CHKERRQ(ierr);
  if (ndm) {
    ierr = DMDestroy(&dm); CHKERRQ(ierr);
    dm   = ndm;
  }
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    args: -dm_plex_box_faces 3,3 -dm_plex_reorder rcm -dm_view ::ascii_info_detail

    test:
      suffix: 0

    test:
      suffix: 1
      nsize: 2

TEST*/
