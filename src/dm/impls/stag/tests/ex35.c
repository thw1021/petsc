static char help[] = "Test DMStag coordinates with periodic boundary conditions";

#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  DM             dm;
  Vec            coord, coord_local;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dim = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);

  if (dim == 1) {
    ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,2,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,2,2,PETSC_DECIDE,PETSC_DECIDE,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,2,2,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Supply -dim option with value 1, 2, or 3\n");

  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);CHKERRQ(ierr);

  ierr = DMStagSetUniformCoordinatesExplicit(dm,-1.0,1.0,-2.0,2.0,-3.0,3.0);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm,&coord_local);CHKERRQ(ierr);
  {
    PetscMPIInt size, rank;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
    for (PetscMPIInt r = 0; r<size; ++r) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d] Local Coordinates\n",r);CHKERRQ(ierr);
      if (r == rank) {
        ierr = VecView(coord_local,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    }
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Global Coordinates:\n");CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm,&coord);CHKERRQ(ierr);
  ierr = VecView(coord,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: seq
      nsize: 1
      args: -dim {1,2,3}

   test:
      suffix: par_1d
      nsize: 2
      args: -dim 1

   test:
      suffix: par_2d
      nsize: 4
      args: -dim 2

   test:
      suffix: par_3d
      nsize: 8
      args: -dim 3

TEST*/
