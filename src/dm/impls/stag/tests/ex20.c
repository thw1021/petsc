static char help[] = "Test DMStag transfer operators, on a faces-only grid.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscInt       dim;
  PetscBool      flg,dump;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option\n");CHKERRQ(ierr);
    return 1;
  }
  if (dim == 1) {
    ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,1,0,DMSTAG_STENCIL_BOX,1,NULL,&dm);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,PETSC_DECIDE,PETSC_DECIDE,0,1,0,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,0,0,1,0,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option with value 1, 2, or 3\n");CHKERRQ(ierr);
      return 1;
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  /* Flags to dump binary or ASCII output */
  dump = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-dump",&dump,NULL);CHKERRQ(ierr);

  /* Directly create a coarsened DM and transfer operators */
  {
    DM dmCoarse;
    ierr = DMCoarsen(dm,MPI_COMM_NULL,&dmCoarse);CHKERRQ(ierr);
    {
      Mat Ai;
      ierr = DMCreateInterpolation(dmCoarse,dm,&Ai,NULL);CHKERRQ(ierr);
      if (dump) {
        PetscViewer viewer;
        ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"matI.pbin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        ierr = MatView(Ai,viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&Ai);CHKERRQ(ierr);
    }
    {
      Mat Ar;
      ierr = DMCreateRestriction(dmCoarse,dm,&Ar);CHKERRQ(ierr);
      if (dump) {
        PetscViewer viewer;
        ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"matR.pbin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        ierr = MatView(Ar,viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&Ar);CHKERRQ(ierr);
    }
    ierr = DMDestroy(&dmCoarse);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -dim 1

   test:
      suffix: 2
      nsize: 1
      args: -dim 2

TEST*/
