static char help[] = "(Partially) test DMStag default interpolation, 2d faces-only.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>
#include <petscksp.h>

PetscErrorCode CreateSystem(DM dm,Mat *A,Vec *b);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dm,dmCoarse;
  Mat            Ai;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,4,PETSC_DECIDE,PETSC_DECIDE,0,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMCoarsen(dm,MPI_COMM_NULL,&dmCoarse);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dmCoarse,dm,&Ai,NULL);CHKERRQ(ierr);

  /* See what happens to a constant value on each sub-grid */
  {
    Vec      localCoarse,globalCoarse,globalFine,localFine;
    ierr = DMGetGlobalVector(dm,&globalFine);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dmCoarse,&globalCoarse);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmCoarse,&localCoarse);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&localFine);CHKERRQ(ierr);
    ierr = VecSet(localCoarse,-1.0);CHKERRQ(ierr);
    ierr = VecSet(localFine,-1.0);CHKERRQ(ierr);
    {
      PetscInt i,j,startx,starty,nx,ny,extrax,extray;
      PetscInt p,vx,vy;
      PetscScalar ***arr;
      ierr = DMStagGetCorners(dmCoarse,&startx,&starty,NULL,&nx,&ny,NULL,&extrax,&extray,NULL);CHKERRQ(ierr);
      ierr = DMStagVecGetArray(dmCoarse,localCoarse,&arr);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmCoarse,DMSTAG_LEFT,0,&vx);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmCoarse,DMSTAG_DOWN,0,&vy);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dmCoarse,DMSTAG_ELEMENT,0,&p);CHKERRQ(ierr);
      for (j=starty; j<starty+ny+extray; ++j) {
        for (i=startx; i<startx+nx+extrax; ++i) {
          arr[j][i][vy] = (i<startx+nx)                  ? 10.0 : -1;
          arr[j][i][vx] = (j<starty+ny)                  ? 20.0 : -1;
          arr[j][i][p ] = (i<startx+nx) && (j<starty+ny) ? 30.0 : -1;
        }
      }
      ierr = DMStagVecRestoreArray(dmCoarse,localCoarse,&arr);CHKERRQ(ierr);
    }
    ierr = DMLocalToGlobal(dmCoarse,localCoarse,INSERT_VALUES,globalCoarse);CHKERRQ(ierr);
    ierr = MatInterpolate(Ai,globalCoarse,globalFine);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dm,globalFine,INSERT_VALUES,localFine);CHKERRQ(ierr);
    {
      PetscInt i,j,startx,starty,nx,ny,extrax,extray;
      PetscInt p,vx,vy;
      PetscScalar ***arr;
      ierr = DMStagGetCorners(dm,&startx,&starty,NULL,&nx,&ny,NULL,&extrax,&extray,NULL);CHKERRQ(ierr);
      ierr = DMStagVecGetArrayRead(dm,localFine,&arr);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dm,DMSTAG_LEFT,0,&vx);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dm,DMSTAG_DOWN,0,&vy);CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dm,DMSTAG_ELEMENT,0,&p);CHKERRQ(ierr);
      for (j=starty; j<starty+ny+extray; ++j) {
        for (i=startx; i<startx+nx+extrax; ++i) {
          const PetscScalar expected_vy = (i<startx+nx)                  ? 10.0 : -1;
          const PetscScalar expected_vx = (j<starty+ny)                  ? 20.0 : -1;
          const PetscScalar expected_p  = (i<startx+nx) && (j<starty+ny) ? 30.0 : -1;
          if (arr[j][i][vy] != expected_vy) {
            ierr = PetscPrintf(PETSC_COMM_SELF,"wrong %" PetscInt_FMT " %" PetscInt_FMT "\n",i,j);
          }
          if (arr[j][i][vx] != expected_vx) {
            ierr = PetscPrintf(PETSC_COMM_SELF,"wrong %" PetscInt_FMT " %" PetscInt_FMT "\n",i,j);
          }
          if (arr[j][i][p ] != expected_p) {
            ierr = PetscPrintf(PETSC_COMM_SELF,"wrong %" PetscInt_FMT " %" PetscInt_FMT "\n",i,j);
          }
        }
      }
      ierr = DMStagVecRestoreArrayRead(dm,localFine,&arr);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(dmCoarse,&localCoarse);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localFine);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmCoarse,&globalCoarse);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&globalFine);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&Ai);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmCoarse);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args:

   test:
      suffix: 2
      nsize: 4
      args: -stag_grid_x 8 -stag_grid_y 4

TEST*/
