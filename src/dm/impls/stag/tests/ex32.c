static char help[] = "Test DMStagRestrictSimple()\n\n";

#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dm,dm_coarse;
  Vec            vec,vec_coarse,vec_local,vec_local_coarse;
  PetscInt       dim,size_coarse;
  PetscReal      norm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dim = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);
  switch (dim) {
    case 1:
      ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,2,3,DMSTAG_STENCIL_BOX,1,NULL,&dm);CHKERRQ(ierr);
      break;
    case 2:
      ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,8,16,PETSC_DECIDE,PETSC_DECIDE,2,3,4,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
      break;
    case 3:
      ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,8,12,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,3,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented!");
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMCoarsen(dm,MPI_COMM_NULL,&dm_coarse);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm,&vec);CHKERRQ(ierr);
  ierr = VecSet(vec,1.0);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&vec_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,vec,INSERT_VALUES,vec_local);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm_coarse,&vec_coarse);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm_coarse,&vec_local_coarse);CHKERRQ(ierr);

  ierr = DMStagRestrictSimple(dm,vec_local,dm_coarse,vec_local_coarse);CHKERRQ(ierr);

  ierr = DMLocalToGlobal(dm_coarse,vec_local_coarse,INSERT_VALUES,vec_coarse);CHKERRQ(ierr);

  ierr = VecGetSize(vec_coarse,&size_coarse);CHKERRQ(ierr);
  ierr = VecNorm(vec_coarse,NORM_1,&norm);CHKERRQ(ierr);
  if ((norm - size_coarse)/((PetscReal) size_coarse) > PETSC_MACHINE_EPSILON * 10.0) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Numerical test failed");
  ierr = VecDestroy(&vec_coarse);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_local_coarse);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_local);CHKERRQ(ierr);
  ierr = DMDestroy(&dm_coarse);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1d
      nsize: 1
      args: -dim 1

   test:
      suffix: 1d_par
      nsize: 4
      args: -dim 1

   test:
      suffix: 2d
      nsize: 1
      args: -dim 2

   test:
      suffix: 2d_par
      nsize: 2
      args: -dim 2

   test:
      suffix: 2d_par_2
      nsize: 8
      args: -dim 2

TEST*/
