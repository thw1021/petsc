static char help[] = "Element closure restrictions in tensor/lexicographic/spectral-element ordering using DMPlex\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   section;
  PetscFE        fe;
  PetscInt       dim,c,cStart,cEnd;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMGetDimension(dm, &dim));

  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMAddField(dm,NULL,(PetscObject)fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL));
  CHKERRQ(DMGetLocalSection(dm,&section));
  CHKERRQ(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  for (c=cStart; c<cEnd; c++) {
    PetscInt numindices,*indices;
    CHKERRQ(DMPlexGetClosureIndices(dm,section,section,c,PETSC_TRUE,&numindices,&indices,NULL,NULL));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Element #%D\n",c-cStart));
    CHKERRQ(PetscIntView(numindices,indices,PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(DMPlexRestoreClosureIndices(dm,section,section,c,PETSC_TRUE,&numindices,&indices,NULL,NULL));
  }

  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 1d_q2
    args: -dm_plex_dim 1 -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_box_faces 2
  test:
    suffix: 2d_q1
    args: -dm_plex_dim 2 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 2,2
  test:
    suffix: 2d_q2
    args: -dm_plex_dim 2 -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_box_faces 2,2
  test:
    suffix: 2d_q3
    args: -dm_plex_dim 2 -petscspace_degree 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1
  test:
    suffix: 3d_q1
    args: -dm_plex_dim 3 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1

TEST*/