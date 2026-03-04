static char help[] = "Tests DMPlexCreateColoring().\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt depth;
  PetscInt distance;
} AppCtx;


PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->depth = 0;
  options->distance = 1;
  PetscOptionsBegin(comm, "", "DMPlexCreateColoring() Test Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-depth", "Stratum depth defining the nodes in the connectivity graph", "ex104.c", options->depth, &options->depth, NULL));
  PetscCall(PetscOptionsInt("-distance", "Coloring distance", "ex104.c", options->distance, &options->distance, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM             dm;
  const PetscInt faces[2]       = {4, 4};
  DMBoundaryType periodicity[2] = {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE};
  AppCtx         user;
  PetscInt ncolors = 0;
  IS *iscolors = NULL;
  ISColoring coloring = NULL;

  PetscFunctionBeginUser;
  /* Create a BoxMesh */
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, faces, NULL, NULL, periodicity, PETSC_TRUE, 0, PETSC_FALSE, &dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "ExampleBoxMesh"));
  /* Color the DMPlex */
  PetscCall(DMPlexCreateColoring(dm, user.depth, user.distance, &coloring));
  PetscCall(ISColoringGetIS(coloring, PETSC_USE_POINTER, &ncolors, &iscolors));
  for (PetscInt c=0; c < ncolors; c++) {
     PetscCall(ISViewFromOptions(iscolors[c], NULL, "-iscoloring_view"));
  }
  PetscCall(ISColoringRestoreIS(coloring, PETSC_USE_POINTER, &iscolors));
  PetscCall(ISColoringDestroy(&coloring));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Serial tests
  test:
    suffix: depth-0_distance-1
    args: -depth 0 -distance 1 -iscoloring_view
  test:
    suffix: depth-0_distance-2
    args: -depth 0 -distance 2 -iscoloring_view
  test:
    suffix: depth-1_distance-1
    args: -depth 1 -distance 1 -iscoloring_view
  test:
    suffix: depth-1_distance-2
    args: -depth 1 -distance 2 -iscoloring_view
  test:
    suffix: depth-2_distance-1
    args: -depth 2 -distance 1 -iscoloring_view
  # Parallel tests
  test:
    suffix: parallel-depth-0_distance-1
    args: -depth 0 -distance 1 -iscoloring_view
  test:
    suffix: parallel-depth-0_distance-2
    nsize: 2
    args: -depth 0 -distance 2 -iscoloring_view
  test:
    suffix: parallel-depth-1_distance-1
    nsize: 2
    args: -depth 1 -distance 1 -iscoloring_view
  test:
    suffix: parallel-depth-1_distance-2
    nsize: 2
    args: -depth 1 -distance 2 -iscoloring_view
  test:
    suffix: parallel-depth-2_distance-1
    nsize: 2
    args: -depth 2 -distance 1 -iscoloring_view

TEST*/
