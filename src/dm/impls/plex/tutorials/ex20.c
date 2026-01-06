static char help[] = "Test DMPlexGetLETKFLocalizationMatrix\n\n";

#include <petscdmplex.h>
#include <petscmat.h>

int main(int argc, char **argv)
{
  DM             dm;
  Mat            H, Q;
  PetscInt       dim      = 1;
  PetscInt       faces[3] = {8, 8, 0};
  PetscReal      lower[3] = {0.0, 0.0, 0.0};
  PetscReal      upper[3] = {1.0, 1.0, 0.0};
  DMBoundaryType bdt[3]   = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscInt       vStart, vEnd, numVertices;
  PetscInt       numGlobalObs, numObservations;
  PetscInt       M, N;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This test requires exactly 1 MPI process");

  /* Get dimension from options */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dm_plex_dim", &dim, NULL));

  /* Create a simple box mesh */
  if (dim == 1) {
    faces[0] = 16;
    upper[0] = 1.0;
  } else if (dim == 2) {
    faces[0] = 8;
    faces[1] = 8;
    upper[0] = 1.0;
    upper[1] = 1.0;
  } else if (dim == 3) {
    faces[0] = 6;
    faces[1] = 6;
    faces[2] = 6;
    upper[0] = 1.0;
    upper[1] = 1.0;
    upper[2] = 1.0;
  }

  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces, lower, upper, bdt, PETSC_TRUE, 0, PETSC_TRUE, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* Get number of vertices */
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  numVertices = vEnd - vStart;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of vertices: %" PetscInt_FMT "\n", numVertices));

  /* Compute number of observations: select every other vertex in each dimension */
  /* For a grid with (faces[d]+1) vertices per dimension, we select (faces[d]+1)/2 observations */
  numGlobalObs = 1;
  for (PetscInt d = 0; d < dim; d++) numGlobalObs *= (faces[d] / 2 + 1);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of observations: %" PetscInt_FMT "\n", numGlobalObs));

  /* Set number of local observations to use: 3^dim */
  numObservations = 1;
  for (PetscInt d = 0; d < dim; d++) numObservations *= 3;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of local observations per vertex: %" PetscInt_FMT "\n", numObservations));

  /* Create observation operator H: numGlobalObs x numVertices */
  /* H selects every other vertex in each dimension */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &H));
  PetscCall(MatSetSizes(H, numGlobalObs, numVertices, numGlobalObs, numVertices));
  PetscCall(MatSetType(H, MATMPIAIJ));
  PetscCall(MatMPIAIJSetPreallocation(H, 1, NULL, 0, NULL));
  PetscCall(MatSetUp(H));

  /* Fill H matrix: select observations from grid vertices */
  /* For a structured grid, we can compute which vertices to select */
  {
    Vec                coordinates;
    PetscSection       coordSection;
    const PetscScalar *coordArray;
    PetscInt           obsIdx = 0;
    PetscInt           offset;

    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(VecGetArrayRead(coordinates, &coordArray));

    /* For each vertex, check if it should be an observation */
    for (PetscInt v = vStart; v < vEnd; v++) {
      PetscReal coords[3] = {0.0, 0.0, 0.0};
      PetscBool isObs     = PETSC_TRUE;

      PetscCall(PetscSectionGetOffset(coordSection, v, &offset));
      for (PetscInt d = 0; d < dim; d++) coords[d] = PetscRealPart(coordArray[offset + d]);

      /* Check if this vertex is at an observation location (every other grid point) */
      for (PetscInt d = 0; d < dim; d++) {
        PetscReal gridSpacing = upper[d] / faces[d];
        PetscInt  gridIdx     = (PetscInt)(coords[d] / gridSpacing + 0.5);
        if (gridIdx % 2 != 0) {
          isObs = PETSC_FALSE;
          break;
        }
      }

      if (isObs && obsIdx < numGlobalObs) {
        PetscCall(MatSetValue(H, obsIdx, v - vStart, 1.0, INSERT_VALUES));
        obsIdx++;
      }
    }

    PetscCall(VecRestoreArrayRead(coordinates, &coordArray));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Created %" PetscInt_FMT " observations\n", obsIdx));
  }

  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectSetName((PetscObject)H, "H_observation_operator"));
  PetscCall(MatGetSize(H, &M, &N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "H matrix size: %" PetscInt_FMT " x %" PetscInt_FMT "\n", M, N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "H matrix:\n"));
  PetscCall(MatView(H, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatViewFromOptions(H, NULL, "-h_view"));

  /* Perturb interior vertex coordinates */
  {
    Vec           coordinates;
    PetscSection  coordSection;
    PetscScalar  *coordArray;
    unsigned long seed = 123456789;

    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(VecGetArray(coordinates, &coordArray));

    for (PetscInt v = vStart; v < vEnd; v++) {
      PetscReal coords[3] = {0.0, 0.0, 0.0};
      PetscInt  offset;
      PetscBool isInterior = PETSC_TRUE;

      PetscCall(PetscSectionGetOffset(coordSection, v, &offset));
      for (PetscInt d = 0; d < dim; d++) coords[d] = PetscRealPart(coordArray[offset + d]);

      /* Check if vertex is on the boundary */
      for (PetscInt d = 0; d < dim; d++) {
        PetscReal gridSpacing = upper[d] / faces[d];
        PetscInt  gridIdx     = (PetscInt)(coords[d] / gridSpacing + 0.5);
        if (gridIdx == 0 || gridIdx == faces[d]) {
          isInterior = PETSC_FALSE;
          break;
        }
      }

      if (isInterior) {
        for (PetscInt d = 0; d < dim; d++) {
          PetscReal noise, gridSpacing = upper[d] / faces[d];

          seed  = (1103515245 * seed + 12345) % 2147483648;
          noise = (PetscReal)seed / 2147483648.0;
          coordArray[offset + d] += (noise - 0.5) * 0.001 * gridSpacing;
        }
      }
    }
    PetscCall(VecRestoreArray(coordinates, &coordArray));
  }

  /* Call the LETKF localization function */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nComputing LETKF localization matrix...\n"));
  PetscCall(DMPlexGetLETKFLocalizationMatrix(dm, numObservations, numGlobalObs, H, &Q));
  PetscCall(PetscObjectSetName((PetscObject)Q, "Q_localization"));

  /* Check the output matrix */
  PetscCall(MatGetSize(Q, &M, &N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nLocalization matrix Q: %" PetscInt_FMT " x %" PetscInt_FMT "\n", M, N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Q matrix:\n"));
  PetscCall(MatView(Q, PETSC_VIEWER_STDOUT_WORLD));

  /* View the matrix if requested */
  PetscCall(MatViewFromOptions(Q, NULL, "-q_view"));

  /* Check sparsity pattern */
  {
    MatInfo info;
    PetscCall(MatGetInfo(Q, MAT_GLOBAL_SUM, &info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Q matrix nonzeros: %g\n", info.nz_used));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Q matrix allocated nonzeros: %g\n", info.nz_allocated));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Average nonzeros per row: %g\n", info.nz_used / M));
  }

  /* Verify some properties */
  {
    PetscReal norm;
    PetscCall(MatNorm(Q, NORM_FROBENIUS, &norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Q Frobenius norm: %g\n", (double)norm));
  }

  /* Cleanup */
  PetscCall(MatDestroy(&Q));
  PetscCall(MatDestroy(&H));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: kokkos !complex !single
    suffix: 1
    diff_args: -j
    args: -dm_plex_dim 1

  test:
    requires: kokkos !complex !single
    suffix: 2
    diff_args: -j
    args: -dm_plex_dim 2

  test:
    requires: kokkos !complex
    suffix: single
    diff_args: -j
    args: -dm_plex_dim 2

TEST*/
