static char help[] = "Test DMPlexGetLETKFLocalizationMatrix.\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;
  Mat            H, Q;
  PetscInt       numobservations;
  PetscInt       dim      = 1;
  PetscInt       faces[3] = {10, 4, 4};
  PetscReal      lower[3] = {0.0, 0.0, 0.0};
  PetscReal      upper[3] = {1.0, 1.0, 1.0};
  DMBoundaryType bdt[3]   = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Get dimension from options */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dm_plex_dim", &dim, NULL));

  /* Set faces based on dimension */
  if (dim == 1) {
    faces[0] = 10;
  } else if (dim == 2) {
    faces[0] = 4;
    faces[1] = 4;
  } else if (dim == 3) {
    faces[0] = 4;
    faces[1] = 4;
    faces[2] = 4;
  }

  /* Create the mesh using DMPlexCreateBoxMesh like ex21 */
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, faces, lower, upper, bdt, PETSC_TRUE, 0, PETSC_TRUE, &dm));
  PetscCall(DMSetFromOptions(dm));

  /* Verify dimension matches */
  PetscInt dmDim;
  PetscCall(DMGetDimension(dm, &dmDim));
  PetscCheck(dmDim == dim, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "DM dimension %" PetscInt_FMT " does not match requested dimension %" PetscInt_FMT, dmDim, dim);

  /* Set number of local observations to use: 3^dim */
  numobservations = 1;
  for (PetscInt d = 0; d < dim; d++) numobservations *= 3;

  /* Get number of vertices */
  PetscInt vStart, vEnd, numVertices;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  numVertices = vEnd - vStart;

  /* Create a section for vertices (required for Global Point mapping) */
  PetscSection section;
  PetscInt     pStart, pEnd;
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &section));
  PetscCall(PetscSectionSetNumFields(section, 1));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) { PetscCall(PetscSectionSetDof(section, v, 1)); }
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));

  /* Create global section */
  PetscSection globalSection;
  PetscCall(DMGetGlobalSection(dm, &globalSection));

  /* Count observations (every other vertex in each dimension) */
  PetscInt numlocalobs = 0;
  {
    Vec                coordinates;
    PetscSection       coordSection;
    const PetscScalar *coordArray;
    PetscInt           offset;

    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(VecGetArrayRead(coordinates, &coordArray));

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
      if (isObs) numlocalobs++;
    }
    PetscCall(VecRestoreArrayRead(coordinates, &coordArray));
  }

  /* Create H matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &H));
  PetscCall(MatSetSizes(H, numlocalobs, PETSC_DECIDE, PETSC_DECIDE, numVertices));
  PetscCall(MatSetType(H, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(H, 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(H, 1, NULL, 0, NULL));

  /* Fill H matrix */
  {
    Vec                coordinates;
    PetscSection       coordSection;
    const PetscScalar *coordArray;
    PetscInt           obsIdx = 0;
    PetscInt           offset;

    PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
    PetscCall(DMGetCoordinateSection(dm, &coordSection));
    PetscCall(VecGetArrayRead(coordinates, &coordArray));

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

      if (isObs) {
        PetscCall(MatSetValue(H, obsIdx, v - vStart, 1.0, INSERT_VALUES));
        obsIdx++;
      }
    }
    PetscCall(VecRestoreArrayRead(coordinates, &coordArray));
  }
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));

  /* View H */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Observation Operator H:\n"));
  PetscCall(MatView(H, PETSC_VIEWER_STDOUT_WORLD));

  /* Call the function */
  PetscCall(DMPlexGetLETKFLocalizationMatrix(dm, numobservations, numlocalobs, H, &Q));

  /* View Q */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization Matrix Q:\n"));
  PetscCall(MatView(Q, PETSC_VIEWER_STDOUT_WORLD));

  /* Cleanup */
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Q));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: kokkos !complex
    suffix: 1
    diff_args: -j
    args: -dm_plex_dim 1
    
  test:
    requires: kokkos !complex
    suffix: 2
    diff_args: -j
    args: -dm_plex_dim 2

TEST*/
