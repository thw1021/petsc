#include <petsc/private/dmpleximpl.h>
#include <petscdmplex.h>
#include <petscmat.h>
#include <cmath>
#include <cstdlib>

typedef struct {
  PetscReal distance;
  PetscInt  obs_index;
} DistObsPair;

static int CompareDistObsPair(const void *a, const void *b)
{
  DistObsPair *pair_a = (DistObsPair *)a;
  DistObsPair *pair_b = (DistObsPair *)b;
  if (pair_a->distance < pair_b->distance) return -1;
  if (pair_a->distance > pair_b->distance) return 1;
  return 0;
}

static PetscReal GaspariCohn(PetscReal distance, PetscReal radius)
{
  if (radius <= 0.0) return 0.0;
  PetscReal r      = distance / radius;
  PetscReal weight = 0.0;

  if (r >= 2.0) {
    weight = 0.0;
  } else if (r >= 1.0) {
    // Region [1, 2]
    PetscReal r2 = r * r;
    PetscReal r3 = r2 * r;
    PetscReal r4 = r3 * r;
    PetscReal r5 = r4 * r;
    weight       = (1.0 / 12.0) * r5 - (0.5) * r4 + (0.625) * r3 + (5.0 / 3.0) * r2 - 5.0 * r + 4.0 - (2.0 / 3.0) / r;
  } else {
    // Region [0, 1]
    PetscReal r2 = r * r;
    PetscReal r3 = r2 * r;
    PetscReal r4 = r3 * r;
    PetscReal r5 = r4 * r;
    weight       = -0.25 * r5 + 0.5 * r4 + 0.625 * r3 - (5.0 / 3.0) * r2 + 1.0;
  }
  return weight;
}

/*@
  DMPlexGetLETKFLocalizationMatrix - Compute localization weight matrix for LETKF

  Collective

  Input Parameters:
+ plex - The DMPlex object
. numobservations - Number of nearest observations to use per vertex
. numglobalobs - Total number of observations
- H - Observation operator matrix

  Output Parameter:
. Q - Localization weight matrix (sparse, AIJ format)

  Notes:
  The output matrix Q has dimensions (numVertices x numglobalobs) where
  numVertices is the number of vertices in the DMPlex. Each row contains
  exactly numobservations non-zero entries corresponding to the nearest
  observations, weighted by the Gaspari-Cohn fifth-order piecewise 
  rational function.

  The observation locations are computed as H * V where V is the vector
  of vertex coordinates. The localization weights ensure smooth tapering
  of observation influence with distance.

  Level: intermediate

.seealso: `DMPLEX`, `DMPlexGetDepthStratum()`, `DMGetCoordinatesLocal()`
@*/
PetscErrorCode DMPlexGetLETKFLocalizationMatrix(DM plex, PetscInt numobservations, PetscInt numglobalobs, Mat H, Mat *Q)
{
  PetscInt      dim, vStart, vEnd, numVertices, v, d, k;
  PetscInt      M, N;
  Vec           coordinates;
  Vec          *obs_vecs;
  PetscScalar **obs_coords;
  PetscScalar  *vertex_coords;
  DistObsPair  *dist_obs_pairs;
  PetscInt     *col_indices;
  PetscScalar  *values;
  PetscInt      localRows, globalRows;
  MPI_Comm      comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(plex, DM_CLASSID, 1);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  PetscAssertPointer(Q, 5);

  PetscCall(PetscObjectGetComm((PetscObject)plex, &comm));
  PetscCall(DMGetCoordinateDim(plex, &dim));
  PetscCall(DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd));
  numVertices = vEnd - vStart;

  /* Check H dimensions */
  PetscCall(MatGetSize(H, &M, &N));
  PetscCheck(M == numglobalobs, comm, PETSC_ERR_ARG_SIZ, "H matrix rows %" PetscInt_FMT " != numglobalobs %" PetscInt_FMT, M, numglobalobs);

  PetscCall(DMGetCoordinates(plex, &coordinates));
  PetscCheck(coordinates, comm, PETSC_ERR_ARG_WRONGSTATE, "DM must have coordinates");

  /* Allocate storage for observation locations */
  PetscCall(PetscMalloc1(dim, &obs_vecs));
  PetscCall(PetscMalloc1(dim, &obs_coords));

  /* Compute observation locations per dimension */
  for (d = 0; d < dim; ++d) {
    Vec coord_comp;
    PetscCall(MatCreateVecs(H, &coord_comp, &obs_vecs[d]));
    PetscCall(VecStrideGather(coordinates, d, coord_comp, INSERT_VALUES));
    PetscCall(MatMult(H, coord_comp, obs_vecs[d]));
    PetscCall(VecGetArray(obs_vecs[d], &obs_coords[d]));
    PetscCall(VecDestroy(&coord_comp));
  }

  /* Create output matrix Q */
  localRows = numVertices;
  PetscCallMPI(MPIU_Allreduce(&localRows, &globalRows, 1, MPIU_INT, MPI_SUM, comm));

  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, localRows, PETSC_DECIDE, globalRows, numglobalobs));
  PetscCall(MatSetType(*Q, MATMPIAIJ));
  PetscCall(MatMPIAIJSetPreallocation(*Q, numobservations, NULL, numobservations, NULL));
  PetscCall(MatSetUp(*Q));

  /* Temporary storage for loop */
  PetscCall(PetscMalloc1(dim, &vertex_coords));
  PetscCall(PetscMalloc1(numglobalobs, &dist_obs_pairs));
  PetscCall(PetscMalloc1(numobservations, &col_indices));
  PetscCall(PetscMalloc1(numobservations, &values));

  /* Get local coordinates array for vertex access */
  Vec          localCoords;
  PetscScalar *local_coords_array;
  PetscSection coordSection;
  PetscCall(DMGetCoordinatesLocal(plex, &localCoords));
  PetscCall(DMGetCoordinateSection(plex, &coordSection));
  PetscCall(VecGetArray(localCoords, &local_coords_array));

  /* Main loop over vertices */
  PetscSection globalSection;
  PetscCall(DMGetGlobalSection(plex, &globalSection));

  for (v = vStart; v < vEnd; ++v) {
    PetscInt globalRow;

    /* Get global DOF for this vertex */
    PetscCall(PetscSectionGetOffset(globalSection, v, &globalRow));

    /* Get vertex coordinates */
    PetscInt off;
    PetscCall(PetscSectionGetOffset(coordSection, v, &off));
    for (d = 0; d < dim; ++d) { vertex_coords[d] = local_coords_array[off + d]; }

    /* Compute distances to all observations */
    for (PetscInt j = 0; j < numglobalobs; ++j) {
      PetscReal dist = 0.0;
      for (d = 0; d < dim; ++d) {
        PetscReal diff = PetscRealPart(vertex_coords[d] - obs_coords[d][j]);
        dist += diff * diff;
      }
      dist                        = std::sqrt(dist);
      dist_obs_pairs[j].distance  = dist;
      dist_obs_pairs[j].obs_index = j;
    }

    /* Sort by distance */
    qsort(dist_obs_pairs, numglobalobs, sizeof(DistObsPair), CompareDistObsPair);

    /* Determine radius from k-th nearest neighbor */
    /* Use distance to the k-th neighbor as the cutoff (r=1.0) */
    /* So radius = dist_k */
    PetscReal radius = dist_obs_pairs[numobservations - 1].distance;

    /* If radius is 0 (co-located), we need a fallback or handle it.
       If multiple observations are at 0 distance, radius is 0.
       GaspariCohn returns 0 if radius <= 0.
       But we want non-zero weight for 0 distance.
       If radius is 0, it means the cutoff is 0.
       Let's use a small epsilon if radius is 0?
       Or if distance is 0, weight is 1.
       But we need a radius for the function.
       If dist_k is 0, then all k neighbors are at 0.
       We can set radius to 1.0 (arbitrary) and distance 0 gives weight 1.
       But what about the (k+1)-th neighbor? If it's also 0?
       We only select k.
       So if dist_k == 0, we can set radius = 1.0. */
    if (radius == 0.0) radius = 1.0;

    /* Select nearest observations and compute weights */
    for (k = 0; k < numobservations; ++k) {
      PetscInt  obs_idx  = dist_obs_pairs[k].obs_index;
      PetscReal distance = dist_obs_pairs[k].distance;
      PetscReal weight   = GaspariCohn(distance, radius);

      col_indices[k] = obs_idx;
      values[k]      = weight;
    }

    /* Insert row into matrix Q */
    PetscCall(MatSetValues(*Q, 1, &globalRow, numobservations, col_indices, values, INSERT_VALUES));
  }

  /* Cleanup loop storage */
  PetscCall(VecRestoreArray(localCoords, &local_coords_array));
  PetscCall(PetscFree(vertex_coords));
  PetscCall(PetscFree(dist_obs_pairs));
  PetscCall(PetscFree(col_indices));
  PetscCall(PetscFree(values));

  /* Cleanup Phase 2 storage */
  for (d = 0; d < dim; ++d) {
    PetscCall(VecRestoreArray(obs_vecs[d], &obs_coords[d]));
    PetscCall(VecDestroy(&obs_vecs[d]));
  }
  PetscCall(PetscFree(obs_vecs));
  PetscCall(PetscFree(obs_coords));

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
