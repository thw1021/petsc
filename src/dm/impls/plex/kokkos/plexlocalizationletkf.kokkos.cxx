#include <petscdmplex.h>
#include <petscmat.h>
#include <petsc/private/dmpleximpl.h>
#include <algorithm>
#include <vector>
#include <cmath>

/* Gaspari-Cohn 5th-order piecewise rational function for localization
   Input: distance d, cutoff radius R
   Output: correlation weight (0 to 1)
*/
static PetscReal GaspariCohn(PetscReal d, PetscReal R)
{
  PetscFunctionBegin;
  if (R <= 0.0) PetscFunctionReturn(0.0);

  const PetscReal r = d / R; // Normalized distance

  if (r >= 2.0) PetscFunctionReturn(0.0);

  const PetscReal r2 = r * r;
  const PetscReal r3 = r2 * r;
  const PetscReal r4 = r3 * r;
  const PetscReal r5 = r4 * r;

  if (r <= 1.0) {
    PetscFunctionReturn(1.0 - (5.0 / 3.0) * r2 + (5.0 / 8.0) * r3 + 0.5 * r4 - 0.25 * r5);
  } else {
    PetscFunctionReturn(4.0 - 5.0 * r + (5.0 / 3.0) * r2 + (5.0 / 8.0) * r3 - 0.5 * r4 - (2.0 / 3.0) / r - 2.0);
  }
}

/* Structure to hold distance and observation index pairs */
typedef struct {
  PetscReal distance;
  PetscInt  index;
} DistanceIndexPair;

/* Comparison function for sorting */
static bool CompareDistanceIndexPair(const DistanceIndexPair &a, const DistanceIndexPair &b)
{
  return a.distance < b.distance;
}

/*@
  DMPlexGetLETKFLocalizationMatrix - Compute the LETKF localization matrix Q

  Input Parameters:
+ plex            - The DMPlex object
. numobservations - Number of closest observations to use per vertex
. numglobslobs    - Total number of global observations
- H               - Observation operator matrix (scalar matrix)

  Output Parameter:
. Q - The localization matrix (MPIAIJ format)

  Level: advanced

.seealso: `DMPlexComputeCellGeometryFVM()`, `DMGetCoordinatesLocal()`
@*/
PetscErrorCode DMPlexGetLETKFLocalizationMatrix(DM plex, PetscInt numobservations, PetscInt numglobslobs, Mat H, Mat *Q)
{
  MPI_Comm                       comm;
  Vec                            coordinates;
  PetscSection                   coordSection;
  const PetscScalar             *coordArray;
  PetscInt                       dim, vStart, vEnd, numVertices;
  PetscInt                       offset;
  Vec                           *V_comp = NULL, *O_comp = NULL;
  PetscScalar                  **global_O_comp = NULL;
  Mat                            Qmat;
  PetscInt                       localRows, globalRows;
  std::vector<DistanceIndexPair> distances;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(plex, DM_CLASSID, 1);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  PetscAssertPointer(Q, 5);

  PetscCall(PetscObjectGetComm((PetscObject)plex, &comm));

  /* Check that numobservations is valid */
  PetscCheck(numobservations > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "numobservations must be positive, got %" PetscInt_FMT, numobservations);
  PetscCheck(numobservations <= numglobslobs, comm, PETSC_ERR_ARG_OUTOFRANGE, "numobservations (%" PetscInt_FMT ") must be <= numglobslobs (%" PetscInt_FMT ")", numobservations, numglobslobs);

  /* Get spatial dimension */
  PetscCall(DMGetCoordinateDim(plex, &dim));
  PetscCheck(dim >= 1 && dim <= 3, comm, PETSC_ERR_ARG_OUTOFRANGE, "Coordinate dimension must be 1, 2, or 3, got %" PetscInt_FMT, dim);

  /* Get vertex range */
  PetscCall(DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd));
  numVertices = vEnd - vStart;

  /* Get coordinates */
  PetscCall(DMGetCoordinatesLocal(plex, &coordinates));
  PetscCall(DMGetCoordinateSection(plex, &coordSection));
  PetscCall(VecGetArrayRead(coordinates, &coordArray));

  /* Allocate arrays for component vectors */
  PetscCall(PetscMalloc2(dim, &V_comp, dim, &O_comp));
  PetscCall(PetscMalloc1(dim, &global_O_comp));

  /* Create component vectors and compute observation locations */
  for (PetscInt d = 0; d < dim; d++) {
    PetscScalar *v_array, *o_array;
    VecScatter   scatter;
    Vec          O_comp_global;
    PetscInt     n;

    /* Create vector for this coordinate component */
    PetscCall(VecCreate(comm, &V_comp[d]));
    PetscCall(VecSetSizes(V_comp[d], numVertices, PETSC_DETERMINE));
    PetscCall(VecSetFromOptions(V_comp[d]));

    /* Extract component d from interlaced coordinates */
    PetscCall(VecGetArray(V_comp[d], &v_array));
    for (PetscInt v = vStart; v < vEnd; v++) {
      PetscCall(PetscSectionGetOffset(coordSection, v, &offset));
      v_array[v - vStart] = coordArray[offset + d];
    }
    PetscCall(VecRestoreArray(V_comp[d], &v_array));

    /* Compute observation locations: O_comp = H * V_comp */
    PetscCall(MatCreateVecs(H, NULL, &O_comp[d]));
    PetscCall(MatMult(H, V_comp[d], O_comp[d]));

    /* Gather observation locations to all processors */
    PetscCall(VecScatterCreateToAll(O_comp[d], &scatter, &O_comp_global));
    PetscCall(VecScatterBegin(scatter, O_comp[d], O_comp_global, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, O_comp[d], O_comp_global, INSERT_VALUES, SCATTER_FORWARD));

    /* Get global observation array */
    PetscCall(VecGetSize(O_comp_global, &n));
    PetscCheck(n == numglobslobs, comm, PETSC_ERR_ARG_INCOMP, "Observation vector size %" PetscInt_FMT " does not match numglobslobs %" PetscInt_FMT, n, numglobslobs);

    PetscCall(VecGetArray(O_comp_global, &o_array));
    PetscCall(PetscMalloc1(numglobslobs, &global_O_comp[d]));
    for (PetscInt obs = 0; obs < numglobslobs; obs++) global_O_comp[d][obs] = o_array[obs];
    PetscCall(VecRestoreArray(O_comp_global, &o_array));

    PetscCall(VecDestroy(&O_comp_global));
    PetscCall(VecScatterDestroy(&scatter));
  }

  /* Create localization matrix Q */
  localRows = numVertices;
  PetscCallMPI(MPIU_Allreduce(&localRows, &globalRows, 1, MPIU_INT, MPI_SUM, comm));

  PetscCall(MatCreate(comm, &Qmat));
  PetscCall(MatSetSizes(Qmat, localRows, PETSC_DECIDE, globalRows, numglobslobs));
  PetscCall(MatSetType(Qmat, MATMPIAIJ));
  PetscCall(MatMPIAIJSetPreallocation(Qmat, numobservations, NULL, numobservations, NULL));
  PetscCall(MatSetUp(Qmat));

  /* Compute localization weights for each vertex */
  distances.resize(numglobslobs);
  std::vector<PetscInt>    col_indices(numobservations);
  std::vector<PetscScalar> values(numobservations);

  for (PetscInt v = vStart; v < vEnd; v++) {
    PetscReal      vertex_coords[3] = {0.0, 0.0, 0.0};
    PetscReal      cutoff;
    const PetscInt globalRow = v - vStart; // Convert to 0-based row index

    /* Get vertex coordinates */
    PetscCall(PetscSectionGetOffset(coordSection, v, &offset));
    for (PetscInt d = 0; d < dim; d++) vertex_coords[d] = PetscRealPart(coordArray[offset + d]);

    /* Compute distances to all observations */
    for (PetscInt obs = 0; obs < numglobslobs; obs++) {
      PetscReal dist_sq = 0.0;
      for (PetscInt d = 0; d < dim; d++) {
        const PetscReal diff = vertex_coords[d] - PetscRealPart(global_O_comp[d][obs]);
        dist_sq += diff * diff;
      }
      distances[obs].distance = PetscSqrtReal(dist_sq);
      distances[obs].index    = obs;
    }

    /* Partially sort to find k nearest neighbors using nth_element */
    /* This is O(N) instead of O(N log N) since we only need the k smallest (not deterministic, use general‑position) */
    //#if defined(PETSC_USE_DEBUG)
    /* In DEBUG mode, use full sort for verification */
    std::sort(distances.begin(), distances.end(), CompareDistanceIndexPair);
    //else
    //std::nth_element(distances.begin(), distances.begin() + numobservations - 1, distances.end(), CompareDistanceIndexPair);
    /* Sort only the k nearest neighbors for consistent ordering */
    //std::sort(distances.begin(), distances.begin() + numobservations, CompareDistanceIndexPair);
    //#endif

    /* Get cutoff radius from k-th nearest observation */
    cutoff = distances[numobservations - 1].distance;
    if (cutoff == 0.0) cutoff = 1.0; // Handle edge case where all k neighbors are co-located

    /* Compute weights and insert into matrix */
    for (PetscInt i = 0; i < numobservations; i++) {
      const PetscReal weight = GaspariCohn(distances[i].distance, cutoff);
      col_indices[i]         = distances[i].index;
      values[i]              = weight;
    }

    /* Insert all values for this row at once */
    PetscCall(MatSetValues(Qmat, 1, &globalRow, numobservations, col_indices.data(), values.data(), INSERT_VALUES));
  }

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(Qmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Qmat, MAT_FINAL_ASSEMBLY));

  /* Cleanup */
  PetscCall(VecRestoreArrayRead(coordinates, &coordArray));
  for (PetscInt d = 0; d < dim; d++) {
    PetscCall(VecDestroy(&V_comp[d]));
    PetscCall(VecDestroy(&O_comp[d]));
    PetscCall(PetscFree(global_O_comp[d]));
  }
  PetscCall(PetscFree2(V_comp, O_comp));
  PetscCall(PetscFree(global_O_comp));

  *Q = Qmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}
