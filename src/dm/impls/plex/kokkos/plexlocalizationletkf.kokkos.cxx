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
  PetscReal r, r2, r3, r4, r5, weight;

  PetscFunctionBegin;
  if (R <= 0.0) PetscFunctionReturn(0.0);

  r = 2.0 * d / R; // Normalized distance

  if (r >= 2.0) {
    weight = 0.0;
  } else if (r <= 1.0) {
    r2     = r * r;
    r3     = r2 * r;
    r4     = r3 * r;
    r5     = r4 * r;
    weight = 1.0 - (5.0 / 3.0) * r2 + (5.0 / 8.0) * r3 + 0.5 * r4 - 0.25 * r5;
  } else {
    r2     = r * r;
    r3     = r2 * r;
    r4     = r3 * r;
    r5     = r4 * r;
    weight = 4.0 - 5.0 * r + (5.0 / 3.0) * r2 + (5.0 / 8.0) * r3 - 0.5 * r4 - (2.0 / 3.0) / r - 2.0;
  }

  PetscFunctionReturn(weight);
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
+ plex - The DMPlex object
. numobservations - Number of closest observations to use per vertex
. numglobslobs - Total number of global observations
- H - Observation operator matrix (scalar matrix)

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
  PetscInt                       dim, vStart, vEnd, numVertices, v;
  PetscInt                       offset;
  Vec                           *V_comp = NULL, *O_comp = NULL;
  PetscScalar                  **global_O_comp = NULL;
  PetscInt                       d, obs;
  Mat                            Qmat;
  PetscInt                       localRows, globalRows;
  std::vector<DistanceIndexPair> distances;
  PetscReal                      epsilon = 1e-6;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(plex, DM_CLASSID, 1);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  PetscAssertPointer(Q, 5);

  PetscCall(PetscObjectGetComm((PetscObject)plex, &comm));

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
  for (d = 0; d < dim; d++) {
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
    for (v = vStart; v < vEnd; v++) {
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
    for (obs = 0; obs < numglobslobs; obs++) global_O_comp[d][obs] = o_array[obs];
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

  for (v = vStart; v < vEnd; v++) {
    PetscReal vertex_coords[3] = {0.0, 0.0, 0.0};
    PetscReal dmax, cutoff;
    PetscInt  globalRow = v - vStart; // Convert to 0-based row index

    /* Get vertex coordinates */
    PetscCall(PetscSectionGetOffset(coordSection, v, &offset));
    for (d = 0; d < dim; d++) vertex_coords[d] = PetscRealPart(coordArray[offset + d]);

    /* Compute distances to all observations */
    for (obs = 0; obs < numglobslobs; obs++) {
      PetscReal dist_sq = 0.0;
      for (d = 0; d < dim; d++) {
        PetscReal diff = vertex_coords[d] - PetscRealPart(global_O_comp[d][obs]);
        dist_sq += diff * diff;
      }
      distances[obs].distance = PetscSqrtReal(dist_sq);
      distances[obs].index    = obs;
    }

    /* Sort by distance */
    std::sort(distances.begin(), distances.end(), CompareDistanceIndexPair);

    /* Get cutoff radius from k-th nearest observation */
    PetscInt k = PetscMin(numobservations, numglobslobs);
    dmax       = distances[k - 1].distance;
    cutoff     = dmax * (1.0 + epsilon);

    /* Compute weights and insert into matrix */
    for (PetscInt i = 0; i < k; i++) {
      PetscReal weight = GaspariCohn(distances[i].distance, cutoff);
      PetscInt  col    = distances[i].index;

      if (weight > 0.0) PetscCall(MatSetValue(Qmat, globalRow, col, weight, INSERT_VALUES));
    }
  }

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(Qmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Qmat, MAT_FINAL_ASSEMBLY));

  /* Cleanup */
  PetscCall(VecRestoreArrayRead(coordinates, &coordArray));
  for (d = 0; d < dim; d++) {
    PetscCall(VecDestroy(&V_comp[d]));
    PetscCall(VecDestroy(&O_comp[d]));
    PetscCall(PetscFree(global_O_comp[d]));
  }
  PetscCall(PetscFree2(V_comp, O_comp));
  PetscCall(PetscFree(global_O_comp));

  *Q = Qmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}
