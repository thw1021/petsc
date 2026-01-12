#include <petscdmplex.h>
#include <petscmat.h>
#include <petsc/private/dmpleximpl.h>
#if defined(PETSC_HAVE_KOKKOS)
  #include <algorithm>
  #include <vector>
  #include <cmath>
  #include <Kokkos_Core.hpp>

/* Gaspari-Cohn 5th-order piecewise rational function for localization
   Input: distance d, cutoff radius R
   Output: correlation weight (0 to 1)
*/
static PetscReal GaspariCohn(PetscReal d, PetscReal R)
{
  if (R <= 0.0) return 0.0;

  const PetscReal r = d / R; // Normalized distance

  if (r >= 2.0) return 0.0;

  const PetscReal r2 = r * r;
  const PetscReal r3 = r2 * r;
  const PetscReal r4 = r3 * r;
  const PetscReal r5 = r4 * r;

  if (r <= 1.0) {
    return 1.0 - (5.0 / 3.0) * r2 + (5.0 / 8.0) * r3 + 0.5 * r4 - 0.25 * r5;
  } else {
    return (1.0 / 12.0) * r5 - 0.5 * r4 + (5.0 / 8.0) * r3 + (5.0 / 3.0) * r2 - 5.0 * r + 4.0 - (2.0 / 3.0) / r;
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

/*@C
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
  MPI_Comm           comm;
  Vec                coordinates;
  PetscSection       coordSection;
  const PetscScalar *coordArray;
  PetscInt           dim, vStart, vEnd, numVertices;
  PetscInt           offset;
  Vec               *V_comp = NULL, *O_comp = NULL;
  PetscScalar      **global_O_comp = NULL;
  Mat                Qmat;
  PetscInt           localRows, globalRows;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(plex, DM_CLASSID, 1);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  PetscAssertPointer(Q, 5);

  PetscCall(PetscKokkosInitializeCheck());
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

  /* Compute localization weights for each vertex using Kokkos */
  {
    /* Prepare data for Kokkos */
    Kokkos::View<PetscScalar **, Kokkos::HostSpace> global_obs("global_obs", dim, numglobslobs);
    for (PetscInt d = 0; d < dim; d++) {
      for (PetscInt i = 0; i < numglobslobs; i++) global_obs(d, i) = global_O_comp[d][i];
    }

    Kokkos::View<PetscInt *, Kokkos::HostSpace> vertex_offsets("vertex_offsets", numVertices);
    for (PetscInt v = vStart; v < vEnd; v++) {
      PetscInt off;
      PetscCall(PetscSectionGetOffset(coordSection, v, &off));
      vertex_offsets(v - vStart) = off;
    }

    /* Output views */
    Kokkos::View<PetscInt **, Kokkos::HostSpace>    q_cols("q_cols", numVertices, numobservations);
    Kokkos::View<PetscScalar **, Kokkos::HostSpace> q_vals("q_vals", numVertices, numobservations);

    /* Parallel loop over vertices */
    /* We use HostSpace and std::nth_element for efficiency on CPU */
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;

    Kokkos::parallel_for("ComputeLocalization", Kokkos::RangePolicy<ExecSpace>(0, numVertices), [=](const int i) {
      PetscReal vertex_coords[3] = {0.0, 0.0, 0.0};
      PetscInt  off              = vertex_offsets(i);

      /* Get vertex coordinates */
      for (PetscInt d = 0; d < dim; d++) vertex_coords[d] = PetscRealPart(coordArray[off + d]);

      /* Compute distances to all observations */
      /* Use std::vector for scratch memory on host */
      std::vector<DistanceIndexPair> my_distances(numglobslobs);

      for (PetscInt obs = 0; obs < numglobslobs; obs++) {
        PetscReal dist_sq = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
          const PetscReal diff = vertex_coords[d] - PetscRealPart(global_obs(d, obs));
          dist_sq += diff * diff;
        }
        my_distances[obs].distance = PetscSqrtReal(dist_sq);
        my_distances[obs].index    = obs;
      }

      /* Find k nearest neighbors using sort (for compatibility with original behavior) */
      /* Note: std::sort is O(N log N). std::nth_element is O(N). */
      /* We use sort here to attempt to reproduce the reference output which used sort. */
      std::sort(my_distances.begin(), my_distances.end(), CompareDistanceIndexPair);

      /* Get cutoff radius */
      PetscReal cutoff = my_distances[numobservations - 1].distance;
      if (cutoff == 0.0) cutoff = 1.0;

      /* Compute weights */
      for (PetscInt k = 0; k < numobservations; k++) {
        q_cols(i, k) = my_distances[k].index;
        q_vals(i, k) = GaspariCohn(my_distances[k].distance, cutoff);
      }
    });

    /* Insert values into matrix */
    PetscInt rstart, rend;
    PetscCall(MatGetOwnershipRange(Qmat, &rstart, &rend));

    for (PetscInt i = 0; i < numVertices; i++) {
      PetscInt globalRow = rstart + i;
      PetscCall(MatSetValues(Qmat, 1, &globalRow, numobservations, &q_cols(i, 0), &q_vals(i, 0), INSERT_VALUES));
    }
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
#else
/*@C
  DMPlexGetLETKFLocalizationMatrix - Compute localization weight matrix for LETKF

  Collective

  Input Parameters:
+ plex            - The DMPlex object
. numobservations - Number of nearest observations to use per vertex
. numglobalobs    - Total number of observations
- H               - Observation operator matrix

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
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)plex), PETSC_ERR_SUP, "DMPlexGetLETKFLocalizationMatrix() requires Kokkos");
}
#endif
