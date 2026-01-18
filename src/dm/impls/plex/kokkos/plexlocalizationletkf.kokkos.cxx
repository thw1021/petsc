#include <petsc/private/dmpleximpl.h>
#include <petscdmplex.h>
#include <petscmat.h>
#include <petsc_kokkos.hpp>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <Kokkos_Core.hpp>

typedef struct {
  PetscReal distance;
  PetscInt  obs_index;
} DistObsPair;

KOKKOS_INLINE_FUNCTION
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
  DMPlexGetLETKFLocalizationMatrix - Compute localization weight matrix for LETKF [move to ml/da/interface]

  Collective

  Input Parameters:
+ n_obs_vertex - Number of nearest observations to use per vertex (eg, MAX_Q_NUM_LOCAL_OBSERVATIONS in LETKF)
. n_obs_local - Number of local observations
. n_dof - Number of degrees of freedom
. Vecxyz - Array of vectors containing the coordinates
- H - Observation operator matrix

  Output Parameter:
. Q - Localization weight matrix (sparse, AIJ format)

  Notes:
  The output matrix Q has dimensions (n_vert_global x n_obs_global) where
  n_vert_global is the number of vertices in the DMPlex. Each row contains
  exactly n_obs_vertex non-zero entries corresponding to the nearest
  observations, weighted by the Gaspari-Cohn fifth-order piecewise
  rational function.

  The observation locations are computed as H * V where V is the vector
  of vertex coordinates. The localization weights ensure smooth tapering
  of observation influence with distance.

  Kokkos is required for this routine.

  Level: intermediate

.seealso: `DMPLEX`, `DMPlexGetDepthStratum()`, `DMGetCoordinatesLocal()`
@*/
PetscErrorCode DMPlexGetLETKFLocalizationMatrix(PetscInt n_obs_vertex, PetscInt n_obs_local, PetscInt n_dof, Vec Vecxyz[3], Mat H, Mat *Q)
{
  PetscInt dim = 0, n_vert_local, d, N, n_obs_global, n_state_local;
  Vec     *obs_vecs;
  MPI_Comm comm;
  PetscInt n_state_global;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(H, MAT_CLASSID, 5);
  PetscAssertPointer(Q, 6);

  PetscCall(PetscKokkosInitializeCheck());

  PetscCall(PetscObjectGetComm((PetscObject)H, &comm));

  /* Infer dim from the number of vectors in Vecxyz */
  for (d = 0; d < 3; ++d) {
    if (Vecxyz[d]) dim++;
    else break;
  }

  PetscCheck(dim > 0, comm, PETSC_ERR_ARG_WRONG, "Dim must be > 0");

  PetscCall(VecGetSize(Vecxyz[0], &n_state_global));
  PetscCall(VecGetLocalSize(Vecxyz[0], &n_state_local));
  n_vert_local = n_state_local / n_dof;

  /* Check H dimensions */
  PetscCall(MatGetSize(H, &n_obs_global, &N));
  PetscCheck(N == n_state_global, comm, PETSC_ERR_ARG_SIZ, "H number of columns %" PetscInt_FMT " != global state size %" PetscInt_FMT, N, n_state_global);

  /* Allocate storage for observation locations */
  PetscCall(PetscMalloc1(dim, &obs_vecs));

  /* Compute observation locations per dimension */
  for (d = 0; d < dim; ++d) {
    PetscCall(MatCreateVecs(H, NULL, &obs_vecs[d]));
    PetscCall(MatMult(H, Vecxyz[d], obs_vecs[d]));
  }

  /* Create output matrix Q in N/n_dof x P */
  PetscCall(MatCreate(comm, Q));
  PetscCall(MatSetSizes(*Q, n_vert_local, n_obs_local, PETSC_DETERMINE, n_obs_global));
  PetscCall(MatSetType(*Q, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(*Q, n_obs_vertex, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*Q, n_obs_vertex, NULL, n_obs_vertex, NULL));
  PetscCall(MatSetUp(*Q));

  /* Prepare Kokkos Views */
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = ExecSpace::memory_space;

  /* Vertex Coordinates */
  Kokkos::View<PetscScalar **, Kokkos::LayoutRight, MemSpace> vertex_coords_dev("vertex_coords", n_vert_local, dim);
  {
    PetscScalar *raw_coords;

    PetscCall(PetscMalloc1(n_vert_local * dim, &raw_coords));
    for (d = 0; d < dim; ++d) {
      PetscScalar *local_coords_array;
      PetscCall(VecGetArray(Vecxyz[d], &local_coords_array));
      PetscCall(PetscMemcpy(&raw_coords[d * n_vert_local], local_coords_array, n_vert_local * sizeof(PetscScalar)));
      PetscCall(VecRestoreArray(Vecxyz[d], &local_coords_array));
    }
    Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vertex_coords_host(raw_coords, n_vert_local, dim);
    Kokkos::deep_copy(vertex_coords_dev, vertex_coords_host);
    PetscCall(PetscFree(raw_coords));
  }

  /* Observation Coordinates */
  Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace> obs_coords_dev("obs_coords", n_obs_global, dim);
  {
    PetscReal *raw_obs_coords;
    PetscCall(PetscMalloc1(n_obs_global * dim, &raw_obs_coords));

    for (d = 0; d < dim; ++d) {
      VecScatter         ctx;
      Vec                seq_vec;
      const PetscScalar *array;

      PetscCall(VecScatterCreateToAll(obs_vecs[d], &ctx, &seq_vec));
      PetscCall(VecScatterBegin(ctx, obs_vecs[d], seq_vec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(ctx, obs_vecs[d], seq_vec, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecGetArrayRead(seq_vec, &array));
      for (PetscInt j = 0; j < n_obs_global; ++j) raw_obs_coords[j * dim + d] = PetscRealPart(array[j]);
      PetscCall(VecRestoreArrayRead(seq_vec, &array));
      PetscCall(VecScatterDestroy(&ctx));
      PetscCall(VecDestroy(&seq_vec));
    }

    Kokkos::View<PetscReal **, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> obs_coords_host(raw_obs_coords, n_obs_global, dim);
    Kokkos::deep_copy(obs_coords_dev, obs_coords_host);
    PetscCall(PetscFree(raw_obs_coords));
  }

  PetscInt rstart;
  PetscCall(VecGetOwnershipRange(Vecxyz[0], &rstart, NULL));

  /* Output Views */
  Kokkos::View<PetscInt **, Kokkos::LayoutRight, MemSpace>    indices_dev("indices", n_vert_local, n_obs_vertex);
  Kokkos::View<PetscScalar **, Kokkos::LayoutRight, MemSpace> values_dev("values", n_vert_local, n_obs_vertex);

  /* Temporary storage for top-k per vertex */
  Kokkos::View<PetscReal **, Kokkos::LayoutRight, MemSpace> best_dists_dev("best_dists", n_vert_local, n_obs_vertex);
  Kokkos::View<PetscInt **, Kokkos::LayoutRight, MemSpace>  best_idxs_dev("best_idxs", n_vert_local, n_obs_vertex);

  Kokkos::deep_copy(best_dists_dev, 1.0e30);

  /* Main Kernel */
  Kokkos::parallel_for(
    "ComputeLocalization", Kokkos::RangePolicy<ExecSpace>(0, n_vert_local), KOKKOS_LAMBDA(const PetscInt i) {
      PetscReal current_max_dist = 1.0e30;
      PetscInt  count            = 0;

      // Iterate over all observations
      for (PetscInt j = 0; j < n_obs_global; ++j) {
        PetscReal dist2 = 0.0;
        for (PetscInt d = 0; d < dim; ++d) {
          PetscReal diff = PetscRealPart(vertex_coords_dev(i, d)) - obs_coords_dev(j, d);
          dist2 += diff * diff;
        }

        if (count < n_obs_vertex) {
          // Insert sorted
          PetscInt pos = count;
          while (pos > 0 && best_dists_dev(i, pos - 1) > dist2) {
            best_dists_dev(i, pos) = best_dists_dev(i, pos - 1);
            best_idxs_dev(i, pos)  = best_idxs_dev(i, pos - 1);
            pos--;
          }
          best_dists_dev(i, pos) = dist2;
          best_idxs_dev(i, pos)  = j;
          count++;
          if (count == n_obs_vertex) current_max_dist = best_dists_dev(i, n_obs_vertex - 1);
        } else if (dist2 < current_max_dist) {
          // Insert sorted
          PetscInt pos = n_obs_vertex - 1;
          while (pos > 0 && best_dists_dev(i, pos - 1) > dist2) {
            best_dists_dev(i, pos) = best_dists_dev(i, pos - 1);
            best_idxs_dev(i, pos)  = best_idxs_dev(i, pos - 1);
            pos--;
          }
          best_dists_dev(i, pos) = dist2;
          best_idxs_dev(i, pos)  = j;
          current_max_dist       = best_dists_dev(i, n_obs_vertex - 1);
        }
      }

      // Compute weights
      PetscReal radius2 = best_dists_dev(i, n_obs_vertex - 1);
      PetscReal radius  = std::sqrt(radius2);
      if (radius == 0.0) radius = 1.0;

      for (PetscInt k = 0; k < n_obs_vertex; ++k) {
        PetscReal dist    = std::sqrt(best_dists_dev(i, k));
        indices_dev(i, k) = best_idxs_dev(i, k);
        values_dev(i, k)  = GaspariCohn(dist, radius);
      }
    });

  /* Copy back to host and fill matrix */
  Kokkos::View<PetscInt **, Kokkos::LayoutRight, Kokkos::HostSpace>    indices_host = Kokkos::create_mirror_view(indices_dev);
  Kokkos::View<PetscScalar **, Kokkos::LayoutRight, Kokkos::HostSpace> values_host  = Kokkos::create_mirror_view(values_dev);
  Kokkos::deep_copy(indices_host, indices_dev);
  Kokkos::deep_copy(values_host, values_dev);

  for (PetscInt i = 0; i < n_vert_local; ++i) {
    PetscInt globalRow = rstart + i;
    PetscCall(MatSetValues(*Q, 1, &globalRow, n_obs_vertex, &indices_host(i, 0), &values_host(i, 0), INSERT_VALUES));
  }

  /* Cleanup Phase 2 storage */
  for (d = 0; d < dim; ++d) PetscCall(VecDestroy(&obs_vecs[d]));
  PetscCall(PetscFree(obs_vecs));

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}
