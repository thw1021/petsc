#include "letkf_impl.h"
#include <petscblaslapack.h>
#include <Kokkos_Core.hpp>

/* ----------------------------------------------------------------------
   Convert a PETSc sequential dense matrix to a Kokkos dense matrix that lives
   on the *device* (DefaultExecutionSpace).  The conversion is performed by:

      1. Getting the matrix size.
      2. Getting a raw host pointer to the PETSc data (column-major).
      3. Wrapping that pointer with a temporary unmanaged host view.
      4. Allocating an owned device view.
      5. deep_copy( device_view , unmanaged_host_view ).

   The routine uses `PetscCall` for error handling and returns the PETSc error
   code (0 on success).  After the copy the PETSc matrix is restored, so the
   function can be called at any point after the matrix has been assembled.
   ---------------------------------------------------------------------- */
PetscErrorCode PetscDenseToKokkosDevice(Mat                                          A, // PETSc matrix (sequential dense)
                                        Kokkos::View<PetscScalar **, Kokkos::LayoutLeft,
                                                     Kokkos::DefaultExecutionSpace> &dA) // device view (output)
{
  PetscInt     m, n;
  PetscScalar *hostPtr = nullptr; // PETSc raw pointer (col-major)

  /* 1) matrix dimensions ------------------------------------------------ */
  PetscCall(MatGetSize(A, &m, &n));

  /* 2) obtain raw host pointer ------------------------------------------ */
  PetscCall(MatDenseGetArray(A, &hostPtr));

  /* 3) temporary unmanaged host view that aliases the PETSc buffer -------- */
  using unmanaged_host_view = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  unmanaged_host_view hA(hostPtr, m, n); // zero-copy, read-only

  /* 4) allocate the *owned* device view --------------------------------- */
  using device_view = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  dA                = device_view("A_device", m, n); // allocated on the device

  /* 5) deep copy host -> device ------------------------------------------ */
  Kokkos::deep_copy(dA, hA);

  /* 6) give PETSc the pointer back -------------------------------------- */
  PetscCall(MatDenseRestoreArray(A, &hostPtr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeNormalizedInnovationMatrix_Kokkos - Computes S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1)

  Input Parameters:
+ Z          - observation ensemble matrix
. y_mean     - mean of observations
. r_inv_sqrt - R^{-1/2}
. m          - ensemble size
- scale      - 1/sqrt(m-1)

  Output Parameter:
. S - normalized innovation matrix
*/
static PetscErrorCode ComputeNormalizedInnovationMatrix_Kokkos(Mat Z, Vec y_mean, Vec r_inv_sqrt, PetscInt m, PetscScalar scale, Mat S)
{
  const PetscScalar *z_array, *y_array, *r_array;
  PetscScalar       *s_array;
  PetscInt           obs_size, obs_size_local, z_cols, i, j;
  PetscInt           y_local_size, r_local_size;
  PetscInt           lda_z, lda_s;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(Z, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(y_mean, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(r_inv_sqrt, VEC_CLASSID, 3);
  PetscValidLogicalCollectiveInt(Z, m, 4);
  PetscValidLogicalCollectiveScalar(Z, scale, 5);
  PetscValidHeaderSpecific(S, MAT_CLASSID, 6);
  PetscCheck(m > 0, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive, got %" PetscInt_FMT, m);

  /* Get observation size from input matrix Z and validate dimensions */
  PetscCall(MatGetSize(Z, &obs_size, &z_cols));
  PetscCall(MatGetLocalSize(Z, &obs_size_local, NULL));
  PetscCheck(z_cols == m, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Matrix Z has %" PetscInt_FMT " columns but ensemble size is %" PetscInt_FMT, z_cols, m);

  /* Verify vector dimensions match observation size (both global and local) */
  PetscCall(VecGetLocalSize(y_mean, &y_local_size));
  PetscCall(VecGetLocalSize(r_inv_sqrt, &r_local_size));
  PetscCheck(y_local_size == obs_size_local, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Vector y_mean local size %" PetscInt_FMT " does not match matrix local rows %" PetscInt_FMT, y_local_size, obs_size_local);
  PetscCheck(r_local_size == obs_size_local, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Vector r_inv_sqrt local size %" PetscInt_FMT " does not match matrix local rows %" PetscInt_FMT, r_local_size, obs_size_local);

  /* Get direct access to arrays for performance */
  PetscCall(MatDenseGetArrayRead(Z, &z_array));
  PetscCall(MatDenseGetArrayWrite(S, &s_array));
  PetscCall(VecGetArrayRead(y_mean, &y_array));
  PetscCall(VecGetArrayRead(r_inv_sqrt, &r_array));

  /* Get Leading Dimension (LDA) to handle padding/strides correctly */
  PetscCall(MatDenseGetLDA(Z, &lda_z));
  PetscCall(MatDenseGetLDA(S, &lda_s));

  /* Compute normalized innovation: S_ij = (Z_ij - y_mean_i) * scale * r_inv_sqrt_i
     Iterate column-wise (j) then row-wise (i) for optimal cache access with column-major storage */
  for (j = 0; j < m; j++) {
    const PetscScalar *z_col = z_array + j * lda_z;
    PetscScalar       *s_col = s_array + j * lda_s;

    for (i = 0; i < obs_size_local; i++) s_col[i] = (z_col[i] - y_array[i]) * scale * r_array[i];
  }

  /* Restore arrays */
  PetscCall(VecRestoreArrayRead(r_inv_sqrt, &r_array));
  PetscCall(VecRestoreArrayRead(y_mean, &y_array));
  PetscCall(MatDenseRestoreArrayWrite(S, &s_array));
  PetscCall(MatDenseRestoreArrayRead(Z, &z_array));

  /* Finalize assembly */
  PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ExtractLocalObservations_Kokkos - Extracts local observations for a vertex using localization matrix Q

  Input Parameters:
+ Q          - localization matrix (state_size x obs_size), each row has Q_NUM_LOCAL_OBSERVATIONS_MAX non-zeros
. vertex_idx - index of the vertex (row of Q)
. Z_global   - global observation ensemble matrix (obs_size x m)
. y_global   - global observation vector (size obs_size)
. y_mean_global - global observation mean (size obs_size)
. r_inv_sqrt_global - global R^{-1/2} (size obs_size)
. m          - ensemble size

  Output Parameters:
. Z_local    - local observation ensemble (p_local x m), pre-allocated
. y_local    - local observation vector (size p_local), pre-allocated
. y_mean_local - local observation mean (size p_local), pre-allocated
. r_inv_sqrt_local - local R^{-1/2} (size p_local), pre-allocated
. local_obs_indices - indices of local observations (size p_local), pre-allocated
*/
static PetscErrorCode ExtractLocalObservations_Kokkos(Mat Q, PetscInt vertex_idx, Mat Z_global, Vec y_global, Vec y_mean_global, Vec r_inv_sqrt_global, PetscInt m, Mat Z_local, Vec y_local, Vec y_mean_local, Vec r_inv_sqrt_local, PetscInt *local_obs_indices)
{
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscInt           ncols, j, k;
  const PetscScalar *z_global_array, *y_global_array, *y_mean_global_array, *r_inv_sqrt_global_array;
  PetscScalar       *z_local_array, *y_local_array, *y_mean_local_array, *r_inv_sqrt_local_array;
  PetscInt           lda_z_global, lda_z_local;

  PetscFunctionBegin;
  /* Get the row of Q corresponding to this vertex */
  PetscCall(MatGetRow(Q, vertex_idx, &ncols, &cols, &vals));
  PetscCheck(ncols == Q_NUM_LOCAL_OBSERVATIONS_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Vertex %" PetscInt_FMT " has %" PetscInt_FMT " local observations, expected %" PetscInt_FMT, vertex_idx, ncols, (PetscInt)Q_NUM_LOCAL_OBSERVATIONS_MAX);

  /* Store indices */
  for (k = 0; k < ncols; k++) local_obs_indices[k] = cols[k];

  /* Get array access to global data */
  PetscCall(MatDenseGetArrayRead(Z_global, &z_global_array));
  PetscCall(VecGetArrayRead(y_global, &y_global_array));
  PetscCall(VecGetArrayRead(y_mean_global, &y_mean_global_array));
  PetscCall(VecGetArrayRead(r_inv_sqrt_global, &r_inv_sqrt_global_array));

  /* Get array access to local data */
  PetscCall(MatDenseGetArrayWrite(Z_local, &z_local_array));
  PetscCall(VecGetArray(y_local, &y_local_array));
  PetscCall(VecGetArray(y_mean_local, &y_mean_local_array));
  PetscCall(VecGetArray(r_inv_sqrt_local, &r_inv_sqrt_local_array));

  /* Get leading dimensions */
  PetscCall(MatDenseGetLDA(Z_global, &lda_z_global));
  PetscCall(MatDenseGetLDA(Z_local, &lda_z_local));

  /* Extract local observations and weight R^{-1/2} */
  for (k = 0; k < ncols; k++) {
    PetscInt obs_idx = cols[k];

    /* Extract from vectors */
    y_local_array[k]          = y_global_array[obs_idx];
    y_mean_local_array[k]     = y_mean_global_array[obs_idx];
    r_inv_sqrt_local_array[k] = r_inv_sqrt_global_array[obs_idx] * PetscSqrtScalar(vals[k]);

    /* Extract from Z matrix (column-major) WITHOUT weighting */
    for (j = 0; j < m; j++) z_local_array[k + j * lda_z_local] = z_global_array[obs_idx + j * lda_z_global];
  }

  /* Restore arrays */
  PetscCall(VecRestoreArray(r_inv_sqrt_local, &r_inv_sqrt_local_array));
  PetscCall(VecRestoreArray(y_mean_local, &y_mean_local_array));
  PetscCall(VecRestoreArray(y_local, &y_local_array));
  PetscCall(MatDenseRestoreArrayWrite(Z_local, &z_local_array));
  PetscCall(VecRestoreArrayRead(r_inv_sqrt_global, &r_inv_sqrt_global_array));
  PetscCall(VecRestoreArrayRead(y_mean_global, &y_mean_global_array));
  PetscCall(VecRestoreArrayRead(y_global, &y_global_array));
  PetscCall(MatDenseRestoreArrayRead(Z_global, &z_global_array));

  /* Restore Q row */
  PetscCall(MatRestoreRow(Q, vertex_idx, &ncols, &cols, &vals));

  /* Assemble local matrices/vectors */
  PetscCall(MatAssemblyBegin(Z_local, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Z_local, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFLocalAnalysis - Performs local LETKF analysis for all grid points

  Input Parameters:
+ da             - the PetscDA context
. impl           - LETKF implementation data
. m              - ensemble size
. n_vertices     - number of grid points
. X              - global anomaly matrix (state_size x m)
. observation    - observation vector
. Z_global       - global observation ensemble (obs_size x m)
. y_mean_global  - global observation mean
- r_inv_sqrt_global - global R^{-1/2}

  Output:
. da->ensemble - updated with analysis ensemble

  Notes:
  This function performs the local analysis loop for LETKF, processing each grid point
  independently using its local observations defined by the localization matrix Q.
*/
PetscErrorCode PetscDALETKFLocalAnalysis(PetscDA da, PetscDALETKFData *impl, PetscInt m, PetscInt n_vertices, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global)
{
  /* Local analysis workspace */
  Mat       Z_local, S_local, T_sqrt_local, G_local;
  Vec       y_local, y_mean_local, delta_scaled_local, r_inv_sqrt_local;
  Vec       w_local, s_transpose_delta;
  PetscInt *local_obs_indices = NULL;
  PetscInt  i_vertex;
  PetscReal sqrt_m_minus_1, scale;

  PetscFunctionBegin;
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  /* ===================================================================== */
  /* Create local analysis workspace (p_local x m matrices and vectors) */
  /* ===================================================================== */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, impl->p_local, m, NULL, &Z_local));
  PetscCall(MatSetUp(Z_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, impl->p_local, m, NULL, &S_local));
  PetscCall(MatSetUp(S_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &T_sqrt_local));
  PetscCall(MatSetUp(T_sqrt_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &G_local));
  PetscCall(MatSetUp(G_local));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, impl->p_local, &y_local));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, impl->p_local, &y_mean_local));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, impl->p_local, &delta_scaled_local));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, impl->p_local, &r_inv_sqrt_local));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &w_local));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &s_transpose_delta));

  PetscCall(PetscMalloc1(impl->p_local, &local_obs_indices));

  /* ===================================================================== */
  /* LETKF: Loop over all vertices and perform local analysis            */
  /* ===================================================================== */
  for (i_vertex = 0; i_vertex < n_vertices; i_vertex++) {
    /* Extract local observations for this vertex using Q[i_vertex,:] */
    PetscCall(ExtractLocalObservations_Kokkos(impl->Q, i_vertex, Z_global, observation, y_mean_global, r_inv_sqrt_global, m, Z_local, y_local, y_mean_local, r_inv_sqrt_local, local_obs_indices));

    /* Compute local normalized innovation matrix: S_local = R_local^{-1/2} * (Z_local - y_mean_local * 1') / sqrt(m - 1) */
    PetscCall(ComputeNormalizedInnovationMatrix_Kokkos(Z_local, y_mean_local, r_inv_sqrt_local, m, scale, S_local));

    /* Compute local delta_scaled = R_local^{-1/2} * (y_local - y_mean_local) */
    PetscCall(VecWAXPY(delta_scaled_local, -1.0, y_mean_local, y_local));
    PetscCall(VecPointwiseMult(delta_scaled_local, delta_scaled_local, r_inv_sqrt_local));

    /* Factor local T = (I + S_local^T * S_local) */
    PetscCall(PetscDATFactor(da, S_local));

    /* Compute local analysis weights: w_local = T_local^{-1} * S_local^T * delta_scaled_local */
    PetscCall(MatMultTranspose(S_local, delta_scaled_local, s_transpose_delta));
    PetscCall(PetscDAApplyTInverse(da, s_transpose_delta, w_local));

    /* Compute local square-root transform: T_sqrt_local = T_local^{-1/2} * U */
    PetscCall(PetscDAApplySqrtTInverse(da, da->U, T_sqrt_local));

    /* Form local transform G_local = w_local * 1' + sqrt(m - 1) * T_sqrt_local * U
       Instead of creating w_ones_local = w_local * 1', we add w_local to each column of G_local */
    PetscCall(MatCopy(T_sqrt_local, G_local, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(G_local, sqrt_m_minus_1));
    {
      const PetscScalar *w_array;
      PetscScalar       *g_array;
      PetscInt           j, k, lda_g;

      PetscCall(VecGetArrayRead(w_local, &w_array));
      PetscCall(MatDenseGetArrayWrite(G_local, &g_array));
      PetscCall(MatDenseGetLDA(G_local, &lda_g));
      for (j = 0; j < m; j++)
        for (k = 0; k < m; k++) g_array[k + j * lda_g] += w_array[k];
      PetscCall(MatDenseRestoreArrayWrite(G_local, &g_array));
      PetscCall(VecRestoreArrayRead(w_local, &w_array));
    }

    /* LETKF Algorithm 2, Line 7: Update ensemble at grid point i_vertex
       E_a[i,:] = x_bar_f[i] + X_f[i,:] * G_local

       Where:
       - x_bar_f[i] is the forecast mean at grid point i_vertex (from global mean vector)
       - X_f[i,:] is the forecast anomaly row at grid point i_vertex (from global anomaly matrix X)
       - G_local = w_local * 1' + sqrt(m-1) * T_local^{1/2} * U (computed above in G_local)
     */
    {
      Vec                X_row, E_analysis_row;
      const PetscScalar *x_array, *mean_array;
      PetscScalar       *e_array, *x_row_vals, *ea_row_vals;
      PetscInt           j, lda_x, lda_e;

      /* Extract row i_vertex from X: X_f[i_vertex, :] */
      PetscCall(PetscMalloc1(m, &x_row_vals));
      PetscCall(MatDenseGetArrayRead(X, &x_array));
      PetscCall(MatDenseGetLDA(X, &lda_x));
      for (j = 0; j < m; j++) x_row_vals[j] = x_array[i_vertex + j * lda_x];
      PetscCall(MatDenseRestoreArrayRead(X, &x_array));

      /* Create temp vectors for the update */
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m, x_row_vals, &X_row));
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &E_analysis_row));

      /* Apply local transform: E_analysis_row = X_row * G_local^T */
      PetscCall(MatMultTranspose(G_local, X_row, E_analysis_row));

      /* Add local mean: E_a[i_vertex, :] = x_bar_f[i_vertex] + X_f[i_vertex, :] * G_local */
      PetscCall(VecGetArrayRead(impl->mean, &mean_array));
      PetscCall(VecShift(E_analysis_row, mean_array[i_vertex]));
      PetscCall(VecRestoreArrayRead(impl->mean, &mean_array));

      /* Store result back in ensemble[i_vertex, :] */
      PetscCall(MatDenseGetArrayWrite(da->ensemble, &e_array));
      PetscCall(MatDenseGetLDA(da->ensemble, &lda_e));
      PetscCall(VecGetArray(E_analysis_row, &ea_row_vals));
      for (j = 0; j < m; j++) e_array[i_vertex + j * lda_e] = ea_row_vals[j];
      PetscCall(VecRestoreArray(E_analysis_row, &ea_row_vals));
      PetscCall(MatDenseRestoreArrayWrite(da->ensemble, &e_array));

      PetscCall(VecDestroy(&E_analysis_row));
      PetscCall(VecDestroy(&X_row));
      PetscCall(PetscFree(x_row_vals));
    }
  }

  /* Ensemble has been updated directly in the loop above */
  PetscCall(MatAssemblyBegin(da->ensemble, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(da->ensemble, MAT_FINAL_ASSEMBLY));

  /* Cleanup */
  PetscCall(PetscFree(local_obs_indices));
  PetscCall(VecDestroy(&s_transpose_delta));
  PetscCall(VecDestroy(&w_local));
  PetscCall(VecDestroy(&r_inv_sqrt_local));
  PetscCall(VecDestroy(&delta_scaled_local));
  PetscCall(VecDestroy(&y_mean_local));
  PetscCall(VecDestroy(&y_local));
  PetscCall(MatDestroy(&G_local));
  PetscCall(MatDestroy(&T_sqrt_local));
  PetscCall(MatDestroy(&S_local));
  PetscCall(MatDestroy(&Z_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}
