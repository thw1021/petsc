#include "letkf_impl.h"
#include <petscblaslapack.h>

static PetscBool PetscDALETKFPackageInitialized = PETSC_FALSE;

/* ========================================================================== */
/*                       LETKF Implementation Lifecycle                       */
/* ========================================================================== */

static PetscErrorCode PetscDALETKFDestroy(PetscDA da)
{
  PetscDALETKFData *impl;

  PetscFunctionBegin;
  if (da->data) {
    impl = (PetscDALETKFData *)da->data;
    /* Destroy persistent work objects */
    PetscCall(VecDestroy(&impl->mean));
    PetscCall(VecDestroy(&impl->y_mean));
    PetscCall(VecDestroy(&impl->delta_scaled));
    PetscCall(VecDestroy(&impl->w));
    PetscCall(VecDestroy(&impl->r_inv_sqrt));
    PetscCall(MatDestroy(&impl->Z));
    PetscCall(MatDestroy(&impl->S));
    PetscCall(MatDestroy(&impl->T_sqrt));
    PetscCall(MatDestroy(&impl->w_ones));
    PetscCall(MatDestroy(&impl->Q)); // Destroy localization matrix

    PetscCall(PetscFree(da->data));
    da->data = NULL;
  }
  /* Clear the composed function */
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalization_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Package Initialization                           */
/* ========================================================================== */

/*@C
  PetscDALETKFInitializePackage - This function initializes everything in the `PetscDALETKF` package. It is called from `TSInitializePackage()`.

  Level: developer

.seealso: [](ch_ts), `PetscInitialize()`, `PetscDALETKFFinalizePackage()`
@*/
PetscErrorCode PetscDALETKFInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDALETKFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDALETKFPackageInitialized = PETSC_TRUE;
  PetscCall(PetscRegisterFinalize(PetscDALETKFFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDALETKFFinalizePackage - This function destroys everything in the `PetscDALETKF` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ts), `PetscFinalize()`, `PetscDALETKFInitializePackage()`
@*/
PetscErrorCode PetscDALETKFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscDALETKFPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                    LETKF Analysis Algorithm                               */
/* ========================================================================== */

/*
  ComputeEnsembleMean - Direct implementation of ensemble mean computation

  Input Parameters:
+ ensemble - ensemble matrix (state_size x ensemble_size)
. ensemble_size - number of ensemble members

  Output Parameter:
. mean - ensemble mean vector
*/
static PetscErrorCode ComputeEnsembleMean(Mat ensemble, PetscInt ensemble_size, Vec mean)
{
  PetscScalar inv_m;
  Vec         col;
  PetscInt    i;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(ensemble, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 3);

  /* Direct implementation instead of da->ops->computemean */
  inv_m = 1.0 / ensemble_size;
  PetscCall(VecSet(mean, 0.0));
  for (i = 0; i < ensemble_size; i++) {
    PetscCall(MatDenseGetColumnVecRead(ensemble, i, &col));
    PetscCall(VecAXPY(mean, inv_m, col));
    PetscCall(MatDenseRestoreColumnVecRead(ensemble, i, &col));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeAnomalies - Direct implementation of anomaly computation, X = (E - x_mean * 1') / sqrt(m - 1)

  Input Parameters:
+ ensemble - ensemble matrix (state_size x ensemble_size)
. mean - ensemble mean vector
. ensemble_size - number of ensemble members

  Output Parameter:
. X - anomaly matrix (state_size x ensemble_size)
*/
static PetscErrorCode ComputeAnomalies(Mat ensemble, Vec mean, PetscInt ensemble_size, Mat X)
{
  PetscScalar scale;
  Vec         col_in, col_out;
  PetscInt    i;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(ensemble, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 4);

  /* Direct implementation instead of da->ops->computeanomalies */
  scale = 1.0 / PetscSqrtReal(ensemble_size - 1);
  for (i = 0; i < ensemble_size; i++) {
    PetscCall(MatDenseGetColumnVecRead(ensemble, i, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(X, i, &col_out));
    PetscCall(VecWAXPY(col_out, -1.0, mean, col_in));
    PetscCall(VecScale(col_out, scale));
    PetscCall(MatDenseRestoreColumnVecWrite(X, i, &col_out));
    PetscCall(MatDenseRestoreColumnVecRead(ensemble, i, &col_in));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeNormalizedInnovationMatrix - Computes S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1) (CPU version)

  Input Parameters:
+ Z          - observation ensemble matrix
. y_mean     - mean of observations
. r_inv_sqrt - R^{-1/2}
. m          - ensemble size
- scale      - 1/sqrt(m-1)

  Output Parameter:
. S - normalized innovation matrix
*/
static PetscErrorCode ComputeNormalizedInnovationMatrix(Mat Z, Vec y_mean, Vec r_inv_sqrt, PetscInt m, PetscScalar scale, Mat S)
{
  const PetscScalar *z_array, *y_array, *r_array;
  PetscScalar       *s_array;
  PetscInt           obs_size, obs_size_local, z_cols;
  PetscInt           y_local_size, r_local_size;
  PetscInt           lda_z, lda_s, i, j;

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

  /* Compute normalized innovation: S_ij = (Z_ij - y_mean_i) * scale * r_inv_sqrt_i */
  for (j = 0; j < m; j++) {
    for (i = 0; i < obs_size_local; i++) s_array[i + j * lda_s] = (z_array[i + j * lda_z] - y_array[i]) * scale * r_array[i];
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
  ExtractLocalObservations - Extracts local observations for a vertex using localization matrix Q (CPU version)

  Input Parameters:
+ Q          - localization matrix (state_size/ndof x obs_size), each row has Q_NUM_LOCAL_OBSERVATIONS_MAX non-zeros
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
static PetscErrorCode ExtractLocalObservations(Mat Q, PetscInt vertex_idx, Mat Z_global, Vec y_global, Vec y_mean_global, Vec r_inv_sqrt_global, PetscInt m, Mat Z_local, Vec y_local, Vec y_mean_local, Vec r_inv_sqrt_local, PetscInt *local_obs_indices)
{
  const PetscInt    *cols;
  const PetscScalar *vals;
  PetscInt           ncols, k, j;
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
    PetscInt    obs_idx = cols[k];
    PetscScalar weight  = vals[k];

    y_local_array[k]          = y_global_array[obs_idx];
    y_mean_local_array[k]     = y_mean_global_array[obs_idx];
    r_inv_sqrt_local_array[k] = r_inv_sqrt_global_array[obs_idx] * PetscSqrtScalar(weight);

    /* Extract Z matrix row (column-major layout) */
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
  PetscDALETKFLocalAnalysis - Performs local LETKF analysis for all grid points (CPU version)

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
  This is the CPU version that does not use Kokkos acceleration.

  All local analysis workspace objects (Z_local, S_local, T_sqrt_local, G_local, y_local,
  y_mean_local, delta_scaled_local, r_inv_sqrt_local, w_local, s_transpose_delta) are
  created with PETSC_COMM_SELF because the analysis at each vertex is serial and independent.
*/
PetscErrorCode PetscDALETKFLocalAnalysis(PetscDA da, PetscDALETKFData *impl, PetscInt m, PetscInt n_vertices, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global)
{
  /* Local analysis workspace */
  Mat       Z_local, S_local, T_sqrt_local, G_local;
  Vec       y_local, y_mean_local, delta_scaled_local, r_inv_sqrt_local;
  Vec       w_local, s_transpose_delta;
  PetscInt *local_obs_indices = NULL;
  PetscInt  i_grid_point;
  PetscInt  ndof;
  PetscReal sqrt_m_minus_1, scale;

  PetscFunctionBegin;
  ndof           = da->ndof;
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  /* ===================================================================== */
  /* Create local analysis workspace (p_local x m matrices and vectors) */
  /* ===================================================================== */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, impl->p_local, m, NULL, &Z_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)Z_local, "dense_"));
  PetscCall(MatSetFromOptions(Z_local));
  PetscCall(MatSetUp(Z_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, impl->p_local, m, NULL, &S_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)S_local, "dense_"));
  PetscCall(MatSetFromOptions(S_local));
  PetscCall(MatSetUp(S_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &T_sqrt_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)T_sqrt_local, "dense_"));
  PetscCall(MatSetFromOptions(T_sqrt_local));
  PetscCall(MatSetUp(T_sqrt_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &G_local));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)G_local, "dense_"));
  PetscCall(MatSetFromOptions(G_local));
  PetscCall(MatSetUp(G_local));

  /* Create vectors using MatCreateVecs from Z_local (p_local x m) */
  PetscCall(MatCreateVecs(Z_local, &w_local, &y_local));
  PetscCall(VecDuplicate(y_local, &y_mean_local));
  PetscCall(VecDuplicate(y_local, &delta_scaled_local));
  PetscCall(VecDuplicate(y_local, &r_inv_sqrt_local));
  PetscCall(VecDuplicate(w_local, &s_transpose_delta));

  PetscCall(PetscMalloc1(impl->p_local, &local_obs_indices));

  /* ===================================================================== */
  /* LETKF: Loop over all grid points and perform local analysis         */
  /* ===================================================================== */
  for (i_grid_point = 0; i_grid_point < n_vertices; i_grid_point++) {
    /* Extract local observations for this grid point using Q[i_grid_point,:] */
    PetscCall(ExtractLocalObservations(impl->Q, i_grid_point, Z_global, observation, y_mean_global, r_inv_sqrt_global, m, Z_local, y_local, y_mean_local, r_inv_sqrt_local, local_obs_indices));

    /* Compute local normalized innovation matrix: S_local = R_local^{-1/2} * (Z_local - y_mean_local * 1') / sqrt(m - 1) */
    PetscCall(ComputeNormalizedInnovationMatrix(Z_local, y_mean_local, r_inv_sqrt_local, m, scale, S_local));

    /* Compute local delta_scaled = R_local^{-1/2} * (y_local - y_mean_local) */
    PetscCall(VecWAXPY(delta_scaled_local, -1.0, y_mean_local, y_local));
    PetscCall(VecPointwiseMult(delta_scaled_local, delta_scaled_local, r_inv_sqrt_local));

    /* Factor local T = (I + S_local^T * S_local) */
    PetscCall(PetscDATFactor(da, S_local));

    /* Compute local analysis weights: w_local = T_local^{-1} * S_local^T * delta_scaled_local */
    PetscCall(MatMultTranspose(S_local, delta_scaled_local, s_transpose_delta));
    PetscCall(PetscDAApplyTInverse(da, s_transpose_delta, w_local));

    /* Compute local square-root transform: T_sqrt_local = T_local^{-1/2} (U is identity, so pass NULL) */
    PetscCall(PetscDAApplySqrtTInverse(da, NULL, T_sqrt_local));

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

    /* LETKF Algorithm 2, Line 13: Update ensemble at grid point i_grid_point
       E_a[i,:] = x_bar_f[i] + X_f[i,:] * G_local

       Where:
       - x_bar_f[i] is the forecast mean at grid point i_grid_point (ndof values from global mean vector)
       - X_f[i,:] is the forecast anomaly rows at grid point i_grid_point (ndof rows from global anomaly matrix X)
       - G_local = w_local * 1' + sqrt(m-1) * T_local^{1/2} * U (computed above in G_local)
     */
    {
      Mat                X_rows, E_analysis_rows;
      const PetscScalar *x_array, *mean_array;
      PetscScalar       *e_array, *x_rows_array, *ea_rows_array;
      PetscInt           j, k, lda_x, lda_e;

      /* Create temp matrices for the update: ndof x m */
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, ndof, m, NULL, &X_rows));
      PetscCall(MatDuplicate(X_rows, MAT_DO_NOT_COPY_VALUES, &E_analysis_rows));

      /* Extract ndof rows starting at (i_grid_point * ndof) from X: X_f[i_grid_point*ndof:(i_grid_point+1)*ndof, :] */
      PetscCall(MatDenseGetArrayRead(X, &x_array));
      PetscCall(MatDenseGetArray(X_rows, &x_rows_array));
      PetscCall(MatDenseGetLDA(X, &lda_x));
      for (j = 0; j < m; j++) {
        for (k = 0; k < ndof; k++) x_rows_array[k + j * ndof] = x_array[(i_grid_point * ndof + k) + j * lda_x];
      }
      PetscCall(MatDenseRestoreArray(X_rows, &x_rows_array));
      PetscCall(MatDenseRestoreArrayRead(X, &x_array));

      /* Apply local transform: E_analysis_rows = X_rows * G_local^T */
      PetscCall(MatMatMult(X_rows, G_local, MAT_REUSE_MATRIX, PETSC_DEFAULT, &E_analysis_rows));

      /* Add local mean: E_a[i_grid_point*ndof:(i_grid_point+1)*ndof, :] = x_bar_f[i_grid_point*ndof:(i_grid_point+1)*ndof] + X_f[...] * G_local */
      PetscCall(VecGetArrayRead(impl->mean, &mean_array));
      PetscCall(MatDenseGetArray(E_analysis_rows, &ea_rows_array));
      for (j = 0; j < m; j++) {
        for (k = 0; k < ndof; k++) ea_rows_array[k + j * ndof] += mean_array[i_grid_point * ndof + k];
      }
      PetscCall(MatDenseRestoreArray(E_analysis_rows, &ea_rows_array));
      PetscCall(VecRestoreArrayRead(impl->mean, &mean_array));

      /* Store result back in ensemble[i_grid_point*ndof:(i_grid_point+1)*ndof, :] */
      PetscCall(MatDenseGetArrayWrite(da->ensemble, &e_array));
      PetscCall(MatDenseGetLDA(da->ensemble, &lda_e));
      PetscCall(MatDenseGetArrayRead(E_analysis_rows, (const PetscScalar **)&ea_rows_array));
      for (j = 0; j < m; j++) {
        for (k = 0; k < ndof; k++) e_array[(i_grid_point * ndof + k) + j * lda_e] = ea_rows_array[k + j * ndof];
      }
      PetscCall(MatDenseRestoreArrayRead(E_analysis_rows, (const PetscScalar **)&ea_rows_array));
      PetscCall(MatDenseRestoreArrayWrite(da->ensemble, &e_array));

      PetscCall(MatDestroy(&E_analysis_rows));
      PetscCall(MatDestroy(&X_rows));
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

/*
  PetscDALETKFAnalysis - Performs the local ensemble transform Kalman filter (LETKF) analysis

  Collective

  Input Parameters:
+ da          - the `PetscDA` context owning the forecast ensemble and buffers
. observation - observation vector `y` in R^P
- H           - observation operator matrix (P x N), sparse AIJ format

  Notes:
  The observation matrix H maps state to observations: Z = H * E
*/
static PetscErrorCode PetscDALETKFAnalysis(PetscDA da, Vec observation, Mat H)
{
  PetscDALETKFData *impl;
  Mat               X;
  PetscInt          m;
  PetscBool         reallocate = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);
  /* Validate ensemble size */
  m = da->ensemble_size;
  PetscCheck(m > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be > 1, got %" PetscInt_FMT, m);

  impl = (PetscDALETKFData *)da->data;
  PetscCall(PetscInfo(da, "squaroot type %s, %" PetscInt_FMT " ensembles, LETKF localization with p_local=%" PetscInt_FMT "\n", (da->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky", m, impl->p_local));

  /* Check if localization matrix Q is set */
  PetscCheck(impl->Q, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Localization matrix Q not set. Call PetscDALETKFSetLocalization() first.");

  /* Check for reallocation needs */
  if (impl->mean) {
    PetscInt mean_size;
    PetscCall(VecGetSize(impl->mean, &mean_size));
    if (mean_size != da->state_size) reallocate = PETSC_TRUE;
  }
  if (impl->Z) {
    PetscInt z_rows, z_cols;
    PetscCall(MatGetSize(impl->Z, &z_rows, &z_cols));
    if (z_rows != da->obs_size || z_cols != m) reallocate = PETSC_TRUE;
  }

  /* Initialize or reallocate persistent work objects */
  if (!impl->mean || reallocate) {
    PetscCall(VecDestroy(&impl->mean));
    PetscCall(VecDestroy(&impl->y_mean));
    PetscCall(VecDestroy(&impl->delta_scaled));
    PetscCall(VecDestroy(&impl->w));
    PetscCall(VecDestroy(&impl->r_inv_sqrt));
    PetscCall(MatDestroy(&impl->Z));
    PetscCall(MatDestroy(&impl->S));
    PetscCall(MatDestroy(&impl->T_sqrt));
    PetscCall(MatDestroy(&impl->w_ones));

    /* Create mean vector from ensemble matrix (right vector = state space) */
    PetscCall(MatCreateVecs(da->ensemble, NULL, &impl->mean));

    /* Create Z matrix (obs_size x m) */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, &impl->Z));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->Z, "dense_"));
    PetscCall(MatSetFromOptions(impl->Z));
    PetscCall(MatSetUp(impl->Z));

    /* Create observation space vectors from Z matrix (left vector = observation space) */
    PetscCall(MatCreateVecs(impl->Z, NULL, &impl->y_mean));
    PetscCall(VecDuplicate(impl->y_mean, &impl->delta_scaled));
    PetscCall(VecDuplicate(da->obs_error_var, &impl->r_inv_sqrt));

    /* Create S matrix (same layout as Z) */
    PetscCall(MatDuplicate(impl->Z, MAT_DO_NOT_COPY_VALUES, &impl->S));

    /* Create T_sqrt matrix (m x m) - usually small */
    /* T_sqrt will hold the result of applying T^{-1/2} to identity matrix */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &impl->T_sqrt));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->T_sqrt, "dense_"));
    PetscCall(MatSetFromOptions(impl->T_sqrt));
    PetscCall(MatSetUp(impl->T_sqrt));

    /* Create w_ones matrix (m x m) */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &impl->w_ones));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->w_ones, "dense_"));
    PetscCall(MatSetFromOptions(impl->w_ones));
    PetscCall(MatSetUp(impl->w_ones));
  }

  /* ===================================================================== */
  /* Alg 6.4 line 1-2: Compute ensemble mean and scaled anomalies        */
  /* ===================================================================== */
  PetscCall(ComputeEnsembleMean(da->ensemble, m, impl->mean));

  /* Create anomaly matrix X = (E - x_mean * 1') / sqrt(m - 1) */
  PetscCall(MatDuplicate(da->ensemble, MAT_DO_NOT_COPY_VALUES, &X));
  PetscCall(ComputeAnomalies(da->ensemble, impl->mean, m, X));

  /* ===================================================================== */
  /* Alg 6.4 line 3-4: Compute GLOBAL observation ensemble Z = H * E     */
  /* ===================================================================== */
  /* Z = H * E using matrix-matrix multiplication (obs_size x ensemble_size) */
  /* Note: When H is a Kokkos matrix type (e.g., aijkokkos), MatMatMult may fail
     with non-Kokkos dense matrices. Use column-by-column multiplication with
     temporary vectors that are compatible with H's type. */
  {
    Vec      col_in, col_out, temp_in, temp_out;
    PetscInt j;

    /* Create or reuse Z matrix */
    if (!impl->Z) {
      PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, &impl->Z));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->Z, "dense_"));
      PetscCall(MatSetFromOptions(impl->Z));
      PetscCall(MatSetUp(impl->Z));
    } else {
      PetscInt z_rows, z_cols;
      PetscCall(MatGetSize(impl->Z, &z_rows, &z_cols));
      if (z_rows != da->obs_size || z_cols != m) {
        PetscCall(MatDestroy(&impl->Z));
        PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, &impl->Z));
        PetscCall(PetscObjectSetOptionsPrefix((PetscObject)impl->Z, "dense_"));
        PetscCall(MatSetFromOptions(impl->Z));
        PetscCall(MatSetUp(impl->Z));
      }
    }

    /* Create temporary vectors compatible with H's type */
    PetscCall(MatCreateVecs(H, &temp_in, &temp_out));

    /* Compute Z = H * E column by column to avoid Kokkos vector type issues */
    for (j = 0; j < m; j++) {
      PetscCall(MatDenseGetColumnVecRead(da->ensemble, j, &col_in));
      PetscCall(MatDenseGetColumnVecWrite(impl->Z, j, &col_out));

      /* Copy to temp vector, multiply, then copy back */
      PetscCall(VecCopy(col_in, temp_in));
      PetscCall(MatMult(H, temp_in, temp_out));
      PetscCall(VecCopy(temp_out, col_out));

      PetscCall(MatDenseRestoreColumnVecWrite(impl->Z, j, &col_out));
      PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, j, &col_in));
    }
    PetscCall(MatAssemblyBegin(impl->Z, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(impl->Z, MAT_FINAL_ASSEMBLY));

    PetscCall(VecDestroy(&temp_out));
    PetscCall(VecDestroy(&temp_in));
  }

  /* Compute GLOBAL observation mean y_mean = H * x_mean */
  /* Use temporary vector compatible with H's type */
  {
    Vec temp_mean, temp_y_mean;
    PetscCall(MatCreateVecs(H, &temp_mean, &temp_y_mean));
    PetscCall(VecCopy(impl->mean, temp_mean));
    PetscCall(MatMult(H, temp_mean, temp_y_mean));
    PetscCall(VecCopy(temp_y_mean, impl->y_mean));
    PetscCall(VecDestroy(&temp_y_mean));
    PetscCall(VecDestroy(&temp_mean));
  }

  /* ===================================================================== */
  /* Compute GLOBAL R^{-1/2} (assumes diagonal R) */
  /* ===================================================================== */
  PetscCall(VecCopy(da->obs_error_var, impl->r_inv_sqrt));
  PetscCall(VecSqrtAbs(impl->r_inv_sqrt));
  PetscCall(VecReciprocal(impl->r_inv_sqrt));

  /* ===================================================================== */
  /* Perform local analysis for all vertices */
  /* ===================================================================== */

#if defined(PETSC_HAVE_KOKKOS)
  /* Use GPU version only if:
     1. sqrt_type is eigen (GPU version only implements eigen/SVD, not cholesky)
     2. H matrix is a Kokkos type (aijkokkos) */
  {
    PetscBool use_gpu = PETSC_FALSE;
    if (da->sqrt_type == PETSCDA_SQRT_EIGEN) {
      /* Check if H matrix is a Kokkos type */
      PetscCall(PetscObjectTypeCompareAny((PetscObject)da->R, &use_gpu, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
    }

    if (use_gpu) {
      PetscCall(PetscDALETKFLocalAnalysis_GPU(da, impl, m, da->state_size / da->ndof, X, observation, impl->Z, impl->y_mean, impl->r_inv_sqrt));
    } else {
      PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, da->state_size / da->ndof, X, observation, impl->Z, impl->y_mean, impl->r_inv_sqrt));
    }
  }
#else
  /* Without Kokkos, use CPU version */
  PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, da->state_size / da->ndof, X, observation, impl->Z, impl->y_mean, impl->r_inv_sqrt));
#endif
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Model Propagation                                */
/* ========================================================================== */

/*
  PetscDALETKFApplyModel - Advances each ensemble member through the user-supplied
  nonlinear model.

  Collective

  Input Parameters:
+ da        - the `PetscDA` context that stores the ensemble
. model     - routine that evaluates the model `f(x, xnew; ctx)`
- model_ctx - optional context for `model`
*/
static PetscErrorCode PetscDALETKFApplyModel(PetscDA da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
{
  Vec      col_in, col_out, temp;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  /* Create temp vector from ensemble matrix (right vector = state space) */
  PetscCall(MatCreateVecs(da->ensemble, NULL, &temp));

  for (i = 0; i < da->ensemble_size; i++) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, i, &col_in));
    PetscCall(model(col_in, temp, model_ctx));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, i, &col_in));

    PetscCall(MatDenseGetColumnVecWrite(da->ensemble, i, &col_out));
    PetscCall(VecCopy(temp, col_out));
    PetscCall(MatDenseRestoreColumnVecWrite(da->ensemble, i, &col_out));
  }

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Public API Functions                             */
/* ========================================================================== */

static PetscErrorCode PetscDALETKFSetLocalization_LETKF(PetscDA da, Mat Q)
{
  PetscDALETKFData *impl;
  PetscInt          i, nrows, ncols, nnz;

  PetscFunctionBegin;
  /* Get implementation data */
  impl = (PetscDALETKFData *)da->data;
  PetscCheck(impl, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA not properly initialized for LETKF");

  /* Get matrix dimensions */
  PetscCall(MatGetSize(Q, &nrows, &ncols));

  /* Validate matrix dimensions */
  PetscCheck(nrows == da->state_size / da->ndof, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix rows (%" PetscInt_FMT ") must match state size (%" PetscInt_FMT ")", nrows, da->state_size);
  PetscCheck(ncols == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix columns (%" PetscInt_FMT ") must match observation size (%" PetscInt_FMT ")", ncols, da->obs_size);

  /* Validate that each row has exactly Q_NUM_LOCAL_OBSERVATIONS_MAX non-zero entries */
  for (i = 0; i < nrows; i++) {
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(Q, i, &nnz, &cols, &vals));
    PetscCheck(nnz == Q_NUM_LOCAL_OBSERVATIONS_MAX, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Row %" PetscInt_FMT " has %" PetscInt_FMT " non-zeros, expected %" PetscInt_FMT, i, nnz, (PetscInt)Q_NUM_LOCAL_OBSERVATIONS_MAX);
    PetscCall(MatRestoreRow(Q, i, &nnz, &cols, &vals));
  }

  /* Store the localization matrix */
  PetscCall(MatDestroy(&impl->Q));
  PetscCall(MatDuplicate(Q, MAT_COPY_VALUES, &impl->Q));
  impl->p_local = Q_NUM_LOCAL_OBSERVATIONS_MAX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFView - Views a `PetscDALETKF` and its implementation-specific data structure.

  Input Parameters:
+ da     - the `PetscDA` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: beginner

.seealso: [](ch_da), `PetscDAViewFromOptions()`
*/
static PetscErrorCode PetscDALETKFView(PetscDA da, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscDALETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  impl = (PetscDALETKFData *)da->data;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDALETKF Object:\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Square root type: %s\n", (da->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky"));
#if defined(PETSC_HAVE_KOKKOS)
    if (da->sqrt_type == PETSCDA_SQRT_CHOLESKY) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: CPU\n"));
    } else {
      /* Check if R matrix is Kokkos type to determine if GPU will be used */
      if (da->R) {
        PetscBool is_kokkos = PETSC_FALSE;
        PetscCall(PetscObjectTypeCompareAny((PetscObject)da->R, &is_kokkos, MATSEQAIJKOKKOS, MATAIJKOKKOS, ""));
        if (is_kokkos) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: Kokkos\n"));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: CPU\n"));
        }
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: CPU or Kokkos (depending on covarience matrix type)\n"));
      }
    }
#else
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Local analysis: CPU\n"));
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Local observations per vertex: %" PetscInt_FMT "\n", impl->p_local));
    if (impl->batch_size > 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  GPU batch size: %" PetscInt_FMT "\n", impl->batch_size));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  GPU batch size: auto\n"));
    }
    if (impl->Q) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization matrix: set\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization matrix: not set\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASetFromOptions_LETKF(PetscDA da, PetscOptionItems *PetscOptionsObjectPtr)
{
  PetscDALETKFData *impl               = (PetscDALETKFData *)da->data;
  PetscOptionItems  PetscOptionsObject = *PetscOptionsObjectPtr;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscDA LETKF Options");
  PetscCall(PetscOptionsInt("-petscda_letkf_batch_size", "Batch size for GPU processing", "", impl->batch_size, &impl->batch_size, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFInitialize - Installs the LETKF-specific operations on a newly created `PetscDA` object.

  Input Parameter:
. da - the `PetscDA` object to configure

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDARegister()`, `PetscDALETKFAnalysis()`
@*/
PetscErrorCode PetscDALETKFInitialize(PetscDA da)
{
  PetscDALETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscCall(PetscNew(&impl));

  da->data                  = impl;
  da->ops->analysis         = PetscDALETKFAnalysis;
  da->ops->applymodel       = PetscDALETKFApplyModel;
  da->ops->computemean      = NULL; /* We implement this directly in the analysis function */
  da->ops->computeanomalies = NULL; /* We implement this directly in the analysis function */
  da->ops->destroy          = PetscDALETKFDestroy;
  da->ops->view             = PetscDALETKFView;
  da->ops->setfromoptions   = PetscDASetFromOptions_LETKF;

  /* Initialize default values */
  impl->p_local    = Q_NUM_LOCAL_OBSERVATIONS_MAX;
  impl->Q          = NULL;
  impl->batch_size = 0;

  /* Register the method for setting localization */
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalization_C", PetscDALETKFSetLocalization_LETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFSetLocalization - Sets the localization matrix for the LETKF algorithm.

  Collective

  Input Parameters:
+ da - the `PetscDA` context
- Q  - the localization matrix (N x P)

  Level: advanced

.seealso: [](ch_da), `PetscDA`
@*/
PetscErrorCode PetscDALETKFSetLocalization(PetscDA da, Mat Q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(Q, MAT_CLASSID, 2);
  PetscTryMethod(da, "PetscDALETKFSetLocalization_C", (PetscDA, Mat), (da, Q));
  PetscFunctionReturn(PETSC_SUCCESS);
}
