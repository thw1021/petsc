#include "petscda.h"
#include <petsc/private/daimpl.h>
#include <petscblaslapack.h>
#include <Kokkos_Core.hpp>

typedef struct {
  /* Persistent work vectors and matrices to avoid repeated allocation */
  Vec mean;
  Vec y_mean;
  Vec delta_scaled;
  Vec w;
  Vec r_inv_sqrt;
  Mat Z;
  Mat S;
  Mat T_sqrt;
  Mat w_ones;
  Mat Q;            // NEW: Localization matrix (n_grid x n_observations_total)
                    //      Each row has exactly Q_NUM_OBSERVATIONS_MAX non-zeros
  PetscInt p_local; // = Q_NUM_OBSERVATIONS_MAX (number of local observations per grid point)
  PetscInt n_grid;  // Number of grid points (n_grid = state_size / da->ndof)
} PetscDALETKFData;

static PetscFunctionList PetscDALETKFSqrtList           = NULL;
static PetscBool         PetscDALETKFPackageInitialized = PETSC_FALSE;

/* ========================================================================== */
/*                    Helper Functions for LETKF Analysis                     */
/* ========================================================================== */

/*
  ComputeNormalizedInnovationMatrix - Computes S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1)

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
  BroadcastWeightVector - Creates matrix with weight vector replicated across all columns

  Input Parameters:
+ w - weight vector of size m (analysis weights from LETKF update)
- m - ensemble size (number of columns to replicate, must equal vector size)

  Output Parameter:
. w_ones - m x m dense matrix where each column is a copy of w (i.e., w * 1^T)
*/
static PetscErrorCode BroadcastWeightVector(Vec w, PetscInt m, Mat w_ones)
{
  const PetscScalar *w_array;
  PetscScalar       *mat_array;
  PetscInt           w_size, w_size_local, mat_rows_local, mat_cols_local;
  PetscInt           i, lda;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(w, m, 2);
  PetscValidHeaderSpecific(w_ones, MAT_CLASSID, 3);
  PetscCheck(m > 0, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive for broadcasting, got %" PetscInt_FMT, m);
  /* Check for potential overflow in matrix size calculation */
  PetscCheck(m <= PETSC_MAX_INT / m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m = %" PetscInt_FMT " too large", m);

  /* Verify dimensions */
  PetscCall(VecGetSize(w, &w_size));
  PetscCall(VecGetLocalSize(w, &w_size_local));
  PetscCheck(w_size == m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_INCOMP, "Weight vector global size (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ")", w_size, m);

  /* Verify consistent parallel layout between vector and matrix */
  PetscCall(MatGetLocalSize(w_ones, &mat_rows_local, &mat_cols_local));
  PetscCheck(mat_rows_local == w_size_local, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix row distribution (%" PetscInt_FMT ") inconsistent with vector distribution (%" PetscInt_FMT ")", mat_rows_local, w_size_local);
  PetscCheck(mat_cols_local == m, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix local columns (%" PetscInt_FMT ") must equal global columns m (%" PetscInt_FMT ") for MPIDense", mat_cols_local, m);

  /* Access raw arrays for efficient broadcasting */
  PetscCall(VecGetArrayRead(w, &w_array));
  PetscCall(MatDenseGetArrayWrite(w_ones, &mat_array));
  PetscCall(MatDenseGetLDA(w_ones, &lda));

  /* Copy w to each column of w_ones */
  /* Note: MatDense uses column-major storage. We copy the vector w into each column. */
  for (i = 0; i < m; i++) PetscCall(PetscArraycpy(mat_array + i * lda, w_array, w_size_local));

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayWrite(w_ones, &mat_array));
  PetscCall(VecRestoreArrayRead(w, &w_array));

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(w_ones, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(w_ones, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                       Square Root Type Setters                            */
/* ========================================================================== */

static PetscErrorCode PetscDALETKFSetSqrt_Cholesky(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscDALETKFSetSqrtType(da, PETSCDA_SQRT_CHOLESKY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDALETKFSetSqrt_Eigen(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscDALETKFSetSqrtType(da, PETSCDA_SQRT_EIGEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

static PetscErrorCode PetscDASetFromOptions_DASLETKF(PetscDA da, PetscOptionItems *PetscOptions)
{
  PetscOptionItems PetscOptionsObject;
  const char      *defaultType;
  char             typeName[256];
  PetscBool        set              = PETSC_FALSE;
  PetscErrorCode (*setter)(PetscDA) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscOptionsObject = PetscOptions ? *PetscOptions : NULL;

  defaultType = (da->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky";
  PetscCall(PetscStrncpy(typeName, defaultType, sizeof(typeName)));
  PetscCall(PetscOptionsFList("-da_sqrt_type", "Matrix square root factorization", "PetscDALETKFSetSqrtType", PetscDALETKFSqrtList, defaultType, typeName, sizeof(typeName), &set));
  if (set) {
    PetscCall(PetscFunctionListFind(PetscDALETKFSqrtList, typeName, &setter));
    PetscCheck(setter, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDALETKF square-root type \"%s\"", typeName);
    PetscCall((*setter)(da));
  }
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
  PetscCall(PetscFunctionListAdd(&PetscDALETKFSqrtList, "cholesky", PetscDALETKFSetSqrt_Cholesky));
  PetscCall(PetscFunctionListAdd(&PetscDALETKFSqrtList, "eigen", PetscDALETKFSetSqrt_Eigen));
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
  PetscCall(PetscFunctionListDestroy(&PetscDALETKFSqrtList));
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
  ComputeAnomalies - Direct implementation of anomaly computation

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
  ExtractLocalObservations - Extracts local observations for a vertex using localization matrix Q

  Input Parameters:
+ Q          - localization matrix (state_size x obs_size), each row has Q_NUM_OBSERVATIONS_MAX non-zeros
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
  PetscInt           ncols, j, k;
  const PetscScalar *z_global_array, *y_global_array, *y_mean_global_array, *r_inv_sqrt_global_array;
  PetscScalar       *z_local_array, *y_local_array, *y_mean_local_array, *r_inv_sqrt_local_array;
  PetscInt           lda_z_global, lda_z_local;

  PetscFunctionBegin;
  /* Get the row of Q corresponding to this vertex */
  PetscCall(MatGetRow(Q, vertex_idx, &ncols, &cols, &vals));
  PetscCheck(ncols == Q_NUM_OBSERVATIONS_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Vertex %" PetscInt_FMT " has %" PetscInt_FMT " local observations, expected %" PetscInt_FMT, vertex_idx, ncols, (PetscInt)Q_NUM_OBSERVATIONS_MAX);

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

  /* Extract local observations WITHOUT weighting
     Note: Localization weights affect covariances, not observation values */
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
  PetscInt          m, n_vertices, i_vertex;
  PetscScalar       scale, sqrt_m_minus_1;
  PetscBool         reallocate = PETSC_FALSE;

  /* Local analysis workspace */
  Mat       Z_local, S_local, T_sqrt_local, w_ones_local, G_local;
  Vec       y_local, y_mean_local, delta_scaled_local, r_inv_sqrt_local;
  Vec       w_local, s_transpose_delta;
  PetscInt *local_obs_indices = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);
  /* Validate ensemble size */
  m          = da->ensemble_size;
  n_vertices = da->state_size;
  PetscCheck(m > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be > 1, got %" PetscInt_FMT, m);

  impl           = (PetscDALETKFData *)da->data;
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
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

    PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &impl->mean));
    PetscCall(VecSetSizes(impl->mean, PETSC_DECIDE, da->state_size));
    PetscCall(VecSetFromOptions(impl->mean));

    PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &impl->y_mean));
    PetscCall(VecSetSizes(impl->y_mean, PETSC_DECIDE, da->obs_size));
    PetscCall(VecSetFromOptions(impl->y_mean));

    PetscCall(VecDuplicate(impl->y_mean, &impl->delta_scaled));
    PetscCall(VecDuplicate(da->obs_error_var, &impl->r_inv_sqrt));

    /* Create Z matrix (obs_size x m) */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, &impl->Z));
    PetscCall(MatSetUp(impl->Z));

    /* Create S matrix (same layout as Z) */
    PetscCall(MatDuplicate(impl->Z, MAT_DO_NOT_COPY_VALUES, &impl->S));

    /* Create T_sqrt matrix (m x m) - usually small */
    if (da->U) {
      PetscCall(MatDuplicate(da->U, MAT_DO_NOT_COPY_VALUES, &impl->T_sqrt));
    } else {
      PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &impl->T_sqrt));
      PetscCall(MatSetUp(impl->T_sqrt));
    }

    /* Create w_ones matrix (m x m) */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &impl->w_ones));
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
  {
    MatReuse scall = MAT_INITIAL_MATRIX;
    if (impl->Z) {
      PetscInt z_rows, z_cols;
      PetscCall(MatGetSize(impl->Z, &z_rows, &z_cols));
      if (z_rows == da->obs_size && z_cols == m) scall = MAT_REUSE_MATRIX;
      else {
        PetscCall(MatDestroy(&impl->Z));
        scall = MAT_INITIAL_MATRIX;
      }
    }
    PetscCall(MatMatMult(H, da->ensemble, scall, PETSC_DEFAULT, &impl->Z));
  }

  /* Compute GLOBAL observation mean y_mean = H * x_mean */
  PetscCall(MatMult(H, impl->mean, impl->y_mean));

  /* ===================================================================== */
  /* Compute GLOBAL R^{-1/2} (assumes diagonal R) */
  /* ===================================================================== */
  PetscCall(VecCopy(da->obs_error_var, impl->r_inv_sqrt));
  PetscCall(VecSqrtAbs(impl->r_inv_sqrt));
  PetscCall(VecReciprocal(impl->r_inv_sqrt));

  /* ===================================================================== */
  /* Create local analysis workspace (p_local x m matrices and vectors) */
  /* ===================================================================== */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, impl->p_local, m, NULL, &Z_local));
  PetscCall(MatSetUp(Z_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, impl->p_local, m, NULL, &S_local));
  PetscCall(MatSetUp(S_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &T_sqrt_local));
  PetscCall(MatSetUp(T_sqrt_local));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, m, NULL, &w_ones_local));
  PetscCall(MatSetUp(w_ones_local));
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
    PetscCall(ExtractLocalObservations(impl->Q, i_vertex, impl->Z, observation, impl->y_mean, impl->r_inv_sqrt, m, Z_local, y_local, y_mean_local, r_inv_sqrt_local, local_obs_indices));

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

    /* Compute local square-root transform: T_sqrt_local = T_local^{-1/2} * U */
    PetscCall(PetscDAApplySqrtTInverse(da, da->U, T_sqrt_local));

    /* Form local transform G_local = w_local * 1' + sqrt(m - 1) * T_sqrt_local * U */
    PetscCall(BroadcastWeightVector(w_local, m, w_ones_local));
    PetscCall(MatCopy(T_sqrt_local, G_local, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(G_local, sqrt_m_minus_1));
    PetscCall(MatAXPY(G_local, 1.0, w_ones_local, SAME_NONZERO_PATTERN));

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
  PetscCall(MatDestroy(&w_ones_local));
  PetscCall(MatDestroy(&T_sqrt_local));
  PetscCall(MatDestroy(&S_local));
  PetscCall(MatDestroy(&Z_local));
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

  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &temp));
  PetscCall(VecSetSizes(temp, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(temp));

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

/*@
  PetscDALETKFSetSqrtType - Selects the reduced-space square-root algorithm used during the LETKF analysis.

  Logically Collective

  Input Parameters:
+ da   - the `PetscDA` object
- type - either `PETSCDA_SQRT_CHOLESKY` or `PETSCDA_SQRT_EIGEN`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDALETKFGetSqrtType()`, `PetscDALETKFAnalysis()`
@*/
PetscErrorCode PetscDALETKFSetSqrtType(PetscDA da, PetscDASqrtType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCheck(type == PETSCDA_SQRT_CHOLESKY || type == PETSCDA_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDA square-root type %" PetscInt_FMT, (PetscInt)type);

  PetscCall(PetscDASetSqrtType(da, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDALETKFGetSqrtType - Retrieves the current square-root implementation configured for the LETKF analysis.

  Not Collective

  Input Parameters:
. da - the `PetscDA` object

  Output Parameter:
. type - on output, the configured `PetscDASqrtType`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDALETKFSetSqrtType()`
@*/
PetscErrorCode PetscDALETKFGetSqrtType(PetscDA da, PetscDASqrtType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscDAGetSqrtType(da, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
  PetscCheck(nrows == da->state_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix rows (%" PetscInt_FMT ") must match state size (%" PetscInt_FMT ")", nrows, da->state_size);
  PetscCheck(ncols == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Localization matrix columns (%" PetscInt_FMT ") must match observation size (%" PetscInt_FMT ")", ncols, da->obs_size);

  /* Validate that each row has exactly Q_NUM_OBSERVATIONS_MAX non-zero entries */
  for (i = 0; i < nrows; i++) {
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(Q, i, &nnz, &cols, &vals));
    PetscCheck(nnz == Q_NUM_OBSERVATIONS_MAX, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Row %" PetscInt_FMT " has %" PetscInt_FMT " non-zeros, expected %" PetscInt_FMT, i, nnz, (PetscInt)Q_NUM_OBSERVATIONS_MAX);
    PetscCall(MatRestoreRow(Q, i, &nnz, &cols, &vals));
  }

  /* Store the localization matrix */
  PetscCall(MatDestroy(&impl->Q));
  PetscCall(MatDuplicate(Q, MAT_COPY_VALUES, &impl->Q));
  impl->p_local = Q_NUM_OBSERVATIONS_MAX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFView - Views a `PetscDALETKF` and its implementation-specific data structure.

  Input Parameters:
+ da     - the `PetscDA` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: internal

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
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Local observations per vertex: %" PetscInt_FMT "\n", impl->p_local));
    if (impl->Q) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization matrix: set\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization matrix: not set\n"));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFInitialize - Installs the LETKF-specific operations on a newly created `PetscDA` object.

  Input Parameter:
. da - the `PetscDA` object to configure

  Level: internal

.seealso: [](ch_da), `PetscDA`, `PetscDARegister()`, `PetscDALETKFAnalysis()`
*/
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
  da->ops->setfromoptions   = PetscDASetFromOptions_DASLETKF;

  /* Initialize default values */
  impl->p_local = Q_NUM_OBSERVATIONS_MAX;
  impl->Q       = NULL;

  /* Register the method for setting localization */
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "PetscDALETKFSetLocalization_C", PetscDALETKFSetLocalization_LETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}