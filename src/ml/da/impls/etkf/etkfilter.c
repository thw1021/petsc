#include "petscda.h"
#include <petsc/private/daimpl.h>
#include <petscblaslapack.h>

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
} PetscDAETKFData;

static PetscFunctionList PetscDAETKFSqrtList           = NULL;
static PetscBool         PetscDAETKFPackageInitialized = PETSC_FALSE;

/* Tolerance for matrix square root verification in debug mode */
#define MATRIX_SQRT_TOLERANCE_FACTOR (100.0 * PETSC_MACHINE_EPSILON)

/* ========================================================================== */
/*                    Helper Functions for ETKF Analysis                     */
/* ========================================================================== */

/*
  ComputeObservationEnsemble - Applies observation operator H to each ensemble member (Alg 6.4 line 3-4)

  Input Parameters:
+ da                   - the PetscDA context
. observation_operator - user-supplied routine H(x, y; ctx)
- obs_ctx              - optional context for observation_operator

  Output Parameter:
. Z - observation ensemble matrix (obs_size x ensemble_size)
*/
static PetscErrorCode ComputeObservationEnsemble(PetscDA da, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx, Mat Z)
{
  /* Ensemble and observation-related vectors */
  Vec ensemble_member_in, observation_out;
  /* Loop counter and ensemble size */
  PetscInt ensemble_idx, ensemble_size;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(Z, MAT_CLASSID, 4);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Ensemble matrix not initialized");

  /* Extract and validate ensemble size */
  ensemble_size = da->ensemble_size;
  PetscCheck(ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be positive, got %" PetscInt_FMT, ensemble_size);
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Observation size must be positive, got %" PetscInt_FMT, da->obs_size);

  /* Apply observation operator H to each ensemble member: Z_i = H(E_i) */
  for (ensemble_idx = 0; ensemble_idx < ensemble_size; ensemble_idx++) {
    /* Get read-only access to ensemble member */
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, ensemble_idx, &ensemble_member_in));

    /* Get write access to the corresponding column in Z.
       Directly writing to Z avoids allocating a temporary vector and performing a copy. */
    PetscCall(MatDenseGetColumnVecWrite(Z, ensemble_idx, &observation_out));

    /* Apply observation operator: observation_out = H(ensemble_member_in) */
    PetscCall(observation_operator(ensemble_member_in, observation_out, obs_ctx));

    /* Restore vectors */
    PetscCall(MatDenseRestoreColumnVecWrite(Z, ensemble_idx, &observation_out));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, ensemble_idx, &ensemble_member_in));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeNormalizedInnovationMatrix - Computes S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1) [Alg 6.4 line 5]

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
+ w - weight vector of size m (analysis weights from ETKF update)
- m - ensemble size (number of columns to replicate, must equal vector size)

  Output Parameter:
. w_ones - m x m dense matrix where each column is a copy of w (i.e., w * 1^T)

  Notes:
  This function constructs the broadcast matrix w * 1^T, where w is the m-dimensional
  weight vector and 1 is an m-dimensional vector of ones. This matrix is a fundamental
  component in the ETKF transform\: G = w * 1^T + sqrt(m-1) * T^{1/2} * U.

  The implementation uses direct array access for performance, avoiding the overhead of
  repeated vector wrapping and copying. This is particularly efficient for dense matrices
  where memory is contiguous column-wise.

  Complexity\: O(m^2) time and memory.

  Level\: developer

.seealso\: [`PetscDAETKFAnalysis()`](etkfilter.c:837), [`MatDenseGetArrayWrite()`](petscmat.h)
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

/*
  UpdateEnsembleWithTransform - Updates ensemble via ETKF transform: E = mean * 1' + X * G [Alg 6.4 line 9]

  Input Parameters:
+ mean     - ensemble mean vector (size state_size), must be initialized
. X        - scaled anomaly matrix (state_size x ensemble_size), X = (E - mean*1')/sqrt(m-1)
. G        - ETKF transform matrix (ensemble_size x ensemble_size), G = w*1' + sqrt(m-1)*T^{1/2}*U
. m        - ensemble size (number of columns in ensemble), must be > 0
- ensemble - ensemble matrix to update in-place (state_size x ensemble_size)

  Notes:
  This function performs the final step (Step 10) of the ETKF analysis algorithm from
  Asch, M., Bocquet, M., and Nodet, M., transforming the forecast ensemble into the analysis ensemble.
  The operation E^a = mean + X * G is computed using matrix-matrix multiplication followed
  by column-wise addition to efficiently handle large state spaces.

  Error Handling\:
  - Validates all input dimensions for consistency
  - Checks for positive ensemble size
  - Ensures proper matrix/vector initialization
  - Handles parallel assembly correctly

  Performance Considerations\:
  - Memory\: Creates one temporary matrix X_G of size (state_size x m)
  - Time complexity\: O(state_size * m^2) for matrix multiply + O(state_size * m) for additions
  - Optimization\: Uses direct array access for dense matrices to avoid Vec overhead
  - Parallel\: Fully parallelizable across both matrix multiply and column updates

  Level\: developer

.seealso\: [`PetscDAETKFAnalysis()`](etkfilter.c:522), [`ComputeAnalysisWeights()`](etkfilter.c:178),
[`BroadcastWeightVector()`](etkfilter.c:245), [`MatMatMult()`](petscmat.h), [`MatDenseGetArrayRead()`](petscmat.h)
*/
static PetscErrorCode UpdateEnsembleWithTransform(Vec mean, Mat X, Mat G, PetscInt m, Mat ensemble)
{
  Mat                X_G;
  const PetscScalar *xg_array, *mean_array;
  PetscScalar       *ens_array;
  PetscInt           x_rows, x_cols, g_rows, g_cols, ens_rows, ens_cols;
  PetscInt           n_local_ens, n_local_xg, mean_local_size;
  PetscInt           lda_ens, lda_xg;
  PetscInt           mean_size, i, j;

  PetscFunctionBegin;
  /* Validate input parameters for correct types and null pointers */
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(G, MAT_CLASSID, 3);
  PetscValidLogicalCollectiveInt(X, m, 4);
  PetscValidHeaderSpecific(ensemble, MAT_CLASSID, 5);

  /* Retrieve and validate matrix dimensions for compatibility */
  PetscCall(MatGetSize(X, &x_rows, &x_cols));
  PetscCall(MatGetSize(G, &g_rows, &g_cols));
  PetscCall(MatGetSize(ensemble, &ens_rows, &ens_cols));
  PetscCall(VecGetSize(mean, &mean_size));

  /* Verify dimension consistency across all inputs */
  PetscCheck(x_cols == m, PetscObjectComm((PetscObject)X), PETSC_ERR_ARG_INCOMP, "Anomaly matrix X columns (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", x_cols, m);
  PetscCheck(g_rows == m, PetscObjectComm((PetscObject)G), PETSC_ERR_ARG_INCOMP, "Transform matrix G rows (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", g_rows, m);
  PetscCheck(g_cols == m, PetscObjectComm((PetscObject)G), PETSC_ERR_ARG_INCOMP, "Transform matrix G must be square, got %" PetscInt_FMT " x %" PetscInt_FMT, g_rows, g_cols);
  PetscCheck(ens_rows == x_rows, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_INCOMP, "Ensemble rows (%" PetscInt_FMT ") must match anomaly matrix X rows (%" PetscInt_FMT ")", ens_rows, x_rows);
  PetscCheck(ens_cols == m, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_INCOMP, "Ensemble columns (%" PetscInt_FMT ") must equal ensemble size (%" PetscInt_FMT ")", ens_cols, m);
  PetscCheck(mean_size == x_rows, PetscObjectComm((PetscObject)mean), PETSC_ERR_ARG_INCOMP, "Mean vector size (%" PetscInt_FMT ") must match state size (%" PetscInt_FMT ")", mean_size, x_rows);

  /* Compute transformed anomaly matrix: X_G = X * G (state_size x m) */
  PetscCall(MatMatMult(X, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X_G));

  /* Access underlying data arrays for direct performance access
     This avoids creating/destroying m Vec objects and calling VecWAXPY m times. */
  PetscCall(MatDenseGetArrayRead(X_G, &xg_array));
  PetscCall(MatDenseGetArrayWrite(ensemble, &ens_array));
  PetscCall(VecGetArrayRead(mean, &mean_array));

  /* Get local dimensions and strides for array traversal */
  PetscCall(MatGetLocalSize(ensemble, &n_local_ens, NULL));
  PetscCall(MatGetLocalSize(X_G, &n_local_xg, NULL));
  PetscCall(VecGetLocalSize(mean, &mean_local_size));
  PetscCall(MatDenseGetLDA(ensemble, &lda_ens));
  PetscCall(MatDenseGetLDA(X_G, &lda_xg));

  /* Verify local dimensions match before direct array access */
  PetscCheck(n_local_ens == n_local_xg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local row size mismatch: ensemble (%" PetscInt_FMT ") vs X_G (%" PetscInt_FMT ")", n_local_ens, n_local_xg);
  PetscCheck(n_local_ens == mean_local_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local row size mismatch: ensemble (%" PetscInt_FMT ") vs mean (%" PetscInt_FMT ")", n_local_ens, mean_local_size);

  /* Update each ensemble member: E_ij = (XG)_ij + mean_i
     Loop over columns (j) and rows (i) of the local data block */
  for (j = 0; j < m; j++) {
    const PetscScalar *xg_col  = xg_array + j * lda_xg;
    PetscScalar       *ens_col = ens_array + j * lda_ens;
    for (i = 0; i < n_local_ens; i++) ens_col[i] = xg_col[i] + mean_array[i];
  }

  /* Restore arrays and finalize assembly */
  PetscCall(VecRestoreArrayRead(mean, &mean_array));
  PetscCall(MatDenseRestoreArrayWrite(ensemble, &ens_array));
  PetscCall(MatDenseRestoreArrayRead(X_G, &xg_array));

  PetscCall(MatAssemblyBegin(ensemble, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ensemble, MAT_FINAL_ASSEMBLY));

  /* Clean up temporary transformed anomaly matrix */
  PetscCall(MatDestroy(&X_G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                       Square Root Type Setters                            */
/* ========================================================================== */

static PetscErrorCode PetscDAETKFSetSqrt_Cholesky(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscDAETKFSetSqrtType(da, PETSCDA_SQRT_CHOLESKY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDAETKFSetSqrt_Eigen(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscDAETKFSetSqrtType(da, PETSCDA_SQRT_EIGEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                       ETKF Implementation Lifecycle                       */
/* ========================================================================== */

static PetscErrorCode PetscDAETKFDestroy(PetscDA da)
{
  PetscDAETKFData *impl;

  PetscFunctionBegin;
  if (da->data) {
    impl = (PetscDAETKFData *)da->data;
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

    PetscCall(PetscFree(da->data));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASetFromOptions_DASETKF(PetscDA da, PetscOptionItems *PetscOptions)
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
  PetscCall(PetscOptionsFList("-da_etkf_sqrt_type", "Matrix square root factorization", "PetscDAETKFSetSqrtType", PetscDAETKFSqrtList, defaultType, typeName, sizeof(typeName), &set));
  if (set) {
    PetscCall(PetscFunctionListFind(PetscDAETKFSqrtList, typeName, &setter));
    PetscCheck(setter, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDAETKF square-root type \"%s\"", typeName);
    PetscCall((*setter)(da));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Package Initialization                           */
/* ========================================================================== */

/*@C
  PetscDAETKFInitializePackage - This function initializes everything in the `PetscDAETKF` package. It is called from `TSInitializePackage()`.

  Level: developer

.seealso: [](ch_ts), `PetscInitialize()`, `PetscDAETKFFinalizePackage()`
@*/
PetscErrorCode PetscDAETKFInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDAETKFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDAETKFPackageInitialized = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&PetscDAETKFSqrtList, "cholesky", PetscDAETKFSetSqrt_Cholesky));
  PetscCall(PetscFunctionListAdd(&PetscDAETKFSqrtList, "eigen", PetscDAETKFSetSqrt_Eigen));
  PetscCall(PetscRegisterFinalize(PetscDAETKFFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDAETKFFinalizePackage - This function destroys everything in the `PetscDAETKF` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ts), `PetscFinalize()`, `PetscDAETKFInitializePackage()`
@*/
PetscErrorCode PetscDAETKFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscDAETKFPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&PetscDAETKFSqrtList));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* ========================================================================== */
/*                    ETKF Analysis Algorithm (Algorithm 6.4)                */
/* ========================================================================== */

/*
  PetscDAETKFAnalysis - Performs the ensemble transform Kalman filter (ETKF) analysis defined by Algorithm 6.4 in Asch, M., Bocquet, M., and Nodet, M.

  Collective

  Input Parameters:
+ da                   - the `PetscDA` context owning the forecast ensemble and buffers
. observation          - observation vector `y`
. observation_operator - user-supplied routine `H(x, y; ctx)` that maps a state to observation space
- obs_ctx              - optional context for `observation_operator`

  Notes:
  The implementation follows the book's deterministic ETKF steps\:
  Step 1 computes the state mean, Step 2 the state anomalies, Steps 3-4 build the normalized innovation statistics,
  Step 5 assembles the reduced-space inverse, Step 6 forms the analysis weights, Steps 7-9 construct the square-root
  transform, and Step 10 applies the transform to refresh every ensemble member.

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFApplyModel()`, `PetscDAComputeMean()`,
`PetscDAComputeAnomalies()`
*/
static PetscErrorCode PetscDAETKFAnalysis(PetscDA da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  PetscDAETKFData *impl;
  Mat              X;
  PetscInt         m;
  PetscScalar      inv_m, scale, sqrt_m_minus_1;
  PetscBool        reallocate = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);
  /* Validate ensemble size */
  m = da->ensemble_size;
  PetscCheck(m > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be > 1, got %" PetscInt_FMT, m);

  impl           = (PetscDAETKFData *)da->data;
  inv_m          = 1.0 / ((PetscScalar)m);
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  PetscCall(PetscInfo(da, "squaroot type %s, %d ensembles\n", (da->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky", (int)m));

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
    /* Note: T_sqrt will hold the result of applying T^{-1/2} to U. Since U is typically Identity, T_sqrt is m x m */
    /* Wait, da->U is usually identity but could be general. Assuming da->U is m x m. */
    if (da->U) {
      PetscCall(MatDuplicate(da->U, MAT_DO_NOT_COPY_VALUES, &impl->T_sqrt));
    } else {
      /* Fallback if U is not set yet (though it should be) */
      /* Assuming U would be compatible with T (m x m) */
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
  PetscCall(PetscDAComputeEnsembleMean(da, impl->mean));

  /* X = (E - x_mean * 1') / sqrt(m - 1) */
  /* Note: PetscDAComputeAnomalies creates a NEW matrix X every time.
     We should probably optimize this too in the future, but for now we follow the API. */
  PetscCall(PetscDAComputeAnomalies(da, impl->mean, &X));

  /* ===================================================================== */
  /* Alg 6.4 line 3-4: Compute observation ensemble Z = H(x_i^f)         */
  /* ===================================================================== */
  PetscCall(ComputeObservationEnsemble(da, observation_operator, obs_ctx, impl->Z));

  /* Compute observation mean y_mean = (1/m) * sum(Z_i) */
  PetscCall(MatGetRowSum(impl->Z, impl->y_mean));
  PetscCall(VecScale(impl->y_mean, inv_m));

  /* ===================================================================== */
  /* Alg 6.4 line 5-6: Build normalized innovation statistics            */
  /* ===================================================================== */
  /* Compute R^{-1/2} (assumes diagonal R) */
  PetscCall(VecCopy(da->obs_error_var, impl->r_inv_sqrt));
  PetscCall(VecSqrtAbs(impl->r_inv_sqrt));
  PetscCall(VecReciprocal(impl->r_inv_sqrt));

  /* S = R^{-1/2} * (Z - y_mean * 1') / sqrt(m - 1) */
  PetscCall(ComputeNormalizedInnovationMatrix(impl->Z, impl->y_mean, impl->r_inv_sqrt, m, scale, impl->S));

  /* delta_scaled = R^{-1/2} * (y^o - y_mean) [Alg 6.4 line 6] */
  PetscCall(VecWAXPY(impl->delta_scaled, -1.0, impl->y_mean, observation));
  PetscCall(VecPointwiseMult(impl->delta_scaled, impl->delta_scaled, impl->r_inv_sqrt));

  /* ===================================================================== */
  /* Alg 6.4 line 7: Factor T = (I + S^T S) and store factorization (T is not inverted here but solved later) */
  /* ===================================================================== */
  /* Apply inflation: T = (1/rho) * I + S^T S */
  /* Note: Inflation is handled inside PetscDATFactor by shifting the diagonal of T */
  PetscCall(PetscDATFactor(da, impl->S));

  /* ===================================================================== */
  /* Alg 6.4 line 8: Compute analysis weights w = T^{-1} * S^T * delta_scaled */
  /* ===================================================================== */
  {
    Vec s_transpose_delta;
    PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &s_transpose_delta));
    PetscCall(VecSetSizes(s_transpose_delta, PETSC_DECIDE, m));
    PetscCall(VecSetFromOptions(s_transpose_delta));
    PetscCall(MatMultTranspose(impl->S, impl->delta_scaled, s_transpose_delta));

    /* w is created/reused inside if not passed? No, we need to handle w */
    if (!impl->w) PetscCall(VecDuplicate(s_transpose_delta, &impl->w));

    PetscCall(PetscDAApplyTInverse(da, s_transpose_delta, impl->w));
    PetscCall(VecDestroy(&s_transpose_delta));
  }

  /* ===================================================================== */
  /* Alg 6.4 line 9: Compute square-root transform T^{1/2} U = T^{1/2}     */
  /* ===================================================================== */
  PetscCall(PetscDAApplySqrtTInverse(da, da->U, impl->T_sqrt));

  /* ===================================================================== */
  /* Alg 6.4 line 9: Form transform G = w * 1' + sqrt(m - 1) * T^{1/2} * U */
  /* ===================================================================== */
  /* w_ones = w * 1' (broadcast weight vector to all columns) */
  PetscCall(BroadcastWeightVector(impl->w, m, impl->w_ones));

  /* sqrt(m-1) T^{1/2} * U */
  PetscCall(MatScale(impl->T_sqrt, sqrt_m_minus_1));

  /* G = w_ones + T_sqrt_U */
  /* We can accumulate into w_ones to act as G */
  PetscCall(MatAXPY(impl->w_ones, 1.0, impl->T_sqrt, SAME_NONZERO_PATTERN));

  /* ===================================================================== */
  /* Alg 6.4 line 9: Update ensemble E = x_mean * 1' + X * G             */
  /* ===================================================================== */
  PetscCall(UpdateEnsembleWithTransform(impl->mean, X, impl->w_ones, m, da->ensemble));

  /* Cleanup temporary X matrix */
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Model Propagation                                */
/* ========================================================================== */

/*
  PetscDAETKFApplyModel - Advances each ensemble member through the user-supplied
  nonlinear model (Algorithm 6.4, Step 10 forecast propagation).

  Collective

  Input Parameters:
+ da        - the `PetscDA` context that stores the ensemble
. model     - routine that evaluates the model `f(x, xnew; ctx)`
- model_ctx - optional context for `model`

  Notes:
  This routine overwrites every ensemble column with the model result supplied by `model`.
  It is typically called immediately after `PetscDAETKFAnalysis()` to start the next forecast cycle.

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFAnalysis()`
*/
static PetscErrorCode PetscDAETKFApplyModel(PetscDA da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
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
  PetscDAETKFSetSqrtType - Selects the reduced-space square-root algorithm used during the ETKF analysis.

  Logically Collective

  Input Parameters:
+ da   - the `PetscDA` object
- type - either `PETSCDA_SQRT_CHOLESKY` or `PETSCDA_SQRT_EIGEN`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFGetSqrtType()`, `PetscDAETKFAnalysis()`
@*/
PetscErrorCode PetscDAETKFSetSqrtType(PetscDA da, PetscDASqrtType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCheck(type == PETSCDA_SQRT_CHOLESKY || type == PETSCDA_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDA square-root type %" PetscInt_FMT, (PetscInt)type);

  PetscCall(PetscDASetSqrtType(da, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAETKFGetSqrtType - Retrieves the current square-root implementation configured for the ETKF analysis.

  Not Collective

  Input Parameters:
. da - the `PetscDA` object

  Output Parameter:
. type - on output, the configured `PetscDASqrtType`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFSetSqrtType()`
@*/
PetscErrorCode PetscDAETKFGetSqrtType(PetscDA da, PetscDASqrtType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscDAGetSqrtType(da, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAETKFView - Views a `PetscDAETKF` and its implementation-specific data structure.

  Input Parameters:
+ da     - the `PetscDA` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: internal

.seealso: [](ch_da), `PetscDAViewFromOptions()`
*/
static PetscErrorCode PetscDAETKFView(PetscDA da, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDAETKF Object:\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Square root type: %s\n", (da->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAETKFInitialize - Installs the ETKF-specific operations on a newly created `PetscDA` object.

  Input Parameter:
. da - the `PetscDA` object to configure

  Level: internal

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFRegister()`, `PetscDAETKFAnalysis()`
*/
static PetscErrorCode PetscDAETKFInitialize(PetscDA da)
{
  PetscDAETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscCall(PetscNew(&impl));

  da->data                  = impl;
  da->ops->analysis         = PetscDAETKFAnalysis;
  da->ops->applymodel       = PetscDAETKFApplyModel;
  da->ops->computemean      = NULL;
  da->ops->computeanomalies = NULL;
  da->ops->destroy          = PetscDAETKFDestroy;
  da->ops->view             = PetscDAETKFView;
  da->ops->setfromoptions   = PetscDASetFromOptions_DASETKF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDAETKFRegister(void)
{
  PetscFunctionBegin;
  PetscCall(PetscDARegister(PETSCDAETKF, PetscDAETKFInitialize));
  PetscCall(PetscDAETKFInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}
