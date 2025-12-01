#include "petscdataassimilator.h"
#include <petsc/private/dataassimilatorimpl.h>

typedef struct {
  PetscDataAssimilatorETKFSqrtType sqrt_type;
} PetscDataAssimilatorETKFData;

static PetscFunctionList PetscDataAssimilatorETKFSqrtList           = NULL;
static PetscBool         PetscDataAssimilatorETKFPackageInitialized = PETSC_FALSE;

/* ========================================================================== */
/*                    Helper Functions for ETKF Analysis                     */
/* ========================================================================== */

/*
  ComputeObservationEnsemble - Applies observation operator H to each ensemble member

  Input Parameters:
+ da                   - the PetscDataAssimilator context
. observation_operator - user-supplied routine H(x, y; ctx)
- obs_ctx              - optional context for observation_operator

  Output Parameter:
. Z - observation ensemble matrix (obs_size x ensemble_size)
*/
static PetscErrorCode ComputeObservationEnsemble(PetscDataAssimilator da, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx, Mat *Z)
{
  /* Ensemble and observation-related vectors */
  Vec ensemble_member_in, observation_out, temp_observation;
  /* Loop counter and ensemble size */
  PetscInt ensemble_idx, ensemble_size;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(Z, 4);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Ensemble matrix not initialized");

  /* Extract and validate ensemble size */
  ensemble_size = da->ensemble_size;
  PetscCheck(ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be positive, got %" PetscInt_FMT, ensemble_size);
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Observation size must be positive, got %" PetscInt_FMT, da->obs_size);

  /* Create output observation ensemble matrix Z (obs_size x ensemble_size) */
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, ensemble_size, NULL, Z));
  PetscCall(MatSetUp(*Z));

  /* Create temporary observation vector for operator application */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &temp_observation));
  PetscCall(VecSetSizes(temp_observation, PETSC_DECIDE, da->obs_size));
  PetscCall(VecSetFromOptions(temp_observation));

  /* Apply observation operator H to each ensemble member: Z_i = H(E_i) */
  for (ensemble_idx = 0; ensemble_idx < ensemble_size; ensemble_idx++) {
    /* Get read-only access to ensemble member */
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, ensemble_idx, &ensemble_member_in));

    /* Apply observation operator: temp_observation = H(ensemble_member_in) */
    PetscCall(observation_operator(ensemble_member_in, temp_observation, obs_ctx));

    /* Release ensemble member */
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, ensemble_idx, &ensemble_member_in));

    /* Write result to observation ensemble matrix */
    PetscCall(MatDenseGetColumnVecWrite(*Z, ensemble_idx, &observation_out));
    PetscCall(VecCopy(temp_observation, observation_out));
    PetscCall(MatDenseRestoreColumnVecWrite(*Z, ensemble_idx, &observation_out));
  }

  /* Clean up temporary vector */
  PetscCall(VecDestroy(&temp_observation));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeInverseSquareRootR - Computes R^{-1/2} from diagonal observation error variance

  Input Parameter:
. obs_error_var - diagonal of observation error covariance matrix R

  Output Parameter:
. r_inv_sqrt - element-wise R^{-1/2}
*/
static PetscErrorCode ComputeInverseSquareRootR(Vec obs_error_var, Vec *r_inv_sqrt)
{
  PetscFunctionBegin;
  PetscCall(VecDuplicate(obs_error_var, r_inv_sqrt));
  PetscCall(VecCopy(obs_error_var, *r_inv_sqrt));
  PetscCall(VecSqrtAbs(*r_inv_sqrt));
  PetscCall(VecReciprocal(*r_inv_sqrt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
static PetscErrorCode ComputeNormalizedInnovationMatrix(Mat Z, Vec y_mean, Vec r_inv_sqrt, PetscInt m, PetscScalar scale, Mat *S)
{
  Vec      z_column, s_column;
  PetscInt ensemble_idx, obs_size, obs_size_local, z_cols;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(Z, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(y_mean, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(r_inv_sqrt, VEC_CLASSID, 3);
  PetscValidLogicalCollectiveInt(Z, m, 4);
  PetscValidLogicalCollectiveScalar(Z, scale, 5);
  PetscAssertPointer(S, 6);
  PetscCheck(m > 0, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive, got %" PetscInt_FMT, m);

  /* Get observation size from input matrix Z and validate dimensions */
  PetscCall(MatGetSize(Z, &obs_size, &z_cols));
  PetscCall(MatGetLocalSize(Z, &obs_size_local, NULL));
  PetscCheck(z_cols == m, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Matrix Z has %" PetscInt_FMT " columns but ensemble size is %" PetscInt_FMT, z_cols, m);

  /* Verify vector dimensions match observation size */
  PetscInt y_mean_size, r_inv_sqrt_size;
  PetscCall(VecGetSize(y_mean, &y_mean_size));
  PetscCall(VecGetSize(r_inv_sqrt, &r_inv_sqrt_size));
  PetscCheck(y_mean_size == obs_size, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Vector y_mean size %" PetscInt_FMT " does not match observation size %" PetscInt_FMT, y_mean_size, obs_size);
  PetscCheck(r_inv_sqrt_size == obs_size, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Vector r_inv_sqrt size %" PetscInt_FMT " does not match observation size %" PetscInt_FMT, r_inv_sqrt_size, obs_size);

  /* Create output matrix S with same distribution as Z */
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)Z), obs_size_local, PETSC_DECIDE, obs_size, m, NULL, S));
  PetscCall(MatSetUp(*S));

  /* Compute normalized innovation for each ensemble member */
  for (ensemble_idx = 0; ensemble_idx < m; ensemble_idx++) {
    /* Get read-only access to Z column and write access to S column */
    PetscCall(MatDenseGetColumnVecRead(Z, ensemble_idx, &z_column));
    PetscCall(MatDenseGetColumnVecWrite(*S, ensemble_idx, &s_column));

    /* S_i = R^{-1/2} * (Z_i - y_mean) / sqrt(m-1) */
    /* First: s_column = Z_i - y_mean */
    PetscCall(VecWAXPY(s_column, -1.0, y_mean, z_column));
    /* Second: s_column *= scale (= 1/sqrt(m-1)) */
    PetscCall(VecScale(s_column, scale));
    /* Third: s_column = s_column .* r_inv_sqrt (element-wise multiplication) */
    PetscCall(VecPointwiseMult(s_column, s_column, r_inv_sqrt));

    /* Restore column vectors */
    PetscCall(MatDenseRestoreColumnVecRead(Z, ensemble_idx, &z_column));
    PetscCall(MatDenseRestoreColumnVecWrite(*S, ensemble_idx, &s_column));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeAnalysisWeights - Computes analysis weight vector w = T * S^T * delta_scaled

  Input Parameters:
+ T            - reduced-space inverse matrix (I + S^T S)^{-1}, m x m SPD matrix
. S            - normalized innovation matrix, obs_size x m
- delta_scaled - scaled innovation vector R^{-1/2}(y^o - y_mean), size obs_size

  Output Parameter:
. w - analysis weight vector, size m

  Notes:
  This function performs the two-step matrix-vector multiplication\:
  1. Compute S^T * delta_scaled (projection to reduced space)
  2. Apply inverse T to get weights w = T * (S^T * delta_scaled)

  The result is used in Step 6 of Algorithm 6.4 from Law, Stuart, and Zygalakis
  to compute the ensemble analysis weights.
*/
static PetscErrorCode ComputeAnalysisWeights(Mat T, Mat S, Vec delta_scaled, Vec *w)
{
  Vec      s_transpose_delta;
  PetscInt t_rows, t_cols, s_rows, s_cols, delta_size;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(T, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(S, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(delta_scaled, VEC_CLASSID, 3);
  PetscAssertPointer(w, 4);

  /* Validate matrix dimensions for compatibility */
  PetscCall(MatGetSize(T, &t_rows, &t_cols));
  PetscCall(MatGetSize(S, &s_rows, &s_cols));
  PetscCall(VecGetSize(delta_scaled, &delta_size));

  PetscCheck(t_rows == t_cols, PetscObjectComm((PetscObject)T), PETSC_ERR_ARG_INCOMP, "Matrix T must be square, got %" PetscInt_FMT " x %" PetscInt_FMT, t_rows, t_cols);
  PetscCheck(s_cols == t_rows, PetscObjectComm((PetscObject)T), PETSC_ERR_ARG_INCOMP, "Matrix S columns (%" PetscInt_FMT ") must match T rows (%" PetscInt_FMT ")", s_cols, t_rows);
  PetscCheck(delta_size == s_rows, PetscObjectComm((PetscObject)S), PETSC_ERR_ARG_INCOMP, "Vector delta_scaled size (%" PetscInt_FMT ") must match S rows (%" PetscInt_FMT ")", delta_size, s_rows);

  /* Step 1: Compute S^T * delta_scaled (project innovation to reduced space) */
  PetscCall(MatCreateVecs(T, NULL, &s_transpose_delta));
  PetscCall(MatMultTranspose(S, delta_scaled, s_transpose_delta));

  /* Step 2: Compute w = T * (S^T * delta_scaled) (apply inverse transform) */
  PetscCall(MatCreateVecs(T, w, NULL));
  PetscCall(MatMult(T, s_transpose_delta, *w));

  /* Cleanup intermediate vector */
  PetscCall(VecDestroy(&s_transpose_delta));
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

  The implementation uses column-wise vector operations following PETSc best practices,
  which ensures\:
  - Proper parallel distribution and communication
  - Efficient memory access patterns
  - Consistency with other PETSc matrix operations

  Complexity\: O(m^2) time for sequential replication, O(m^2/p) parallel time where p
  is the number of processes. Memory\: O(m^2) total, O(m^2/p) per process.

  Alternative Implementation Considered\:
  Direct array manipulation via [`MatDenseGetArrayWrite()`](petscmat.h) could reduce function
  call overhead but sacrifices code clarity and parallel safety. The current approach
  prioritizes maintainability and correctness.

  Level\: developer

.seealso\: [`ComputeAnalysisWeights()`](etkfilter.c:178), [`MatDenseGetColumnVecWrite()`](petscmat.h), [`PetscDataAssimilatorETKFAnalysis()`](etkfilter.c:522)
*/
static PetscErrorCode BroadcastWeightVector(Vec w, PetscInt m, Mat *w_ones)
{
  Vec      col_out;
  PetscInt i, w_size, w_size_local;
  PetscInt mat_rows_local, mat_cols_local;

  PetscFunctionBegin;
  /* Validate input parameters for type correctness and null pointers */
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(w, m, 2);
  PetscAssertPointer(w_ones, 3);

  /* Validate ensemble size is physically meaningful */
  PetscCheck(m > 0, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive for broadcasting, got %" PetscInt_FMT, m);
  PetscCheck(m < PETSC_MAX_INT / m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m = %" PetscInt_FMT " too large, m*m would overflow", m);

  /* Verify vector dimensions match ensemble size (global and local) */
  PetscCall(VecGetSize(w, &w_size));
  PetscCall(VecGetLocalSize(w, &w_size_local));
  PetscCheck(w_size == m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_INCOMP, "Weight vector global size (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ")", w_size, m);

  /* Create dense matrix with parallel distribution matching weight vector */
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)w), w_size_local, PETSC_DECIDE, m, m, NULL, w_ones));
  PetscCall(MatSetUp(*w_ones));

  /* Verify consistent parallel layout between vector and matrix */
  PetscCall(MatGetLocalSize(*w_ones, &mat_rows_local, &mat_cols_local));
  PetscCheck(mat_rows_local == w_size_local, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix row distribution (%" PetscInt_FMT ") inconsistent with vector distribution (%" PetscInt_FMT ")", mat_rows_local, w_size_local);

  /* Broadcast weight vector: copy w to each of the m columns */
  for (i = 0; i < m; i++) {
    /* Obtain write access to column i of the output matrix */
    PetscCall(MatDenseGetColumnVecWrite(*w_ones, i, &col_out));

    /* Copy weight vector to this column: w_ones[:, i] = w */
    PetscCall(VecCopy(w, col_out));

    /* Release column vector, marking it as modified */
    PetscCall(MatDenseRestoreColumnVecWrite(*w_ones, i, &col_out));
  }

  /* Finalize matrix assembly for parallel consistency and communication */
  PetscCall(MatAssemblyBegin(*w_ones, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*w_ones, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  UpdateEnsembleWithTransform - Updates ensemble via ETKF transform: E = mean * 1' + X * G

  Input Parameters:
+ mean     - ensemble mean vector (size state_size)
. X        - scaled anomaly matrix (state_size x ensemble_size), X = (E - mean*1')/sqrt(m-1)
. G        - ETKF transform matrix (ensemble_size x ensemble_size), G = w*1' + sqrt(m-1)*T^{1/2}*U
. m        - ensemble size (number of columns in ensemble)
- ensemble - ensemble matrix to update in-place (state_size x ensemble_size)

  Notes:
  This function performs the final step (Step 10) of the ETKF analysis algorithm from
  Law, Stuart, and Zygalakis, transforming the forecast ensemble into the analysis ensemble.
  The operation E^a = mean + X * G is computed column-wise to efficiently handle large
  state spaces while minimizing memory footprint.

  The computation proceeds as\:
  1. Compute the transformed anomaly matrix\: X_G = X * G (state_size x m)
  2. For each ensemble member i\: E^a_i = mean + (X*G)_i

  Mathematical Background\:
  The transform ensures the analysis ensemble has the correct\:
  - Mean\: E[E^a] = mean + X*G*1 = mean (since G is constructed such that G*1 = 0)
  - Covariance\: Cov(E^a) = X*G*G^T*X^T which approximates the analysis covariance P^a

  Performance Considerations\:
  - Memory\: Creates one temporary matrix X_G of size (state_size x m)
  - Time complexity\: O(state_size * m^2) for matrix multiply + O(state_size * m) for additions
  - Parallel\: Fully parallelizable across both matrix multiply and column updates

  Level\: developer

.seealso\: [`PetscDataAssimilatorETKFAnalysis()`](etkfilter.c:522), [`ComputeAnalysisWeights()`](etkfilter.c:178),
[`BroadcastWeightVector()`](etkfilter.c:245), [`MatMatMult()`](petscmat.h), [`VecWAXPY()`](petscvec.h)
*/
static PetscErrorCode UpdateEnsembleWithTransform(Vec mean, Mat X, Mat G, PetscInt m, Mat ensemble)
{
  Mat      X_G;
  Vec      col_in, col_out;
  PetscInt ensemble_idx;
  PetscInt x_rows, x_cols, g_rows, g_cols, ens_rows, ens_cols;
  PetscInt mean_size;

  PetscFunctionBegin;
  /* Validate input parameters for correct types and null pointers */
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(G, MAT_CLASSID, 3);
  PetscValidLogicalCollectiveInt(X, m, 4);
  PetscValidHeaderSpecific(ensemble, MAT_CLASSID, 5);

  /* Validate ensemble size is physically meaningful */
  PetscCheck(m > 0, PetscObjectComm((PetscObject)ensemble), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive, got %" PetscInt_FMT, m);

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

  /* Update each ensemble member: E_i = mean + (X*G)_i */
  for (ensemble_idx = 0; ensemble_idx < m; ensemble_idx++) {
    /* Get read-only access to transformed anomaly column */
    PetscCall(MatDenseGetColumnVecRead(X_G, ensemble_idx, &col_in));

    /* Get write access to ensemble column for in-place update */
    PetscCall(MatDenseGetColumnVecWrite(ensemble, ensemble_idx, &col_out));

    /* Compute: ensemble[:, i] = mean + (X*G)[:, i]
       VecWAXPY performs: col_out = 1.0 * col_in + mean */
    PetscCall(VecWAXPY(col_out, 1.0, mean, col_in));

    /* Restore column vectors, marking ensemble column as modified */
    PetscCall(MatDenseRestoreColumnVecRead(X_G, ensemble_idx, &col_in));
    PetscCall(MatDenseRestoreColumnVecWrite(ensemble, ensemble_idx, &col_out));
  }

  /* Finalize ensemble matrix assembly for parallel consistency */
  PetscCall(MatAssemblyBegin(ensemble, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ensemble, MAT_FINAL_ASSEMBLY));

  /* Clean up temporary transformed anomaly matrix */
  PetscCall(MatDestroy(&X_G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                       Square Root Type Setters                            */
/* ========================================================================== */

static PetscErrorCode PetscDataAssimilatorETKFSetSqrt_Cholesky(PetscDataAssimilator da)
{
  PetscFunctionBegin;
  PetscCall(PetscDataAssimilatorETKFSetSqrtType(da, PETSCDAETKF_SQRT_CHOLESKY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDataAssimilatorETKFSetSqrt_Eigen(PetscDataAssimilator da)
{
  PetscFunctionBegin;
  PetscCall(PetscDataAssimilatorETKFSetSqrtType(da, PETSCDAETKF_SQRT_EIGEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                       ETKF Implementation Lifecycle                       */
/* ========================================================================== */

static PetscErrorCode PetscDataAssimilatorETKFDestroy(PetscDataAssimilator da)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(da->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDataAssimilatorSetFromOptions_DAETKF(PetscDataAssimilator da, PetscOptionItems *PetscOptions)
{
  PetscDataAssimilatorETKFData *impl;
  PetscOptionItems              PetscOptionsObject;
  const char                   *defaultType;
  char                          typeName[256];
  PetscBool                     set              = PETSC_FALSE;
  PetscErrorCode (*setter)(PetscDataAssimilator) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssert(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDataAssimilator data structure not initialized");

  impl               = (PetscDataAssimilatorETKFData *)da->data;
  PetscOptionsObject = PetscOptions ? *PetscOptions : NULL;

  defaultType = (impl->sqrt_type == PETSCDAETKF_SQRT_EIGEN) ? "eigen" : "cholesky";
  PetscCall(PetscStrncpy(typeName, defaultType, sizeof(typeName)));
  PetscCall(PetscOptionsFList("-dataassimilator_etkf_sqrt_type", "Matrix square root factorization", "PetscDataAssimilatorETKFSetSqrtType", PetscDataAssimilatorETKFSqrtList, defaultType, typeName, sizeof(typeName), &set));
  if (set) {
    PetscCall(PetscFunctionListFind(PetscDataAssimilatorETKFSqrtList, typeName, &setter));
    PetscCheck(setter, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDataAssimilatorETKF square-root type \"%s\"", typeName);
    PetscCall((*setter)(da));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Package Initialization                           */
/* ========================================================================== */

/*@C
  PetscDataAssimilatorETKFInitializePackage - This function initializes everything in the `PetscDataAssimilatorETKF` package. It is called from `TSInitializePackage()`.

  Level: developer

.seealso: [](ch_ts), `PetscInitialize()`, `PetscDataAssimilatorETKFFinalizePackage()`
@*/
PetscErrorCode PetscDataAssimilatorETKFInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDataAssimilatorETKFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDataAssimilatorETKFPackageInitialized = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&PetscDataAssimilatorETKFSqrtList, "cholesky", PetscDataAssimilatorETKFSetSqrt_Cholesky));
  PetscCall(PetscFunctionListAdd(&PetscDataAssimilatorETKFSqrtList, "eigen", PetscDataAssimilatorETKFSetSqrt_Eigen));
  PetscCall(PetscRegisterFinalize(PetscDataAssimilatorETKFFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDataAssimilatorETKFFinalizePackage - This function destroys everything in the `PetscDataAssimilatorETKF` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ts), `PetscFinalize()`, `PetscDataAssimilatorETKFInitializePackage()`
@*/
PetscErrorCode PetscDataAssimilatorETKFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscDataAssimilatorETKFPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&PetscDataAssimilatorETKFSqrtList));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                    ETKF Analysis Algorithm (Algorithm 6.4)                */
/*                                                                            */
/*  This section implements the deterministic Ensemble Transform Kalman      */
/*  Filter (ETKF) analysis step as described in Law, Stuart, and Zygalakis.  */
/*  The algorithm updates the forecast ensemble to produce the analysis      */
/*  ensemble by combining observations with the forecast using optimal       */
/*  Kalman filtering in the reduced ensemble subspace.                       */
/*                                                                            */
/*  Key computational steps:                                                 */
/*    - Observation space projection via H(x) operator                       */
/*    - Reduced-space Kalman gain computation via (I + S^T S)^{-1}          */
/*    - Matrix square root factorization (Cholesky or eigendecomposition)   */
/*    - Transform application to update ensemble members                     */
/*                                                                            */
/* ========================================================================== */

/*
  PetscDataAssimilatorETKFAnalysis - Performs the ensemble transform Kalman filter (ETKF) analysis defined by Algorithm 6.4 in Law, Stuart, and Zygalakis.

  Collective

  Input Parameters:
+ da                   - the `PetscDataAssimilator` context owning the forecast ensemble and buffers
. observation          - observation vector `y`
. observation_operator - user-supplied routine `H(x, y; ctx)` that maps a state to observation space
- obs_ctx              - optional context for `observation_operator`

  Notes:
  The implementation follows the book's deterministic ETKF steps\:
  Step 1 computes the state mean, Step 2 the state anomalies, Steps 3-4 build the normalized innovation statistics,
  Step 5 assembles the reduced-space inverse, Step 6 forms the analysis weights, Steps 7-9 construct the square-root
  transform, and Step 10 applies the transform to refresh every ensemble member.

  Level: advanced

.seealso: [](ch_dataassimilator), `PetscDataAssimilator`, `PetscDataAssimilatorETKFApplyModel()`, `PetscDataAssimilatorComputeMean()`,
`PetscDataAssimilatorComputeAnomalies()`
*/
static PetscErrorCode PetscDataAssimilatorETKFAnalysis(PetscDataAssimilator da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  PetscDataAssimilatorETKFData *impl;
  Vec                           mean, y_mean, delta_scaled, w, r_inv_sqrt;
  Mat                           X, Z, S, T, T_sqrt, I_m, w_ones, T_sqrt_U, G;
  PetscInt                      m;
  PetscScalar                   inv_m, scale, sqrt_m_minus_1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);

  /* Validate ensemble size */
  m = da->ensemble_size;
  PetscCheck(m > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be > 1, got %" PetscInt_FMT, m);

  impl           = (PetscDataAssimilatorETKFData *)da->data;
  inv_m          = 1.0 / ((PetscScalar)m);
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));

  /* ===================================================================== */
  /* Step 1-2: Compute ensemble mean and scaled anomalies                */
  /* ===================================================================== */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(mean));
  PetscCall(PetscDataAssimilatorComputeMean(da, mean));

  /* X = (E - x_mean * 1') / sqrt(m - 1) */
  PetscCall(PetscDataAssimilatorComputeAnomalies(da, &X));

  /* ===================================================================== */
  /* Step 3: Compute observation ensemble Z = H(x_i^f)                   */
  /* ===================================================================== */
  PetscCall(ComputeObservationEnsemble(da, observation_operator, obs_ctx, &Z));

  /* Compute observation mean y_mean = (1/m) * sum(Z_i) */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &y_mean));
  PetscCall(VecSetSizes(y_mean, PETSC_DECIDE, da->obs_size));
  PetscCall(VecSetFromOptions(y_mean));
  PetscCall(MatGetRowSum(Z, y_mean));
  PetscCall(VecScale(y_mean, inv_m));

  /* ===================================================================== */
  /* Step 4: Build normalized innovation statistics                       */
  /* ===================================================================== */
  /* Compute R^{-1/2} (assumes diagonal R) */
  PetscCall(ComputeInverseSquareRootR(da->obs_error_var, &r_inv_sqrt));

  /* S = R^{-1/2} * (Z - y_mean * 1') / sqrt(m - 1) */
  PetscCall(ComputeNormalizedInnovationMatrix(Z, y_mean, r_inv_sqrt, m, scale, &S));

  /* delta_scaled = R^{-1/2} * (y^o - y_mean) */
  PetscCall(VecDuplicate(y_mean, &delta_scaled));
  PetscCall(VecWAXPY(delta_scaled, -1.0, y_mean, observation));
  PetscCall(VecPointwiseMult(delta_scaled, delta_scaled, r_inv_sqrt));

  /* ===================================================================== */
  /* Step 5: Form T = (I + S^T S)^{-1} via solve (not explicit inverse)  */
  /* ===================================================================== */
  /* Create identity matrix I_m */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)da->ensemble), &I_m));
  PetscCall(MatSetSizes(I_m, PETSC_DECIDE, PETSC_DECIDE, m, m));
  PetscCall(MatSetType(I_m, MATDENSE));
  PetscCall(MatSetUp(I_m));
  PetscCall(MatZeroEntries(I_m));
  PetscCall(MatShift(I_m, 1.0));
  PetscCall(MatAssemblyBegin(I_m, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(I_m, MAT_FINAL_ASSEMBLY));

  /* T_temp = I + S^T S */
  PetscCall(MatTransposeMatMult(S, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T));
  PetscCall(MatAXPY(T, 1.0, I_m, SAME_NONZERO_PATTERN));

  /* Solve (I + S^T S) X = I for T = X using Cholesky (T is SPD) */
  Mat T_factored;
  PetscCall(MatDuplicate(T, MAT_COPY_VALUES, &T_factored));
  PetscCall(MatCholeskyFactor(T_factored, NULL, NULL));
  PetscCall(MatMatSolve(T_factored, I_m, T));
  PetscCall(MatDestroy(&T_factored));

  /* ===================================================================== */
  /* Step 6: Compute analysis weights w = T * S^T * delta_scaled         */
  /* ===================================================================== */
  PetscCall(ComputeAnalysisWeights(T, S, delta_scaled, &w));

  /* ===================================================================== */
  /* Step 7-9: Compute square-root transform T_sqrt                       */
  /* ===================================================================== */
  switch (impl->sqrt_type) {
  case PETSCDAETKF_SQRT_CHOLESKY:
    PetscCall(PetscDataAssimilatorCholeskySqrt_Private(T, &T_sqrt));
    break;
  case PETSCDAETKF_SQRT_EIGEN:
    PetscCall(PetscDataAssimilatorSymmetricEigenSqrt_Private(T, &T_sqrt));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDataAssimilatorETKF square-root type %" PetscInt_FMT, (PetscInt)impl->sqrt_type);
  }

  /* ===================================================================== */
  /* Step 8: Form transform G = w * 1' + sqrt(m - 1) * T^{1/2} * U       */
  /* ===================================================================== */
  /* w_ones = w * 1' (broadcast weight vector to all columns) */
  PetscCall(BroadcastWeightVector(w, m, &w_ones));

  /* T_sqrt_U = T^{1/2} * U */
  PetscCall(MatMatMult(T_sqrt, da->U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_sqrt_U));
  PetscCall(MatScale(T_sqrt_U, sqrt_m_minus_1));

  /* G = w_ones + T_sqrt_U */
  PetscCall(MatDuplicate(w_ones, MAT_COPY_VALUES, &G));
  PetscCall(MatAXPY(G, 1.0, T_sqrt_U, DIFFERENT_NONZERO_PATTERN));

  /* ===================================================================== */
  /* Step 10: Update ensemble E = x_mean * 1' + X * G                    */
  /* ===================================================================== */
  PetscCall(UpdateEnsembleWithTransform(mean, X, G, m, da->ensemble));

  /* Cleanup */
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&y_mean));
  PetscCall(VecDestroy(&delta_scaled));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&r_inv_sqrt));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Z));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&I_m));
  PetscCall(MatDestroy(&T));
  PetscCall(MatDestroy(&T_sqrt));
  PetscCall(MatDestroy(&w_ones));
  PetscCall(MatDestroy(&T_sqrt_U));
  PetscCall(MatDestroy(&G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Model Propagation                                */
/* ========================================================================== */

/*
  PetscDataAssimilatorETKFApplyModel - Advances each ensemble member through the user-supplied
  nonlinear model (Algorithm 6.4, Step 10 forecast propagation).

  Collective

  Input Parameters:
+ da        - the `PetscDataAssimilator` context that stores the ensemble
. model     - routine that evaluates the model `f(x, xnew; ctx)`
- model_ctx - optional context for `model`

  Notes:
  This routine overwrites every ensemble column with the model result supplied by `model`.
  It is typically called immediately after `PetscDataAssimilatorETKFAnalysis()` to start the next forecast cycle.

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilator`, `PetscDataAssimilatorETKFAnalysis()`
*/
static PetscErrorCode PetscDataAssimilatorETKFApplyModel(PetscDataAssimilator da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
{
  Vec      col_in, col_out, temp;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);

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
  PetscDataAssimilatorETKFSetSqrtType - Selects the reduced-space square-root algorithm used during the ETKF analysis.

  Logically Collective

  Input Parameters:
+ da   - the `PetscDataAssimilator` object
- type - either `PETSCDAETKF_SQRT_CHOLESKY` or `PETSCDAETKF_SQRT_EIGEN`

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilator`, `PetscDataAssimilatorETKFGetSqrtType()`, `PetscDataAssimilatorETKFAnalysis()`
@*/
PetscErrorCode PetscDataAssimilatorETKFSetSqrtType(PetscDataAssimilator da, PetscDataAssimilatorETKFSqrtType type)
{
  PetscDataAssimilatorETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDataAssimilator data structure not initialized");
  PetscCheck(type == PETSCDAETKF_SQRT_CHOLESKY || type == PETSCDAETKF_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDataAssimilatorETKF square-root type %" PetscInt_FMT, (PetscInt)type);

  impl            = (PetscDataAssimilatorETKFData *)da->data;
  impl->sqrt_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorETKFGetSqrtType - Retrieves the current square-root implementation configured for the ETKF analysis.

  Not Collective

  Input Parameters:
. da - the `PetscDataAssimilator` object

  Output Parameter:
. type - on output, the configured `PetscDataAssimilatorETKFSqrtType`

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilator`, `PetscDataAssimilatorETKFSetSqrtType()`
@*/
PetscErrorCode PetscDataAssimilatorETKFGetSqrtType(PetscDataAssimilator da, PetscDataAssimilatorETKFSqrtType *type)
{
  PetscDataAssimilatorETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDataAssimilator data structure not initialized");

  impl  = (PetscDataAssimilatorETKFData *)da->data;
  *type = impl->sqrt_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDataAssimilatorETKFView - Views a `PetscDataAssimilatorETKF` and its implementation-specific data structure.

  Collective

  Input Parameters:
+ da     - the `PetscDataAssimilator` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorViewFromOptions()`
*/
static PetscErrorCode PetscDataAssimilatorETKFView(PetscDataAssimilator da, PetscViewer viewer)
{
  PetscBool                     iascii;
  PetscDataAssimilatorETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)da), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(da, 1, viewer, 2);

  impl = (PetscDataAssimilatorETKFData *)da->data;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDataAssimilatorETKF Object:\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Square root type: %s\n", (impl->sqrt_type == PETSCDAETKF_SQRT_EIGEN) ? "eigen" : "cholesky"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDataAssimilatorETKFInitialize - Installs the ETKF-specific operations on a newly created `PetscDataAssimilator` object.

  Collective

  Input Parameter:
. da - the `PetscDataAssimilator` object to configure

  Level: developer

.seealso: [](ch_dataassimilator), `PetscDataAssimilator`, `PetscDataAssimilatorETKFRegister()`, `PetscDataAssimilatorETKFAnalysis()`
*/
static PetscErrorCode PetscDataAssimilatorETKFInitialize(PetscDataAssimilator da)
{
  PetscDataAssimilatorETKFData *impl;

  PetscFunctionBegin;
  if (da->data) PetscCall(PetscFree(da->data)); // can be called twice (useful?)
  PetscCall(PetscNew(&impl));
  impl->sqrt_type = PETSCDAETKF_SQRT_CHOLESKY;

  da->data                  = impl;
  da->ops->analysis         = PetscDataAssimilatorETKFAnalysis;
  da->ops->applymodel       = PetscDataAssimilatorETKFApplyModel;
  da->ops->computemean      = NULL;
  da->ops->computeanomalies = NULL;
  da->ops->destroy          = PetscDataAssimilatorETKFDestroy;
  da->ops->view             = PetscDataAssimilatorETKFView;
  da->ops->setfromoptions   = PetscDataAssimilatorSetFromOptions_DAETKF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDataAssimilatorETKFRegister(void)
{
  PetscFunctionBegin;
  PetscCall(PetscDataAssimilatorRegister(PETSCDAETKF, PetscDataAssimilatorETKFInitialize));
  PetscCall(PetscDataAssimilatorETKFInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}
