#include "petscdas.h"
#include <petsc/private/dasimpl.h>
#include <petscblaslapack.h>

typedef struct {
  PetscDASETKFSqrtType sqrt_type;
  Mat                  V;               /* Eigen vectors (LAPACK column-major storage) */
  Mat                  L_cholesky;      /* Lower triangular Cholesky factor */
  Vec                  sqrt_eigen_vals; /* Square root of eigen values */
  Mat                  I_StS;           /* T = I + S^T * S matrix */
} PetscDASETKFData;

static PetscFunctionList PetscDASETKFSqrtList           = NULL;
static PetscBool         PetscDASETKFPackageInitialized = PETSC_FALSE;

/* ========================================================================== */
/*                    Helper Functions for ETKF Analysis                     */
/* ========================================================================== */

/*
  ComputeObservationEnsemble - Applies observation operator H to each ensemble member (Alg 6.4 line 3-4)

  Input Parameters:
+ da                   - the PetscDAS context
. observation_operator - user-supplied routine H(x, y; ctx)
- obs_ctx              - optional context for observation_operator

  Output Parameter:
. Z - observation ensemble matrix (obs_size x ensemble_size)
*/
static PetscErrorCode ComputeObservationEnsemble(PetscDAS da, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx, Mat *Z)
{
  /* Ensemble and observation-related vectors */
  Vec ensemble_member_in, observation_out;
  /* Loop counter and ensemble size */
  PetscInt ensemble_idx, ensemble_size;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscAssertPointer(Z, 4);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Ensemble matrix not initialized");

  /* Extract and validate ensemble size */
  ensemble_size = da->ensemble_size;
  PetscCheck(ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be positive, got %" PetscInt_FMT, ensemble_size);
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Observation size must be positive, got %" PetscInt_FMT, da->obs_size);

  /* Create output observation ensemble matrix Z (obs_size x ensemble_size) */
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, ensemble_size, NULL, Z));
  PetscCall(MatSetUp(*Z));

  /* Apply observation operator H to each ensemble member: Z_i = H(E_i) */
  for (ensemble_idx = 0; ensemble_idx < ensemble_size; ensemble_idx++) {
    /* Get read-only access to ensemble member */
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, ensemble_idx, &ensemble_member_in));

    /* Get write access to the corresponding column in Z.
       Directly writing to Z avoids allocating a temporary vector and performing a copy. */
    PetscCall(MatDenseGetColumnVecWrite(*Z, ensemble_idx, &observation_out));

    /* Apply observation operator: observation_out = H(ensemble_member_in) */
    PetscCall(observation_operator(ensemble_member_in, observation_out, obs_ctx));

    /* Restore vectors */
    PetscCall(MatDenseRestoreColumnVecWrite(*Z, ensemble_idx, &observation_out));
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

.seealso\: [`PetscDASETKFAnalysis()`](etkfilter.c:837), [`MatDenseGetArrayWrite()`](petscmat.h)
*/
static PetscErrorCode BroadcastWeightVector(Vec w, PetscInt m, Mat *w_ones)
{
  const PetscScalar *w_array;
  PetscScalar       *mat_array;
  PetscInt           w_size, w_size_local, mat_rows_local, mat_cols_local;
  PetscInt           i, lda;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(w, m, 2);
  PetscAssertPointer(w_ones, 3);
  PetscCheck(m > 0, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive for broadcasting, got %" PetscInt_FMT, m);
  /* Check for potential overflow in matrix size calculation */
  PetscCheck(m <= PETSC_MAX_INT / m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m = %" PetscInt_FMT " too large", m);

  /* Verify dimensions */
  PetscCall(VecGetSize(w, &w_size));
  PetscCall(VecGetLocalSize(w, &w_size_local));
  PetscCheck(w_size == m, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_INCOMP, "Weight vector global size (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ")", w_size, m);

  /* Create dense matrix with parallel distribution matching weight vector */
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)w), w_size_local, PETSC_DECIDE, m, m, NULL, w_ones));
  PetscCall(MatSetUp(*w_ones));

  /* Verify consistent parallel layout between vector and matrix */
  PetscCall(MatGetLocalSize(*w_ones, &mat_rows_local, &mat_cols_local));
  PetscCheck(mat_rows_local == w_size_local, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix row distribution (%" PetscInt_FMT ") inconsistent with vector distribution (%" PetscInt_FMT ")", mat_rows_local, w_size_local);
  PetscCheck(mat_cols_local == m, PetscObjectComm((PetscObject)w), PETSC_ERR_PLIB, "Matrix local columns (%" PetscInt_FMT ") must equal global columns m (%" PetscInt_FMT ") for MPIDense", mat_cols_local, m);

  /* Access raw arrays for efficient broadcasting */
  PetscCall(VecGetArrayRead(w, &w_array));
  PetscCall(MatDenseGetArrayWrite(*w_ones, &mat_array));
  PetscCall(MatDenseGetLDA(*w_ones, &lda));

  /* Copy w to each column of w_ones */
  /* Note: MatDense uses column-major storage. We copy the vector w into each column. */
  for (i = 0; i < m; i++) { PetscCall(PetscArraycpy(mat_array + i * lda, w_array, w_size_local)); }

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayWrite(*w_ones, &mat_array));
  PetscCall(VecRestoreArrayRead(w, &w_array));

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(*w_ones, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*w_ones, MAT_FINAL_ASSEMBLY));
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
  Law, Stuart, and Zygalakis, transforming the forecast ensemble into the analysis ensemble.
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

.seealso\: [`PetscDASETKFAnalysis()`](etkfilter.c:522), [`ComputeAnalysisWeights()`](etkfilter.c:178),
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
    for (i = 0; i < n_local_ens; i++) { ens_col[i] = xg_col[i] + mean_array[i]; }
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

static PetscErrorCode PetscDASETKFSetSqrt_Cholesky(PetscDAS da)
{
  PetscFunctionBegin;
  PetscCall(PetscDASETKFSetSqrtType(da, PETSCDASETKF_SQRT_CHOLESKY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASETKFSetSqrt_Eigen(PetscDAS da)
{
  PetscFunctionBegin;
  PetscCall(PetscDASETKFSetSqrtType(da, PETSCDASETKF_SQRT_EIGEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                       ETKF Implementation Lifecycle                       */
/* ========================================================================== */

static PetscErrorCode PetscDASETKFDestroy(PetscDAS da)
{
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  if (da->data) {
    impl = (PetscDASETKFData *)da->data;
    PetscCall(MatDestroy(&impl->V));
    PetscCall(MatDestroy(&impl->L_cholesky));
    PetscCall(VecDestroy(&impl->sqrt_eigen_vals));
    PetscCall(MatDestroy(&impl->I_StS));
    PetscCall(PetscFree(da->data));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASSetFromOptions_DASETKF(PetscDAS da, PetscOptionItems *PetscOptions)
{
  PetscDASETKFData *impl;
  PetscOptionItems  PetscOptionsObject;
  const char       *defaultType;
  char              typeName[256];
  PetscBool         set              = PETSC_FALSE;
  PetscErrorCode (*setter)(PetscDAS) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscAssert(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDAS data structure not initialized");

  impl               = (PetscDASETKFData *)da->data;
  PetscOptionsObject = PetscOptions ? *PetscOptions : NULL;

  defaultType = (impl->sqrt_type == PETSCDASETKF_SQRT_EIGEN) ? "eigen" : "cholesky";
  PetscCall(PetscStrncpy(typeName, defaultType, sizeof(typeName)));
  PetscCall(PetscOptionsFList("-das_etkf_sqrt_type", "Matrix square root factorization", "PetscDASETKFSetSqrtType", PetscDASETKFSqrtList, defaultType, typeName, sizeof(typeName), &set));
  if (set) {
    PetscCall(PetscFunctionListFind(PetscDASETKFSqrtList, typeName, &setter));
    PetscCheck(setter, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDASETKF square-root type \"%s\"", typeName);
    PetscCall((*setter)(da));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Package Initialization                           */
/* ========================================================================== */

/*@C
  PetscDASETKFInitializePackage - This function initializes everything in the `PetscDASETKF` package. It is called from `TSInitializePackage()`.

  Level: developer

.seealso: [](ch_ts), `PetscInitialize()`, `PetscDASETKFFinalizePackage()`
@*/
PetscErrorCode PetscDASETKFInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDASETKFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDASETKFPackageInitialized = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&PetscDASETKFSqrtList, "cholesky", PetscDASETKFSetSqrt_Cholesky));
  PetscCall(PetscFunctionListAdd(&PetscDASETKFSqrtList, "eigen", PetscDASETKFSetSqrt_Eigen));
  PetscCall(PetscRegisterFinalize(PetscDASETKFFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDASETKFFinalizePackage - This function destroys everything in the `PetscDASETKF` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ts), `PetscFinalize()`, `PetscDASETKFInitializePackage()`
@*/
PetscErrorCode PetscDASETKFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscDASETKFPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&PetscDASETKFSqrtList));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*         T-Matrix Factorization and Application Methods [Alg 6.4 line 7]    */
/* ========================================================================== */

/*
  PetscDASTFactor - Compute and store factorization of T matrix

  Input Parameters:
+ da - the PetscDAS context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  This function computes T = I + S^T * S and stores its factorization based on
  sqrt_type. For CHOLESKY mode, it computes and stores the lower triangular
  Cholesky factor. For EIGEN mode, it computes and stores the
  eigenvectors and square root of eigenvalues.
*/
static PetscErrorCode PetscDASTFactor(PetscDAS da, Mat S)
{
  PetscDASETKFData *impl;
  PetscInt          s_rows, s_cols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(S, MAT_CLASSID, 2);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDAS data structure not initialized");

  impl = (PetscDASETKFData *)da->data;

  /* Validate matrix dimensions for safety */
  PetscCall(MatGetSize(S, &s_rows, &s_cols));
  PetscCheck(s_cols > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Innovation matrix S must have positive columns, got %" PetscInt_FMT, s_cols);

  /* Clean up any previous factorization to avoid memory leaks */
  PetscCall(MatDestroy(&impl->I_StS));
  PetscCall(MatDestroy(&impl->V));
  PetscCall(MatDestroy(&impl->L_cholesky));
  PetscCall(VecDestroy(&impl->sqrt_eigen_vals));

  /* Compute T = I + S^T * S */
  PetscCall(MatTransposeMatMult(S, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &impl->I_StS));
  PetscCall(MatShift(impl->I_StS, 1.0));

  /* Compute and store factorization based on sqrt_type */
  switch (impl->sqrt_type) {
  case PETSCDASETKF_SQRT_CHOLESKY: {
    /* Compute Cholesky factorization: T = L * L^T using LAPACK */
    PetscBLASInt n, lda, info;
    PetscScalar *a_array;
    PetscInt     m_T, N_T;

    PetscCall(MatDuplicate(impl->I_StS, MAT_COPY_VALUES, &impl->L_cholesky));

    /* Get matrix dimensions and convert to BLAS int */
    PetscCall(MatGetSize(impl->L_cholesky, &m_T, &N_T));
    PetscCheck(m_T == N_T, PetscObjectComm((PetscObject)impl->L_cholesky), PETSC_ERR_ARG_WRONG, "Matrix must be square for Cholesky");
    PetscCall(PetscBLASIntCast(N_T, &n));
    lda = n;

    /* Get array from dense matrix */
    PetscCall(MatDenseGetArray(impl->L_cholesky, &a_array));

    /* Compute Cholesky factorization: A = L * L^T (lower triangular) */
    LAPACKpotrf_("L", &n, a_array, &lda, &info);
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky factorization (xPOTRF): info=%d", (int)info);

    /* Restore array and finalize matrix */
    PetscCall(MatDenseRestoreArray(impl->L_cholesky, &a_array));
    PetscCall(MatAssemblyBegin(impl->L_cholesky, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(impl->L_cholesky, MAT_FINAL_ASSEMBLY));
    break;
  }
  case PETSCDASETKF_SQRT_EIGEN: {
    /* Compute eigendecomposition: T = V * D * V^T */
    Vec          eigen_vals;
    PetscBLASInt n, lda, lwork, info;
    PetscScalar *a_array, *work, *eig_array;
    PetscInt     m_V, N_V;
#if defined(PETSC_USE_COMPLEX)
    PetscReal *rwork = NULL;
#endif

    PetscCall(MatCreateVecs(impl->I_StS, &eigen_vals, NULL));
    PetscCall(MatDuplicate(impl->I_StS, MAT_COPY_VALUES, &impl->V));

    /* Inline eigendecomposition computation using LAPACK syev */
    /* Get matrix dimensions */
    PetscCall(MatGetSize(impl->V, &m_V, &N_V));
    PetscCheck(m_V == N_V, PetscObjectComm((PetscObject)impl->V), PETSC_ERR_ARG_WRONG, "Matrix must be square");

    /* Convert to BLAS int */
    PetscCall(PetscBLASIntCast(N_V, &n));
    lda = n;

    /* Get array from dense matrix */
    PetscCall(MatDenseGetArray(impl->V, &a_array));

    /* Get array from eigenvalue vector */
    PetscCall(VecGetArray(eigen_vals, &eig_array));

    /* Query optimal workspace size */
    lwork = -1;
    PetscCall(PetscMalloc1(1, &work));

#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscMalloc1(PetscMax(1, 3 * n - 2), &rwork));
    LAPACKsyev_("V", "U", &n, a_array, &lda, (PetscReal *)eig_array, work, &lwork, rwork, &info);
#else
    LAPACKsyev_("V", "U", &n, a_array, &lda, eig_array, work, &lwork, &info);
#endif
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine xSYEV work query: info=%d", (int)info);

    /* Allocate workspace */
    lwork = (PetscBLASInt)PetscRealPart(work[0]);
    PetscCall(PetscFree(work));
    PetscCall(PetscMalloc1(lwork, &work));

    /* Compute eigendecomposition */
#if defined(PETSC_USE_COMPLEX)
    LAPACKsyev_("V", "U", &n, a_array, &lda, (PetscReal *)eig_array, work, &lwork, rwork, &info);
    PetscCall(PetscFree(rwork));
#else
    LAPACKsyev_("V", "U", &n, a_array, &lda, eig_array, work, &lwork, &info);
#endif
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine xSYEV: info=%d", (int)info);

    /* Cleanup */
    PetscCall(PetscFree(work));
    PetscCall(VecRestoreArray(eigen_vals, &eig_array));
    PetscCall(MatDenseRestoreArray(impl->V, &a_array));
    PetscCall(MatAssemblyBegin(impl->V, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(impl->V, MAT_FINAL_ASSEMBLY));

    /* Compute and store sqrt(eigenvalues) for later use */
    PetscCall(VecDuplicate(eigen_vals, &impl->sqrt_eigen_vals));
    PetscCall(VecCopy(eigen_vals, impl->sqrt_eigen_vals));
    PetscCall(VecSqrtAbs(impl->sqrt_eigen_vals));

    /* Debug verification: Ensure V * D * V^T == T */
    if (PetscDefined(USE_DEBUG)) {
      PetscReal norm_T, norm_diff;
      Mat       V_D, VDVt;

      /* Compute D * V^T by scaling rows */
      PetscCall(MatDuplicate(impl->V, MAT_COPY_VALUES, &V_D));
      PetscCall(MatDiagonalScale(V_D, NULL, eigen_vals));

      /* Compute V * D * V^T */
      PetscCall(MatMatTransposeMult(V_D, impl->V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VDVt));

      /* Compute ||V*D*V^T - T|| / ||T|| */
      PetscCall(MatAXPY(VDVt, -1.0, impl->I_StS, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(impl->I_StS, NORM_FROBENIUS, &norm_T));
      PetscCall(MatNorm(VDVt, NORM_FROBENIUS, &norm_diff));

      if (norm_T > 0) {
        PetscReal relative_error = norm_diff / norm_T;
        PetscCheck(relative_error <= 1.e-10, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "Eigendecomposition verification failed: ||V*D*V^T - T||/||T|| = %g > 1e-10", (double)relative_error);
      }

      /* Cleanup debug matrices */
      PetscCall(MatDestroy(&V_D));
      PetscCall(MatDestroy(&VDVt));
    }

    PetscCall(VecDestroy(&eigen_vals));
    break;
  }
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDASETKF square-root type %" PetscInt_FMT, (PetscInt)impl->sqrt_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDASApplyTInverse - Apply T^{-1} to a vector [Alg 6.4 line 8]

  Input Parameters:
+ da - the PetscDAS context
- sdel  - input vector S^T-delta

  Output Parameter:
. w - output vector w = T^{-1} * sdel

  Notes:_
  This function applies the inverse of T = I + S^T S using the stored
  factorization. For CHOLESKY mode, it uses triangular solves. For EIGEN mode,
  it uses the eigendecomposition.
*/
static PetscErrorCode PetscDASApplyTInverse(PetscDAS da, Vec sdel, Vec *w)
{
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(sdel, VEC_CLASSID, 2);
  PetscAssertPointer(w, 3);

  impl = (PetscDASETKFData *)da->data;
  PetscCheck(impl->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "T matrix not factored. Call PetscDASTFactor first");

  PetscCall(VecDuplicate(sdel, w));

  switch (impl->sqrt_type) {
  case PETSCDASETKF_SQRT_CHOLESKY: {
    /* Solve L * L^T * w = sdel using LAPACK's Cholesky solve (xPOTRS) */
    PetscBLASInt n, lda, nrhs, info;
    PetscScalar *a_array, *b_array;
    PetscInt     m_L, N_L;

    PetscCheck(impl->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

    /* Get dimensions */
    PetscCall(MatGetSize(impl->L_cholesky, &m_L, &N_L));
    PetscCall(PetscBLASIntCast(N_L, &n));
    lda  = n;
    nrhs = 1;

    /* Copy sdel to w for in-place solve */
    PetscCall(VecCopy(sdel, *w));

    /* Get arrays */
    PetscCall(MatDenseGetArray(impl->L_cholesky, &a_array));
    PetscCall(VecGetArray(*w, &b_array));

    /* Solve L * L^T * w = sdel */
    LAPACKpotrs_("L", &n, &nrhs, a_array, &lda, b_array, &n, &info);
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky solve (xPOTRS): info=%d", (int)info);

    /* Restore arrays */
    PetscCall(MatDenseRestoreArray(impl->L_cholesky, &a_array));
    PetscCall(VecRestoreArray(*w, &b_array));
    break;
  }
  case PETSCDASETKF_SQRT_EIGEN: {
    /* Solve using eigendecomposition: T^{-1} = V * D^{-1} * V^T */
    Vec temp;
    PetscCheck(impl->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
    PetscCheck(impl->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

    /* w = T^-1 S^T-delta [sdel] */
    PetscCall(VecDuplicate(sdel, &temp));
    /* temp = V^T * sdel */
    PetscCall(MatMultTranspose(impl->V, sdel, temp));
    /* temp = D^{-1} * temp = D^{-1} * V^T * sdel */
    PetscCall(VecPointwiseDivide(temp, temp, impl->sqrt_eigen_vals));
    PetscCall(VecPointwiseDivide(temp, temp, impl->sqrt_eigen_vals));
    /* w = V * temp = V * D^{-1} * V^T * sdel */
    PetscCall(MatMult(impl->V, temp, *w));
    PetscCall(VecDestroy(&temp));
    break;
  }
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDASETKF square-root type %" PetscInt_FMT, (PetscInt)impl->sqrt_type);
  }
  /* Debugging verification: Test that sdel == T * w */
  if (PetscDefined(USE_DEBUG)) {
    PetscReal norm_T, norm_diff;
    Vec       temp;
    PetscCall(VecDuplicate(sdel, &temp));
    /* Compute Ty = (I + S'S) * w */
    PetscCall(MatMult(impl->I_StS, *w, temp));
    /* Compute difference: t = T w - sdel */
    PetscCall(VecAYPX(temp, -1.0, sdel));
    /* Compute norms for comparison */
    PetscCall(VecNorm(sdel, NORM_2, &norm_T));
    PetscCall(VecNorm(temp, NORM_2, &norm_diff));
    /* Verify is small */
    if (norm_T > 0) {
      PetscReal relative_error = norm_diff / norm_T;
      if (relative_error > 1.e-10) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)da), "WARNING: T^{-1} verification failed! ||T*w - sdel||/||sdel|| = %g\n", (double)relative_error));
        //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)da), "*"));
      } else {
        // PetscCall(PetscPrintf(PetscObjectComm((PetscObject)da), "T^{-1/2} verification passed: ||T*w - sdel|| = %g\n", (double)relative_error));
      }
    }
    PetscCall(VecDestroy(&temp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDASApplySqrtTInverse - Apply T^{-1/2} to a matrix (I) [Alg 6.4 line 9]

  Input Parameters:
+ da - the PetscDAS context
- U  - input matrix

  Output Parameter:
. Y - output matrix Y = T^{-1/2} * U

  Notes:
  This function applies the inverse square root of T = I + S^T * S using the
  stored factorization. For CHOLESKY mode, it uses a forward solve with L.
  For EIGEN mode, use T = V * D * V^T.
*/
static PetscErrorCode PetscDASApplySqrtTInverse(PetscDAS da, Mat U, Mat *Y)
{
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  PetscAssertPointer(Y, 3);

  impl = (PetscDASETKFData *)da->data;
  PetscCheck(impl->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "I_StS matrix not created. Call PetscDASTFactor first");

  switch (impl->sqrt_type) {
  case PETSCDASETKF_SQRT_CHOLESKY: {
    /* T^{-1/2} = L^{-T}, so solve L^T * Y = U using LAPACK triangular solve */
    PetscBLASInt n, lda, nrhs, info;
    PetscScalar *a_array, *b_array;
    PetscInt     m_L, N_L, m_U, N_U;

    PetscCheck(impl->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

    /* Get dimensions */
    PetscCall(MatGetSize(impl->L_cholesky, &m_L, &N_L));
    PetscCall(MatGetSize(U, &m_U, &N_U));
    PetscCheck(m_L == m_U, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Cholesky factor rows (%" PetscInt_FMT ") must match U rows (%" PetscInt_FMT ")", m_L, m_U);

    PetscCall(PetscBLASIntCast(N_L, &n));
    PetscCall(PetscBLASIntCast(N_U, &nrhs));
    lda = n;

    /* Copy U to Y for in-place solve */
    PetscCall(MatDuplicate(U, MAT_COPY_VALUES, Y));

    /* Get arrays */
    PetscCall(MatDenseGetArray(impl->L_cholesky, &a_array));
    PetscCall(MatDenseGetArray(*Y, &b_array));

    /* Solve L^T * Y = U using triangular solve (L is lower, so L^T is upper) */
    LAPACKtrtrs_("L", "T", "N", &n, &nrhs, a_array, &lda, b_array, &n, &info);
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK triangular solve (xTRTRS): info=%d", (int)info);

    /* Restore arrays */
    PetscCall(MatDenseRestoreArray(impl->L_cholesky, &a_array));
    PetscCall(MatDenseRestoreArray(*Y, &b_array));
    PetscCall(MatAssemblyBegin(*Y, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*Y, MAT_FINAL_ASSEMBLY));
    break;
  }
  case PETSCDASETKF_SQRT_EIGEN: {
    Vec diag_inv;
    /* T^-1 = V D^-1 V^T = (V D^-1/2 V^T)^2 : T^{-1/2} = V D^{-1/2} V^T */
    PetscCheck(impl->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
    PetscCheck(impl->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

    /* V_D_2 = V = V * U[I] */
    Mat V_D_2;
    PetscCall(MatMatMult(impl->V, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &V_D_2));
    /* D^{-1/2} */
    PetscCall(VecDuplicate(impl->sqrt_eigen_vals, &diag_inv));
    PetscCall(VecCopy(impl->sqrt_eigen_vals, diag_inv));
    PetscCall(VecReciprocal(diag_inv));
    /* V_D_2 = V * D^{-1/2} */
    PetscCall(MatDiagonalScale(V_D_2, NULL, diag_inv));
    PetscCall(VecDestroy(&diag_inv));
    /* T^{-1/2} = V_D_2 * V^T = V * D^{-1/2} * V^T */
    PetscCall(MatMatTransposeMult(V_D_2, impl->V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, Y));
    PetscCall(MatDestroy(&V_D_2));
    break;
  }
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDASETKF square-root type %" PetscInt_FMT, (PetscInt)impl->sqrt_type);
  }

  /* Debugging verification: Test that T*Y*Y = U^2 (assume U*U = U = I)
     Mathematical property: If Y = T^{-1/2} U, then T * Y * Y = U^2 */
  if (PetscDefined(USE_DEBUG)) {
    Mat       Y2, T_diff;
    PetscReal norm_T, norm_diff;

    /* Compute Y2 = Y * Y' = T^-1 */
    PetscCall(MatMatTransposeMult(*Y, *Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y2));

    /* Compute T_diff = T * T^-1 = U^2 = U = I */
    PetscCall(MatMatMult(Y2, impl->I_StS, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_diff));

    /* Compute difference: T_diff = Y Y' T - U ~ 0 */
    PetscCall(MatAYPX(T_diff, -1.0, U, SAME_NONZERO_PATTERN));

    /* Compute norms for comparison */
    PetscCall(MatNorm(U, NORM_FROBENIUS, &norm_T));
    PetscCall(MatNorm(T_diff, NORM_FROBENIUS, &norm_diff));
    /* Verify that ||Y * Y * T - U[U=I] || / ||U|| is small */
    PetscCheck(norm_diff / norm_T < 1.e-10, PETSC_COMM_SELF, PETSC_ERR_LIB, "T^{1/2} wrong. ||Y*Y*T-I||/||I||>tol %g", (double)norm_diff);
    /* Cleanup */
    PetscCall(MatDestroy(&Y2));
    PetscCall(MatDestroy(&T_diff));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                    ETKF Analysis Algorithm (Algorithm 6.4)                */
/* ========================================================================== */

/*
  PetscDASETKFAnalysis - Performs the ensemble transform Kalman filter (ETKF) analysis defined by Algorithm 6.4 in Law, Stuart, and Zygalakis.

  Collective

  Input Parameters:
+ da                   - the `PetscDAS` context owning the forecast ensemble and buffers
. observation          - observation vector `y`
. observation_operator - user-supplied routine `H(x, y; ctx)` that maps a state to observation space
- obs_ctx              - optional context for `observation_operator`

  Notes:
  The implementation follows the book's deterministic ETKF steps\:
  Step 1 computes the state mean, Step 2 the state anomalies, Steps 3-4 build the normalized innovation statistics,
  Step 5 assembles the reduced-space inverse, Step 6 forms the analysis weights, Steps 7-9 construct the square-root
  transform, and Step 10 applies the transform to refresh every ensemble member.

  Level: advanced

.seealso: [](ch_das), `PetscDAS`, `PetscDASETKFApplyModel()`, `PetscDASComputeMean()`,
`PetscDASComputeAnomalies()`
*/
static PetscErrorCode PetscDASETKFAnalysis(PetscDAS da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  PetscDASETKFData *impl;
  Vec               mean, y_mean, delta_scaled, w, r_inv_sqrt;
  Mat               X, Z, S, T_sqrt, w_ones;
  PetscInt          m;
  PetscScalar       inv_m, scale, sqrt_m_minus_1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);
  /* Validate ensemble size */
  m = da->ensemble_size;
  PetscCheck(m > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be > 1, got %" PetscInt_FMT, m);

  impl           = (PetscDASETKFData *)da->data;
  inv_m          = 1.0 / ((PetscScalar)m);
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  PetscCall(PetscInfo(da, "squaroot type %s, %d ensembles\n", (impl->sqrt_type == PETSCDASETKF_SQRT_EIGEN) ? "eigen" : "cholesky", (int)m));
  /* ===================================================================== */
  /* Alg 6.4 line 1-2: Compute ensemble mean and scaled anomalies        */
  /* ===================================================================== */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(mean));
  PetscCall(PetscDASComputeMean(da, mean));

  /* X = (E - x_mean * 1') / sqrt(m - 1) */
  PetscCall(PetscDASComputeAnomalies(da, &X));

  /* ===================================================================== */
  /* Alg 6.4 line 3-4: Compute observation ensemble Z = H(x_i^f)         */
  /* ===================================================================== */
  PetscCall(ComputeObservationEnsemble(da, observation_operator, obs_ctx, &Z));

  /* Compute observation mean y_mean = (1/m) * sum(Z_i) */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &y_mean));
  PetscCall(VecSetSizes(y_mean, PETSC_DECIDE, da->obs_size));
  PetscCall(VecSetFromOptions(y_mean));
  PetscCall(MatGetRowSum(Z, y_mean));
  PetscCall(VecScale(y_mean, inv_m));

  /* ===================================================================== */
  /* Alg 6.4 line 5-6: Build normalized innovation statistics            */
  /* ===================================================================== */
  /* Compute R^{-1/2} (assumes diagonal R) */
  PetscCall(VecDuplicate(da->obs_error_var, &r_inv_sqrt));
  PetscCall(VecCopy(da->obs_error_var, r_inv_sqrt));
  PetscCall(VecSqrtAbs(r_inv_sqrt));
  PetscCall(VecReciprocal(r_inv_sqrt));

  /* S = R^{-1/2} * (Z - y_mean * 1') / sqrt(m - 1) */
  PetscCall(ComputeNormalizedInnovationMatrix(Z, y_mean, r_inv_sqrt, m, scale, &S));

  /* delta_scaled = R^{-1/2} * (y^o - y_mean) [Alg 6.4 line 6] */
  PetscCall(VecDuplicate(y_mean, &delta_scaled));
  PetscCall(VecWAXPY(delta_scaled, -1.0, y_mean, observation));
  PetscCall(VecPointwiseMult(delta_scaled, delta_scaled, r_inv_sqrt));

  /* ===================================================================== */
  /* Alg 6.4 line 7: Factor T = (I + S^T S) and store factorization (T is not inverted here but solved later) */
  /* ===================================================================== */
  PetscCall(PetscDASTFactor(da, S));

  /* ===================================================================== */
  /* Alg 6.4 line 8: Compute analysis weights w = T * S^T * delta_scaled */
  /* ===================================================================== */
  Vec s_transpose_delta;
  PetscCall(MatCreateVecs(impl->I_StS, NULL, &s_transpose_delta));
  PetscCall(MatMultTranspose(S, delta_scaled, s_transpose_delta));
  PetscCall(PetscDASApplyTInverse(da, s_transpose_delta, &w));
  PetscCall(VecDestroy(&s_transpose_delta));

  /* ===================================================================== */
  /* Alg 6.4 line 9: Compute square-root transform T^{1/2} U = T^{1/2}     */
  /* ===================================================================== */
  PetscCall(PetscDASApplySqrtTInverse(da, da->U, &T_sqrt));
  /* ===================================================================== */
  /* Alg 6.4 line 9: Form transform G = w * 1' + sqrt(m - 1) * T^{1/2} * U */
  /* ===================================================================== */
  /* w_ones = w * 1' (broadcast weight vector to all columns) -- remove */
  PetscCall(BroadcastWeightVector(w, m, &w_ones));
  /* sqrt(m-1) T^{1/2} * U */
  PetscCall(MatScale(T_sqrt, sqrt_m_minus_1));
  /* G = w_ones + T_sqrt_U */
  PetscCall(MatAXPY(w_ones, 1.0, T_sqrt, SAME_NONZERO_PATTERN));
  /* ===================================================================== */
  /* Alg 6.4 line 9: Update ensemble E = x_mean * 1' + X * G             */
  /* ===================================================================== */
  PetscCall(UpdateEnsembleWithTransform(mean, X, w_ones, m, da->ensemble));

  /* Cleanup */
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&y_mean));
  PetscCall(VecDestroy(&delta_scaled));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&r_inv_sqrt));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Z));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&T_sqrt));
  PetscCall(MatDestroy(&w_ones));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                          Model Propagation                                */
/* ========================================================================== */

/*
  PetscDASETKFApplyModel - Advances each ensemble member through the user-supplied
  nonlinear model (Algorithm 6.4, Step 10 forecast propagation).

  Collective

  Input Parameters:
+ da        - the `PetscDAS` context that stores the ensemble
. model     - routine that evaluates the model `f(x, xnew; ctx)`
- model_ctx - optional context for `model`

  Notes:
  This routine overwrites every ensemble column with the model result supplied by `model`.
  It is typically called immediately after `PetscDASETKFAnalysis()` to start the next forecast cycle.

  Level: intermediate

.seealso: [](ch_das), `PetscDAS`, `PetscDASETKFAnalysis()`
*/
static PetscErrorCode PetscDASETKFApplyModel(PetscDAS da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
{
  Vec      col_in, col_out, temp;
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);

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
  PetscDASETKFSetSqrtType - Selects the reduced-space square-root algorithm used during the ETKF analysis.

  Logically Collective

  Input Parameters:
+ da   - the `PetscDAS` object
- type - either `PETSCDASETKF_SQRT_CHOLESKY` or `PETSCDASETKF_SQRT_EIGEN`

  Level: advanced

.seealso: [](ch_das), `PetscDAS`, `PetscDASETKFGetSqrtType()`, `PetscDASETKFAnalysis()`
@*/
PetscErrorCode PetscDASETKFSetSqrtType(PetscDAS da, PetscDASETKFSqrtType type)
{
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDAS data structure not initialized");
  PetscCheck(type == PETSCDASETKF_SQRT_CHOLESKY || type == PETSCDASETKF_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDASETKF square-root type %" PetscInt_FMT, (PetscInt)type);

  impl            = (PetscDASETKFData *)da->data;
  impl->sqrt_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASETKFGetSqrtType - Retrieves the current square-root implementation configured for the ETKF analysis.

  Not Collective

  Input Parameters:
. da - the `PetscDAS` object

  Output Parameter:
. type - on output, the configured `PetscDASETKFSqrtType`

  Level: advanced

.seealso: [](ch_das), `PetscDAS`, `PetscDASETKFSetSqrtType()`
@*/
PetscErrorCode PetscDASETKFGetSqrtType(PetscDAS da, PetscDASETKFSqrtType *type)
{
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDAS data structure not initialized");

  impl  = (PetscDASETKFData *)da->data;
  *type = impl->sqrt_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDASETKFView - Views a `PetscDASETKF` and its implementation-specific data structure.

  Input Parameters:
+ da     - the `PetscDAS` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: internal

.seealso: [](ch_das), `PetscDASViewFromOptions()`
*/
static PetscErrorCode PetscDASETKFView(PetscDAS da, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);

  impl = (PetscDASETKFData *)da->data;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDASETKF Object:\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Square root type: %s\n", (impl->sqrt_type == PETSCDASETKF_SQRT_EIGEN) ? "eigen" : "cholesky"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDASETKFInitialize - Installs the ETKF-specific operations on a newly created `PetscDAS` object.

  Input Parameter:
. da - the `PetscDAS` object to configure

  Level: internal

.seealso: [](ch_das), `PetscDAS`, `PetscDASETKFRegister()`, `PetscDASETKFAnalysis()`
*/
static PetscErrorCode PetscDASETKFInitialize(PetscDAS da)
{
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);

  PetscCall(PetscNew(&impl));
  impl->V               = NULL;
  impl->L_cholesky      = NULL;
  impl->sqrt_eigen_vals = NULL;
  impl->I_StS           = NULL;
  impl->sqrt_type       = PETSCDASETKF_SQRT_EIGEN;

  da->data                  = impl;
  da->ops->analysis         = PetscDASETKFAnalysis;
  da->ops->applymodel       = PetscDASETKFApplyModel;
  da->ops->computemean      = NULL;
  da->ops->computeanomalies = NULL;
  da->ops->destroy          = PetscDASETKFDestroy;
  da->ops->view             = PetscDASETKFView;
  da->ops->setfromoptions   = PetscDASSetFromOptions_DASETKF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDASETKFRegister(void)
{
  PetscFunctionBegin;
  PetscCall(PetscDASRegister(PETSCDASETKF, PetscDASETKFInitialize));
  PetscCall(PetscDASETKFInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}
