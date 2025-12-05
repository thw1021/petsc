#include "petscdas.h"
#include <petsc/private/dasimpl.h>
#include <petscblaslapack.h>

typedef struct {
  PetscDASETKFSqrtType sqrt_type;
  Mat                  V;               /* Eigen vectors (LAPACK column-major storage) */
  Mat                  L_cholesky;      /* Lower triangular Cholesky factor */
  Vec                  sqrt_eigen_vals; /* Square root of eigen values */
  Mat                  I_StS;           /* T = I + S^T * S matrix */

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
} PetscDASETKFData;

static PetscFunctionList PetscDASETKFSqrtList           = NULL;
static PetscBool         PetscDASETKFPackageInitialized = PETSC_FALSE;

/* Tolerance for matrix square root verification in debug mode */
#define MATRIX_SQRT_TOLERANCE_FACTOR (100.0 * PETSC_MACHINE_EPSILON)

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
static PetscErrorCode ComputeObservationEnsemble(PetscDAS da, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx, Mat Z)
{
  /* Ensemble and observation-related vectors */
  Vec ensemble_member_in, observation_out;
  /* Loop counter and ensemble size */
  PetscInt ensemble_idx, ensemble_size;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
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

.seealso\: [`PetscDASETKFAnalysis()`](etkfilter.c:837), [`MatDenseGetArrayWrite()`](petscmat.h)
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
    /* Destroy factorization data */
    PetscCall(MatDestroy(&impl->V));
    PetscCall(MatDestroy(&impl->L_cholesky));
    PetscCall(VecDestroy(&impl->sqrt_eigen_vals));
    PetscCall(MatDestroy(&impl->I_StS));

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
  PetscDASTFactor_Cholesky - Computes Cholesky factorization of T

  Input Parameters:
+ da   - the PetscDAS context
- impl - the internal data structure containing T matrix (I_StS)
*/
static PetscErrorCode PetscDASTFactor_Cholesky(PetscDAS da, PetscDASETKFData *impl)
{
  PetscBLASInt n, lda, info;
  PetscScalar *a_array;
  PetscInt     m_T, N_T;

  PetscFunctionBegin;
  /* Initialize or update L_cholesky matrix */
  if (!impl->L_cholesky) {
    PetscCall(MatDuplicate(impl->I_StS, MAT_COPY_VALUES, &impl->L_cholesky));
  } else {
    PetscCall(MatCopy(impl->I_StS, impl->L_cholesky, SAME_NONZERO_PATTERN));
  }

  /* Get matrix dimensions and convert to BLAS int */
  PetscCall(MatGetSize(impl->L_cholesky, &m_T, &N_T));
  PetscCheck(m_T == N_T, PetscObjectComm((PetscObject)impl->L_cholesky), PETSC_ERR_ARG_WRONG, "Matrix must be square for Cholesky");
  PetscCall(PetscBLASIntCast(N_T, &n));
  lda = n;

  /* Get array from dense matrix */
  PetscCall(MatDenseGetArray(impl->L_cholesky, &a_array));

  /* Compute Cholesky factorization: A = L * L^T (lower triangular) */
  LAPACKpotrf_("L", &n, a_array, &lda, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky factorization (xPOTRF): info=%d. Matrix T is not positive definite.", (int)info);

  /* Restore array and finalize matrix */
  PetscCall(MatDenseRestoreArray(impl->L_cholesky, &a_array));
  PetscCall(MatAssemblyBegin(impl->L_cholesky, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(impl->L_cholesky, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDASTFactor_Eigen - Computes Eigendecomposition of T

  Input Parameters:
+ da   - the PetscDAS context
- impl - the internal data structure containing T matrix (I_StS)
*/
static PetscErrorCode PetscDASTFactor_Eigen(PetscDAS da, PetscDASETKFData *impl)
{
  PetscBLASInt n, lda, lwork, info;
  PetscScalar *a_array, *work, *eig_array;
  PetscInt     m_V, N_V;
#if defined(PETSC_USE_COMPLEX)
  PetscReal *rwork = NULL;
#endif

  PetscFunctionBegin;
  /* Initialize or update V matrix */
  if (!impl->V) {
    PetscCall(MatDuplicate(impl->I_StS, MAT_COPY_VALUES, &impl->V));
  } else {
    PetscCall(MatCopy(impl->I_StS, impl->V, SAME_NONZERO_PATTERN));
  }

  /* Initialize or update eigenvalue vector */
  if (!impl->sqrt_eigen_vals) PetscCall(MatCreateVecs(impl->I_StS, &impl->sqrt_eigen_vals, NULL));

  /* Get matrix dimensions */
  PetscCall(MatGetSize(impl->V, &m_V, &N_V));
  PetscCheck(m_V == N_V, PetscObjectComm((PetscObject)impl->V), PETSC_ERR_ARG_WRONG, "Matrix must be square");
  PetscCall(PetscBLASIntCast(N_V, &n));
  lda = n;

  /* Get arrays */
  PetscCall(MatDenseGetArray(impl->V, &a_array));
  PetscCall(VecGetArray(impl->sqrt_eigen_vals, &eig_array));

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
  PetscCall(VecRestoreArray(impl->sqrt_eigen_vals, &eig_array));
  PetscCall(MatDenseRestoreArray(impl->V, &a_array));

  PetscCall(MatAssemblyBegin(impl->V, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(impl->V, MAT_FINAL_ASSEMBLY));

  /* Compute sqrt(eigenvalues) */
  PetscCall(VecSqrtAbs(impl->sqrt_eigen_vals));

  /* Debug verification: Ensure V * D * V^T == T */
  if (PetscDefined(USE_DEBUG)) {
    PetscReal norm_T, norm_diff;
    Mat       V_D, VDVt;

    /* Compute D * V^T by scaling rows */
    PetscCall(MatDuplicate(impl->V, MAT_COPY_VALUES, &V_D));

    /* Restore D for verification (since sqrt_eigen_vals currently holds sqrt(D)) */
    PetscCall(VecPointwiseMult(impl->sqrt_eigen_vals, impl->sqrt_eigen_vals, impl->sqrt_eigen_vals));

    PetscCall(MatDiagonalScale(V_D, NULL, impl->sqrt_eigen_vals));

    /* Compute V * D * V^T */
    PetscCall(MatMatTransposeMult(V_D, impl->V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VDVt));

    /* Compute ||V*D*V^T - T|| / ||T|| */
    PetscCall(MatAXPY(VDVt, -1.0, impl->I_StS, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(impl->I_StS, NORM_FROBENIUS, &norm_T));
    PetscCall(MatNorm(VDVt, NORM_FROBENIUS, &norm_diff));

    if (norm_T > 0) {
      PetscReal relative_error = norm_diff / norm_T;
      PetscCheck(relative_error < MATRIX_SQRT_TOLERANCE_FACTOR, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "Eigendecomposition verification failed: ||V*D*V^T - T||/||T|| = %g", (double)relative_error);
    }

    /* Restore sqrt(D) back to sqrt_eigen_vals */
    PetscCall(VecSqrtAbs(impl->sqrt_eigen_vals));

    /* Cleanup debug matrices */
    PetscCall(MatDestroy(&V_D));
    PetscCall(MatDestroy(&VDVt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDASTFactor - Compute and store factorization of T matrix

  Input Parameters:
+ da - the PetscDAS context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  This function computes T = I + S^T * S and stores its factorization based on
  the selected sqrt_type.

  - For CHOLESKY mode: computes the lower triangular Cholesky factor L such that T = L * L^T.
  - For EIGEN mode: computes eigenvectors V and eigenvalues D such that T = V * D * V^T.

  The implementation uses matrix reuse (MAT_REUSE_MATRIX) to minimize memory allocation
  overhead when the ensemble size remains constant across analysis cycles.
*/
static PetscErrorCode PetscDASTFactor(PetscDAS da, Mat S)
{
  PetscDASETKFData *impl;
  PetscInt          m, s_rows, s_cols;
  MatReuse          scall      = MAT_INITIAL_MATRIX;
  PetscBool         reallocate = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(S, MAT_CLASSID, 2);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDAS data structure not initialized");

  impl = (PetscDASETKFData *)da->data;

  /* 1. Validate Matrix Dimensions */
  PetscCall(MatGetSize(S, &s_rows, &s_cols));
  m = s_cols; /* Ensemble size */

  PetscCheck(m > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Innovation matrix S must have positive columns, got %" PetscInt_FMT, m);
  PetscCheck(m == da->ensemble_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "S matrix columns (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ") defined in PetscDAS", m, da->ensemble_size);

  /* 2. Manage Resource Reuse */
  /* Check if we can reuse the T matrix (I_StS) and dependent factors */
  if (impl->I_StS) {
    PetscInt t_rows, t_cols;
    PetscCall(MatGetSize(impl->I_StS, &t_rows, &t_cols));

    /* If dimensions have changed, we must fully reallocate */
    if (t_rows != m || t_cols != m) {
      reallocate = PETSC_TRUE;
      PetscCall(PetscInfo(da, "Ensemble size changed (old: %" PetscInt_FMT ", new: %" PetscInt_FMT "), reallocating T matrix and factors\n", t_rows, m));
    } else {
      scall = MAT_REUSE_MATRIX;
    }
  }

  if (reallocate && impl->I_StS) {
    PetscCall(MatDestroy(&impl->I_StS));
    PetscCall(MatDestroy(&impl->V));
    PetscCall(MatDestroy(&impl->L_cholesky));
    PetscCall(VecDestroy(&impl->sqrt_eigen_vals));
    scall = MAT_INITIAL_MATRIX;
  }

  /* 3. Compute T = I + S^T * S */
  /*
     MatTransposeMatMult computes C = A^T * B (here C = S^T * S).
     When using MAT_REUSE_MATRIX, the existing C is overwritten with the new result.
  */
  PetscCall(MatTransposeMatMult(S, S, scall, PETSC_DEFAULT, &impl->I_StS));

  /* Add Identity: T = T + I */
  PetscCall(MatShift(impl->I_StS, 1.0));

  /* 4. Compute Factorization based on strategy */
  switch (impl->sqrt_type) {
  case PETSCDASETKF_SQRT_CHOLESKY:
    PetscCall(PetscDASTFactor_Cholesky(da, impl));
    break;
  case PETSCDASETKF_SQRT_EIGEN:
    PetscCall(PetscDASTFactor_Eigen(da, impl));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDASETKF square-root type %d", (int)impl->sqrt_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplyTInverse_Cholesky - Helper for Cholesky solver path
*/
static PetscErrorCode ApplyTInverse_Cholesky(PetscDAS da, PetscDASETKFData *impl, Vec sdel, Vec w)
{
  PetscBLASInt n, lda, nrhs, info;
  PetscScalar *a_array, *b_array;
  PetscInt     m_L, N_L;

  PetscFunctionBegin;
  PetscCheck(impl->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

  /* Get dimensions */
  PetscCall(MatGetSize(impl->L_cholesky, &m_L, &N_L));
  PetscCall(PetscBLASIntCast(N_L, &n));
  lda  = n;
  nrhs = 1;

  /* Copy sdel to w for in-place solve */
  PetscCall(VecCopy(sdel, w));

  /* Get arrays */
  PetscCall(MatDenseGetArray(impl->L_cholesky, &a_array));
  PetscCall(VecGetArray(w, &b_array));

  /* Solve L * L^T * w = sdel using LAPACK's Cholesky solve (xPOTRS) */
  /* Note: POTRS expects the input B (w) to contain the RHS, and overwrites it with the solution */
  LAPACKpotrs_("L", &n, &nrhs, a_array, &lda, b_array, &n, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky solve (xPOTRS): info=%d", (int)info);

  /* Restore arrays */
  PetscCall(MatDenseRestoreArray(impl->L_cholesky, &a_array));
  PetscCall(VecRestoreArray(w, &b_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplyTInverse_Eigen - Helper for Eigendecomposition solver path
*/
static PetscErrorCode ApplyTInverse_Eigen(PetscDAS da, PetscDASETKFData *impl, Vec sdel, Vec w)
{
  Vec temp;

  PetscFunctionBegin;
  PetscCheck(impl->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
  PetscCheck(impl->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

  /* Allocate temporary vector for projection */
  PetscCall(VecDuplicate(sdel, &temp));

  /* 1. Project onto eigenvectors: temp = V^T * sdel */
  PetscCall(MatMultTranspose(impl->V, sdel, temp));

  /* 2. Scale by inverse eigenvalues: temp = D^{-1} * temp */
  /* We store sqrt(D), so divide twice: temp = (temp / sqrt(D)) / sqrt(D) */
  PetscCall(VecPointwiseDivide(temp, temp, impl->sqrt_eigen_vals));
  PetscCall(VecPointwiseDivide(temp, temp, impl->sqrt_eigen_vals));

  /* 3. Map back to standard basis: w = V * temp */
  PetscCall(MatMult(impl->V, temp, w));

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  VerifyTInverse - Debug helper to verify solution accuracy
*/
static PetscErrorCode VerifyTInverse(PetscDAS da, PetscDASETKFData *impl, Vec sdel, Vec w)
{
  PetscReal norm_T, norm_diff;
  Vec       temp;
  PetscReal relative_error;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(sdel, &temp));

  /* Compute Ty = (I + S'S) * w */
  PetscCall(MatMult(impl->I_StS, w, temp));

  /* Compute difference: t = T w - sdel */
  PetscCall(VecAYPX(temp, -1.0, sdel));

  /* Compute norms for comparison */
  PetscCall(VecNorm(sdel, NORM_2, &norm_T));
  PetscCall(VecNorm(temp, NORM_2, &norm_diff));

  /* Verify is small */
  if (norm_T > 0) {
    relative_error = norm_diff / norm_T;
    /* Using a loose tolerance since we are comparing against factorization */
    if (relative_error > MATRIX_SQRT_TOLERANCE_FACTOR) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)da), "WARNING: T^{-1} verification failed! ||T*w - sdel||/||sdel|| = %g\n", (double)relative_error));
  }

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDASApplyTInverse - Apply T^{-1} to a vector [Alg 6.4 line 8]

  Input Parameters:
+ da - the PetscDAS context
- sdel  - input vector S^T-delta

  Output Parameter:
. w - output vector w = T^{-1} * sdel

  Notes:
  This function applies the inverse of T = I + S^T S using the stored
  factorization. For CHOLESKY mode, it uses triangular solves. For EIGEN mode,
  it uses the eigendecomposition (T^{-1} = V D^{-1} V^T).
*/
static PetscErrorCode PetscDASApplyTInverse(PetscDAS da, Vec sdel, Vec w)
{
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(sdel, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 3);

  impl = (PetscDASETKFData *)da->data;
  PetscCheck(impl->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "T matrix not factored. Call PetscDASTFactor first");

  switch (impl->sqrt_type) {
  case PETSCDASETKF_SQRT_CHOLESKY:
    PetscCall(ApplyTInverse_Cholesky(da, impl, sdel, w));
    break;
  case PETSCDASETKF_SQRT_EIGEN:
    PetscCall(ApplyTInverse_Eigen(da, impl, sdel, w));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDASETKF square-root type %" PetscInt_FMT, (PetscInt)impl->sqrt_type);
  }

  /* Debugging verification: Test that sdel == T * w */
  if (PetscDefined(USE_DEBUG)) PetscCall(VerifyTInverse(da, impl, sdel, w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplySqrtTInverse_Cholesky - Computes Y = L^{-T} * U using forward substitution.

  Notes:
  Since T = L * L^T, T^{-1} = L^{-T} * L^{-1}.
  We uses L^{-T} as the non-symmetric "square root" inverse, i.e., T^{-1/2} = L^{-T}.
  This requires solving L^T * Y = U.
*/
static PetscErrorCode ApplySqrtTInverse_Cholesky(PetscDAS da, PetscDASETKFData *impl, Mat U, Mat Y)
{
  PetscBLASInt       n, lda, nrhs, info;
  const PetscScalar *l_array;
  PetscScalar       *y_array;
  PetscInt           m_L, N_L, m_U, N_U;

  PetscFunctionBegin;
  PetscCheck(impl->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

  /* Get dimensions and validate compatibility */
  PetscCall(MatGetSize(impl->L_cholesky, &m_L, &N_L));
  PetscCall(MatGetSize(U, &m_U, &N_U));
  PetscCheck(m_L == m_U, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Cholesky factor rows (%" PetscInt_FMT ") must match U rows (%" PetscInt_FMT ")", m_L, m_U);

  PetscCall(PetscBLASIntCast(N_L, &n));
  PetscCall(PetscBLASIntCast(N_U, &nrhs));
  lda = n;

  /* Initialize Y with U for in-place solve */
  PetscCall(MatCopy(U, Y, SAME_NONZERO_PATTERN));

  /* Get direct array access */
  PetscCall(MatDenseGetArrayRead(impl->L_cholesky, &l_array));
  PetscCall(MatDenseGetArray(Y, &y_array));

  /* Solve L^T * Y = U using LAPACK triangular solve (L is lower, so L^T is upper)
     TRTRS args: UPLO='L', TRANS='T', DIAG='N' */
  LAPACKtrtrs_("L", "T", "N", &n, &nrhs, (PetscScalar *)l_array, &lda, y_array, &n, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK triangular solve (xTRTRS): info=%d", (int)info);

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayRead(impl->L_cholesky, &l_array));
  PetscCall(MatDenseRestoreArray(Y, &y_array));

  PetscCall(MatAssemblyBegin(Y, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Y, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplySqrtTInverse_Eigen - Computes Y = V * D^{-1/2} * V^T * U.

  Notes:
  This computes the symmetric square root T^{-1/2} = V * D^{-1/2} * V^T.
  The operation is performed as Y = V * (D^{-1/2} * (V^T * U)) to strictly follow
  linear algebra operations for general matrix U.
*/
static PetscErrorCode ApplySqrtTInverse_Eigen(PetscDAS da, PetscDASETKFData *impl, Mat U, Mat Y)
{
  Mat W;
  Vec diag_inv;

  PetscFunctionBegin;
  PetscCheck(impl->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
  PetscCheck(impl->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

  /* Prepare inverse sqrt eigenvalues: D^{-1/2}
     Note: impl->sqrt_eigen_vals currently stores sqrt(D) */
  PetscCall(VecDuplicate(impl->sqrt_eigen_vals, &diag_inv));
  PetscCall(VecCopy(impl->sqrt_eigen_vals, diag_inv));
  PetscCall(VecReciprocal(diag_inv)); /* Now diag_inv contains 1/sqrt(D) = D^{-1/2} */

  /* Step 1: Compute W = V^T * U (Project U onto eigenbasis) */
  PetscCall(MatTransposeMatMult(impl->V, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &W));

  /* Step 2: Scale rows of W by D^{-1/2}: W <- D^{-1/2} * W */
  PetscCall(MatDiagonalScale(W, diag_inv, NULL));

  /* Step 3: Compute Y = V * W (Project back to standard basis)
     Y = V * (D^{-1/2} * V^T * U) */
  /* Note: Y is already allocated, so we use MAT_REUSE_MATRIX or MAT_COPY_VALUES if the structure is compatible?
     MatMatMult with MAT_REUSE_MATRIX expects Y to be already assembled and correct size.
     We are writing into Y. */
  {
    Mat Y_temp;
    PetscCall(MatMatMult(impl->V, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y_temp));
    PetscCall(MatCopy(Y_temp, Y, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Y_temp));
  }

  /* Cleanup */
  PetscCall(MatDestroy(&W));
  PetscCall(VecDestroy(&diag_inv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDASApplySqrtTInverse - Apply T^{-1/2} to a matrix U [Alg 6.4 line 9]

  Input Parameters:
+ da - the PetscDAS context
- U  - input matrix (usually Identity, but can be general)

  Output Parameter:
. Y - output matrix Y = T^{-1/2} * U

  Notes:
  This function applies the inverse square root of T = I + S^T * S using the
  stored factorization.

  - For CHOLESKY mode: Computes Y = L^{-T} U
  - For EIGEN mode: Computes Y = V D^{-1/2} V^T U

  Both results satisfy Y^T * T * Y = U^T * U, preserving the metric.
*/
static PetscErrorCode PetscDASApplySqrtTInverse(PetscDAS da, Mat U, Mat Y)
{
  PetscDASETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(Y, MAT_CLASSID, 3);

  impl = (PetscDASETKFData *)da->data;
  PetscCheck(impl->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "I_StS matrix not created. Call PetscDASTFactor first");

  switch (impl->sqrt_type) {
  case PETSCDASETKF_SQRT_CHOLESKY:
    PetscCall(ApplySqrtTInverse_Cholesky(da, impl, U, Y));
    break;
  case PETSCDASETKF_SQRT_EIGEN:
    PetscCall(ApplySqrtTInverse_Eigen(da, impl, U, Y));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDASETKF square-root type %" PetscInt_FMT, (PetscInt)impl->sqrt_type);
  }

  /* Debugging verification: Check that metric is preserved
     Verify that Y^T * T * Y = U^T * U */
  if (PetscDefined(USE_DEBUG)) {
    Mat       YtTY, UtU, T_Y;
    PetscReal norm_ref, norm_diff;

    /* Compute LHS: Y^T * T * Y */
    PetscCall(MatMatMult(impl->I_StS, Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_Y));   /* T * Y */
    PetscCall(MatTransposeMatMult(Y, T_Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &YtTY)); /* Y^T * (T * Y) */

    /* Compute RHS: U^T * U */
    PetscCall(MatTransposeMatMult(U, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UtU));

    /* Compute difference: Diff = LHS - RHS */
    PetscCall(MatAXPY(YtTY, -1.0, UtU, SAME_NONZERO_PATTERN));

    /* Check norms */
    PetscCall(MatNorm(UtU, NORM_FROBENIUS, &norm_ref));
    PetscCall(MatNorm(YtTY, NORM_FROBENIUS, &norm_diff));

    if (norm_ref > 0.0) PetscCheck(norm_diff / norm_ref < MATRIX_SQRT_TOLERANCE_FACTOR, PETSC_COMM_SELF, PETSC_ERR_PLIB, "T^{-1/2} verification failed. ||Y^T*T*Y - U^T*U||/||U^T*U|| = %g", (double)(norm_diff / norm_ref));

    /* Cleanup debug matrices */
    PetscCall(MatDestroy(&T_Y));
    PetscCall(MatDestroy(&YtTY));
    PetscCall(MatDestroy(&UtU));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                    ETKF Analysis Algorithm (Algorithm 6.4)                */
/* ========================================================================== */

/*
  PetscDASETKFAnalysis - Performs the ensemble transform Kalman filter (ETKF) analysis defined by Algorithm 6.4 in Asch, M., Bocquet, M., and Nodet, M.

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
  Mat               X;
  PetscInt          m;
  PetscScalar       inv_m, scale, sqrt_m_minus_1;
  PetscBool         reallocate = PETSC_FALSE;

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
  PetscCall(PetscDASComputeMean(da, impl->mean));

  /* X = (E - x_mean * 1') / sqrt(m - 1) */
  /* Note: PetscDASComputeAnomalies creates a NEW matrix X every time.
     We should probably optimize this too in the future, but for now we follow the API. */
  PetscCall(PetscDASComputeAnomalies(da, &X));

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
  PetscCall(PetscDASTFactor(da, impl->S));

  /* ===================================================================== */
  /* Alg 6.4 line 8: Compute analysis weights w = T * S^T * delta_scaled */
  /* ===================================================================== */
  {
    Vec s_transpose_delta;
    PetscCall(MatCreateVecs(impl->I_StS, NULL, &s_transpose_delta));
    PetscCall(MatMultTranspose(impl->S, impl->delta_scaled, s_transpose_delta));

    /* w is created/reused inside if not passed? No, we need to handle w */
    if (!impl->w) PetscCall(VecDuplicate(s_transpose_delta, &impl->w));

    PetscCall(PetscDASApplyTInverse(da, s_transpose_delta, impl->w));
    PetscCall(VecDestroy(&s_transpose_delta));
  }

  /* ===================================================================== */
  /* Alg 6.4 line 9: Compute square-root transform T^{1/2} U = T^{1/2}     */
  /* ===================================================================== */
  PetscCall(PetscDASApplySqrtTInverse(da, da->U, impl->T_sqrt));

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
