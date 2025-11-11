#include "petscda.h"
#include <petsc/private/daimpl.h>

typedef struct {
  PetscDAETKFSqrtType sqrt_type;
} PetscDAETKFData;

static PetscFunctionList PetscDAETKFSqrtList           = NULL;
static PetscBool         PetscDAETKFPackageInitialized = PETSC_FALSE;

/* ========================================================================== */
/*                    Helper Functions for ETKF Analysis                     */
/* ========================================================================== */

/*
  ComputeObservationEnsemble - Applies observation operator H to each ensemble member

  Input Parameters:
+ da                   - the PetscDA context
. observation_operator - user-supplied routine H(x, y; ctx)
- obs_ctx              - optional context for observation_operator

  Output Parameter:
. Z - observation ensemble matrix (obs_size x ensemble_size)
*/
static PetscErrorCode ComputeObservationEnsemble(PetscDA da, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx, Mat *Z)
{
  Vec      col_in, col_out, temp_vec;
  PetscInt i, m;

  PetscFunctionBegin;
  m = da->ensemble_size;

  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, Z));
  PetscCall(MatSetUp(*Z));

  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &temp_vec));
  PetscCall(VecSetSizes(temp_vec, PETSC_DECIDE, da->obs_size));
  PetscCall(VecSetFromOptions(temp_vec));

  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, i, &col_in));
    PetscCall(observation_operator(col_in, temp_vec, obs_ctx));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, i, &col_in));

    PetscCall(MatDenseGetColumnVecWrite(*Z, i, &col_out));
    PetscCall(VecCopy(temp_vec, col_out));
    PetscCall(MatDenseRestoreColumnVecWrite(*Z, i, &col_out));
  }

  PetscCall(VecDestroy(&temp_vec));
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
  Vec         col_in, col_out;
  PetscInt    i, obs_size, obs_size_local;
  PetscMPIInt size;

  PetscFunctionBegin;
  /* Get observation size from input matrix Z */
  PetscCall(MatGetSize(Z, &obs_size, NULL));
  PetscCall(MatGetLocalSize(Z, &obs_size_local, NULL));

  /* For sequential matrices, use local size; for parallel, use PETSC_DECIDE */
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)Z), &size));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)Z), obs_size_local, PETSC_DECIDE, obs_size, m, NULL, S));

  PetscCall(MatSetUp(*S));

  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(Z, i, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(*S, i, &col_out));

    /* S_i = (Z_i - y_mean) / sqrt(m-1) */
    PetscCall(VecWAXPY(col_out, -1.0, y_mean, col_in));
    PetscCall(VecScale(col_out, scale));
    /* Apply R^{-1/2} */
    PetscCall(VecPointwiseMult(col_out, col_out, r_inv_sqrt));

    PetscCall(MatDenseRestoreColumnVecRead(Z, i, &col_in));
    PetscCall(MatDenseRestoreColumnVecWrite(*S, i, &col_out));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeAnalysisWeights - Computes weight vector w = T * S^T * delta_scaled

  Input Parameters:
+ T            - reduced-space inverse matrix (I + S^T S)^{-1}
. S            - normalized innovation matrix
- delta_scaled - scaled innovation vector R^{-1/2}(y^o - y_mean)

  Output Parameter:
. w - analysis weight vector
*/
static PetscErrorCode ComputeAnalysisWeights(Mat T, Mat S, Vec delta_scaled, Vec *w)
{
  Vec S_T_delta;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(T, NULL, &S_T_delta));
  PetscCall(MatMultTranspose(S, delta_scaled, S_T_delta));

  PetscCall(MatCreateVecs(T, w, NULL));
  PetscCall(MatMult(T, S_T_delta, *w));

  PetscCall(VecDestroy(&S_T_delta));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  BroadcastWeightVector - Creates matrix with weight vector replicated across all columns

  Input Parameters:
+ w - weight vector of size m
- m - ensemble size

  Output Parameter:
. w_ones - m x m matrix where each column is w
*/
static PetscErrorCode BroadcastWeightVector(Vec w, PetscInt m, Mat *w_ones)
{
  const PetscScalar *w_array;
  PetscInt           i, j;

  PetscFunctionBegin;
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)w), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, w_ones));
  PetscCall(MatSetUp(*w_ones));

  PetscCall(VecGetArrayRead(w, &w_array));
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) PetscCall(MatSetValue(*w_ones, i, j, w_array[i], INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(w, &w_array));

  PetscCall(MatAssemblyBegin(*w_ones, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*w_ones, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  UpdateEnsembleWithTransform - Updates ensemble: E = mean * 1' + X * G

  Input Parameters:
+ mean  - ensemble mean vector
. X     - anomaly matrix
. G     - transform matrix
. m     - ensemble size
- ensemble - ensemble matrix to update (in-place)
*/
static PetscErrorCode UpdateEnsembleWithTransform(Vec mean, Mat X, Mat G, PetscInt m, Mat ensemble)
{
  Mat      X_G;
  Vec      col_in, col_out;
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(MatMatMult(X, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X_G));

  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(X_G, i, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(ensemble, i, &col_out));

    PetscCall(VecWAXPY(col_out, 1.0, mean, col_in));

    PetscCall(MatDenseRestoreColumnVecRead(X_G, i, &col_in));
    PetscCall(MatDenseRestoreColumnVecWrite(ensemble, i, &col_out));
  }

  PetscCall(MatDestroy(&X_G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                       Square Root Type Setters                            */
/* ========================================================================== */

static PetscErrorCode PetscDAETKFSetSqrt_Cholesky(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscDAETKFSetSqrtType(da, PETSCDAETKF_SQRT_CHOLESKY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDAETKFSetSqrt_Eigen(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscDAETKFSetSqrtType(da, PETSCDAETKF_SQRT_EIGEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                       ETKF Implementation Lifecycle                       */
/* ========================================================================== */

static PetscErrorCode PetscDAETKFDestroy(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(da->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASetFromOptions_DAETKF(PetscDA da, PetscOptionItems *PetscOptions)
{
  PetscDAETKFData *impl;
  PetscOptionItems PetscOptionsObject;
  const char      *defaultType;
  char             typeName[256];
  PetscBool        set              = PETSC_FALSE;
  PetscErrorCode (*setter)(PetscDA) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssert(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA data structure not initialized");

  impl               = (PetscDAETKFData *)da->data;
  PetscOptionsObject = PetscOptions ? *PetscOptions : NULL;

  defaultType = (impl->sqrt_type == PETSCDAETKF_SQRT_EIGEN) ? "eigen" : "cholesky";
  PetscCall(PetscStrncpy(typeName, defaultType, sizeof(typeName)));
  PetscCall(PetscOptionsFList("-petscdaetkf_sqrt_type", "Matrix square root factorization", "PetscDAETKFSetSqrtType", PetscDAETKFSqrtList, defaultType, typeName, sizeof(typeName), &set));
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
/*                          ETKF Analysis Algorithm                          */
/* ========================================================================== */

/*
  PetscDAETKFAnalysis - Performs the ensemble transform Kalman filter (ETKF) analysis defined by Algorithm 6.4 in Law, Stuart, and Zygalakis.

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
  Vec              mean, y_mean, delta_scaled, w, r_inv_sqrt;
  Mat              X, Z, S, T, T_sqrt, I_m, w_ones, T_sqrt_U, G;
  PetscInt         m;
  PetscScalar      inv_m, scale, sqrt_m_minus_1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);
  PetscAssertPointer(observation_operator, 3);

  /* Validate ensemble size */
  m = da->ensemble_size;
  PetscCheck(m > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be > 1, got %" PetscInt_FMT, m);

  impl           = (PetscDAETKFData *)da->data;
  inv_m          = 1.0 / ((PetscScalar)m);
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));

  /* ===================================================================== */
  /* Step 1-2: Compute ensemble mean and scaled anomalies                */
  /* ===================================================================== */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(mean));
  PetscCall(PetscDAComputeMean(da, mean));

  /* X = (E - x_mean * 1') / sqrt(m - 1) */
  PetscCall(PetscDAComputeAnomalies(da, &X));

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
    PetscCall(PetscDACholeskySqrt_Private(T, &T_sqrt));
    break;
  case PETSCDAETKF_SQRT_EIGEN:
    PetscCall(PetscDASymmetricEigenSqrt_Private(T, &T_sqrt));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDAETKF square-root type %" PetscInt_FMT, (PetscInt)impl->sqrt_type);
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
  PetscAssertPointer(model, 2);

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
- type - either `PETSCDAETKF_SQRT_CHOLESKY` or `PETSCDAETKF_SQRT_EIGEN`

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFGetSqrtType()`, `PetscDAETKFAnalysis()`
@*/
PetscErrorCode PetscDAETKFSetSqrtType(PetscDA da, PetscDAETKFSqrtType type)
{
  PetscDAETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA data structure not initialized");
  PetscCheck(type == PETSCDAETKF_SQRT_CHOLESKY || type == PETSCDAETKF_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDAETKF square-root type %" PetscInt_FMT, (PetscInt)type);

  impl            = (PetscDAETKFData *)da->data;
  impl->sqrt_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAETKFGetSqrtType - Retrieves the current square-root implementation configured for the ETKF analysis.

  Not Collective

  Input Parameters:
. da - the `PetscDA` object

  Output Parameter:
. type - on output, the configured `PetscDAETKFSqrtType`

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFSetSqrtType()`
@*/
PetscErrorCode PetscDAETKFGetSqrtType(PetscDA da, PetscDAETKFSqrtType *type)
{
  PetscDAETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCheck(da->data, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "PetscDA data structure not initialized");

  impl  = (PetscDAETKFData *)da->data;
  *type = impl->sqrt_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAETKFView - Views a `PetscDAETKF` and its implementation-specific data structure.

  Collective

  Input Parameters:
+ da     - the `PetscDA` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: beginner

.seealso: [](ch_da), `PetscDAViewFromOptions()`
*/
static PetscErrorCode PetscDAETKFView(PetscDA da, PetscViewer viewer)
{
  PetscBool        iascii;
  PetscDAETKFData *impl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)da), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(da, 1, viewer, 2);

  impl = (PetscDAETKFData *)da->data;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDAETKF Object:\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Square root type: %s\n", (impl->sqrt_type == PETSCDAETKF_SQRT_EIGEN) ? "eigen" : "cholesky"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAETKFInitialize - Installs the ETKF-specific operations on a newly created `PetscDA` object.

  Collective

  Input Parameter:
. da - the `PetscDA` object to configure

  Level: developer

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFRegister()`, `PetscDAETKFAnalysis()`
*/
static PetscErrorCode PetscDAETKFInitialize(PetscDA da)
{
  PetscDAETKFData *impl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&impl));
  impl->sqrt_type = PETSCDAETKF_SQRT_EIGEN;

  da->data                  = impl;
  da->ops->analysis         = PetscDAETKFAnalysis;
  da->ops->applymodel       = PetscDAETKFApplyModel;
  da->ops->computemean      = NULL;
  da->ops->computeanomalies = NULL;
  da->ops->destroy          = PetscDAETKFDestroy;
  da->ops->view             = PetscDAETKFView;
  da->ops->setfromoptions   = PetscDASetFromOptions_DAETKF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDAETKFRegister(void)
{
  PetscFunctionBegin;
  PetscCall(PetscDARegister(PETSCDAETKF, PetscDAETKFInitialize));
  PetscCall(PetscDAETKFInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}
