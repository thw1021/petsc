#include "petscda.h"
#include <petsc/private/daimpl.h>

typedef struct {
  PetscDAETKFSqrtType sqrt_type;
} PetscDAETKFData;

static PetscFunctionList PetscDAETKFSqrtList = NULL;
static PetscBool         PetscDAETKFPackageInitialized;

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

static PetscErrorCode PetscDAETKFDestroy(PetscDA da)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(da->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASetFromOptions_DAETKF(PetscDA da, PetscOptionItems *PetscOptions)
{
  PetscDAETKFData  *impl               = (PetscDAETKFData *)da->data;
  PetscOptionItems  PetscOptionsObject = PetscOptions ? *PetscOptions : NULL;
  const char       *defaultType;
  char              typeName[64];
  PetscBool         set;
  PetscErrorCode (*setter)(PetscDA);

  PetscFunctionBegin;
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

.seealso: [](ch_ts), `PetscFinalize()`, `PetscDAETKFInitiallizePackage()`
@*/
PetscErrorCode PetscDAETKFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscDAETKFPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&PetscDAETKFSqrtList));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAETKFAnalysis - Performs the ensemble transform Kalman filter (ETKF) analysis defined by Algorithm 6.4 in Law, Stuart, and Zygalakis.

  Collective

  Input Parameters:
+ da                   - the `PetscDA` context owning the forecast ensemble and
buffers . observation          - observation vector `y` . observation_operator -
user-supplied routine `H(x, y; ctx)` that maps a state to observation space
- obs_ctx              - optional context for `observation_operator`

  Notes:
  The implementation follows the book's deterministic ETKF steps verbatim:
  Step 1 computes the state mean, Step 2 the state anomalies, Steps 3-4 build the normalized innovation statistics, Step 5 assembles the reduced-space inverse, Step 6 forms the analysis weights, Steps 7-9 construct the square-root
transform, and Step 10 applies the transform to refresh every ensemble member.

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFApplyModel()`, `PetscDAComputeMean()`,
`PetscDAComputeAnomalies()`
*/
static PetscErrorCode PetscDAETKFAnalysis(PetscDA da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  PetscDAETKFData *impl = (PetscDAETKFData *)da->data;
  Vec              mean, y_mean, delta, delta_scaled, col_in, col_out, w, temp_vec;
  Mat              X, Z, S, T, T_sqrt;
  PetscInt         i, m;
  PetscScalar      inv_m, scale, sqrt_m_minus_1;
  PetscScalar     *r_inv_sqrt_array, *w_array;
  Vec              r_inv_sqrt;
  Mat              I_m, w_ones, T_sqrt_U, X_Y;
  Vec              S_T_delta;

  PetscFunctionBegin;
  /* Map of ETKF analysis steps to Algorithm 6.4:
   *   1-2: build ensemble mean X and observation ensemble Z (lines below)
   *   3-4: compute normalized innovation statistics (S, delta_scaled)
   *   5: form T = (Y_f Y_f^T + R)^(-1) via factorization + solve
   *   6-9: produce weight vector w and square-root transform T_sqrt
   *  10: update ensemble states with mean + anomalies * transform
   */

  m              = da->ensemble_size;
  inv_m          = 1.0 / ((PetscScalar)m);
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));

  /* compute the prior/ensemble mean */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(mean));
  PetscCall(PetscDAComputeMean(da, mean));

  /* scaled anomalies X = (E - x_bar)/sqrt(m - 1) */
  PetscCall(PetscDAComputeAnomalies(da, &X));

  /* Z = H(x_i^f). */
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, &Z));
  PetscCall(MatSetUp(Z));

  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &temp_vec));
  PetscCall(VecSetSizes(temp_vec, PETSC_DECIDE, da->obs_size));
  PetscCall(VecSetFromOptions(temp_vec));

  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, i, &col_in));
    PetscCall(observation_operator(col_in, temp_vec, obs_ctx)); /* H(x_i^f) */
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, i, &col_in));

    PetscCall(MatDenseGetColumnVecWrite(Z, i, &col_out));
    PetscCall(VecCopy(temp_vec, col_out));
    PetscCall(MatDenseRestoreColumnVecWrite(Z, i, &col_out));
  }

  /* observation mean y_bar */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &y_mean));
  PetscCall(VecSetSizes(y_mean, PETSC_DECIDE, da->obs_size));
  PetscCall(VecSetFromOptions(y_mean));
  PetscCall(MatGetRowSum(Z, y_mean)); /* accumulate 1/m * H(x_i^f). */
  PetscCall(VecScale(y_mean, inv_m));

  // r_inv_sqrt: R^-1/2
  PetscCall(VecDuplicate(da->obs_error_var, &r_inv_sqrt));
  PetscCall(VecCopy(da->obs_error_var, r_inv_sqrt)); /* Step 3: reuse diag(R) to build R^{-1/2}. */
  PetscCall(VecGetArray(r_inv_sqrt, &r_inv_sqrt_array));
  for (i = 0; i < da->obs_size; i++) r_inv_sqrt_array[i] = 1.0 / PetscSqrtReal(PetscRealPart(r_inv_sqrt_array[i])); /* assumes R diagonal. */
  PetscCall(VecRestoreArray(r_inv_sqrt, &r_inv_sqrt_array));

  // create S = R^-1/2 (Z - y_bar * 1)/sqrt(m - 1)
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->obs_size, m, NULL, &S));
  PetscCall(MatSetUp(S));
  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(Z, i, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(S, i, &col_out));

    PetscCall(VecWAXPY(col_out, -1.0, y_mean, col_in));        /* H(x_i^f) - y_bar */
    PetscCall(VecScale(col_out, scale));                       /* 1/sqrt(m-1) scaling */
    PetscCall(VecPointwiseMult(col_out, col_out, r_inv_sqrt)); /* Step 3: S = R^{-1/2} Y_f */

    PetscCall(MatDenseRestoreColumnVecRead(Z, i, &col_in));
    PetscCall(MatDenseRestoreColumnVecWrite(S, i, &col_out));
  }

  /* Innovation y^o - y_bar. */
  delta = y_mean;
  PetscCall(VecAYPX(delta, -1.0, observation));
  delta_scaled = delta;
  PetscCall(VecPointwiseMult(delta_scaled, delta_scaled, r_inv_sqrt)); /* delta. */

  // I_m
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &I_m));
  PetscCall(MatSetUp(I_m));
  PetscCall(MatZeroEntries(I_m));
  for (i = 0; i < m; i++) PetscCall(MatSetValue(I_m, i, i, 1.0, INSERT_VALUES)); /* Identity in reduced space. */
  PetscCall(MatAssemblyBegin(I_m, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(I_m, MAT_FINAL_ASSEMBLY));

  // T = (I + S' S)^-1
  PetscCall(MatTransposeMatMult(S, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T));
  PetscCall(MatAXPY(T, 1.0, I_m, SAME_NONZERO_PATTERN)); /* Algorithm 3.1 Step 6: T = (S^T S + I) (doc/manual/da_ex1.tex:84). */

  // Invert T -- todo: optimize to not invert T but solve with Cholesky or eigen data
  PetscCall(MatLUFactor(T, NULL, NULL, NULL));
  Mat T_inv;
  PetscCall(MatDuplicate(I_m, MAT_COPY_VALUES, &T_inv));
  PetscCall(MatMatSolve(T, I_m, T_inv));
  PetscCall(MatDestroy(&T));
  T = T_inv; // T = (I + S'S)^-1

  // make w = T S^T delta
  PetscCall(MatCreateVecs(T, NULL, &S_T_delta));
  PetscCall(MatMultTranspose(S, delta_scaled, S_T_delta)); /* Algorithm 3.1 Step 7: S^T delta */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &w));
  PetscCall(VecSetSizes(w, PETSC_DECIDE, m));
  PetscCall(VecSetFromOptions(w));
  PetscCall(MatMult(T, S_T_delta, w)); // w = T S^T delta

  /* T_sqrt = square-root of T */
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
  // w * 1
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &w_ones));
  PetscCall(MatSetUp(w_ones));
  PetscCall(VecGetArray(w, &w_array));
  for (i = 0; i < m; i++) {
    for (PetscInt j = 0; j < m; j++) PetscCall(MatSetValue(w_ones, i, j, w_array[i], INSERT_VALUES)); /* w replicated across columns (w * 1'). */
  }
  PetscCall(VecRestoreArray(w, &w_array));
  PetscCall(MatAssemblyBegin(w_ones, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(w_ones, MAT_FINAL_ASSEMBLY));

  // w_ones: G = w * 1 + (m - 1) T^1/2 U -- U = I -- remove
  PetscCall(MatMatMult(T_sqrt, da->U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_sqrt_U));
  PetscCall(MatScale(T_sqrt_U, sqrt_m_minus_1)); /* sqrt(m - 1) T^.5 U */
  PetscCall(MatAXPY(w_ones, 1.0, T_sqrt_U, DIFFERENT_NONZERO_PATTERN));
  // X_Y = X G
  PetscCall(MatMatMult(X, w_ones, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X_Y));
  // da->ensemble: E = mean + X G
  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(X_Y, i, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(da->ensemble, i, &col_out));

    PetscCall(VecWAXPY(col_out, 1.0, mean, col_in));

    PetscCall(MatDenseRestoreColumnVecRead(X_Y, i, &col_in));
    PetscCall(MatDenseRestoreColumnVecWrite(da->ensemble, i, &col_out));
  }

  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&y_mean));
  PetscCall(VecDestroy(&temp_vec));
  PetscCall(VecDestroy(&r_inv_sqrt));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&S_T_delta));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Z));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&I_m));
  PetscCall(MatDestroy(&T));
  PetscCall(MatDestroy(&T_sqrt));
  PetscCall(MatDestroy(&w_ones));
  PetscCall(MatDestroy(&T_sqrt_U));
  PetscCall(MatDestroy(&X_Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAETKFApplyModel - Advances each ensemble member through the user-supplied
  nonlinear model (Algorithm 6.4, Step 10 forecast propagation).

  Collective

  Input Parameters:
+ da     - the `PetscDA` context that stores the ensemble
. model  - routine that evaluates the model `f(x, xnew; ctx)`
- model_ctx - optional context for `model`

  Notes:
  This routine overwrites every ensemble column with the model result supplied by `model`. It is typically called immediately after `PetscDAETKFAnalysis()` to start the next forecast cycle.

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PetscDAETKFAnalysis()`
*/
static PetscErrorCode PetscDAETKFApplyModel(PetscDA da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
{
  Vec      col_in, col_out, temp;
  PetscInt i;

  PetscFunctionBegin;
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
  PetscDAETKFData *impl = (PetscDAETKFData *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCheck(type == PETSCDAETKF_SQRT_CHOLESKY || type == PETSCDAETKF_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDAETKF square-root type %" PetscInt_FMT, (PetscInt)type);

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
  PetscDAETKFData *impl = (PetscDAETKFData *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);

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
  PetscDAETKFData *impl = (PetscDAETKFData *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)da), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(da, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDAEKF Object:\n"));
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
