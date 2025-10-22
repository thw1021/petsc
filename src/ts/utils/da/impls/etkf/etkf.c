#include "petscda.h"
#include <petsc/private/daimpl.h>

typedef struct {
  DAETKFSqrtType sqrt_type;
} DAETKFData;

static PetscFunctionList DAETKFSqrtList = NULL;
static PetscBool         DAETKFPackageInitialized;

static PetscErrorCode DAETKFSetSqrt_Cholesky(DA da)
{
  PetscFunctionBegin;
  PetscCall(DAETKFSetSqrtType(da, DAETKF_SQRT_CHOLESKY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DAETKFSetSqrt_Eigen(DA da)
{
  PetscFunctionBegin;
  PetscCall(DAETKFSetSqrtType(da, DAETKF_SQRT_EIGEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DAETKFDestroy(DA da)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(da->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DASetFromOptions_DAETKF(DA da, PetscOptionItems *PetscOptions)
{
  DAETKFData      *impl               = (DAETKFData *)da->data;
  PetscOptionItems PetscOptionsObject = PetscOptions ? *PetscOptions : NULL;
  const char      *defaultType;
  char             typeName[64];
  PetscBool        set;
  PetscErrorCode (*setter)(DA);

  PetscFunctionBegin;
  defaultType = (impl->sqrt_type == DAETKF_SQRT_EIGEN) ? "eigen" : "cholesky";
  PetscCall(PetscStrncpy(typeName, defaultType, sizeof(typeName)));
  PetscCall(PetscOptionsFList("-daetkf_sqrt_type", "Matrix square root factorization", "DAETKFSetSqrtType", DAETKFSqrtList, defaultType, typeName, sizeof(typeName), &set));
  if (set) {
    PetscCall(PetscFunctionListFind(DAETKFSqrtList, typeName, &setter));
    PetscCheck(setter, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DAETKF square-root type \"%s\"", typeName);
    PetscCall((*setter)(da));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DAETKFInitializePackage - This function initializes everything in the `DAETKF` package. It is called from `TSInitializePackage()`.

  Level: developer

.seealso: [](ch_ts), `PetscInitialize()`, `DAETKFFinalizePackage()`
@*/
PetscErrorCode DAETKFInitializePackage(void)
{
  PetscFunctionBegin;
  if (DAETKFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  DAETKFPackageInitialized = PETSC_TRUE;
  PetscCall(PetscFunctionListAdd(&DAETKFSqrtList, "cholesky", DAETKFSetSqrt_Cholesky));
  PetscCall(PetscFunctionListAdd(&DAETKFSqrtList, "eigen", DAETKFSetSqrt_Eigen));
  PetscCall(PetscRegisterFinalize(DAETKFFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DAETKFFinalizePackage - This function destroys everything in the `DAETKF` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ts), `PetscFinalize()`, `DAETKFInitiallizePackage()`
@*/
PetscErrorCode DAETKFFinalizePackage(void)
{
  PetscFunctionBegin;
  DAETKFPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&DAETKFSqrtList));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DAETKFAnalysis - Performs the ensemble transform Kalman filter (ETKF) analysis defined by Algorithm 6.4 in Law, Stuart, and Zygalakis.

  Collective

  Input Parameters:
+ da                   - the `DA` context owning the forecast ensemble and
buffers . observation          - observation vector `y` . observation_operator -
user-supplied routine `H(x, y; ctx)` that maps a state to observation space
- obs_ctx              - optional context for `observation_operator`

  Notes:
  The implementation follows the book's deterministic ETKF steps verbatim:
  Step 1 computes the state mean, Step 2 the state anomalies, Steps 3-4 build the normalized innovation statistics, Step 5 assembles the reduced-space inverse, Step 6 forms the analysis weights, Steps 7-9 construct the square-root
transform, and Step 10 applies the transform to refresh every ensemble member.

  Level: advanced

.seealso: [](ch_da), `DA`, `DAETKFApplyModel()`, `DAComputeMean()`,
`DAComputeAnomalies()`
*/
static PetscErrorCode DAETKFAnalysis(DA da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  DAETKFData  *impl = (DAETKFData *)da->data;
  Vec          mean, y_mean, delta, delta_scaled, col_in, col_out, w, temp_vec;
  Mat          X, Z, S, T, T_sqrt;
  PetscInt     i, m;
  PetscScalar  inv_m, scale, sqrt_m_minus_1;
  PetscScalar *r_inv_sqrt_array, *w_array;
  Vec          r_inv_sqrt;
  Mat          I_m, w_ones, T_sqrt_U, X_Y;
  Vec          S_T_delta;

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

  /* Step 1: compute the prior ensemble mean in state space. */
  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(mean));
  PetscCall(DAComputeMean(da, mean));

  /* Step 2: form anomalies X scaled by 1/sqrt(m-1); columns span the ensemble
   * subspace. */
  PetscCall(DAComputeAnomalies(da, &X));

  /* Buffer to hold raw observation-space evaluations H(x_i^f). */
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

  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &y_mean));
  PetscCall(VecSetSizes(y_mean, PETSC_DECIDE, da->obs_size));
  PetscCall(VecSetFromOptions(y_mean));
  PetscCall(VecSet(y_mean, 0.0)); /* Step 3: observation-space mean y_bar. */

  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(Z, i, &col_in));
    PetscCall(VecAXPY(y_mean, inv_m, col_in)); /* accumulate 1/m * H(x_i^f). */
    PetscCall(MatDenseRestoreColumnVecRead(Z, i, &col_in));
  }

  PetscCall(VecDuplicate(da->obs_error_var, &r_inv_sqrt));
  PetscCall(VecCopy(da->obs_error_var, r_inv_sqrt)); /* Step 3: reuse diag(R) to build R^{-1/2}. */
  PetscCall(VecGetArray(r_inv_sqrt, &r_inv_sqrt_array));
  for (i = 0; i < da->obs_size; i++) r_inv_sqrt_array[i] = 1.0 / PetscSqrtReal(PetscRealPart(r_inv_sqrt_array[i])); /* assumes R diagonal. */
  PetscCall(VecRestoreArray(r_inv_sqrt, &r_inv_sqrt_array));

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

  delta = y_mean;
  PetscCall(VecAYPX(delta, -1.0, observation)); /* Innovation y^o - y_bar. */

  delta_scaled = delta;
  PetscCall(VecPointwiseMult(delta_scaled, delta_scaled, r_inv_sqrt)); /* Step 4: del_tilda. */

  PetscCall(MatTransposeMatMult(S, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T)); /* Step 5: S S'. */

  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &I_m));
  PetscCall(MatSetUp(I_m));
  PetscCall(MatZeroEntries(I_m));
  for (i = 0; i < m; i++) PetscCall(MatSetValue(I_m, i, i, 1.0, INSERT_VALUES)); /* Identity in reduced space. */
  PetscCall(MatAssemblyBegin(I_m, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(I_m, MAT_FINAL_ASSEMBLY));

  // optimize to not invert T but solve with Cholesky or eigen data -- TODO
  PetscCall(MatAXPY(T, 1.0, I_m, SAME_NONZERO_PATTERN)); /* Step 5: T = S S' + I. */

  PetscCall(MatLUFactor(T, NULL, NULL, NULL));
  Mat T_inv;
  PetscCall(MatDuplicate(I_m, MAT_COPY_VALUES, &T_inv));
  PetscCall(MatMatSolve(T, I_m, T_inv));
  PetscCall(MatDestroy(&T));
  T = T_inv;

  PetscCall(MatCreateVecs(T, NULL, &S_T_delta));
  PetscCall(MatMultTranspose(S, delta_scaled, S_T_delta)); /* S' del_tilda. */

  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &w));
  PetscCall(VecSetSizes(w, PETSC_DECIDE, m));
  PetscCall(VecSetFromOptions(w));
  PetscCall(MatMult(T, S_T_delta, w)); /* Step 6: weights in ensemble space. */

  switch (impl->sqrt_type) {
  case DAETKF_SQRT_CHOLESKY:
    PetscCall(DACholeskySqrt_Private(T, &T_sqrt)); /* Step 7a. */
    break;
  case DAETKF_SQRT_EIGEN:
    PetscCall(DASymmetricEigenSqrt_Private(T, &T_sqrt)); /* Step 7b. */
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported DAETKF square-root type %" PetscInt_FMT, (PetscInt)impl->sqrt_type);
  }

  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, m, m, NULL, &w_ones));
  PetscCall(MatSetUp(w_ones));
  PetscCall(VecGetArray(w, &w_array));
  for (i = 0; i < m; i++) {
    for (PetscInt j = 0; j < m; j++) PetscCall(MatSetValue(w_ones, i, j, w_array[i], INSERT_VALUES)); /* w replicated across columns (w * 1'). */
  }
  PetscCall(VecRestoreArray(w, &w_array));
  PetscCall(MatAssemblyBegin(w_ones, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(w_ones, MAT_FINAL_ASSEMBLY));

  PetscCall(MatMatMult(T_sqrt, da->U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_sqrt_U));
  PetscCall(MatScale(T_sqrt_U, sqrt_m_minus_1)); /* Step 8: rotate and undo scaling. */

  PetscCall(MatAXPY(w_ones, 1.0, T_sqrt_U, DIFFERENT_NONZERO_PATTERN)); /* Step 9: deterministic transform. */

  PetscCall(MatMatMult(X, w_ones, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X_Y)); /* Transformed anomalies. */

  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(X_Y, i, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(da->ensemble, i, &col_out));

    PetscCall(VecWAXPY(col_out, 1.0, mean, col_in)); /* Step 10: final analysis member. */

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
  DAETKFApplyModel - Advances each ensemble member through the user-supplied
  nonlinear model (Algorithm 6.4, Step 10 forecast propagation).

  Collective

  Input Parameters:
+ da     - the `DA` context that stores the ensemble
. model  - routine that evaluates the model `f(x, xnew; ctx)`
- model_ctx - optional context for `model`

  Notes:
  This routine overwrites every ensemble column with the model result supplied by `model`. It is typically called immediately after `DAETKFAnalysis()` to start the next forecast cycle.

  Level: intermediate

.seealso: [](ch_da), `DA`, `DAETKFAnalysis()`
*/
static PetscErrorCode DAETKFApplyModel(DA da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
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
  DAETKFSetSqrtType - Selects the reduced-space square-root algorithm used during the ETKF analysis.

  Logically Collective

  Input Parameters:
+ da   - the `DA` object
- type - either `DAETKF_SQRT_CHOLESKY` or `DAETKF_SQRT_EIGEN`

  Level: intermediate

.seealso: [](ch_da), `DA`, `DAETKFGetSqrtType()`, `DAETKFAnalysis()`
@*/
PetscErrorCode DAETKFSetSqrtType(DA da, DAETKFSqrtType type)
{
  DAETKFData *impl = (DAETKFData *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscCheck(type == DAETKF_SQRT_CHOLESKY || type == DAETKF_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid DAETKF square-root type %" PetscInt_FMT, (PetscInt)type);

  impl->sqrt_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAETKFGetSqrtType - Retrieves the current square-root implementation configured for the ETKF analysis.

  Not Collective

  Input Parameters:
+ da - the `DA` object

  Output Parameter:
. type - on output, the configured `DAETKFSqrtType`

  Level: intermediate

.seealso: [](ch_da), `DA`, `DAETKFSetSqrtType()`
@*/
PetscErrorCode DAETKFGetSqrtType(DA da, DAETKFSqrtType *type)
{
  DAETKFData *impl = (DAETKFData *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(type, 2);

  *type = impl->sqrt_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DAETKFInitialize - Installs the ETKF-specific operations on a newly created `DA` object.

  Collective

  Input Parameter:
. da - the `DA` object to configure

  Level: developer

.seealso: [](ch_da), `DA`, `DAETKFRegister()`, `DAETKFAnalysis()`
*/
static PetscErrorCode DAETKFInitialize(DA da)
{
  DAETKFData *impl;

  PetscFunctionBegin;
  PetscCall(PetscNew(&impl));
  impl->sqrt_type = DAETKF_SQRT_EIGEN;

  da->data                  = impl;
  da->ops->analysis         = DAETKFAnalysis;
  da->ops->applymodel       = DAETKFApplyModel;
  da->ops->computemean      = NULL;
  da->ops->computeanomalies = NULL;
  da->ops->destroy          = DAETKFDestroy;
  da->ops->view             = NULL;
  da->ops->setfromoptions   = DASetFromOptions_DAETKF;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DAETKFRegister(void)
{
  PetscFunctionBegin;
  PetscCall(DARegister(DAETKF, DAETKFInitialize));
  PetscCall(DAETKFInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}
