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
  PetscInt          m, n_vertices;
  PetscScalar       scale, sqrt_m_minus_1;
  PetscBool         reallocate = PETSC_FALSE;

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
  /* Perform local analysis for all vertices */
  /* ===================================================================== */

#if defined(PETSC_HAVE_KOKKOS)
  PetscCall(PetscDALETKFLocalAnalysis(da, impl, m, n_vertices, scale, sqrt_m_minus_1, X, observation, impl->Z, impl->y_mean, impl->r_inv_sqrt));
#else
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "KOKKOS require for DALETKF")
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Local observations per vertex: %" PetscInt_FMT "\n", impl->p_local));
    if (impl->Q) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization matrix: set\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization matrix: not set\n"));
    }
  }
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
  da->ops->setfromoptions   = NULL;

  /* Initialize default values */
  impl->p_local = Q_NUM_LOCAL_OBSERVATIONS_MAX;
  impl->Q       = NULL;

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
