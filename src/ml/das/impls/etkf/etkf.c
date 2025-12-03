#include <petsc/private/dasimpl.h>
#include <petscblaslapack.h>

/* Helper function to compute ensemble mean and perturbations (Algorithm 6.4 Lines 1-2) */
static PetscErrorCode ComputeEnsembleMeanAndPerturbations(PetscDAS das, Vec *mean, Mat *X)
{
  PetscInt       i, m = das->ensemble_size;
  PetscReal      scale;

  PetscFunctionBegin;
  /* Compute ensemble mean */
  if (!*mean) PetscCall(VecDuplicate(das->ensemble[0], mean));
  PetscCall(VecSet(*mean, 0.0));
  for (i = 0; i < m; i++) PetscCall(VecAXPY(*mean, 1.0, das->ensemble[i]));
  scale = 1.0 / m;
  PetscCall(VecScale(*mean, scale));

  /* Compute normalized perturbations X[:,i] = (ensemble[i] - mean) / sqrt(m-1) */
  if (!*X) {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)das), PETSC_DECIDE, PETSC_DECIDE, das->state_size, m, NULL, X));
    PetscCall(MatSetUp(*X));
  }
  scale = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  for (i = 0; i < m; i++) {
    Vec col;
    PetscCall(MatDenseGetColumnVecWrite(*X, i, &col));
    PetscCall(VecWAXPY(col, -1.0, *mean, das->ensemble[i]));
    PetscCall(VecScale(col, scale));
    PetscCall(MatDenseRestoreColumnVecWrite(*X, i, &col));
  }
  PetscCall(MatAssemblyBegin(*X, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*X, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Helper function to transform ensemble to observation space (Algorithm 6.4 Lines 3-5) */
static PetscErrorCode TransformToObservationSpace(PetscDAS das, Mat *Z, Mat *S)
{
  PetscInt       i, m = das->ensemble_size, p;
  Vec            y_mean, col_z, col_s;
  PetscReal      scale;

  PetscFunctionBegin;
  PetscCall(MatGetSize(das->obs_operator, &p, NULL));

  /* Line 3: Apply observation operator Z[:,i] = H * ensemble[i] */
  if (!*Z) {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)das), PETSC_DECIDE, PETSC_DECIDE, p, m, NULL, Z));
    PetscCall(MatSetUp(*Z));
  }
  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecWrite(*Z, i, &col_z));
    PetscCall(MatMult(das->obs_operator, das->ensemble[i], col_z));
    PetscCall(MatDenseRestoreColumnVecWrite(*Z, i, &col_z));
  }
  PetscCall(MatAssemblyBegin(*Z, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Z, MAT_FINAL_ASSEMBLY));

  /* Line 4: Compute observation mean y_mean = (1/m) sum(Z[:,i]) */
  PetscCall(VecDuplicate(das->obs_error_cov_diag, &y_mean));
  PetscCall(VecSet(y_mean, 0.0));
  for (i = 0; i < m; i++) {
    PetscCall(MatDenseGetColumnVecRead(*Z, i, &col_z));
    PetscCall(VecAXPY(y_mean, 1.0, col_z));
    PetscCall(MatDenseRestoreColumnVecRead(*Z, i, &col_z));
  }
  scale = 1.0 / m;
  PetscCall(VecScale(y_mean, scale));

  /* Line 5: Compute S = R^(-1/2) * (Z - y_mean*1^T) / sqrt(m-1) */
  /* Use cached R^(-1/2) diagonal for efficiency */
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;
  
  if (!*S) {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)das), PETSC_DECIDE, PETSC_DECIDE, p, m, NULL, S));
    PetscCall(MatSetUp(*S));
  }
  scale = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  for (i = 0; i < m; i++) {
    Vec temp;
    PetscCall(VecDuplicate(y_mean, &temp));
    PetscCall(MatDenseGetColumnVecRead(*Z, i, &col_z));
    PetscCall(VecWAXPY(temp, -1.0, y_mean, col_z));
    PetscCall(MatDenseRestoreColumnVecRead(*Z, i, &col_z));

    /* Apply R^(-1/2) element-wise (R is diagonal) */
    PetscCall(VecPointwiseMult(temp, temp, etkf->R_inv_sqrt));
    
    PetscCall(MatDenseGetColumnVecWrite(*S, i, &col_s));
    PetscCall(VecCopy(temp, col_s));
    PetscCall(VecScale(col_s, scale));
    PetscCall(MatDenseRestoreColumnVecWrite(*S, i, &col_s));
    
    PetscCall(VecDestroy(&temp));
  }
  PetscCall(MatAssemblyBegin(*S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*S, MAT_FINAL_ASSEMBLY));
  PetscCall(VecDestroy(&y_mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Helper function to compute innovation (Algorithm 6.4 Line 6) */
static PetscErrorCode ComputeInnovation(PetscDAS das, Vec observation, Vec *delta)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;
  Vec            y_mean;
  PetscInt       i, m = das->ensemble_size, p;

  PetscFunctionBegin;
  PetscCall(MatGetSize(das->obs_operator, &p, NULL));

  /* Compute observation mean from Z (already computed in TransformToObservationSpace) */
  PetscCall(VecDuplicate(das->obs_error_cov_diag, &y_mean));
  PetscCall(VecSet(y_mean, 0.0));
  for (i = 0; i < m; i++) {
    Vec col_z;
    PetscCall(MatDenseGetColumnVecRead(etkf->Z, i, &col_z));
    PetscCall(VecAXPY(y_mean, 1.0, col_z));
    PetscCall(MatDenseRestoreColumnVecRead(etkf->Z, i, &col_z));
  }
  PetscCall(VecScale(y_mean, 1.0 / m));

  /* Compute innovation delta = R^(-1/2) * (y - y_mean) */
  if (!*delta) PetscCall(VecDuplicate(das->obs_error_cov_diag, delta));

  /* delta = R^(-1/2) * (observation - y_mean) using diagonal R */
  PetscCall(VecWAXPY(*delta, -1.0, y_mean, observation));
  PetscCall(VecPointwiseMult(*delta, *delta, etkf->R_inv_sqrt));

  PetscCall(VecDestroy(&y_mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Helper function to build and compute eigendecomposition of T = (I + S^T S)^(-1) (Algorithm 6.4 Line 7) */
static PetscErrorCode BuildAndFactorMatrix(PetscDAS das, Mat *I_plus_StS, Mat *V_T, Vec *D)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;
  PetscInt       m    = das->ensemble_size;

  PetscFunctionBegin;
  /* Compute I + S^T * S directly - always recreate to avoid accumulation issues */
  if (*I_plus_StS) PetscCall(MatDestroy(I_plus_StS));
  PetscCall(MatTransposeMatMult(etkf->S, etkf->S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, I_plus_StS));
  PetscCall(MatShift(*I_plus_StS, 1.0));

  /* Compute eigendecomposition of I + S^T S: (I + S^T S) = V D V^T */
  /* Always recreate V_T and D since LAPACK modifies them */
  if (*V_T) PetscCall(MatDestroy(V_T));
  if (*D) PetscCall(VecDestroy(D));
  PetscCall(MatDuplicate(*I_plus_StS, MAT_COPY_VALUES, V_T));
  PetscCall(MatCreateVecs(*I_plus_StS, D, NULL));
  
  PetscScalar *eig_vals, *eig_vecs;
  PetscBLASInt n_blas, lwork = -1, info_lapack;
  PetscScalar  work_query, *work;
  PetscCall(PetscBLASIntCast(m, &n_blas));
  PetscCall(MatDenseGetArray(*V_T, &eig_vecs));
  PetscCall(VecGetArray(*D, &eig_vals));
  
  /* Query optimal workspace */
  LAPACKsyev_("V", "U", &n_blas, eig_vecs, &n_blas, eig_vals, &work_query, &lwork, &info_lapack);
  lwork = (PetscBLASInt)PetscRealPart(work_query);
  PetscCall(PetscMalloc1(lwork, &work));
  
  /* Compute eigendecomposition */
  LAPACKsyev_("V", "U", &n_blas, eig_vecs, &n_blas, eig_vals, work, &lwork, &info_lapack);
  PetscCall(PetscFree(work));
  PetscCheck(info_lapack == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK syev failed with info = %d", (int)info_lapack);
  
  PetscCall(VecRestoreArray(*D, &eig_vals));
  PetscCall(MatDenseRestoreArray(*V_T, &eig_vecs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Helper function to solve for weights using eigendecomposition (Algorithm 6.4 Line 8) */
static PetscErrorCode SolveForWeights(PetscDAS das, Mat V_T, Vec D, Vec *w)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;
  Vec            rhs, temp;
  PetscInt       m = das->ensemble_size, j;

  PetscFunctionBegin;
  /* Compute rhs = S^T * delta */
  PetscCall(MatCreateVecs(etkf->S, &rhs, NULL));
  PetscCall(MatMultTranspose(etkf->S, etkf->delta, rhs));

  /* Solve (I + S^T S) * w = rhs using eigendecomposition: w = V * D^(-1) * V^T * rhs */
  if (!*w) { PetscCall(VecDuplicate(rhs, w)); }
  PetscCall(VecDuplicate(rhs, &temp));

  /* temp = V^T * rhs */
  PetscCall(MatMultTranspose(V_T, rhs, temp));
  
  /* Apply D^(-1) */
  PetscScalar *temp_arr, *eig_vals;
  PetscCall(VecGetArray(temp, &temp_arr));
  PetscCall(VecGetArray(D, &eig_vals));
  for (j = 0; j < m; j++) {
    PetscCheck(PetscRealPart(eig_vals[j]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Eigenvalue %d is non-positive: %g", (int)j, (double)PetscRealPart(eig_vals[j]));
    temp_arr[j] /= eig_vals[j];
  }
  PetscCall(VecRestoreArray(D, &eig_vals));
  PetscCall(VecRestoreArray(temp, &temp_arr));
  
  /* w = V * (D^(-1) * temp) */
  PetscCall(MatMult(V_T, temp, *w));

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Helper function to update ensemble using eigendecomposition (Algorithm 6.4 Line 9) */
static PetscErrorCode UpdateEnsembleWithEigen(PetscDAS das, Mat V_T, Vec D)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;
  PetscInt       i, m = das->ensemble_size;
  Vec            temp, x_update, t_sqrt_col;
  PetscReal      scale = PetscSqrtReal((PetscReal)(m - 1));
  Mat            T_sqrt;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(V_T, &temp, NULL));
  PetscCall(VecDuplicate(das->ensemble[0], &x_update));

  /* Compute T^(1/2) = V * D^(-1/2) * V^T where T = (I + S^T S)^(-1) */
  /* First create V * D^(-1/2) by scaling columns of V */
  Mat V_D_sqrt;
  PetscCall(MatDuplicate(V_T, MAT_COPY_VALUES, &V_D_sqrt));
  
  PetscScalar *eig_vals;
  PetscCall(VecGetArray(D, &eig_vals));
  for (i = 0; i < m; i++) {
    Vec col;
    PetscCheck(PetscRealPart(eig_vals[i]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Eigenvalue %d is non-positive: %g", (int)i, (double)PetscRealPart(eig_vals[i]));
    PetscCall(MatDenseGetColumnVecWrite(V_D_sqrt, i, &col));
    PetscCall(VecScale(col, 1.0 / PetscSqrtScalar(eig_vals[i])));
    PetscCall(MatDenseRestoreColumnVecWrite(V_D_sqrt, i, &col));
  }
  PetscCall(VecRestoreArray(D, &eig_vals));

  /* Compute T^(1/2) = V * D^(-1/2) * V^T */
  PetscCall(MatTransposeMatMult(V_D_sqrt, V_T, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_sqrt));
  PetscCall(MatDestroy(&V_D_sqrt));

  /* Update each ensemble member: E[:, i] = mean + X * (w + sqrt(m-1) * T^(1/2)[:, i]) */
  /* Each column i of T^(1/2) provides the random perturbation for ensemble member i */
  for (i = 0; i < m; i++) {
    /* Get column i of T^(1/2) */
    PetscCall(MatDenseGetColumnVecRead(T_sqrt, i, &t_sqrt_col));
    
    /* Compute w + sqrt(m-1) * T^(1/2)[:, i] */
    PetscCall(VecCopy(etkf->w, temp));
    PetscCall(VecAXPY(temp, scale, t_sqrt_col));
    PetscCall(MatDenseRestoreColumnVecRead(T_sqrt, i, &t_sqrt_col));
    
    /* Apply to perturbations: X * temp */
    PetscCall(MatMult(etkf->X, temp, x_update));
    
    /* Final ensemble update: ensemble[i] = mean + X * (w + sqrt(m-1) * T^(1/2)[:, i]) */
    PetscCall(VecWAXPY(das->ensemble[i], 1.0, x_update, etkf->ensemble_mean));
  }

  PetscCall(MatDestroy(&T_sqrt));
  PetscCall(VecDestroy(&temp));
  PetscCall(VecDestroy(&x_update));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Main assimilation function implementing Algorithm 6.4 */
static PetscErrorCode PetscDASAssimilate_ETKF(PetscDAS das, Vec observation)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;

  PetscFunctionBegin;
  /* Lines 1-2: Compute ensemble mean and perturbations */
  PetscCall(ComputeEnsembleMeanAndPerturbations(das, &etkf->ensemble_mean, &etkf->X));

  /* Lines 3-5: Transform to observation space */
  PetscCall(TransformToObservationSpace(das, &etkf->Z, &etkf->S));

  /* Line 6: Compute innovation */
  PetscCall(ComputeInnovation(das, observation, &etkf->delta));

  /* Line 7: Build and compute eigendecomposition of I + S^T S */
  PetscCall(BuildAndFactorMatrix(das, &etkf->I_plus_StS, &etkf->V_T, &etkf->D_T));

  /* Line 8: Solve for weights using eigendecomposition */
  PetscCall(SolveForWeights(das, etkf->V_T, etkf->D_T, &etkf->w));

  /* Line 9: Update ensemble using eigendecomposition */
  PetscCall(UpdateEnsembleWithEigen(das, etkf->V_T, etkf->D_T));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Forecast function (Algorithm 6.4 Line 10) */
static PetscErrorCode PetscDASForecast_ETKF(PetscDAS das, Mat model_operator)
{
  PetscInt i;
  Vec      temp;

  PetscFunctionBegin;
  if (model_operator) {
    /* Linear forecast: ensemble[i] = model_operator * ensemble[i] */
    PetscCall(VecDuplicate(das->ensemble[0], &temp));
    for (i = 0; i < das->ensemble_size; i++) {
      PetscCall(MatMult(model_operator, das->ensemble[i], temp));
      PetscCall(VecCopy(temp, das->ensemble[i]));
    }
    PetscCall(VecDestroy(&temp));
  }
  /* For nonlinear models, the user should apply their model directly to the ensemble */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASSetUp_ETKF(PetscDAS das)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;

  PetscFunctionBegin;
  /* Check that required parameters are set */
  PetscCheck(das->ensemble_size > 0, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_WRONGSTATE, "Ensemble size must be set");
  PetscCheck(das->state_size > 0, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_WRONGSTATE, "State size must be set");
  PetscCheck(das->obs_operator, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_WRONGSTATE, "Observation operator must be set");
  PetscCheck(das->obs_error_cov_diag, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_WRONGSTATE, "Observation error covariance diagonal must be set");
  PetscCheck(das->ensemble, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_WRONGSTATE, "Ensemble must be set");

  /* Compute R^(-1/2) diagonal once */
  PetscCall(VecDuplicate(das->obs_error_cov_diag, &etkf->R_inv_sqrt));
  PetscCall(VecCopy(das->obs_error_cov_diag, etkf->R_inv_sqrt));
  PetscCall(VecSqrtAbs(etkf->R_inv_sqrt));
  PetscCall(VecReciprocal(etkf->R_inv_sqrt));

  /* Initialize ETKF-specific data if needed */
  etkf->ensemble_mean = NULL;
  etkf->X             = NULL;
  etkf->Z             = NULL;
  etkf->S             = NULL;
  etkf->delta         = NULL;
  etkf->I_plus_StS    = NULL;
  etkf->V_T           = NULL;
  etkf->D_T           = NULL;
  etkf->w             = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASReset_ETKF(PetscDAS das)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&etkf->ensemble_mean));
  PetscCall(MatDestroy(&etkf->X));
  PetscCall(MatDestroy(&etkf->Z));
  PetscCall(MatDestroy(&etkf->S));
  PetscCall(VecDestroy(&etkf->delta));
  PetscCall(MatDestroy(&etkf->I_plus_StS));
  PetscCall(MatDestroy(&etkf->V_T));
  PetscCall(VecDestroy(&etkf->D_T));
  PetscCall(VecDestroy(&etkf->w));
  PetscCall(VecDestroy(&etkf->R_inv_sqrt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASDestroy_ETKF(PetscDAS das)
{
  PetscFunctionBegin;
  PetscCall(PetscDASReset_ETKF(das));
  PetscCall(PetscFree(das->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASView_ETKF(PetscDAS das, PetscViewer viewer)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ETKF Data Assimilation\n"));
    if (etkf->inflation != 1.0) PetscCall(PetscViewerASCIIPrintf(viewer, "  Inflation factor: %g\n", (double)etkf->inflation));
    if (etkf->use_localization) PetscCall(PetscViewerASCIIPrintf(viewer, "  Localization radius: %g\n", (double)etkf->loc_radius));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDASSetFromOptions_ETKF(PetscDAS das, PetscOptionItems PetscOptionsObject)
{
  PetscDAS_ETKF *etkf = (PetscDAS_ETKF *)das->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "ETKF Data Assimilation options");
  PetscCall(PetscOptionsReal("-das_etkf_inflation", "Multiplicative inflation factor", "PetscDASETKFSetInflation", etkf->inflation, &etkf->inflation, &flg));
  PetscCall(PetscOptionsReal("-das_etkf_localization_radius", "Localization radius", "", etkf->loc_radius, &etkf->loc_radius, &etkf->use_localization));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASETKFSetInflation - Set the multiplicative inflation factor for ETKF

  Logically Collective

  Input Parameters:
+ das       - the `PetscDAS` context (must be of type ETKF)
- inflation - the inflation factor (>= 1.0)

  Level: intermediate

.seealso: `PetscDAS`, `PETSCDASETKF`
@*/
PetscErrorCode PetscDASETKFSetInflation(PetscDAS das, PetscReal inflation)
{
  PetscDAS_ETKF *etkf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscValidLogicalCollectiveReal(das, inflation, 2);
  PetscCheck(inflation >= 1.0, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_OUTOFRANGE, "Inflation factor must be >= 1.0");
  etkf            = (PetscDAS_ETKF *)das->data;
  etkf->inflation = inflation;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PETSCDASETKF - Ensemble Transform Kalman Filter (ETKF) data assimilation

   Options Database:
+  -das_etkf_inflation <value>           - Multiplicative covariance inflation factor
-  -das_etkf_localization_radius <value> - Localization radius

   Level: beginner

   Notes:
   Implements Algorithm 6.4 ETKF from data assimilation literature.
   Uses eigenvalue decomposition for matrix operations.
   All matrices are stored in dense format for eigenvalue computation.

.seealso: `PetscDASCreate()`, `PetscDAS`, `PetscDASSetType()`
M*/
PETSC_EXTERN PetscErrorCode PetscDASCreate_ETKF(PetscDAS das)
{
  PetscDAS_ETKF *etkf;

  PetscFunctionBegin;
  PetscCall(PetscNew(&etkf));
  das->data = (void *)etkf;

  das->ops->setup          = PetscDASSetUp_ETKF;
  das->ops->setfromoptions = PetscDASSetFromOptions_ETKF;
  das->ops->assimilate     = PetscDASAssimilate_ETKF;
  das->ops->forecast       = PetscDASForecast_ETKF;
  das->ops->reset          = PetscDASReset_ETKF;
  das->ops->destroy        = PetscDASDestroy_ETKF;
  das->ops->view           = PetscDASView_ETKF;

  etkf->inflation        = 1.0; /* Default: no inflation */
  etkf->use_localization = PETSC_FALSE;
  etkf->loc_radius       = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
