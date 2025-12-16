/*
   Data Assimilation Interface - Ensemble-based data assimilation methods for PETSc

   This file provides the public interface for the PETSc Data Assimilation (DA) package,
   which implements ensemble-based methods for state estimation in dynamical systems.
   The primary implementation is the Ensemble Transform Kalman Filter (ETKF), a
   deterministic square-root filter that avoids stochastic perturbations.

   The ETKF algorithm is based on:
   Asch, M., Bocquet, M., and Nodet, M. (2016).
   "Data Assimilation: Methods, Algorithms, and Applications"
   SIAM, Philadelphia, PA. doi:10.1137/1.9781611974546
   Specifically Algorithm 6.4 (ETKF).

   Key features:
   - Ensemble-based state estimation with configurable ensemble sizes
   - Support for nonlinear observation operators
   - Deterministic square-root updates for numerical stability
   - Modular design allowing for multiple DA algorithm implementations
*/
#include <petsc/private/daimpl.h>
#include <petscblaslapack.h>

PetscClassId      PETSCDA_CLASSID          = 0;
PetscBool         PetscDARegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscDAList              = NULL;

static PetscBool PetscDAPackageInitialized = PETSC_FALSE;

/*@C
  PetscDAInitializePackage - This function initializes everything in the `PetscDA`
  package. called on the first call to `PetscDACreate()` when using static or shared
  libraries.

  Level: developer

.seealso: `PetscDAFinalizePackage()`, `PetscInitialize()`
@*/
PetscErrorCode PetscDAInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDAPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);

  PetscDAPackageInitialized = PETSC_TRUE;
  PetscCall(PetscClassIdRegister("Data Assimilation", &PETSCDA_CLASSID));
  PetscCall(PetscDARegisterAll());
  PetscCall(PetscRegisterFinalize(PetscDAFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDAFinalizePackage - This function finalizes everything in the `PetscDA` package. It
  is called from `PetscFinalize()`.

  Level: developer

.seealso: `PetscDAInitializePackage()`, `PetscInitialize()`
@*/
PetscErrorCode PetscDAFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscDAList));
  PetscDARegisterAllCalled  = PETSC_FALSE;
  PetscDAPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDAETKFRegister(void);
PETSC_EXTERN PetscErrorCode PetscDALETKFRegister(void);

/*@C
  PetscDARegister - Registers a constructor for a `PetscDA` implementation with the
  dispatcher.

  Not Collective

  Input Parameters:
+ sname    - name associated with the implementation
- function - routine that creates the implementation and installs method table

  Level: developer

.seealso: [](ch_da), `PetscDARegisterAll()`, `PetscDASetType()`
@*/
PetscErrorCode PetscDARegister(const char sname[], PetscErrorCode (*function)(PetscDA))
{
  PetscFunctionBegin;
  PetscCall(PetscDAInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscDAList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDARegisterAll - Registers all data assimilation backends that were compiled in.

  Not Collective

  Level: developer

.seealso: [](ch_da), `PetscDARegister()`
@*/
PetscErrorCode PetscDARegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDARegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDARegisterAllCalled = PETSC_TRUE;
  PetscCall(PetscDAETKFRegister()); // add new methods here
  PetscCall(PetscDALETKFRegister());
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  PetscDASetOptionsPrefix - Sets the prefix used for searching for all
  PetscDA options in the database.

  Logically Collective

  Input Parameters:
+ das - the `PetscDA` context
- p   - the prefix string to prepend to all PetscDA option requests

  Level: advanced

.seealso: `PetscDA`, `PetscDASetFromOptions()`, `PetscDAAppendOptionsPrefix()`, `PetscDAGetOptionsPrefix()`
@*/
PetscErrorCode PetscDASetOptionsPrefix(PetscDA das, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDA_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAAppendOptionsPrefix - Appends to the prefix used for searching for all PetscDA options in the database.

  Logically Collective

  Input Parameters:
+ das - the `PetscDA` context
- p   - the prefix string to prepend to all `PetscDA` option requests

  Level: advanced

.seealso: `PetscDA`, `PetscDASetFromOptions()`, `PetscDASetOptionsPrefix()`, `PetscDAGetOptionsPrefix()`
@*/
PetscErrorCode PetscDAAppendOptionsPrefix(PetscDA das, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDA_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetOptionsPrefix - Gets the prefix used for searching for all
  PetscDA options in the database

  Not Collective

  Input Parameter:
. das - the `PetscDA` context

  Output Parameter:
. p - pointer to the prefix string used

  Level: advanced

.seealso: `PetscDA`, `PetscDASetFromOptions()`, `PetscDASetOptionsPrefix()`, `PetscDAAppendOptionsPrefix()`
@*/
PetscErrorCode PetscDAGetOptionsPrefix(PetscDA das, const char *p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDA_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute mean of ensemble */
static PetscErrorCode PetscDAComputeEnsembleMean_Default(PetscDA da, Vec mean)
{
  PetscScalar inv_m;
  PetscInt    m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before computing the ensemble mean");
  PetscCheck(da->ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Ensemble size must be positive");

  m     = da->ensemble_size;
  inv_m = 1.0 / (PetscScalar)m;
  /* Compute observation mean mean = (1/m) * sum(E_i) */
  PetscCall(MatGetRowSum(da->ensemble, mean));
  PetscCall(VecScale(mean, inv_m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Alg 6.4 line 2: Compute anomalies X = (E - mean) / sqrt(m-1)
static PetscErrorCode PetscDAComputeAnomalies_Default(PetscDA da, Vec mean_in, Mat *anomalies_out)
{
  Vec       mean = NULL;
  Vec       col_in, col_out;
  Mat       anomalies;
  MPI_Comm  comm;
  PetscReal scale;
  PetscInt  ensemble_size;
  PetscInt  j;
  PetscBool mean_created = PETSC_FALSE;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (mean_in) PetscValidHeaderSpecific(mean_in, VEC_CLASSID, 2);
  PetscAssertPointer(anomalies_out, 3);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before computing anomalies");
  PetscCheck(da->ensemble_size > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least 2 to form anomalies");
  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "State size must be positive");

  /* Cache frequently-used values for clarity and efficiency */
  ensemble_size = da->ensemble_size;
  comm          = PetscObjectComm((PetscObject)da->ensemble);

  /*
    Compute normalization scale for anomalies.
    Alg 6.4 line 2: anomalies are normalized by 1/sqrt(m-1) so that
    the anomalies matrix X satisfies X*X^T = ensemble covariance matrix.
    This ensures proper statistical properties for ensemble-based methods.
  */
  scale = 1.0 / PetscSqrtReal((PetscReal)(ensemble_size - 1));

  /* Use provided mean or create and compute it */
  if (mean_in) {
    mean = mean_in;
  } else {
    /* Create and compute ensemble mean vector */
    PetscCall(VecCreate(comm, &mean));
    PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
    PetscCall(VecSetFromOptions(mean));
    mean_created = PETSC_TRUE;

    /* Alg 6.4 line 1: \bar{x} = (1/m)\sum_j x^{(j)} */
    PetscCall(PetscDAComputeEnsembleMean(da, mean));
  }

  /* Allocate anomalies matrix (state_size x ensemble_size) */
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, da->state_size, ensemble_size, NULL, &anomalies));
  PetscCall(MatSetUp(anomalies));

  /*
    Form anomalies by subtracting mean from each ensemble member and scaling.
    For each column j: anomaly_j = (ensemble_j - mean) / sqrt(m-1)
  */
  for (j = 0; j < ensemble_size; ++j) { // should be locals only
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, j, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(anomalies, j, &col_out));

    /* Alg 6.4 line 2: subtract the mean column-wise to form x^{(j)} - \bar{x} */
    PetscCall(VecWAXPY(col_out, -1.0, mean, col_in));
    /* Alg 6.4 line 2: scale anomalies by 1/\sqrt{m-1} */
    PetscCall(VecScale(col_out, scale));

    PetscCall(MatDenseRestoreColumnVecWrite(anomalies, j, &col_out));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, j, &col_in));
  }

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(anomalies, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(anomalies, MAT_FINAL_ASSEMBLY));

  /* Transfer ownership to output and clean up temporary resources */
  *anomalies_out = anomalies;
  if (mean_created) PetscCall(VecDestroy(&mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDACreate - Creates a new `PetscDA` object for ensemble-based data assimilation.

  Collective

  Input Parameter:
. comm - MPI communicator used to create the object

  Output Parameter:
. da_out - newly created `PetscDA` object

  Level: beginner

.seealso: [](ch_da), `PetscDADestroy()`, `PetscDASetType()`, `PetscDASetUp()`
@*/
PetscErrorCode PetscDACreate(MPI_Comm comm, PetscDA *da_out)
{
  PetscDA da;

  PetscFunctionBegin;
  PetscAssertPointer(da_out, 2);

  PetscCall(PetscDAInitializePackage());

  PetscCall(PetscHeaderCreate(da, PETSCDA_CLASSID, "PetscDA", "Data Assimilation", "DA", comm, PetscDADestroy, PetscDAView));
  PetscCall(PetscMemzero(da->ops, sizeof(*da->ops)));
  da->ops->computemean      = PetscDAComputeEnsembleMean_Default;
  da->ops->computeanomalies = PetscDAComputeAnomalies_Default;

  da->ensemble_size = 0;
  da->state_size    = 0;
  da->obs_size      = 0;
  da->ndof          = 1;
  da->ensemble      = NULL;
  da->obs_error_var = NULL;
  da->U             = NULL;
  da->assembled     = PETSC_FALSE;
  da->data          = NULL;
  da->inflation     = 1.0;

  /* Initialize T-matrix factorization fields */
  da->sqrt_type       = PETSCDA_SQRT_EIGEN;
  da->V               = NULL;
  da->L_cholesky      = NULL;
  da->sqrt_eigen_vals = NULL;
  da->I_StS           = NULL;

  *da_out = da;

  PetscCall(PetscDASetType(da, PETSCDAETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDADestroy - Destroys a `PetscDA` object and releases its resources.

  Collective

  Input Parameter:
. da - pointer to the `PetscDA` object to destroy

  Level: beginner

.seealso: [](ch_da), `PetscDACreate()`
@*/
PetscErrorCode PetscDADestroy(PetscDA *da)
{
  PetscFunctionBegin;
  if (!da || !*da) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*da, PETSCDA_CLASSID, 1);
  if (--((PetscObject)*da)->refct > 0) {
    *da = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if ((*da)->ops->destroy) PetscCall((*(*da)->ops->destroy)(*da));

  PetscCall(MatDestroy(&(*da)->ensemble));
  PetscCall(VecDestroy(&(*da)->obs_error_var));
  PetscCall(MatDestroy(&(*da)->U));

  /* Destroy T-matrix factorization data */
  PetscCall(MatDestroy(&(*da)->V));
  PetscCall(MatDestroy(&(*da)->L_cholesky));
  PetscCall(VecDestroy(&(*da)->sqrt_eigen_vals));
  PetscCall(MatDestroy(&(*da)->I_StS));

  PetscCall(PetscHeaderDestroy(da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetType - Sets the data assimilation implementation used by a `PetscDA` object.

  Collective

  Input Parameters:
+ da   - the `PetscDA` context
- type - name of the implementation (for example `PETSCDAETKF`)

  Level: intermediate

.seealso: [](ch_da), `PetscDAGetType()`, `PetscDARegister()`
@*/
PetscErrorCode PetscDASetType(PetscDA da, PetscDAType type)
{
  PetscErrorCode (*r)(PetscDA);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)da, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscDARegisterAll());
  PetscCall(PetscFunctionListFind(PetscDAList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDA type: %s", type);

  PetscTryTypeMethod(da, destroy);
  da->ops->destroy = NULL;
  da->data         = NULL;

  PetscCall((*r)(da));
  PetscCall(PetscObjectChangeTypeName((PetscObject)da, type));

  if (!da->ops->computemean) da->ops->computemean = PetscDAComputeEnsembleMean_Default;
  if (!da->ops->computeanomalies) da->ops->computeanomalies = PetscDAComputeAnomalies_Default;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetType - Gets the name of the implementation currently associated with a `PetscDA`.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. type - pointer that will receive the type name (may be `NULL`)

  Level: intermediate

.seealso: [](ch_da), `PetscDASetType()`
@*/
PetscErrorCode PetscDAGetType(PetscDA da, PetscDAType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (type) {
    PetscAssertPointer(type, 2);
    *type = ((PetscObject)da)->type_name;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetFromOptions - Configures a `PetscDA` object from the options database.

  Collective

  Input Parameter:
. da - the `PetscDA` context to set up

  Level: intermediate

.seealso: [](ch_da), `PetscDASetType()`, `PetscObjectOptionsBegin()`
@*/
PetscErrorCode PetscDASetFromOptions(PetscDA da)
{
  char      type_name[256];
  PetscBool type_set;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscObjectOptionsBegin((PetscObject)da);

  /* Allow runtime selection of data assimilation type */
  PetscCall(PetscOptionsFList("-petscda_type", "Data assimilation method", "PetscDASetType", PetscDAList, ((PetscObject)da)->type_name, type_name, sizeof(type_name), &type_set));
  if (type_set) PetscCall(PetscDASetType(da, type_name));

  PetscCall(PetscOptionsReal("-petscda_inflation", "Inflation factor", "PetscDASetInflation", da->inflation, &da->inflation, NULL));

  if (da->ops->setfromoptions) PetscCall((*da->ops->setfromoptions)(da, &PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetSizes - Sets the state, observation, and ensemble dimensions used by a `PetscDA`.

  Collective

  Input Parameters:
+ da            - the `PetscDA` context
. state_size    - number of state components
. obs_size      - number of observation components
- ensemble_size - number of ensemble members

  Level: beginner

.seealso: [](ch_da), `PetscDAGetSizes()`, `PetscDASetUp()`
@*/
PetscErrorCode PetscDASetSizes(PetscDA da, PetscInt state_size, PetscInt obs_size, PetscInt ensemble_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, state_size, 2);
  PetscValidLogicalCollectiveInt(da, obs_size, 3);
  PetscValidLogicalCollectiveInt(da, ensemble_size, 4);

  PetscCheck(!da->assembled, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Cannot change sizes after PetscDASetUp() has been called");

  da->state_size    = state_size;
  da->obs_size      = obs_size;
  da->ensemble_size = ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetSizes - Retrieves the dimension settings associated with a `PetscDA`.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameters:
+ state_size    - number of state components (may be `NULL`)
. obs_size      - number of observation components (may be `NULL`)
- ensemble_size - number of ensemble members (may be `NULL`)

  Level: beginner

.seealso: [](ch_da), `PetscDASetSizes()`
@*/
PetscErrorCode PetscDAGetSizes(PetscDA da, PetscInt *state_size, PetscInt *obs_size, PetscInt *ensemble_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (state_size) *state_size = da->state_size;
  if (obs_size) *obs_size = da->obs_size;
  if (ensemble_size) *ensemble_size = da->ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetNDOF - Set the number of degrees of freedom per grid point

  Logically Collective

  Input Parameters:
+ da   - the PetscDA context
- ndof - number of degrees of freedom per grid point (e.g., 2 for shallow water with h and hu)

  Notes:
  This must be called before PetscDASetUp(). The default is 1 (scalar field).

  Level: intermediate

.seealso: `PetscDA`, `PetscDAGetNDOF()`, `PetscDASetUp()`, `PetscDASetSizes()`
@*/
PetscErrorCode PetscDASetNDOF(PetscDA da, PetscInt ndof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, ndof, 2);
  PetscCheck(!da->assembled, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Cannot set ndof after PetscDASetUp() has been called");
  PetscCheck(ndof > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "ndof must be positive, got %" PetscInt_FMT, ndof);
  da->ndof = ndof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetNDOF - Get the number of degrees of freedom per grid point

  Not Collective

  Input Parameter:
. da - the PetscDA context

  Output Parameter:
. ndof - number of degrees of freedom per grid point

  Level: intermediate

.seealso: `PetscDA`, `PetscDASetNDOF()`
@*/
PetscErrorCode PetscDAGetNDOF(PetscDA da, PetscInt *ndof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(ndof, 2);
  *ndof = da->ndof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetUp - Allocates internal data structures for a `PetscDA` based on the previously provided sizes.

  Collective

  Input Parameter:
. da - the `PetscDA` context to assemble

  Level: beginner

.seealso: [](ch_da), `PetscDASetSizes()`, `PetscDASetType()`
@*/
PetscErrorCode PetscDASetUp(PetscDA da)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  if (da->assembled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set state size before calling PetscDASetUp()");
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set observation size before calling PetscDASetUp()");
  PetscCheck(da->ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set ensemble size before calling PetscDASetUp()");

  comm = PetscObjectComm((PetscObject)da);

  if (!da->ensemble) {
    PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, da->state_size, da->ensemble_size, NULL, &da->ensemble));
    PetscCall(MatSetUp(da->ensemble));
  }

  if (!da->obs_error_var) {
    PetscCall(VecCreate(comm, &da->obs_error_var));
    PetscCall(VecSetSizes(da->obs_error_var, PETSC_DECIDE, da->obs_size));
    PetscCall(VecSetFromOptions(da->obs_error_var));
    PetscCall(VecSet(da->obs_error_var, 1.0));
  }

  if (!da->U) {
    PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, da->ensemble_size, da->ensemble_size, NULL, &da->U));
    PetscCall(MatSetUp(da->U));
    /* Initialize U as identity matrix */
    PetscCall(MatZeroEntries(da->U));
    PetscCall(MatShift(da->U, 1.0));
    PetscCall(MatAssemblyBegin(da->U, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(da->U, MAT_FINAL_ASSEMBLY));
  }

  da->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAView - Views a `PetscDA` and its implementation-specific data structure.

  Collective

  Input Parameters:
+ da     - the `PetscDA` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: beginner

.seealso: [](ch_da), `PetscDAViewFromOptions()`
@*/
PetscErrorCode PetscDAView(PetscDA da, PetscViewer viewer)
{
  PetscBool   iascii;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)da), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(da, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da), &size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDA Object: %" PetscInt_FMT " MPI process%s\n", (PetscInt)size, size > 1 ? "es" : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  State size: %" PetscInt_FMT "\n", da->state_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Observation size: %" PetscInt_FMT "\n", da->obs_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Ensemble size: %" PetscInt_FMT "\n", da->ensemble_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Assembled: %s\n", da->assembled ? "true" : "false"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Inflation: %g\n", (double)da->inflation));
  }

  if (da->ops->view) PetscCall((*da->ops->view)(da, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAViewFromOptions - Processes command-line options to determine if a `PetscDA` should be viewed.

  Collective

  Input Parameters:
+ da     - the `PetscDA` context
. obj    - optional object that provides the prefix for options
- option - option name to check (may be `NULL`)

  Level: beginner

.seealso: [](ch_da), `PetscDAView()`, `PetscObjectViewFromOptions()`
@*/
PetscErrorCode PetscDAViewFromOptions(PetscDA da, PetscObject obj, const char option[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)da, obj, option));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetObsErrorVariance - Sets the observation-error variances associated with a `PetscDA`.

  Collective

  Input Parameters:
+ da            - the `PetscDA` context
- obs_error_var - vector containing observation error variances (assumes R is a diagonal matrix)

  Level: beginner

.seealso: [](ch_da), `PetscDAGetObsErrorVariance()`
@*/
PetscErrorCode PetscDASetObsErrorVariance(PetscDA da, Vec obs_error_var)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(obs_error_var, VEC_CLASSID, 2);

  if (!da->obs_error_var) PetscCall(VecDuplicate(obs_error_var, &da->obs_error_var));
  PetscCall(VecCopy(obs_error_var, da->obs_error_var));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetObsErrorVariance - Returns a borrowed reference to the observation-error variance vector.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. obs_error_var - pointer to the variance vector managed by the `PetscDA`

  Level: beginner

.seealso: [](ch_da), `PetscDASetObsErrorVariance()`
@*/
PetscErrorCode PetscDAGetObsErrorVariance(PetscDA da, Vec *obs_error_var)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(obs_error_var, 2);
  *obs_error_var = da->obs_error_var;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetInflation - Sets the inflation factor for the data assimilation method.

  Logically Collective

  Input Parameters:
+ da        - the `PetscDA` context
- inflation - the inflation factor (must be >= 1.0)

  Level: intermediate

.seealso: [](ch_da), `PetscDAGetInflation()`
@*/
PetscErrorCode PetscDASetInflation(PetscDA da, PetscReal inflation)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveReal(da, inflation, 2);
  PetscCheck(inflation >= 1.0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Inflation factor must be >= 1.0, got %g", (double)inflation);
  da->inflation = inflation;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetInflation - Gets the inflation factor for the data assimilation method.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. inflation - the inflation factor

  Level: intermediate

.seealso: [](ch_da), `PetscDASetInflation()`
@*/
PetscErrorCode PetscDAGetInflation(PetscDA da, PetscReal *inflation)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(inflation, 2);
  *inflation = da->inflation;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetOrthogonalTransform - Installs the ensemble-space orthogonal matrix used in deterministic square-root updates.

  Collective

  Input Parameters:
+ da - the `PetscDA` context
- U  - orthogonal matrix to store (referenced internally)

  Level: developer

.seealso: [](ch_da), `PetscDAGetOrthogonalTransform()`, `PetscDAETKFAnalysis()`
@*/
PetscErrorCode PetscDASetOrthogonalTransform(PetscDA da, Mat U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);

  if (da->U) PetscCall(MatDestroy(&da->U));
  PetscCall(PetscObjectReference((PetscObject)U));
  da->U = U;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetOrthogonalTransform - Retrieves the orthogonal matrix currently stored in a `PetscDA`.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. U - pointer that will receive the matrix (may be `NULL`)

  Level: developer

.seealso: [](ch_da), `PetscDASetOrthogonalTransform()`
@*/
PetscErrorCode PetscDAGetOrthogonalTransform(PetscDA da, Mat *U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(U, 2);
  *U = da->U;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetEnsembleMember - Returns a read-only view of an ensemble member stored in the `PetscDA`.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
- member_idx - index of the requested member (0 <= idx < ensemble_size)

  Output Parameter:
. member - read-only vector view; call `PetscDARestoreEnsembleMember()` when done

  Level: intermediate

.seealso: [](ch_da), `PetscDARestoreEnsembleMember()`, `PetscDASetEnsembleMember()`
@*/
PetscErrorCode PetscDAGetEnsembleMember(PetscDA da, PetscInt member_idx, Vec *member)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(member, 3);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before accessing ensemble members");
  PetscCheck(member_idx >= 0 && member_idx < da->ensemble_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, da->ensemble_size);

  PetscCall(MatDenseGetColumnVecRead(da->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDARestoreEnsembleMember - Returns a column view obtained with `PetscDAGetEnsembleMember()`.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
. member_idx - index that was previously requested
- member     - location that holds the view to restore

  Level: intermediate

.seealso: [](ch_da), `PetscDAGetEnsembleMember()`
@*/
PetscErrorCode PetscDARestoreEnsembleMember(PetscDA da, PetscInt member_idx, Vec *member)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(member, 3);

  PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASetEnsembleMember - Overwrites an ensemble member with user-provided state data.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
. member_idx - index of the entry to modify
- member     - vector containing the new state values

  Level: intermediate

.seealso: [](ch_da), `PetscDAGetEnsembleMember()`
@*/
PetscErrorCode PetscDASetEnsembleMember(PetscDA da, PetscInt member_idx, Vec member)
{
  Vec col;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(member, VEC_CLASSID, 3);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before setting ensemble members");
  PetscCheck(member_idx >= 0 && member_idx < da->ensemble_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, da->ensemble_size);

  PetscCall(MatDenseGetColumnVecWrite(da->ensemble, member_idx, &col));
  PetscCall(VecCopy(member, col));
  PetscCall(MatDenseRestoreColumnVecWrite(da->ensemble, member_idx, &col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAComputeMean - Computes the ensemble mean state for a `PetscDA`.

  Collective

  Input Parameters:
+ da   - the `PetscDA` context
- mean - vector that will hold the ensemble mean

  Level: intermediate

.seealso: [](ch_da), `PetscDAComputeAnomalies()`
@*/
PetscErrorCode PetscDAComputeEnsembleMean(PetscDA da, Vec mean)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);

  PetscUseTypeMethod(da, computemean, mean);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  InitializeEnsemble - Initialize ensemble members with Gaussian perturbations

  Input Parameters:
+ daas          - PetscDA context
. x0            - Background state
. ensemble_size - Number of ensemble members
. obs_error_std - Standard deviation for perturbations
- rng           - Random number generator

  Notes:
  Each ensemble member is initialized as x0 + Gaussian(0, obs_error_std)
*/
PetscErrorCode InitializeEnsemble(PetscDA daas, Vec x0, PetscInt ensemble_size, PetscReal obs_error_std, PetscRandom rng)
{
  Vec       member, col, x_mean;
  PetscInt  i;
  PetscReal scale;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(rng, PETSC_RANDOM_CLASSID, 5);
  PetscCall(VecDuplicate(x0, &member));
  PetscCall(VecDuplicate(x0, &x_mean));

  /* Scale factor to maintain consistent ensemble spread across different ensemble sizes.
     After removing the sample mean, the ensemble variance is approximately:
       Var_final ≈ Var_initial * (m-1)/m
     To maintain consistent initial spread regardless of m, we scale by sqrt(m/(m-1)).
     This ensures the final ensemble spread is approximately obs_error_std^2. */
  scale = PetscSqrtReal((PetscReal)ensemble_size / (PetscReal)(ensemble_size - 1));

  /* Populate the Gaussian draws with scaled standard deviation */
  for (i = 0; i < ensemble_size; i++) {
    PetscCall(VecSetRandomGaussian(member, rng, 0.0, obs_error_std * scale));
    PetscCall(PetscDASetEnsembleMember(daas, i, member));
  }
  /* get mean of perturbations */
  PetscCall(PetscDAComputeEnsembleMean(daas, x_mean));
  /* remove mean and add x0 */
  for (i = 0; i < ensemble_size; i++) {
    PetscCall(MatDenseGetColumnVecWrite(daas->ensemble, i, &col));
    PetscCall(VecAXPY(col, -1.0, x_mean));
    PetscCall(VecAXPY(col, 1.0, x0));
    PetscCall(MatDenseRestoreColumnVecWrite(daas->ensemble, i, &col));
  }

  PetscCall(VecDestroy(&member));
  PetscCall(VecDestroy(&x_mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAComputeAnomalies - Forms the state-space anomalies matrix for a `PetscDA`.

  Collective

  Input Parameters:
+ da   - the `PetscDA` context
- mean - optional mean state vector (pass `NULL` to compute internally)

  Output Parameter:
. anomalies - location to store the newly created anomalies matrix

  Notes:
  If `mean` is `NULL`, the function will create a temporary vector and compute
  the ensemble mean using `PetscDAComputeEnsembleMean()`. If `mean` is provided,
  it will be used directly, which can improve performance when the mean has
  already been computed.

  Level: intermediate

.seealso: [](ch_da), `PetscDAComputeEnsembleMean()`
@*/
PetscErrorCode PetscDAComputeAnomalies(PetscDA da, Vec mean, Mat *anomalies)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (mean) PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscAssertPointer(anomalies, 3);

  PetscUseTypeMethod(da, computeanomalies, mean, anomalies);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAAnalysis - Executes the analysis (update) step using sparse observation matrix H

  Collective

  Input Parameters:
+ da          - the `PetscDA` context
. observation - observation vector y in R^P
- H           - observation operator matrix (P x N), sparse AIJ format

  Notes:
  The observation matrix H maps from state space (N dimensions) to observation
  space (P dimensions): y = H*x + noise
  
  H must be a sparse AIJ matrix (will be AIJKokkos for GPU support).
  
  For identity observations (observe entire state), create H as:
    MatCreateAIJ(..., n, n, 1, NULL, 0, NULL, &H)
    for i=0 to n-1: MatSetValue(H, i, i, 1.0, INSERT_VALUES)
  
  For partial observations, set appropriate rows and columns to observe
  specific state components.

  Level: intermediate

.seealso: [](ch_da), `PetscDAApplyModel()`, `PetscDASetObsErrorVariance()`
@*/
PetscErrorCode PetscDAAnalysis(PetscDA da, Vec observation, Mat H)
{
  PetscInt h_rows, h_cols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 3);

  /* Validate H dimensions match PetscDA configuration */
  PetscCall(MatGetSize(H, &h_rows, &h_cols));
  PetscCheck(h_rows == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "H matrix rows (%" PetscInt_FMT ") must match obs_size (%" PetscInt_FMT ")", h_rows, da->obs_size);
  PetscCheck(h_cols == da->state_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "H matrix cols (%" PetscInt_FMT ") must match state_size (%" PetscInt_FMT ")", h_cols, da->state_size);

  PetscUseTypeMethod(da, analysis, observation, H);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDAApplyModel - Advances every ensemble member through the user-supplied forecast model.

  Collective

  Input Parameters:
+ da        - the `PetscDA` context
. model     - routine that evaluates the model map `f(x, xnew; ctx)`
- model_ctx - optional context for `model`

  Level: intermediate

.seealso: [](ch_da), `PetscDAAnalysis()`
@*/
PetscErrorCode PetscDAApplyModel(PetscDA da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscUseTypeMethod(da, applymodel, model, model_ctx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecSetRandomGaussian - Fills a vector with Gaussian random values of the given mean and standard deviation.

  Collective

  Input Parameters:
+ v       - the vector to fill
. rng     - PETSc random number generator
. mean    - desired mean of the Gaussian samples
- std_dev - desired standard deviation

  Level: developer

  Notes:
  Uses the Box-Muller transform to generate normally distributed random numbers
  from uniform random numbers. Handles edge cases where uniform random values
  approach 0 or 1.

.seealso: [](ch_da), `PetscRandomSetInterval()`, `VecSetRandom()`
@*/
PetscErrorCode VecSetRandomGaussian(Vec v, PetscRandom rng, PetscReal mean, PetscReal std_dev)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(rng, PETSC_RANDOM_CLASSID, 2);
  PetscCall(VecSetRandomGaussian_Private(v, rng, mean, std_dev));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetRandomGaussian_Private(Vec v, PetscRandom rng, PetscReal mean, PetscReal std_dev)
{
  PetscInt        n, i;
  PetscScalar    *array;
  PetscReal       u1, u2;
  PetscReal       gauss_sample1, gauss_sample2, magnitude, theta;
  const PetscReal min_uniform     = PETSC_MACHINE_EPSILON;
  const PetscInt  max_retry_count = 100;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscCheck(PetscIsInfOrNanReal(mean) == PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Mean must be a finite real number");
  PetscCheck(std_dev >= 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Standard deviation must be non-negative, got %g", (double)std_dev);
  PetscCheck(PetscIsInfOrNanReal(std_dev) == PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Standard deviation must be a finite real number");

  PetscCall(VecGetLocalSize(v, &n));

  /* Handle empty vector case efficiently */
  if (n == 0) PetscFunctionReturn(PETSC_SUCCESS);

  /* Handle zero standard deviation case: all values become mean */
  if (std_dev == 0.0) {
    PetscCall(VecSet(v, mean));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(VecGetArray(v, &array));

  /*
    Generate Gaussian-distributed random values using the Box-Muller transform.
    This transform converts pairs of uniform random variables U1, U2 ~ Uniform(0,1)
    into pairs of independent standard normal variables Z0, Z1 ~ N(0,1):
      Z0 = sqrt(-2 * ln(U1)) * cos(2pi * U2)
      Z1 = sqrt(-2 * ln(U1)) * sin(2pi * U2)
    Then scale and shift to get desired mean and standard deviation.
  */
  for (i = 0; i < n; i += 2) {
    PetscInt retry_count = 0;

    /*
      Generate U1 and ensure it's not too close to 0 to avoid log(0) singularity.
      Add retry limit to prevent infinite loops in case of RNG failure.
    */
    do {
      PetscCall(PetscRandomGetValueReal(rng, &u1));
      retry_count++;
      PetscCheck(retry_count < max_retry_count, PETSC_COMM_SELF, PETSC_ERR_LIB, "Random number generator failed to produce valid values after %" PetscInt_FMT " attempts", (PetscInt)max_retry_count);
    } while (u1 < min_uniform);

    PetscCall(PetscRandomGetValueReal(rng, &u2));

    /*
      Apply Box-Muller transform:
      - magnitude: sqrt(-2 * ln(U1)) represents the radial distance from origin
      - theta: 2pi * U2 represents the angle uniformly distributed on [0, 2pi]
      - Converting from polar to Cartesian coordinates yields two independent samples
    */
    magnitude     = PetscSqrtReal(-2.0 * PetscLogReal(u1));
    theta         = 2.0 * PETSC_PI * u2;
    gauss_sample1 = magnitude * PetscCosReal(theta);
    gauss_sample2 = magnitude * PetscSinReal(theta);

    /* Scale and shift to achieve desired mean and standard deviation */
    array[i] = mean + std_dev * gauss_sample1;
    if (i + 1 < n) array[i + 1] = mean + std_dev * gauss_sample2;
  }

  PetscCall(VecRestoreArray(v, &array));
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

/* ========================================================================== */
/*         T-Matrix Factorization and Application Methods [Alg 6.4 line 7]    */
/* ========================================================================== */

/* Tolerance for matrix square root verification in debug mode
   Use a more relaxed tolerance to account for accumulated floating-point errors
   in multiple matrix operations (Y^T * T * Y involves 3 matrix multiplications).
   A tolerance of 1e-2 (1%) is reasonable for numerical verification. */
#define MATRIX_SQRT_TOLERANCE_FACTOR 1.0e-2

/*
  PetscDATFactor_Cholesky - Computes Cholesky factorization of T

  Input Parameters:
+ da - the PetscDA context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  Computes the lower triangular Cholesky factor L such that T = L * L^T.
*/
static PetscErrorCode PetscDATFactor_Cholesky(PetscDA da)
{
  PetscBLASInt n, lda, info;
  PetscScalar *a_array;
  PetscInt     m_T, N_T;

  PetscFunctionBegin;
  /* Initialize or update L_cholesky matrix */
  if (!da->L_cholesky) {
    PetscCall(MatDuplicate(da->I_StS, MAT_COPY_VALUES, &da->L_cholesky));
  } else {
    PetscCall(MatCopy(da->I_StS, da->L_cholesky, SAME_NONZERO_PATTERN));
  }

  /* Get matrix dimensions and convert to BLAS int */
  PetscCall(MatGetSize(da->L_cholesky, &m_T, &N_T));
  PetscCheck(m_T == N_T, PetscObjectComm((PetscObject)da->L_cholesky), PETSC_ERR_ARG_WRONG, "Matrix must be square for Cholesky");
  PetscCall(PetscBLASIntCast(N_T, &n));
  lda = n;

  /* Get array from dense matrix */
  PetscCall(MatDenseGetArray(da->L_cholesky, &a_array));

  /* Compute Cholesky factorization: A = L * L^T (lower triangular) */
  LAPACKpotrf_("L", &n, a_array, &lda, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky factorization (xPOTRF): info=%" PetscInt_FMT ". Matrix T is not positive definite.", (PetscInt)info);

  /* Restore array and finalize matrix */
  PetscCall(MatDenseRestoreArray(da->L_cholesky, &a_array));
  PetscCall(MatAssemblyBegin(da->L_cholesky, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(da->L_cholesky, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDATFactor_Eigen - Computes Eigendecomposition of T

  Input Parameters:
+ da - the PetscDA context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  Computes eigenvectors V and eigenvalues D such that T = V * D * V^T.
*/
static PetscErrorCode PetscDATFactor_Eigen(PetscDA da)
{
  PetscBLASInt n, lda, lwork, info;
  PetscScalar *a_array, *work, *eig_array;
  PetscInt     m_V, N_V;
#if defined(PETSC_USE_COMPLEX)
  PetscReal *rwork = NULL;
#endif

  PetscFunctionBegin;
  /* Initialize or update V matrix */
  if (!da->V) {
    PetscCall(MatDuplicate(da->I_StS, MAT_COPY_VALUES, &da->V));
  } else {
    PetscCall(MatCopy(da->I_StS, da->V, SAME_NONZERO_PATTERN));
  }

  /* Initialize or update eigenvalue vector */
  if (!da->sqrt_eigen_vals) PetscCall(MatCreateVecs(da->I_StS, &da->sqrt_eigen_vals, NULL));

  /* Get matrix dimensions */
  PetscCall(MatGetSize(da->V, &m_V, &N_V));
  PetscCheck(m_V == N_V, PetscObjectComm((PetscObject)da->V), PETSC_ERR_ARG_WRONG, "Matrix must be square");
  PetscCall(PetscBLASIntCast(N_V, &n));
  lda = n;

  /* Get arrays */
  PetscCall(MatDenseGetArray(da->V, &a_array));
  PetscCall(VecGetArray(da->sqrt_eigen_vals, &eig_array));

  /* Query optimal workspace size */
  lwork = -1;
  PetscCall(PetscMalloc1(1, &work));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(PetscMax(1, 3 * n - 2), &rwork));
  LAPACKsyev_("V", "U", &n, a_array, &lda, (PetscReal *)eig_array, work, &lwork, rwork, &info);
#else
  LAPACKsyev_("V", "U", &n, a_array, &lda, eig_array, work, &lwork, &info);
#endif
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine xSYEV work query: info=%" PetscInt_FMT, (PetscInt)info);

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
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine xSYEV: info=%" PetscInt_FMT, (PetscInt)info);

  /* Cleanup */
  PetscCall(PetscFree(work));
  PetscCall(VecRestoreArray(da->sqrt_eigen_vals, &eig_array));
  PetscCall(MatDenseRestoreArray(da->V, &a_array));

  PetscCall(MatAssemblyBegin(da->V, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(da->V, MAT_FINAL_ASSEMBLY));

  /* Compute sqrt(eigenvalues) */
  PetscCall(VecSqrtAbs(da->sqrt_eigen_vals));

  /* Debug verification: Ensure V * D * V^T == T */
  if (PetscDefined(USE_DEBUG)) {
    PetscReal norm_T, norm_diff, relative_error;
    Mat       V_D, VDVt;

    /* Compute D * V^T by scaling rows */
    PetscCall(MatDuplicate(da->V, MAT_COPY_VALUES, &V_D));

    /* Restore D for verification (since sqrt_eigen_vals currently holds sqrt(D)) */
    PetscCall(VecPointwiseMult(da->sqrt_eigen_vals, da->sqrt_eigen_vals, da->sqrt_eigen_vals));

    PetscCall(MatDiagonalScale(V_D, NULL, da->sqrt_eigen_vals));

    /* Compute V * D * V^T */
    PetscCall(MatMatTransposeMult(V_D, da->V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VDVt));

    /* Compute ||V*D*V^T - T|| / ||T|| */
    PetscCall(MatAXPY(VDVt, -1.0, da->I_StS, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(da->I_StS, NORM_FROBENIUS, &norm_T));
    PetscCall(MatNorm(VDVt, NORM_FROBENIUS, &norm_diff));

    PetscCheck(norm_T > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "T = 0");
    relative_error = norm_diff / norm_T;
    PetscCheck(relative_error < MATRIX_SQRT_TOLERANCE_FACTOR, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "Eigendecomposition verification failed: ||V*D*V^T - T||/||T|| = %g", (double)relative_error);

    /* Restore sqrt(D) back to sqrt_eigen_vals */
    PetscCall(VecSqrtAbs(da->sqrt_eigen_vals));

    /* Cleanup debug matrices */
    PetscCall(MatDestroy(&V_D));
    PetscCall(MatDestroy(&VDVt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDATFactor - Compute and store factorization of T matrix

  Collective

  Input Parameters:
+ da - the PetscDA context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  This function computes T = I + S^T * S and stores its factorization based on
  the selected sqrt_type.

  - For CHOLESKY mode: computes the lower triangular Cholesky factor L such that T = L * L^T.
  - For EIGEN mode: computes eigenvectors V and eigenvalues D such that T = V * D * V^T.

  The implementation uses matrix reuse (MAT_REUSE_MATRIX) to minimize memory allocation
  overhead when the ensemble size remains constant across analysis cycles.

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAApplyTInverse()`, `PetscDAApplySqrtTInverse()`
@*/
PetscErrorCode PetscDATFactor(PetscDA da, Mat S)
{
  PetscInt  m, s_rows, s_cols;
  MatReuse  scall      = MAT_INITIAL_MATRIX;
  PetscBool reallocate = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(S, MAT_CLASSID, 2);

  /* 1. Validate Matrix Dimensions */
  PetscCall(MatGetSize(S, &s_rows, &s_cols));
  m = s_cols; /* Ensemble size */

  PetscCheck(m > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Innovation matrix S must have positive columns, got %" PetscInt_FMT, m);
  PetscCheck(m == da->ensemble_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "S matrix columns (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ") defined in PetscDA", m, da->ensemble_size);

  /* 2. Manage Resource Reuse */
  /* Check if we can reuse the T matrix (I_StS) and dependent factors */
  if (da->I_StS) {
    PetscInt t_rows, t_cols;
    PetscCall(MatGetSize(da->I_StS, &t_rows, &t_cols));

    /* If dimensions have changed, we must fully reallocate */
    if (t_rows != m || t_cols != m) {
      reallocate = PETSC_TRUE;
      PetscCall(PetscInfo(da, "Ensemble size changed (old: %" PetscInt_FMT ", new: %" PetscInt_FMT "), reallocating T matrix and factors\n", t_rows, m));
    } else {
      scall = MAT_REUSE_MATRIX;
    }
  }

  if (reallocate && da->I_StS) {
    PetscCall(MatDestroy(&da->I_StS));
    PetscCall(MatDestroy(&da->V));
    PetscCall(MatDestroy(&da->L_cholesky));
    PetscCall(VecDestroy(&da->sqrt_eigen_vals));
    scall = MAT_INITIAL_MATRIX;
  }

  /* 3. Compute T = I + S^T * S */
  /*
     MatTransposeMatMult computes C = A^T * B (here C = S^T * S).
     When using MAT_REUSE_MATRIX, the existing C is overwritten with the new result.
  */
  PetscCall(MatTransposeMatMult(S, S, scall, PETSC_DEFAULT, &da->I_StS));

  /* Add Identity: T = T + (1/rho)*I */
  PetscCall(MatShift(da->I_StS, 1.0 / da->inflation));

  /* 4. Compute Factorization based on strategy */
  switch (da->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(PetscDATFactor_Cholesky(da));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(PetscDATFactor_Eigen(da));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)da->sqrt_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplyTInverse_Cholesky - Helper for Cholesky solver path
*/
static PetscErrorCode ApplyTInverse_Cholesky(PetscDA da, Vec sdel, Vec w)
{
  PetscBLASInt n, lda, nrhs, info;
  PetscScalar *a_array, *b_array;
  PetscInt     m_L, N_L;

  PetscFunctionBegin;
  PetscCheck(da->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

  /* Get dimensions */
  PetscCall(MatGetSize(da->L_cholesky, &m_L, &N_L));
  PetscCall(PetscBLASIntCast(N_L, &n));
  lda  = n;
  nrhs = 1;

  /* Copy sdel to w for in-place solve */
  PetscCall(VecCopy(sdel, w));

  /* Get arrays */
  PetscCall(MatDenseGetArray(da->L_cholesky, &a_array));
  PetscCall(VecGetArray(w, &b_array));

  /* Solve L * L^T * w = sdel using LAPACK's Cholesky solve (xPOTRS) */
  /* Note: POTRS expects the input B (w) to contain the RHS, and overwrites it with the solution */
  LAPACKpotrs_("L", &n, &nrhs, a_array, &lda, b_array, &n, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky solve (xPOTRS): info=%" PetscInt_FMT, (PetscInt)info);

  /* Restore arrays */
  PetscCall(MatDenseRestoreArray(da->L_cholesky, &a_array));
  PetscCall(VecRestoreArray(w, &b_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplyTInverse_Eigen - Helper for Eigendecomposition solver path
*/
static PetscErrorCode ApplyTInverse_Eigen(PetscDA da, Vec sdel, Vec w)
{
  Vec temp;

  PetscFunctionBegin;
  PetscCheck(da->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
  PetscCheck(da->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

  /* Allocate temporary vector for projection */
  PetscCall(VecDuplicate(sdel, &temp));

  /* 1. Project onto eigenvectors: temp = V^T * sdel */
  PetscCall(MatMultTranspose(da->V, sdel, temp));

  /* 2. Scale by inverse eigenvalues: temp = D^{-1} * temp */
  /* We store sqrt(D), so divide twice: temp = (temp / sqrt(D)) / sqrt(D) */
  PetscCall(VecPointwiseDivide(temp, temp, da->sqrt_eigen_vals));
  PetscCall(VecPointwiseDivide(temp, temp, da->sqrt_eigen_vals));

  /* 3. Map back to standard basis: w = V * temp */
  PetscCall(MatMult(da->V, temp, w));

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAApplyTInverse - Apply T^{-1} to a vector [Alg 6.4 line 8]

  Collective

  Input Parameters:
+ da   - the PetscDA context
- sdel - input vector S^T-delta

  Output Parameter:
. w - output vector w = T^{-1} * sdel

  Notes:
  This function applies the inverse of T = I + S^T S using the stored
  factorization. For CHOLESKY mode, it uses triangular solves. For EIGEN mode,
  it uses the eigendecomposition (T^{-1} = V D^{-1} V^T).

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDATFactor()`, `PetscDAApplySqrtTInverse()`
@*/
PetscErrorCode PetscDAApplyTInverse(PetscDA da, Vec sdel, Vec w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(sdel, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 3);

  PetscCheck(da->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "T matrix not factored. Call PetscDATFactor first");

  switch (da->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(ApplyTInverse_Cholesky(da, sdel, w));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(ApplyTInverse_Eigen(da, sdel, w));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)da->sqrt_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplySqrtTInverse_Cholesky - Computes Y = L^{-T} * U using forward substitution.

  Notes:
  Since T = L * L^T, T^{-1} = L^{-T} * L^{-1}.
  We uses L^{-T} as the non-symmetric "square root" inverse, i.e., T^{-1/2} = L^{-T}.
  This requires solving L^T * Y = U.
*/
static PetscErrorCode ApplySqrtTInverse_Cholesky(PetscDA da, Mat U, Mat Y)
{
  PetscBLASInt       n, lda, nrhs, info;
  const PetscScalar *l_array;
  PetscScalar       *y_array;
  PetscInt           m_L, N_L, m_U, N_U;

  PetscFunctionBegin;
  PetscCheck(da->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

  /* Get dimensions and validate compatibility */
  PetscCall(MatGetSize(da->L_cholesky, &m_L, &N_L));
  PetscCall(MatGetSize(U, &m_U, &N_U));
  PetscCheck(m_L == m_U, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Cholesky factor rows (%" PetscInt_FMT ") must match U rows (%" PetscInt_FMT ")", m_L, m_U);

  PetscCall(PetscBLASIntCast(N_L, &n));
  PetscCall(PetscBLASIntCast(N_U, &nrhs));
  lda = n;

  /* Initialize Y with U for in-place solve */
  PetscCall(MatCopy(U, Y, SAME_NONZERO_PATTERN));

  /* Get direct array access */
  PetscCall(MatDenseGetArrayRead(da->L_cholesky, &l_array));
  PetscCall(MatDenseGetArray(Y, &y_array));

  /* Solve L^T * Y = U using LAPACK triangular solve (L is lower, so L^T is upper)
     TRTRS args: UPLO='L', TRANS='T', DIAG='N' */
  LAPACKtrtrs_("L", "T", "N", &n, &nrhs, (PetscScalar *)l_array, &lda, y_array, &n, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK triangular solve (xTRTRS): info=%" PetscInt_FMT, (PetscInt)info);

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayRead(da->L_cholesky, &l_array));
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
static PetscErrorCode ApplySqrtTInverse_Eigen(PetscDA da, Mat U, Mat Y)
{
  Mat W;
  Vec diag_inv;

  PetscFunctionBegin;
  PetscCheck(da->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
  PetscCheck(da->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

  /* Prepare inverse sqrt eigenvalues: D^{-1/2}
     Note: da->sqrt_eigen_vals currently stores sqrt(D) */
  PetscCall(VecDuplicate(da->sqrt_eigen_vals, &diag_inv));
  PetscCall(VecCopy(da->sqrt_eigen_vals, diag_inv));
  PetscCall(VecReciprocal(diag_inv)); /* Now diag_inv contains 1/sqrt(D) = D^{-1/2} */

  /* Step 1: Compute W = V^T * U (Project U onto eigenbasis) */
  PetscCall(MatTransposeMatMult(da->V, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &W));

  /* Step 2: Scale rows of W by D^{-1/2}: W <- D^{-1/2} * W */
  PetscCall(MatDiagonalScale(W, diag_inv, NULL));

  /* Step 3: Compute Y = V * W (Project back to standard basis)
     Y = V * (D^{-1/2} * V^T * U) */
  {
    Mat Y_temp;
    PetscCall(MatMatMult(da->V, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y_temp));
    PetscCall(MatCopy(Y_temp, Y, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Y_temp));
  }

  /* Cleanup */
  PetscCall(MatDestroy(&W));
  PetscCall(VecDestroy(&diag_inv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAApplySqrtTInverse - Apply T^{-1/2} to a matrix U [Alg 6.4 line 9]

  Collective

  Input Parameters:
+ da - the PetscDA context
- U  - input matrix (usually Identity, but can be general)

  Output Parameter:
. Y - output matrix Y = T^{-1/2} * U

  Notes:
  This function applies the inverse square root of T = I + S^T * S using the
  stored factorization.

  - For CHOLESKY mode: Computes Y = L^{-T} U
  - For EIGEN mode: Computes Y = V D^{-1/2} V^T U

  Both results satisfy Y^T * T * Y = U^T * U, preserving the metric.

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDATFactor()`, `PetscDAApplyTInverse()`
@*/
PetscErrorCode PetscDAApplySqrtTInverse(PetscDA da, Mat U, Mat Y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(Y, MAT_CLASSID, 3);

  PetscCheck(da->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "I_StS matrix not created. Call PetscDATFactor first");

  switch (da->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(ApplySqrtTInverse_Cholesky(da, U, Y));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(ApplySqrtTInverse_Eigen(da, U, Y));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)da->sqrt_type);
  }

  /* Debugging verification: Check that metric is preserved
     Verify that Y^T * T * Y = U^T * U */
  if (PetscDefined(USE_DEBUG)) {
    Mat       YtTY, UtU, T_Y;
    PetscReal norm_ref, norm_diff;

    /* Compute LHS: Y^T * T * Y */
    PetscCall(MatMatMult(da->I_StS, Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_Y));     /* T * Y */
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

/*@
  PetscDASetSqrtType - Selects the reduced-space square-root algorithm used during analysis.

  Logically Collective

  Input Parameters:
+ da   - the `PetscDA` object
- type - either `PETSCDA_SQRT_CHOLESKY` or `PETSCDA_SQRT_EIGEN`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDAGetSqrtType()`
@*/
PetscErrorCode PetscDASetSqrtType(PetscDA da, PetscDASqrtType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCheck(type == PETSCDA_SQRT_CHOLESKY || type == PETSCDA_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDA square-root type %" PetscInt_FMT, (PetscInt)type);

  da->sqrt_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAGetSqrtType - Retrieves the current square-root implementation configured for analysis.

  Not Collective

  Input Parameters:
. da - the `PetscDA` object

  Output Parameter:
. type - on output, the configured `PetscDASqrtType`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PetscDASetSqrtType()`
@*/
PetscErrorCode PetscDAGetSqrtType(PetscDA da, PetscDASqrtType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);

  *type = da->sqrt_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}
