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

PetscClassId      PETSCDATAASSIMILATOR_CLASSID          = 0;
PetscBool         PetscDataAssimilatorRegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscDataAssimilatorList              = NULL;

static PetscBool PetscDataAssimilatorPackageInitialized = PETSC_FALSE;

/* Tolerance for matrix square root verification in debug mode */
#define MATRIX_SQRT_TOLERANCE_FACTOR 100.0

/* Tolerance for eigenvalue negativity check */
#define EIGENVALUE_TOLERANCE_FACTOR 10.0

/*@C
  PetscDataAssimilatorInitializePackage - This function initializes everything in the `PetscDataAssimilator`
  package. called on the first call to `PetscDataAssimilatorCreate()` when using static or shared
  libraries.

  Level: developer

.seealso: `PetscDataAssimilatorFinalizePackage()`, `PetscInitialize()`
@*/
PetscErrorCode PetscDataAssimilatorInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDataAssimilatorPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);

  PetscDataAssimilatorPackageInitialized = PETSC_TRUE;
  PetscCall(PetscClassIdRegister("Data Assimilation", &PETSCDATAASSIMILATOR_CLASSID));
  PetscCall(PetscDataAssimilatorRegisterAll());
  PetscCall(PetscRegisterFinalize(PetscDataAssimilatorFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDataAssimilatorFinalizePackage - This function finalizes everything in the `PetscDataAssimilator` package. It
  is called from `PetscFinalize()`.

  Level: developer

.seealso: `PetscDataAssimilatorInitializePackage()`, `PetscInitialize()`
@*/
PetscErrorCode PetscDataAssimilatorFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscDataAssimilatorList));
  PetscDataAssimilatorRegisterAllCalled  = PETSC_FALSE;
  PetscDataAssimilatorPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDataAssimilatorETKFRegister(void);

/*@C
  PetscDataAssimilatorRegister - Registers a constructor for a `PetscDataAssimilator` implementation with the
  dispatcher.

  Not Collective

  Input Parameters:
+ sname    - name associated with the implementation
- function - routine that creates the implementation and installs method table

  Level: developer

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorRegisterAll()`, `PetscDataAssimilatorSetType()`
@*/
PetscErrorCode PetscDataAssimilatorRegister(const char sname[], PetscErrorCode (*function)(PetscDataAssimilator))
{
  PetscFunctionBegin;
  PetscCall(PetscDataAssimilatorInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscDataAssimilatorList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorRegisterAll - Registers all data assimilation backends that were compiled in.

  Not Collective

  Level: developer

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorRegister()`
@*/
PetscErrorCode PetscDataAssimilatorRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDataAssimilatorRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDataAssimilatorRegisterAllCalled = PETSC_TRUE;
  PetscCall(PetscDataAssimilatorETKFRegister());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute mean of ensemble */
static PetscErrorCode PetscDataAssimilatorComputeMean_Default(PetscDataAssimilator da, Vec mean)
{
  Vec         member;
  PetscScalar inv_m;
  PetscInt    m, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDataAssimilatorSetUp() must be called before computing the ensemble mean");
  PetscCheck(da->ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Ensemble size must be positive");

  m     = da->ensemble_size;
  inv_m = 1.0 / (PetscScalar)m;

  PetscCall(VecSet(mean, 0.0));
  for (j = 0; j < m; ++j) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, j, &member));
    PetscCall(VecAXPY(mean, inv_m, member));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, j, &member));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDataAssimilatorComputeAnomalies_Default(PetscDataAssimilator da, Mat *anomalies_out)
{
  Vec       mean;
  Vec       col_in, col_out;
  Mat       anomalies;
  MPI_Comm  comm;
  PetscReal scale;
  PetscInt  ensemble_size;
  PetscInt  j;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(anomalies_out, 2);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDataAssimilatorSetUp() must be called before computing anomalies");
  PetscCheck(da->ensemble_size > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least 2 to form anomalies");
  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "State size must be positive");

  /* Cache frequently-used values for clarity and efficiency */
  ensemble_size = da->ensemble_size;
  comm          = PetscObjectComm((PetscObject)da->ensemble);

  /*
    Compute normalization scale for anomalies.
    Algorithm line 14: anomalies are normalized by 1/sqrt(m-1) so that
    the anomalies matrix X satisfies X*X^T = ensemble covariance matrix.
    This ensures proper statistical properties for ensemble-based methods.
  */
  scale = 1.0 / PetscSqrtReal((PetscReal)(ensemble_size - 1));

  /* Create and compute ensemble mean vector */
  PetscCall(VecCreate(comm, &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(mean));

  /* Algorithm line 12: \bar{x} = (1/m)\sum_j x^{(j)} */
  PetscCall(PetscDataAssimilatorComputeMean(da, mean));

  /* Allocate anomalies matrix (state_size x ensemble_size) */
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, da->state_size, ensemble_size, NULL, &anomalies));
  PetscCall(MatSetUp(anomalies));

  /*
    Form anomalies by subtracting mean from each ensemble member and scaling.
    For each column j: anomaly_j = (ensemble_j - mean) / sqrt(m-1)
  */
  for (j = 0; j < ensemble_size; ++j) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, j, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(anomalies, j, &col_out));

    /* Algorithm line 13: subtract the mean column-wise to form x^{(j)} - \bar{x} */
    PetscCall(VecWAXPY(col_out, -1.0, mean, col_in));
    /* Algorithm line 14: scale anomalies by 1/\sqrt{m-1} */
    PetscCall(VecScale(col_out, scale));

    PetscCall(MatDenseRestoreColumnVecWrite(anomalies, j, &col_out));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, j, &col_in));
  }

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(anomalies, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(anomalies, MAT_FINAL_ASSEMBLY));

  /* Transfer ownership to output and clean up temporary resources */
  *anomalies_out = anomalies;
  PetscCall(VecDestroy(&mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorCreate - Creates a new `PetscDataAssimilator` object for ensemble-based data assimilation.

  Collective

  Input Parameter:
. comm - MPI communicator used to create the object

  Output Parameter:
. da_out - newly created `PetscDataAssimilator` object

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorDestroy()`, `PetscDataAssimilatorSetType()`, `PetscDataAssimilatorSetUp()`
@*/
PetscErrorCode PetscDataAssimilatorCreate(MPI_Comm comm, PetscDataAssimilator *da_out)
{
  PetscDataAssimilator da;

  PetscFunctionBegin;
  PetscAssertPointer(da_out, 2);

  PetscCall(PetscDataAssimilatorInitializePackage());

  PetscCall(PetscHeaderCreate(da, PETSCDATAASSIMILATOR_CLASSID, "PetscDataAssimilator", "Data Assimilation", "DataAssimilation", comm, PetscDataAssimilatorDestroy, PetscDataAssimilatorView));
  PetscCall(PetscMemzero(da->ops, sizeof(*da->ops)));
  da->ops->computemean      = PetscDataAssimilatorComputeMean_Default;
  da->ops->computeanomalies = PetscDataAssimilatorComputeAnomalies_Default;

  da->ensemble_size = 0;
  da->state_size    = 0;
  da->obs_size      = 0;
  da->ensemble      = NULL;
  da->obs_error_var = NULL;
  da->U             = NULL;
  da->assembled     = PETSC_FALSE;
  da->data          = NULL;

  *da_out = da;

  PetscCall(PetscDataAssimilatorSetType(da, PETSCDAETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorDestroy - Destroys a `PetscDataAssimilator` object and releases its resources.

  Collective

  Input Parameter:
. da - pointer to the `PetscDataAssimilator` object to destroy

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorCreate()`
@*/
PetscErrorCode PetscDataAssimilatorDestroy(PetscDataAssimilator *da)
{
  PetscFunctionBegin;
  if (!da || !*da) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*da, PETSCDATAASSIMILATOR_CLASSID, 1);
  if (--((PetscObject)*da)->refct > 0) {
    *da = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if ((*da)->ops->destroy) PetscCall((*(*da)->ops->destroy)(*da));

  PetscCall(MatDestroy(&(*da)->ensemble));
  PetscCall(VecDestroy(&(*da)->obs_error_var));
  PetscCall(MatDestroy(&(*da)->U));

  PetscCall(PetscHeaderDestroy(da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorSetType - Sets the data assimilation implementation used by a `PetscDataAssimilator` object.

  Collective

  Input Parameters:
+ da   - the `PetscDataAssimilator` context
- type - name of the implementation (for example `PETSCDAETKF`)

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorGetType()`, `PetscDataAssimilatorRegister()`
@*/
PetscErrorCode PetscDataAssimilatorSetType(PetscDataAssimilator da, PetscDataAssimilatorType type)
{
  PetscErrorCode (*r)(PetscDataAssimilator);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)da, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscDataAssimilatorRegisterAll());
  PetscCall(PetscFunctionListFind(PetscDataAssimilatorList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDataAssimilator type: %s", type);

  if (da->ops->destroy) PetscCall((*da->ops->destroy)(da));

  PetscCall(PetscObjectChangeTypeName((PetscObject)da, type));
  PetscCall((*r)(da));

  if (!da->ops->computemean) da->ops->computemean = PetscDataAssimilatorComputeMean_Default;
  if (!da->ops->computeanomalies) da->ops->computeanomalies = PetscDataAssimilatorComputeAnomalies_Default;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorGetType - Gets the name of the implementation currently associated with a `PetscDataAssimilator`.

  Not Collective

  Input Parameter:
. da - the `PetscDataAssimilator` context

  Output Parameter:
. type - pointer that will receive the type name (may be `NULL`)

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorSetType()`
@*/
PetscErrorCode PetscDataAssimilatorGetType(PetscDataAssimilator da, PetscDataAssimilatorType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  if (type) {
    PetscAssertPointer(type, 2);
    *type = ((PetscObject)da)->type_name;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorSetFromOptions - Configures a `PetscDataAssimilator` object from the options database.

  Collective

  Input Parameter:
. da - the `PetscDataAssimilator` context to set up

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorSetType()`, `PetscObjectOptionsBegin()`
@*/
PetscErrorCode PetscDataAssimilatorSetFromOptions(PetscDataAssimilator da)
{
  char      type_name[256];
  PetscBool type_set;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);

  PetscObjectOptionsBegin((PetscObject)da);

  /* Allow runtime selection of data assimilation type */
  PetscCall(PetscOptionsFList("-petscdataassimilator_type", "Data assimilation method", "PetscDataAssimilatorSetType", PetscDataAssimilatorList, ((PetscObject)da)->type_name, type_name, sizeof(type_name), &type_set));
  if (type_set) PetscCall(PetscDataAssimilatorSetType(da, type_name));

  if (da->ops->setfromoptions) PetscCall((*da->ops->setfromoptions)(da, &PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorSetSizes - Sets the state, observation, and ensemble dimensions used by a `PetscDataAssimilator`.

  Collective

  Input Parameters:
+ da            - the `PetscDataAssimilator` context
. state_size    - number of state components
. obs_size      - number of observation components
- ensemble_size - number of ensemble members

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorGetSizes()`, `PetscDataAssimilatorSetUp()`
@*/
PetscErrorCode PetscDataAssimilatorSetSizes(PetscDataAssimilator da, PetscInt state_size, PetscInt obs_size, PetscInt ensemble_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, state_size, 2);
  PetscValidLogicalCollectiveInt(da, obs_size, 3);
  PetscValidLogicalCollectiveInt(da, ensemble_size, 4);

  PetscCheck(!da->assembled, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Cannot change sizes after PetscDataAssimilatorSetUp() has been called");

  da->state_size    = state_size;
  da->obs_size      = obs_size;
  da->ensemble_size = ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorGetSizes - Retrieves the dimension settings associated with a `PetscDataAssimilator`.

  Not Collective

  Input Parameter:
. da - the `PetscDataAssimilator` context

  Output Parameters:
+ state_size    - number of state components (may be `NULL`)
. obs_size      - number of observation components (may be `NULL`)
- ensemble_size - number of ensemble members (may be `NULL`)

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorSetSizes()`
@*/
PetscErrorCode PetscDataAssimilatorGetSizes(PetscDataAssimilator da, PetscInt *state_size, PetscInt *obs_size, PetscInt *ensemble_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  if (state_size) *state_size = da->state_size;
  if (obs_size) *obs_size = da->obs_size;
  if (ensemble_size) *ensemble_size = da->ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorSetUp - Allocates internal data structures for a `PetscDataAssimilator` based on the previously provided sizes.

  Collective

  Input Parameter:
. da - the `PetscDataAssimilator` context to assemble

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorSetSizes()`, `PetscDataAssimilatorSetType()`
@*/
PetscErrorCode PetscDataAssimilatorSetUp(PetscDataAssimilator da)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);

  if (da->assembled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set state size before calling PetscDataAssimilatorSetUp()");
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set observation size before calling PetscDataAssimilatorSetUp()");
  PetscCheck(da->ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set ensemble size before calling PetscDataAssimilatorSetUp()");

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
  PetscDataAssimilatorView - Views a `PetscDataAssimilator` and its implementation-specific data structure.

  Collective

  Input Parameters:
+ da     - the `PetscDataAssimilator` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorViewFromOptions()`
@*/
PetscErrorCode PetscDataAssimilatorView(PetscDataAssimilator da, PetscViewer viewer)
{
  PetscBool   iascii;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)da), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(da, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da), &size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDataAssimilator Object: %d MPI process%s\n", size, size > 1 ? "es" : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  State size: %" PetscInt_FMT "\n", da->state_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Observation size: %" PetscInt_FMT "\n", da->obs_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Ensemble size: %" PetscInt_FMT "\n", da->ensemble_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Assembled: %s\n", da->assembled ? "true" : "false"));
  }

  if (da->ops->view) PetscCall((*da->ops->view)(da, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorViewFromOptions - Processes command-line options to determine if a `PetscDataAssimilator` should be viewed.

  Collective

  Input Parameters:
+ da     - the `PetscDataAssimilator` context
. obj    - optional object that provides the prefix for options
- option - option name to check (may be `NULL`)

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorView()`, `PetscObjectViewFromOptions()`
@*/
PetscErrorCode PetscDataAssimilatorViewFromOptions(PetscDataAssimilator da, PetscObject obj, const char option[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)da, obj, option));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorSetObsErrorVariance - Sets the observation-error variances associated with a `PetscDataAssimilator`.

  Collective

  Input Parameters:
+ da            - the `PetscDataAssimilator` context
- obs_error_var - vector containing observation error variances

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorGetObsErrorVariance()`
@*/
PetscErrorCode PetscDataAssimilatorSetObsErrorVariance(PetscDataAssimilator da, Vec obs_error_var)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscValidHeaderSpecific(obs_error_var, VEC_CLASSID, 2);

  if (!da->obs_error_var) PetscCall(VecDuplicate(obs_error_var, &da->obs_error_var));
  PetscCall(VecCopy(obs_error_var, da->obs_error_var));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorGetObsErrorVariance - Returns a borrowed reference to the observation-error variance vector.

  Not Collective

  Input Parameter:
. da - the `PetscDataAssimilator` context

  Output Parameter:
. obs_error_var - pointer to the variance vector managed by the `PetscDataAssimilator`

  Level: beginner

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorSetObsErrorVariance()`
@*/
PetscErrorCode PetscDataAssimilatorGetObsErrorVariance(PetscDataAssimilator da, Vec *obs_error_var)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(obs_error_var, 2);
  *obs_error_var = da->obs_error_var;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorSetOrthogonalTransform - Installs the ensemble-space orthogonal matrix used in deterministic square-root updates.

  Collective

  Input Parameters:
+ da - the `PetscDataAssimilator` context
- U  - orthogonal matrix to store (referenced internally)

  Level: developer

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorGetOrthogonalTransform()`, `PetscDataAssimilatorETKFAnalysis()`
@*/
PetscErrorCode PetscDataAssimilatorSetOrthogonalTransform(PetscDataAssimilator da, Mat U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);

  if (da->U) PetscCall(MatDestroy(&da->U));
  PetscCall(PetscObjectReference((PetscObject)U));
  da->U = U;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorGetOrthogonalTransform - Retrieves the orthogonal matrix currently stored in a `PetscDataAssimilator`.

  Not Collective

  Input Parameter:
. da - the `PetscDataAssimilator` context

  Output Parameter:
. U - pointer that will receive the matrix (may be `NULL`)

  Level: developer

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorSetOrthogonalTransform()`
@*/
PetscErrorCode PetscDataAssimilatorGetOrthogonalTransform(PetscDataAssimilator da, Mat *U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(U, 2);
  *U = da->U;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorGetEnsembleMember - Returns a read-only view of an ensemble member stored in the `PetscDataAssimilator`.

  Collective

  Input Parameters:
+ da         - the `PetscDataAssimilator` context
- member_idx - index of the requested member (0 <= idx < ensemble_size)

  Output Parameter:
. member - read-only vector view; call `PetscDataAssimilatorRestoreEnsembleMember()` when done

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorRestoreEnsembleMember()`, `PetscDataAssimilatorSetEnsembleMember()`
@*/
PetscErrorCode PetscDataAssimilatorGetEnsembleMember(PetscDataAssimilator da, PetscInt member_idx, Vec *member)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(member, 3);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDataAssimilatorSetUp() must be called before accessing ensemble members");
  PetscCheck(member_idx >= 0 && member_idx < da->ensemble_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, da->ensemble_size);

  PetscCall(MatDenseGetColumnVecRead(da->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorRestoreEnsembleMember - Returns a column view obtained with `PetscDataAssimilatorGetEnsembleMember()`.

  Collective

  Input Parameters:
+ da         - the `PetscDataAssimilator` context
. member_idx - index that was previously requested
- member     - location that holds the view to restore

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorGetEnsembleMember()`
@*/
PetscErrorCode PetscDataAssimilatorRestoreEnsembleMember(PetscDataAssimilator da, PetscInt member_idx, Vec *member)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(member, 3);

  PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorSetEnsembleMember - Overwrites an ensemble member with user-provided state data.

  Collective

  Input Parameters:
+ da         - the `PetscDataAssimilator` context
. member_idx - index of the entry to modify
- member     - vector containing the new state values

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorGetEnsembleMember()`
@*/
PetscErrorCode PetscDataAssimilatorSetEnsembleMember(PetscDataAssimilator da, PetscInt member_idx, Vec member)
{
  Vec col;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscValidHeaderSpecific(member, VEC_CLASSID, 3);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDataAssimilatorSetUp() must be called before setting ensemble members");
  PetscCheck(member_idx >= 0 && member_idx < da->ensemble_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, da->ensemble_size);

  PetscCall(MatDenseGetColumnVecWrite(da->ensemble, member_idx, &col));
  PetscCall(VecCopy(member, col));
  PetscCall(MatDenseRestoreColumnVecWrite(da->ensemble, member_idx, &col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorComputeMean - Computes the ensemble mean state for a `PetscDataAssimilator`.

  Collective

  Input Parameters:
+ da   - the `PetscDataAssimilator` context
- mean - vector that will hold the ensemble mean

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorComputeAnomalies()`
@*/
PetscErrorCode PetscDataAssimilatorComputeMean(PetscDataAssimilator da, Vec mean)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);

  PetscUseTypeMethod(da, computemean, mean);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDataAssimilatorComputeAnomalies - Forms the state-space anomalies matrix for a `PetscDataAssimilator`.

  Collective

  Input Parameters:
+ da        - the `PetscDataAssimilator` context
- anomalies - location to store the newly created anomalies matrix

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorComputeMean()`
@*/
PetscErrorCode PetscDataAssimilatorComputeAnomalies(PetscDataAssimilator da, Mat *anomalies)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscAssertPointer(anomalies, 2);

  PetscUseTypeMethod(da, computeanomalies, anomalies);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDataAssimilatorAnalysis - Executes the analysis (update) step of the configured data assimilation method.

  Collective

  Input Parameters:
+ da                   - the `PetscDataAssimilator` context
. observation          - observation vector
. observation_operator - routine that evaluates the observation model `H(x)`
- obs_ctx              - optional context for `observation_operator`

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorApplyModel()`, `PetscDataAssimilatorETKFAnalysis()`
@*/
PetscErrorCode PetscDataAssimilatorAnalysis(PetscDataAssimilator da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);

  PetscUseTypeMethod(da, analysis, observation, observation_operator, obs_ctx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDataAssimilatorApplyModel - Advances every ensemble member through the user-supplied forecast model.

  Collective

  Input Parameters:
+ da        - the `PetscDataAssimilator` context
. model     - routine that evaluates the model map `f(x, xnew; ctx)`
- model_ctx - optional context for `model`

  Level: intermediate

.seealso: [](ch_dataassimilator), `PetscDataAssimilatorAnalysis()`
@*/
PetscErrorCode PetscDataAssimilatorApplyModel(PetscDataAssimilator da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDATAASSIMILATOR_CLASSID, 1);

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

.seealso: [](ch_vec), `PetscRandomSetInterval()`, `VecSetRandom()`
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
      PetscCheck(retry_count < max_retry_count, PETSC_COMM_SELF, PETSC_ERR_LIB, "Random number generator failed to produce valid values after %d attempts", (int)max_retry_count);
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

/*
  PetscDataAssimilatorCholeskySqrt_Private - Computes the lower triangular Cholesky factorization of a symmetric positive definite matrix.

  Input Parameter:
. A - symmetric positive definite matrix to factorize

  Output Parameter:
. L_out - lower triangular Cholesky factor such that A = L * L^T

  Notes:
  This function uses LAPACK's potrf routine for the Cholesky decomposition.
  The input matrix A must be symmetric and positive definite for the factorization to succeed.
  In debug mode, the result is verified by checking ||L*L^T - A||_F.

  The Cholesky factorization is preferred over eigendecomposition-based square roots when:
  - The matrix is known to be positive definite
  - A lower triangular factor is specifically needed
  - Performance is critical (Cholesky is O(n^3/3) vs O(n^3) for eigendecomposition)

  Developer Notes:
  This is a private function used internally by the data assimilation module.
  For a general symmetric matrix square root that handles semi-definite matrices,
  use PetscDataAssimilatorSymmetricEigenSqrt_Private instead.
*/
PetscErrorCode PetscDataAssimilatorCholeskySqrt_Private(Mat A, Mat *L_out)
{
  Mat          L = NULL;
  PetscInt     m, n, i, j;
  PetscScalar *array = NULL;
  PetscBLASInt bn, info;
  PetscBool    is_dense;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(L_out, 2);

  /* Verify matrix properties required for Cholesky factorization */
  PetscCall(MatGetSize(A, &m, &n));
  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be square for Cholesky factorization, got %" PetscInt_FMT " x %" PetscInt_FMT, m, n);
  PetscCheck(n > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Matrix dimension must be positive, got %" PetscInt_FMT, n);

  /* Verify matrix type - Cholesky requires dense storage */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &is_dense));
  if (!is_dense) PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &is_dense));
  PetscCheck(is_dense, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be dense for Cholesky factorization");

  /* Create working copy to preserve input matrix */
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &L));
  PetscCall(MatDenseGetArray(L, &array));

  /* Perform Cholesky factorization using LAPACK */
  PetscCall(PetscBLASIntCast(n, &bn));
  PetscCallBLAS("LAPACKpotrf", LAPACKpotrf_("L", &bn, array, &bn, &info));

  /* Handle LAPACK error codes with detailed diagnostics */
  if (info != 0) {
    PetscCall(MatDenseRestoreArray(L, &array));
    PetscCall(MatDestroy(&L));
    if (info < 0) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK potrf: illegal argument at position %" PetscBLASInt_FMT, -info);
    } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "LAPACK potrf: matrix is not positive definite, leading minor of order %" PetscBLASInt_FMT " is not positive", info);
    }
  }

  /*
    Zero out upper triangle to obtain strict lower triangular result.
    LAPACK potrf stores the result in the lower triangle and leaves the
    upper triangle unchanged. We explicitly zero it for clarity and to
    ensure the output is a proper lower triangular matrix.

    Performance note: This loop is O(n^2) but negligible compared to the
    O(n^3/3) cost of the Cholesky factorization itself.
  */
  for (j = 0; j < n; j++) {
    for (i = 0; i < j; i++) array[i + j * n] = 0.0;
  }

  PetscCall(MatDenseRestoreArray(L, &array));

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));

  /*
    Verify correctness in debug mode by checking ||L*L^T - A||_F <= tolerance.
    This helps catch numerical issues and validates the implementation.
  */
  if (PetscDefined(USE_DEBUG)) {
    Mat       sqrtA_check;
    PetscReal normA, normDiff, tolerance;

    /* Reconstruct A from L*L^T */
    PetscCall(MatMatTransposeMult(L, L, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sqrtA_check));

    /* Compute verification metrics */
    PetscCall(MatNorm(A, NORM_FROBENIUS, &normA));
    PetscCall(MatAXPY(sqrtA_check, -1.0, A, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(sqrtA_check, NORM_FROBENIUS, &normDiff));

    /* Set tolerance relative to matrix magnitude */
    tolerance = MATRIX_SQRT_TOLERANCE_FACTOR * PETSC_MACHINE_EPSILON * PetscMax(1.0, normA);

    PetscCall(MatDestroy(&sqrtA_check));
    PetscCheck(normDiff <= tolerance, PETSC_COMM_SELF, PETSC_ERR_LIB, "Matrix square root verification failed: ||L*L^T - A||_F = %g exceeds tolerance %g (||A||_F = %g)", (double)normDiff, (double)tolerance, (double)normA);
  }

  *L_out = L;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDataAssimilatorSymmetricEigenSqrt_Private - Computes the symmetric matrix square root using eigendecomposition.

  Input Parameter:
. A - symmetric matrix to compute the square root of

  Output Parameter:
. sqrtA_out - matrix square root such that sqrtA * sqrtA = A

  Notes:
  This function uses LAPACK's syev routine to compute eigenvalues and eigenvectors.
  For a symmetric matrix A, the square root is computed as:
    A = V * D * V^T  (eigendecomposition)
    sqrt(A) = V * sqrt(D) * V^T

  The function handles semi-definite matrices by clamping small negative eigenvalues
  (within numerical tolerance) to zero. This is more robust than Cholesky factorization
  for matrices that may not be strictly positive definite.

  In debug mode, the result is verified by checking ||sqrtA*sqrtA - A||_F.

  Performance: O(n^3) for eigendecomposition, plus O(n^3) for matrix multiplications.

  Developer Notes:
  - Prefer PetscDataAssimilatorCholeskySqrt_Private for positive definite matrices (faster)
  - This function is more robust for semi-definite or nearly singular matrices
  - All eigenvalues must be non-negative (within tolerance) for the operation to succeed
*/
PetscErrorCode PetscDataAssimilatorSymmetricEigenSqrt_Private(Mat A, Mat *sqrtA_out)
{
  Mat          sqrtA = NULL, eigenvectors = NULL, scaled_eigenvectors = NULL;
  Vec          sqrt_eigenvalues = NULL;
  PetscInt     matrix_rows, matrix_cols, i;
  PetscScalar *workspace = NULL, *eigvec_array = NULL, *sqrt_eigval_array = NULL;
  PetscReal   *eigenvalues = NULL;
  PetscBLASInt blas_n, workspace_size, lapack_info;
  PetscReal    eigenvalue_tolerance;
  PetscBool    is_dense;

  PetscFunctionBegin;
  /* Validate input parameters */
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(sqrtA_out, 2);

  /* Verify matrix is square */
  PetscCall(MatGetSize(A, &matrix_rows, &matrix_cols));
  PetscCheck(matrix_rows == matrix_cols, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be square for eigenvalue decomposition, got %" PetscInt_FMT " x %" PetscInt_FMT, matrix_rows, matrix_cols);
  PetscCheck(matrix_rows > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Matrix dimension must be positive, got %" PetscInt_FMT, matrix_rows);

  /* Verify matrix type - eigendecomposition requires dense storage */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &is_dense));
  if (!is_dense) PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &is_dense));
  PetscCheck(is_dense, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be dense for eigenvalue decomposition");

  /* Handle edge case: 1x1 matrix */
  if (matrix_rows == 1) {
    PetscScalar val;
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &sqrtA));
    PetscCall(MatDenseGetArray(sqrtA, &eigvec_array));
    val = eigvec_array[0];
    PetscCheck(PetscRealPart(val) >= -EIGENVALUE_TOLERANCE_FACTOR * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix square root failed: value %g is negative", (double)PetscRealPart(val));
    eigvec_array[0] = PetscSqrtScalar(PetscMax((double)PetscRealPart(val), (double)0.0));
    PetscCall(MatDenseRestoreArray(sqrtA, &eigvec_array));
    PetscCall(MatAssemblyBegin(sqrtA, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(sqrtA, MAT_FINAL_ASSEMBLY));
    *sqrtA_out = sqrtA;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Allocate workspace for LAPACK syev (requires 3*n elements per documentation) */
  PetscCall(PetscBLASIntCast(3 * matrix_rows, &workspace_size));
  PetscCall(PetscMalloc1(workspace_size, &workspace));

  /*
    Create a copy of A to hold eigenvectors.
    LAPACK syev overwrites the input matrix with eigenvectors.
  */
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &eigenvectors));
  PetscCall(MatDenseGetArray(eigenvectors, &eigvec_array));

  /* Create vector to store square roots of eigenvalues */
  PetscCall(MatCreateVecs(eigenvectors, NULL, &sqrt_eigenvalues));
  PetscCall(VecGetArrayWrite(sqrt_eigenvalues, &sqrt_eigval_array));

  PetscCall(PetscBLASIntCast(matrix_rows, &blas_n));

  /* Perform eigendecomposition using LAPACK */
#if defined(PETSC_USE_COMPLEX)
  {
    PetscReal *rwork = NULL;
    PetscInt   ridx;

    /*
      Complex-valued path requires:
      - Separate real workspace (rwork) of size 3*n-2
      - Real eigenvalue buffer (eigenvalues are always real for Hermitian matrices)
    */
    PetscCall(PetscMalloc1(3 * matrix_rows - 2, &rwork));
    PetscCall(PetscMalloc1(matrix_rows, &eigenvalues));

    /* Call LAPACK: compute eigenvalues and eigenvectors */
    PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &blas_n, eigvec_array, &blas_n, eigenvalues, workspace, &workspace_size, rwork, &lapack_info));

    /* Copy real eigenvalues to complex array */
    for (ridx = 0; ridx < matrix_rows; ridx++) sqrt_eigval_array[ridx] = eigenvalues[ridx];

    PetscCall(PetscFree(rwork));
  }
#else
  /*
    Real arithmetic path: LAPACK writes eigenvalues directly to output array.
    This avoids an extra allocation and copy.
  */
  eigenvalues = (PetscReal *)sqrt_eigval_array;
  PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &blas_n, eigvec_array, &blas_n, eigenvalues, workspace, &workspace_size, &lapack_info));
#endif

  /* Check LAPACK return status */
  if (lapack_info != 0) {
    PetscCall(VecRestoreArrayWrite(sqrt_eigenvalues, &sqrt_eigval_array));
    PetscCall(MatDenseRestoreArray(eigenvectors, &eigvec_array));
    PetscCall(MatDestroy(&eigenvectors));
    PetscCall(VecDestroy(&sqrt_eigenvalues));
    PetscCall(PetscFree(workspace));
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscFree(eigenvalues));
#endif
    if (lapack_info < 0) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK syev: illegal argument at position %" PetscBLASInt_FMT, -lapack_info);
    } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_CONV_FAILED, "LAPACK syev: failed to converge, %" PetscBLASInt_FMT " off-diagonal elements did not converge to zero", lapack_info);
    }
  }

  /*
    Compute square root of eigenvalues with robust handling of numerical noise.
    For symmetric matrices, eigenvalues should be real and non-negative.
    We allow small negative eigenvalues (within tolerance) and clamp them to zero.
  */
  eigenvalue_tolerance = EIGENVALUE_TOLERANCE_FACTOR * PETSC_MACHINE_EPSILON;
  for (i = 0; i < matrix_rows; i++) {
    PetscReal eigenvalue           = eigenvalues[i];
    PetscReal negativity_threshold = eigenvalue_tolerance * PetscMax(1.0, PetscAbsReal(eigenvalue));

    /* Check that eigenvalue is not significantly negative */
    PetscCheck(eigenvalue >= -negativity_threshold, PETSC_COMM_SELF, PETSC_ERR_LIB,
               "Matrix square root failed: eigenvalue[%" PetscInt_FMT "] = %g is negative beyond tolerance %g. "
               "Matrix may not be symmetric or positive semi-definite.",
               i, (double)eigenvalue, (double)negativity_threshold);

    /* Compute sqrt(eigenvalue), clamping small negative values to zero */
    eigenvalues[i] = (eigenvalue > eigenvalue_tolerance) ? PetscSqrtReal(eigenvalue) : 0.0;
  }

#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree(eigenvalues));
#endif

  /* Restore arrays before matrix operations */
  PetscCall(VecRestoreArrayWrite(sqrt_eigenvalues, &sqrt_eigval_array));
  PetscCall(MatDenseRestoreArray(eigenvectors, &eigvec_array));

  /*
    Reconstruct matrix square root: sqrt(A) = V * sqrt(D) * V^T
    where V contains eigenvectors as columns and sqrt(D) is diagonal.

    Algorithm:
    1. Create scaled_eigenvectors = V * sqrt(D) by column scaling
    2. Compute sqrtA = scaled_eigenvectors * V^T via matrix-transpose-mult
  */
  PetscCall(MatDuplicate(eigenvectors, MAT_COPY_VALUES, &scaled_eigenvectors));
  PetscCall(MatDiagonalScale(scaled_eigenvectors, NULL, sqrt_eigenvalues));
  PetscCall(MatMatTransposeMult(scaled_eigenvectors, eigenvectors, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sqrtA));

  /* Clean up intermediate matrices and vectors */
  PetscCall(MatDestroy(&scaled_eigenvectors));
  PetscCall(VecDestroy(&sqrt_eigenvalues));

  /* Finalize matrix assembly */
  PetscCall(MatAssemblyBegin(sqrtA, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(sqrtA, MAT_FINAL_ASSEMBLY));

  /*
    Verification in debug mode: check ||sqrtA * sqrtA - A||_F <= tolerance
    This validates both the implementation and numerical stability.
  */
  if (PetscDefined(USE_DEBUG)) {
    Mat       verification_matrix = NULL;
    PetscReal norm_original, norm_difference, verification_tolerance;

    /* Compute sqrtA * sqrtA */
    PetscCall(MatMatMult(sqrtA, sqrtA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &verification_matrix));

    /* Compute ||A||_F and ||sqrtA*sqrtA - A||_F */
    PetscCall(MatNorm(A, NORM_FROBENIUS, &norm_original));
    PetscCall(MatAXPY(verification_matrix, -1.0, A, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(verification_matrix, NORM_FROBENIUS, &norm_difference));

    /* Set relative tolerance */
    verification_tolerance = MATRIX_SQRT_TOLERANCE_FACTOR * PETSC_MACHINE_EPSILON * PetscMax(1.0, norm_original);

    PetscCall(MatDestroy(&verification_matrix));

    PetscCheck(norm_difference <= verification_tolerance, PETSC_COMM_SELF, PETSC_ERR_LIB,
               "Matrix square root verification failed: ||sqrtA*sqrtA - A||_F = %g exceeds tolerance %g "
               "(||A||_F = %g, relative error = %g)",
               (double)norm_difference, (double)verification_tolerance, (double)norm_original, (double)(norm_difference / PetscMax(norm_original, 1.0)));
  }

  /* Clean up and return result */
  PetscCall(MatDestroy(&eigenvectors));
  PetscCall(PetscFree(workspace));

  *sqrtA_out = sqrtA;
  PetscFunctionReturn(PETSC_SUCCESS);
}
