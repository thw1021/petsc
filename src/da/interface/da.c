/* Deterministic ensemble Kalman filter support for SWE-GPT5.
   Implements the ETKF package described in Algorithm 6.4 of Asch, Bocquet, and
   Nodet (2016) "Data Assimilation" (SIAM, doi:10.1137/1.9781611974546). */
#include <petsc/private/daimpl.h> /*I "petscda.h"  I*/
#include <petscblaslapack.h>

PetscClassId      PETSCDA_CLASSID          = 0;
PetscBool         PetscDARegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscDAList              = NULL;

static PetscBool PetscDAPackageInitialized = PETSC_FALSE;

/* Tolerance for matrix square root verification in debug mode */
#define MATRIX_SQRT_TOLERANCE_FACTOR 100.0

/* Tolerance for eigenvalue negativity check */
#define EIGENVALUE_TOLERANCE_FACTOR 10.0

/*@C
  PetscDAInitializePackage - This function initializes everything in the `PetscDA`
  package. It is called from `PetscDLLibraryRegister_petscda()` when using dynamic
  libraries, and on the first call to `PetscDACreate()` when using static or shared
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

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers PetscDA.

*/
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscda(void)
{
  PetscFunctionBegin;
  PetscCall(PetscDAInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
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
  PetscCall(PetscDAETKFRegister());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute mean of ensemble */
static PetscErrorCode PetscDAComputeMean_Default(PetscDA da, Vec mean)
{
  Vec         member;
  PetscScalar inv_m;
  PetscInt    m, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before computing the ensemble mean");
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

static PetscErrorCode PetscDAComputeAnomalies_Default(PetscDA da, Mat *anomalies_out)
{
  Vec       mean, col_in, col_out;
  Mat       anomalies;
  MPI_Comm  comm;
  PetscReal scale;
  PetscInt  m, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(anomalies_out, 2);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before computing anomalies");
  PetscCheck(da->ensemble_size > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least 2 to form anomalies");

  m    = da->ensemble_size;
  comm = PetscObjectComm((PetscObject)da->ensemble);

  /* Algorithm line 14: anomalies are normalized by 1/sqrt(m-1) so that X X^T equals the ensemble covariance. */
  scale = 1.0 / PetscSqrtReal((PetscReal)(m - 1));

  PetscCall(VecCreate(comm, &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(mean));

  /* Algorithm line 12: \bar{x} = (1/m)\sum_j x^{(j)} */
  PetscCall(PetscDAComputeMean(da, mean));

  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, da->state_size, m, NULL, &anomalies));
  PetscCall(MatSetUp(anomalies));

  for (j = 0; j < m; ++j) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, j, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(anomalies, j, &col_out));

    /* Algorithm line 13: subtract the mean column-wise to form x^{(j)} - \bar{x} */
    PetscCall(VecWAXPY(col_out, -1.0, mean, col_in));
    /* Algorithm line 14: scale anomalies by 1/\sqrt{m-1} */
    PetscCall(VecScale(col_out, scale));

    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, j, &col_in));
    PetscCall(MatDenseRestoreColumnVecWrite(anomalies, j, &col_out));
  }

  PetscCall(MatAssemblyBegin(anomalies, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(anomalies, MAT_FINAL_ASSEMBLY));

  *anomalies_out = anomalies;
  PetscCall(VecDestroy(&mean));
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

  PetscCall(PetscHeaderCreate(da, PETSCDA_CLASSID, "PetscDA", "Data Assimilation", "DataAssimilation", comm, PetscDADestroy, PetscDAView));
  PetscCall(PetscMemzero(da->ops, sizeof(*da->ops)));
  da->ops->computemean      = PetscDAComputeMean_Default;
  da->ops->computeanomalies = PetscDAComputeAnomalies_Default;

  da->ensemble_size = 0;
  da->state_size    = 0;
  da->obs_size      = 0;
  da->ensemble      = NULL;
  da->obs_error_var = NULL;
  da->U             = NULL;
  da->assembled     = PETSC_FALSE;
  da->data          = NULL;

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

  if (da->ops->destroy) PetscCall((*da->ops->destroy)(da));

  PetscCall(PetscObjectChangeTypeName((PetscObject)da, type));
  PetscCall((*r)(da));

  if (!da->ops->computemean) da->ops->computemean = PetscDAComputeMean_Default;
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);

  PetscObjectOptionsBegin((PetscObject)da);
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "PetscDA Object: %d MPI process%s\n", size, size > 1 ? "es" : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  State size: %" PetscInt_FMT "\n", da->state_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Observation size: %" PetscInt_FMT "\n", da->obs_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Ensemble size: %" PetscInt_FMT "\n", da->ensemble_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Assembled: %s\n", da->assembled ? "true" : "false"));
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
- obs_error_var - vector containing observation error variances

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
PetscErrorCode PetscDAComputeMean(PetscDA da, Vec mean)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);

  PetscUseTypeMethod(da, computemean, mean);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAComputeAnomalies - Forms the state-space anomalies matrix for a `PetscDA`.

  Collective

  Input Parameters:
+ da        - the `PetscDA` context
- anomalies - location to store the newly created anomalies matrix

  Level: intermediate

.seealso: [](ch_da), `PetscDAComputeMean()`
@*/
PetscErrorCode PetscDAComputeAnomalies(PetscDA da, Mat *anomalies)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(anomalies, 2);

  PetscUseTypeMethod(da, computeanomalies, anomalies);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDAAnalysis - Executes the analysis (update) step of the configured data assimilation method.

  Collective

  Input Parameters:
+ da                   - the `PetscDA` context
. observation          - observation vector
. observation_operator - routine that evaluates the observation model `H(x)`
- obs_ctx              - optional context for `observation_operator`

  Level: intermediate

.seealso: [](ch_da), `PetscDAApplyModel()`, `PetscDAETKFAnalysis()`
@*/
PetscErrorCode PetscDAAnalysis(PetscDA da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);

  PetscUseTypeMethod(da, analysis, observation, observation_operator, obs_ctx);
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
  PetscReal       u1, u2, z0, z1, radius;
  const PetscReal min_uniform = PETSC_MACHINE_EPSILON;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArray(v, &array));

  for (i = 0; i < n; i += 2) {
    /* Get uniform random values, ensuring they're not too close to 0 to avoid log(0) */
    do {
      PetscCall(PetscRandomGetValueReal(rng, &u1));
    } while (u1 < min_uniform);

    PetscCall(PetscRandomGetValueReal(rng, &u2));

    /* Box-Muller transform */
    radius = PetscSqrtReal(-2.0 * PetscLogReal(u1));
    z0     = radius * PetscCosReal(2.0 * PETSC_PI * u2);
    z1     = radius * PetscSinReal(2.0 * PETSC_PI * u2);

    array[i] = mean + std_dev * z0;
    if (i + 1 < n) array[i + 1] = mean + std_dev * z1;
  }

  PetscCall(VecRestoreArray(v, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDACholeskySqrt_Private(Mat A, Mat *L_out)
{
  Mat          L;
  PetscInt     m, n, i, j;
  PetscScalar *array;
  PetscBLASInt bn, info;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, &n));
  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be square for Cholesky factorization");

  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &L));
  PetscCall(MatDenseGetArray(L, &array));

  PetscCall(PetscBLASIntCast(n, &bn));
  LAPACKpotrf_("L", &bn, array, &bn, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK potrf failed with info = %" PetscBLASInt_FMT, info);

  /* Zero out upper triangle to get lower triangular result */
  for (j = 0; j < n; j++) {
    for (i = 0; i < j; i++) array[i + j * n] = 0.0;
  }

  PetscCall(MatDenseRestoreArray(L, &array));
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));

  /* Verify correctness in debug mode */
  if (PetscDefined(USE_DEBUG)) {
    Mat       sqrtA = L, sqrtA_check;
    PetscReal normA, normDiff, tolerance;
    PetscCall(MatMatTransposeMult(sqrtA, sqrtA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sqrtA_check));
    PetscCall(MatNorm(A, NORM_FROBENIUS, &normA));
    PetscCall(MatAXPY(sqrtA_check, -1.0, A, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(sqrtA_check, NORM_FROBENIUS, &normDiff));
    tolerance = MATRIX_SQRT_TOLERANCE_FACTOR * PETSC_MACHINE_EPSILON * PetscMax(1.0, normA);
    PetscCall(MatDestroy(&sqrtA_check));
    PetscCheck(normDiff <= tolerance, PETSC_COMM_SELF, PETSC_ERR_LIB, "Matrix square root verification failed: ||sqrtA*sqrtA^T - A||_F = %g (||A||_F = %g, tolerance = %g)", (double)normDiff, (double)normA, (double)tolerance);
  }

  *L_out = L;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDASymmetricEigenSqrt_Private(Mat A, Mat *sqrtA_out)
{
  Mat          sqrtA, V, VSqrtD;
  Vec          sqrtD;
  PetscInt     m, n, i;
  PetscScalar *work, *varray, *sqrtvals;
  PetscReal   *eigvals;
  PetscBLASInt bn, lwork, info;
  PetscReal    eps;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, &n));
  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be square for eigenvalue decomposition");

  /* Workspace length for LAPACKsyev: 3*n follows the routine documentation. */
  PetscCall(PetscBLASIntCast(3 * n, &lwork));
  PetscCall(PetscMalloc1(lwork, &work));

  /* Copy A because LAPACKsyev overwrites its input with eigenvectors. */
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &V));
  PetscCall(MatDenseGetArray(V, &varray));

  PetscCall(MatCreateVecs(V, NULL, &sqrtD));
  PetscCall(VecGetArrayWrite(sqrtD, &sqrtvals));

  PetscCall(PetscBLASIntCast(n, &bn));

#if defined(PETSC_USE_COMPLEX)
  {
    PetscReal *rwork;
    PetscInt   ridx;
    /* Complex-valued path needs an auxiliary real rwork array and a separate real eigenvalue buffer. */
    PetscCall(PetscMalloc1(3 * n - 2, &rwork));
    PetscCall(PetscMalloc1(n, &eigvals));
    PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &bn, varray, &bn, eigvals, work, &lwork, rwork, &info));
    for (ridx = 0; ridx < n; ridx++) sqrtvals[ridx] = eigvals[ridx];
    PetscCall(PetscFree(rwork));
  }
#else
  /* In real arithmetic LAPACK writes eigenvalues directly into sqrtvals. */
  eigvals = (PetscReal *)sqrtvals;
  PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &bn, varray, &bn, eigvals, work, &lwork, &info));
#endif
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK syev failed with info = %" PetscBLASInt_FMT, info);

  /* Compute square root of eigenvalues and check for negative eigenvalues */
  eps = EIGENVALUE_TOLERANCE_FACTOR * PETSC_MACHINE_EPSILON;
  for (i = 0; i < n; i++) {
    PetscReal eig       = eigvals[i];
    PetscReal threshold = eps * PetscMax(1.0, PetscAbsReal(eig));
    PetscCheck(eig >= -threshold, PETSC_COMM_SELF, PETSC_ERR_LIB, "Matrix square root failed: eigenvalue %g is negative beyond tolerance %g", (double)eig, (double)threshold);
    eigvals[i] = (eig > threshold) ? PetscSqrtReal(PetscMax(eig, (PetscReal)0.0)) : 0.0;
  }

#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree(eigvals));
#endif

  PetscCall(VecRestoreArrayWrite(sqrtD, &sqrtvals));
  PetscCall(MatDenseRestoreArray(V, &varray));

  PetscCall(MatDuplicate(V, MAT_COPY_VALUES, &VSqrtD));
  /* Form V * sqrt(D) by scaling each eigenvector column. */
  PetscCall(MatDiagonalScale(VSqrtD, NULL, sqrtD));
  /* Reconstruct sqrt(A) = (V sqrt(D)) * V^T. */
  PetscCall(MatMatTransposeMult(VSqrtD, V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sqrtA));
  PetscCall(MatDestroy(&VSqrtD));
  PetscCall(VecDestroy(&sqrtD));

  PetscCall(MatAssemblyBegin(sqrtA, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(sqrtA, MAT_FINAL_ASSEMBLY));

  /* Verify correctness in debug mode */
  if (PetscDefined(USE_DEBUG)) {
    Mat       sqrtA_check;
    PetscReal normA, normDiff, tolerance;
    PetscCall(MatMatMult(sqrtA, sqrtA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sqrtA_check));
    PetscCall(MatNorm(A, NORM_FROBENIUS, &normA));
    PetscCall(MatAXPY(sqrtA_check, -1.0, A, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(sqrtA_check, NORM_FROBENIUS, &normDiff));
    tolerance = MATRIX_SQRT_TOLERANCE_FACTOR * PETSC_MACHINE_EPSILON * PetscMax(1.0, normA);
    PetscCall(MatDestroy(&sqrtA_check));
    PetscCheck(normDiff <= tolerance, PETSC_COMM_SELF, PETSC_ERR_LIB, "Matrix square root verification failed: ||sqrtA*sqrtA - A||_F = %g (||A||_F = %g, tolerance = %g)", (double)normDiff, (double)normA, (double)tolerance);
  }

  PetscCall(MatDestroy(&V));
  PetscCall(PetscFree(work));

  *sqrtA_out = sqrtA;
  PetscFunctionReturn(PETSC_SUCCESS);
}
