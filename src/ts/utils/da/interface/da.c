/* Deterministic ensemble Kalman filter support for SWE-GPT5.
   Implements the ETKF package described in Algorithm 6.4 of Asch, Bocquet, and
   Nodet (2016) "Data Assimilation" (SIAM, doi:10.1137/1.9781611974546). */
#include <petsc/private/daimpl.h> /*I "petscda.h"  I*/
#include <petscblaslapack.h>

PetscClassId      DA_CLASSID          = 0;
PetscBool         DARegisterAllCalled = PETSC_FALSE;
PetscFunctionList DAList              = NULL;

static PetscBool DAPackageInitialized = PETSC_FALSE;

/*@C
  DAInitializePackage - This function initializes everything in the `DA`
  package. It is called from `PetscDLLibraryRegister_petscda()` when using dynamic
  libraries, and on the first call to `DACreate()` when using static or shared
  libraries.

  Level: developer

.seealso: `DAFinalizePackage()`, `PetscInitialize()`
@*/
PetscErrorCode DAInitializePackage(void)
{
  PetscFunctionBegin;
  if (DAPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);

  DAPackageInitialized = PETSC_TRUE;
  PetscCall(PetscClassIdRegister("Data Assimilation", &DA_CLASSID));
  PetscCall(DARegisterAll());
  PetscCall(PetscRegisterFinalize(DAFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers DA.

*/
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscda(void)
{
  PetscFunctionBegin;
  PetscCall(DAInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
/*@C
  DAFinalizePackage - This function finalizes everything in the `DA` package. It
  is called from `PetscFinalize()`.

  Level: developer

.seealso: `DAInitializePackage()`, `PetscInitialize()`
@*/
PetscErrorCode DAFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DAList));
  DARegisterAllCalled  = PETSC_FALSE;
  DAPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DAETKFRegister(void);

/*@C
  DARegister - Registers a constructor for a `DA` implementation with the
  dispatcher.

  Not Collective

  Input Parameters:
+ sname    - name associated with the implementation
- function - routine that creates the implementation and installs method table

  Level: developer

.seealso: [](ch_da), `DARegisterAll()`, `DASetType()`
@*/
PetscErrorCode DARegister(const char sname[], PetscErrorCode (*function)(DA))
{
  PetscFunctionBegin;
  PetscCall(DAInitializePackage());
  PetscCall(PetscFunctionListAdd(&DAList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DARegisterAll - Registers all data assimilation backends that were compiled in.

  Not Collective

  Level: developer

.seealso: [](ch_da), `DARegister()`
@*/
PetscErrorCode DARegisterAll(void)
{
  PetscFunctionBegin;
  if (DARegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  DARegisterAllCalled = PETSC_TRUE;
  PetscCall(DAETKFRegister());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute mean of ensemble */
static PetscErrorCode DAComputeMean_Default(DA da, Vec mean)
{
  Vec         member;
  PetscScalar inv_m;
  PetscInt    m, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "DASetUp() must be called before computing the ensemble mean");
  PetscCheck(da->ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Ensemble size must be positive");

  PetscCall(VecSet(mean, 0.0));
  m     = da->ensemble_size;
  inv_m = 1.0 / (PetscScalar)m;
  for (j = 0; j < m; ++j) {
    PetscCall(MatDenseGetColumnVecRead(da->ensemble, j, &member));
    PetscCall(VecAXPY(mean, inv_m, member));
    PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, j, &member));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DAComputeAnomalies_Default(DA da, Mat *anomalies_out)
{
  Vec       mean, col_in, col_out;
  Mat       anomalies;
  PetscReal scale;
  PetscInt  m, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(anomalies_out, 2);
  PetscCheck(da->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "DASetUp() must be called before computing anomalies");
  PetscCheck(da->ensemble_size > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least 2 to form anomalies");

  m = da->ensemble_size;
  /* Algorithm line 14: anomalies are normalized by 1/sqrt(m-1) so that X X^T equals the ensemble covariance. */
  scale = 1.0 / PetscSqrtReal((PetscReal)(m - 1));

  PetscCall(VecCreate(PetscObjectComm((PetscObject)da->ensemble), &mean));
  PetscCall(VecSetSizes(mean, PETSC_DECIDE, da->state_size));
  PetscCall(VecSetFromOptions(mean));
  /* Algorithm line 12: \bar{x} = (1/m)\sum_j x^{(j)} */
  PetscCall(DAComputeMean(da, mean));

  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da->ensemble), PETSC_DECIDE, PETSC_DECIDE, da->state_size, m, NULL, &anomalies));
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
  DACreate - Creates a new `DA` object for ensemble-based data assimilation.

  Collective

  Input Parameter:
. comm - MPI communicator used to create the object

  Output Parameter:
. da_out - newly created `DA` object

  Level: beginner

.seealso: [](ch_da), `DADestroy()`, `DASetType()`, `DASetUp()`
@*/
PetscErrorCode DACreate(MPI_Comm comm, DA *da_out)
{
  DA da;

  PetscFunctionBegin;
  PetscAssertPointer(da_out, 2);

  PetscCall(DAInitializePackage());

  PetscCall(PetscHeaderCreate(da, DA_CLASSID, "DA", "Data Assimilation", "DataAssimilation", comm, DADestroy, DAView));
  PetscCall(PetscMemzero(da->ops, sizeof(*da->ops)));
  da->ops->computemean      = DAComputeMean_Default;
  da->ops->computeanomalies = DAComputeAnomalies_Default;

  da->ensemble_size = 0;
  da->state_size    = 0;
  da->obs_size      = 0;
  da->ensemble      = NULL;
  da->obs_error_var = NULL;
  da->U             = NULL;
  da->assembled     = PETSC_FALSE;
  da->data          = NULL;

  *da_out = da;

  PetscCall(DASetType(da, DAETKF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DADestroy - Destroys a `DA` object and releases its resources.

  Collective

  Input Parameter:
. da - pointer to the `DA` object to destroy

  Level: beginner

.seealso: [](ch_da), `DACreate()`
@*/
PetscErrorCode DADestroy(DA *da)
{
  PetscFunctionBegin;
  if (!da || !*da) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*da, DA_CLASSID, 1);
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
  DASetType - Sets the data assimilation implementation used by a `DA` object.

  Collective

  Input Parameters:
+ da   - the `DA` context
- type - name of the implementation (for example `DAETKF`)

  Level: intermediate

.seealso: [](ch_da), `DAGetType()`, `DARegister()`
@*/
PetscErrorCode DASetType(DA da, DAType type)
{
  PetscErrorCode (*r)(DA);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)da, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DARegisterAll());
  PetscCall(PetscFunctionListFind(DAList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DA type: %s", type);

  if (da->ops->destroy) PetscCall((*da->ops->destroy)(da));

  PetscCall(PetscObjectChangeTypeName((PetscObject)da, type));
  PetscCall((*r)(da));

  if (!da->ops->computemean) da->ops->computemean = DAComputeMean_Default;
  if (!da->ops->computeanomalies) da->ops->computeanomalies = DAComputeAnomalies_Default;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAGetType - Gets the name of the implementation currently associated with a `DA`.

  Not Collective

  Input Parameter:
. da - the `DA` context

  Output Parameter:
. type - pointer that will receive the type name (may be `NULL`)

  Level: intermediate

.seealso: [](ch_da), `DASetType()`
@*/
PetscErrorCode DAGetType(DA da, DAType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  if (type) {
    PetscAssertPointer(type, 2);
    *type = ((PetscObject)da)->type_name;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DASetFromOptions - Configures a `DA` object from the options database.

  Collective

  Input Parameter:
. da - the `DA` context to set up

  Level: intermediate

.seealso: [](ch_da), `DASetType()`, `PetscObjectOptionsBegin()`
@*/
PetscErrorCode DASetFromOptions(DA da)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);

  PetscObjectOptionsBegin((PetscObject)da);
  if (da->ops->setfromoptions) PetscCall((*da->ops->setfromoptions)(da, &PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DASetSizes - Sets the state, observation, and ensemble dimensions used by a `DA`.

  Collective

  Input Parameters:
+ da            - the `DA` context
. state_size    - number of state components
. obs_size      - number of observation components
- ensemble_size - number of ensemble members

  Level: beginner

.seealso: [](ch_da), `DAGetSizes()`, `DASetUp()`
@*/
PetscErrorCode DASetSizes(DA da, PetscInt state_size, PetscInt obs_size, PetscInt ensemble_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, state_size, 2);
  PetscValidLogicalCollectiveInt(da, obs_size, 3);
  PetscValidLogicalCollectiveInt(da, ensemble_size, 4);

  PetscCheck(!da->assembled, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Cannot change sizes after DASetUp() has been called");

  da->state_size    = state_size;
  da->obs_size      = obs_size;
  da->ensemble_size = ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAGetSizes - Retrieves the dimension settings associated with a `DA`.

  Not Collective

  Input Parameter:
. da - the `DA` context

  Output Parameters:
+ state_size    - number of state components (may be `NULL`)
. obs_size      - number of observation components (may be `NULL`)
- ensemble_size - number of ensemble members (may be `NULL`)

  Level: beginner

.seealso: [](ch_da), `DASetSizes()`
@*/
PetscErrorCode DAGetSizes(DA da, PetscInt *state_size, PetscInt *obs_size, PetscInt *ensemble_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  if (state_size) *state_size = da->state_size;
  if (obs_size) *obs_size = da->obs_size;
  if (ensemble_size) *ensemble_size = da->ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DASetUp - Allocates internal data structures for a `DA` based on the previously provided sizes.

  Collective

  Input Parameter:
. da - the `DA` context to assemble

  Level: beginner

.seealso: [](ch_da), `DASetSizes()`, `DASetType()`
@*/
PetscErrorCode DASetUp(DA da)
{
  PetscInt     i, j;
  PetscScalar *uarray;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);

  if (da->assembled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set state size before calling DASetUp()");
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set observation size before calling DASetUp()");
  PetscCheck(da->ensemble_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set ensemble size before calling DASetUp()");

  if (!da->ensemble) {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da), PETSC_DECIDE, PETSC_DECIDE, da->state_size, da->ensemble_size, NULL, &da->ensemble));
    PetscCall(MatSetUp(da->ensemble));
  }

  if (!da->obs_error_var) {
    PetscCall(VecCreate(PetscObjectComm((PetscObject)da), &da->obs_error_var));
    PetscCall(VecSetSizes(da->obs_error_var, PETSC_DECIDE, da->obs_size));
    PetscCall(VecSetFromOptions(da->obs_error_var));
    PetscCall(VecSet(da->obs_error_var, 1.0));
  }

  if (!da->U) {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)da), PETSC_DECIDE, PETSC_DECIDE, da->ensemble_size, da->ensemble_size, NULL, &da->U));
    PetscCall(MatSetUp(da->U));

    PetscCall(MatDenseGetArray(da->U, &uarray));
    for (i = 0; i < da->ensemble_size; i++) {
      for (j = 0; j < da->ensemble_size; j++) uarray[i * da->ensemble_size + j] = (i == j) ? 1.0 : 0.0;
    }
    PetscCall(MatDenseRestoreArray(da->U, &uarray));
    PetscCall(MatAssemblyBegin(da->U, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(da->U, MAT_FINAL_ASSEMBLY));
  }

  da->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAView - Views a `DA` and its implementation-specific data structure.

  Collective

  Input Parameters:
+ da     - the `DA` context
- viewer - the `PetscViewer` to use (or `NULL` for standard output)

  Level: beginner

.seealso: [](ch_da), `DAViewFromOptions()`
@*/
PetscErrorCode DAView(DA da, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)da), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(da, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "DA Object: %d MPI processes\n", PetscObjectComm((PetscObject)da) == MPI_COMM_SELF ? 1 : PetscGlobalSize));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  State size: %" PetscInt_FMT "\n", da->state_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Observation size: %" PetscInt_FMT "\n", da->obs_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Ensemble size: %" PetscInt_FMT "\n", da->ensemble_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Assembled: %s\n", da->assembled ? "true" : "false"));
  }

  if (da->ops->view) PetscCall((*da->ops->view)(da, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAViewFromOptions - Processes command-line options to determine if a `DA` should be viewed.

  Collective

  Input Parameters:
+ da     - the `DA` context
. obj    - optional object that provides the prefix for options
- option - option name to check (may be `NULL`)

  Level: beginner

.seealso: [](ch_da), `DAView()`, `PetscObjectViewFromOptions()`
@*/
PetscErrorCode DAViewFromOptions(DA da, PetscObject obj, const char option[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)da, obj, option));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DASetObsErrorVariance - Sets the observation-error variances associated with a `DA`.

  Collective

  Input Parameters:
+ da            - the `DA` context
- obs_error_var - vector containing observation error variances

  Level: beginner

.seealso: [](ch_da), `DAGetObsErrorVariance()`
@*/
PetscErrorCode DASetObsErrorVariance(DA da, Vec obs_error_var)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscValidHeaderSpecific(obs_error_var, VEC_CLASSID, 2);

  if (!da->obs_error_var) PetscCall(VecDuplicate(obs_error_var, &da->obs_error_var));
  PetscCall(VecCopy(obs_error_var, da->obs_error_var));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAGetObsErrorVariance - Returns a borrowed reference to the observation-error variance vector.

  Not Collective

  Input Parameter:
. da - the `DA` context

  Output Parameter:
. obs_error_var - pointer to the variance vector managed by the `DA`

  Level: beginner

.seealso: [](ch_da), `DASetObsErrorVariance()`
@*/
PetscErrorCode DAGetObsErrorVariance(DA da, Vec *obs_error_var)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(obs_error_var, 2);
  *obs_error_var = da->obs_error_var;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DASetOrthogonalTransform - Installs the ensemble-space orthogonal matrix used in deterministic square-root updates.

  Collective

  Input Parameters:
+ da - the `DA` context
- U  - orthogonal matrix to store (referenced internally)

  Level: developer

.seealso: [](ch_da), `DAGetOrthogonalTransform()`, `DAETKFAnalysis()`
@*/
PetscErrorCode DASetOrthogonalTransform(DA da, Mat U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);

  if (da->U) PetscCall(MatDestroy(&da->U));
  PetscCall(PetscObjectReference((PetscObject)U));
  da->U = U;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAGetOrthogonalTransform - Retrieves the orthogonal matrix currently stored in a `DA`.

  Not Collective

  Input Parameter:
. da - the `DA` context

  Output Parameter:
. U - pointer that will receive the matrix (may be `NULL`)

  Level: developer

.seealso: [](ch_da), `DASetOrthogonalTransform()`
@*/
PetscErrorCode DAGetOrthogonalTransform(DA da, Mat *U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(U, 2);
  *U = da->U;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAGetEnsembleMember - Returns a read-only view of an ensemble member stored in the `DA`.

  Collective

  Input Parameters:
+ da         - the `DA` context
- member_idx - index of the requested member (0 <= idx < ensemble_size)

  Output Parameter:
. member - read-only vector view; call `DARestoreEnsembleMember()` when done

  Level: intermediate

.seealso: [](ch_da), `DARestoreEnsembleMember()`, `DASetEnsembleMember()`
@*/
PetscErrorCode DAGetEnsembleMember(DA da, PetscInt member_idx, Vec *member)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(member, 3);

  PetscCheck(member_idx >= 0 && member_idx < da->ensemble_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, da->ensemble_size);

  PetscCall(MatDenseGetColumnVecRead(da->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DARestoreEnsembleMember - Returns a column view obtained with `DAGetEnsembleMember()`.

  Collective

  Input Parameters:
+ da         - the `DA` context
. member_idx - index that was previously requested
- member     - location that holds the view to restore

  Level: intermediate

.seealso: [](ch_da), `DAGetEnsembleMember()`
@*/
PetscErrorCode DARestoreEnsembleMember(DA da, PetscInt member_idx, Vec *member)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(member, 3);

  PetscCall(MatDenseRestoreColumnVecRead(da->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DASetEnsembleMember - Overwrites an ensemble member with user-provided state data.

  Collective

  Input Parameters:
+ da         - the `DA` context
. member_idx - index of the entry to modify
- member     - vector containing the new state values

  Level: intermediate

.seealso: [](ch_da), `DAGetEnsembleMember()`
@*/
PetscErrorCode DASetEnsembleMember(DA da, PetscInt member_idx, Vec member)
{
  Vec col;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscValidHeaderSpecific(member, VEC_CLASSID, 3);

  PetscCheck(member_idx >= 0 && member_idx < da->ensemble_size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index out of range");

  PetscCall(MatDenseGetColumnVecWrite(da->ensemble, member_idx, &col));
  PetscCall(VecCopy(member, col));
  PetscCall(MatDenseRestoreColumnVecWrite(da->ensemble, member_idx, &col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAComputeMean - Computes the ensemble mean state for a `DA`.

  Collective

  Input Parameters:
+ da   - the `DA` context
- mean - vector that will hold the ensemble mean

  Level: intermediate

.seealso: [](ch_da), `DAComputeAnomalies()`
@*/
PetscErrorCode DAComputeMean(DA da, Vec mean)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);

  PetscUseTypeMethod(da, computemean, mean);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DAComputeAnomalies - Forms the state-space anomalies matrix for a `DA`.

  Collective

  Input Parameters:
+ da        - the `DA` context
- anomalies - location to store the newly created anomalies matrix

  Level: intermediate

.seealso: [](ch_da), `DAComputeMean()`
@*/
PetscErrorCode DAComputeAnomalies(DA da, Mat *anomalies)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(anomalies, 2);

  PetscUseTypeMethod(da, computeanomalies, anomalies);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DAAnalysis - Executes the analysis (update) step of the configured data assimilation method.

  Collective

  Input Parameters:
+ da                   - the `DA` context
. observation          - observation vector
. observation_operator - routine that evaluates the observation model `H(x)`
- obs_ctx              - optional context for `observation_operator`

  Level: intermediate

.seealso: [](ch_da), `DAApplyModel()`, `DAETKFAnalysis()`
@*/
PetscErrorCode DAAnalysis(DA da, Vec observation, PetscErrorCode (*observation_operator)(Vec, Vec, void *), void *obs_ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);

  PetscUseTypeMethod(da, analysis, observation, observation_operator, obs_ctx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DAApplyModel - Advances every ensemble member through the user-supplied forecast model.

  Collective

  Input Parameters:
+ da        - the `DA` context
. model     - routine that evaluates the model map `f(x, xnew; ctx)`
- model_ctx - optional context for `model`

  Level: intermediate

.seealso: [](ch_da), `DAAnalysis()`
@*/
PetscErrorCode DAApplyModel(DA da, PetscErrorCode (*model)(Vec, Vec, void *), void *model_ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DA_CLASSID, 1);
  PetscAssertPointer(model_ctx, 3);

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

.seealso: [](ch_vec), `PetscRandomSetInterval()`, `VecSetRandom()`
@*/
PetscErrorCode VecSetRandomGaussian(Vec v, PetscRandom rng, PetscReal mean, PetscReal std_dev)
{
  return VecSetRandomGaussian_Private(v, rng, mean, std_dev);
}

PetscErrorCode VecSetRandomGaussian_Private(Vec v, PetscRandom rng, PetscReal mean, PetscReal std_dev)
{
  PetscInt     n, i;
  PetscScalar *array;
  PetscReal    u1, u2, z0, z1;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArray(v, &array));

  for (i = 0; i < n; i += 2) {
    PetscCall(PetscRandomGetValueReal(rng, &u1));
    PetscCall(PetscRandomGetValueReal(rng, &u2));

    z0 = PetscSqrtReal(-2.0 * PetscLogReal(u1)) * PetscCosReal(2.0 * PETSC_PI * u2);
    z1 = PetscSqrtReal(-2.0 * PetscLogReal(u1)) * PetscSinReal(2.0 * PETSC_PI * u2);

    array[i] = mean + std_dev * z0;
    if (i + 1 < n) array[i + 1] = mean + std_dev * z1;
  }

  PetscCall(VecRestoreArray(v, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DACholeskySqrt_Private(Mat A, Mat *L_out)
{
  Mat          L;
  PetscInt     m, n, i, j;
  PetscScalar *array;
  PetscBLASInt bn, info;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, &n));
  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be square");

  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &L));
  PetscCall(MatDenseGetArray(L, &array));

  PetscCall(PetscBLASIntCast(n, &bn));
  LAPACKpotrf_("L", &bn, array, &bn, &info);
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK potrf failed with info = %" PetscBLASInt_FMT, info);

  for (j = 0; j < n; j++) {
    for (i = 0; i < j; i++) array[i + j * n] = 0.0;
  }

  PetscCall(MatDenseRestoreArray(L, &array));
  PetscCall(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY));

  // check correctness
  if (PetscDefined(USE_DEBUG)) {
    Mat       sqrtA = L, sqrtA_check;
    PetscReal normA, normDiff, tolerance;
    PetscCall(MatMatTransposeMult(sqrtA, sqrtA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sqrtA_check));
    PetscCall(MatNorm(A, NORM_FROBENIUS, &normA));
    PetscCall(MatAXPY(sqrtA_check, -1.0, A, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(sqrtA_check, NORM_FROBENIUS, &normDiff));
    tolerance = 100.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, normA);
    PetscCall(MatDestroy(&sqrtA_check));
    PetscCheck(normDiff <= tolerance, PETSC_COMM_SELF, PETSC_ERR_LIB,
               "Matrix square root verification failed: ||sqrtA*sqrtA - A||_F "
               "= %g (||A||_F = %g)",
               (double)normDiff, (double)normA);
  }

  *L_out = L;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DASymmetricEigenSqrt_Private(Mat A, Mat *sqrtA_out)
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
  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix must be square");

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
    /* Complex-valued path needs an auxiliary real rwork array and a separate real eigenvalue buffer. */
    PetscCall(PetscMalloc1(3 * n - 2, &rwork));
    PetscCall(PetscMalloc(n, &eigvals));
    PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &bn, varray, &bn, eigvals, work, &lwork, rwork, &info));
    for (int i = 0; i < n; i++) sqrtvals[i] = eigvals[i];
    PetscCall(PetscFree(rwork));
  }
#else
  /* In real arithmetic LAPACK writes eigenvalues directly into sqrtvals. */
  eigvals = (PetscReal *)sqrtvals;
  PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &bn, varray, &bn, eigvals, work, &lwork, &info));
#endif
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK syev failed with info = %" PetscBLASInt_FMT, info);

  eps = 10.0 * PETSC_MACHINE_EPSILON;
  for (i = 0; i < n; i++) {
    PetscReal eig       = eigvals[i];
    PetscReal threshold = eps * PetscMax(1.0, PetscAbsReal(eig));
    PetscCheck(eig >= -threshold, PETSC_COMM_SELF, PETSC_ERR_LIB, "Matrix square root failed: eigenvalue %g is negative beyond tolerance", (double)eig);
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

  // check correctness
  if (PetscDefined(USE_DEBUG)) {
    Mat       sqrtA_check;
    PetscReal normA, normDiff, tolerance;
    PetscCall(MatMatMult(sqrtA, sqrtA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &sqrtA_check));
    PetscCall(MatNorm(A, NORM_FROBENIUS, &normA));
    PetscCall(MatAXPY(sqrtA_check, -1.0, A, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(sqrtA_check, NORM_FROBENIUS, &normDiff));
    tolerance = 100.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, normA);
    PetscCall(MatDestroy(&sqrtA_check));
    PetscCheck(normDiff <= tolerance, PETSC_COMM_SELF, PETSC_ERR_LIB,
               "Matrix square root verification failed: ||sqrtA*sqrtA - A||_F "
               "= %g (||A||_F = %g)",
               (double)normDiff, (double)normA);
  }

  PetscCall(MatDestroy(&V));
  PetscCall(PetscFree(work));

  *sqrtA_out = sqrtA;
  PetscFunctionReturn(PETSC_SUCCESS);
}
