#include <petsc/private/dasimpl.h>

PetscBool         PetscDASRegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscDASList              = NULL;

PetscClassId PETSCDAS_CLASSID;

/* Logging support */
PetscLogEvent PetscDAS_SetUp, PetscDAS_Assimilate, PetscDAS_Forecast;

/*@C
  PetscDASRegister - Adds a method to the `PetscDAS` package.

  Not collective

  Input Parameters:
+ sname    - name of a new user-defined data assimilation method
- function - routine to create method context

  Level: advanced

.seealso: `PetscDASRegisterAll()`
@*/
PetscErrorCode PetscDASRegister(const char sname[], PetscErrorCode (*function)(PetscDAS))
{
  PetscFunctionBegin;
  PetscCall(PetscDASInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscDASList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASCreate - Creates a `PetscDAS` object.

  Collective

  Input Parameter:
. comm - the MPI communicator that will share the `PetscDAS` object

  Output Parameter:
. newdas - the new `PetscDAS` object

  Level: beginner

.seealso: `PetscDASAssimilate()`, `PetscDASForecast()`, `PetscDAS`
@*/
PetscErrorCode PetscDASCreate(MPI_Comm comm, PetscDAS *newdas)
{
  PetscDAS das;

  PetscFunctionBegin;
  PetscAssertPointer(newdas, 2);
  *newdas = NULL;
  PetscCall(PetscDASInitializePackage());

  PetscCall(PetscHeaderCreate(das, PETSCDAS_CLASSID, "PetscDAS", "Data Assimilation System", "PetscDAS", comm, PetscDASDestroy, PetscDASView));

  das->setupcalled   = PETSC_FALSE;
  das->assimilated   = PETSC_FALSE;
  das->data          = NULL;
  das->ensemble      = NULL;
  das->ensemble_size = 0;
  das->state_size    = 0;
  das->observation        = NULL;
  das->obs_operator       = NULL;
  das->obs_error_cov_diag = NULL;
  das->random             = NULL;

  *newdas = das;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASView - Prints information about the `PetscDAS` object

  Collective

  Input Parameters:
+ das    - the `PetscDAS` context
- viewer - a `PetscViewer` context

  Options Database Key:
. -das_view - Calls `PetscDASView()` at the end of `PetscDASAssimilate()`

  Level: beginner

.seealso: `PetscDAS`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode PetscDASView(PetscDAS das, PetscViewer viewer)
{
  PetscBool    isascii, isstring;
  PetscDASType type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(((PetscObject)das)->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(das, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)das, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Ensemble size: %" PetscInt_FMT "\n", das->ensemble_size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "State size: %" PetscInt_FMT "\n", das->state_size));
    PetscTryTypeMethod(das, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(PetscDASGetType(das, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " PetscDASType: %-7.7s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASViewFromOptions - View a `PetscDAS` object based on values in the options database

  Collective

  Input Parameters:
+ das  - the `PetscDAS` context
. obj  - Optional object that provides the prefix for the options database
- name - command line option

  Level: intermediate

.seealso: `PetscDAS`, `PetscDASView`, `PetscObjectViewFromOptions()`, `PetscDASCreate()`
@*/
PetscErrorCode PetscDASViewFromOptions(PetscDAS das, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)das, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASSetFromOptions - Sets `PetscDAS` options from the options database.

  Collective

  Input Parameter:
. das - the `PetscDAS` context

  Options Database Keys:
. -das_type <type> - the particular type of data assimilation method to be used

  Level: beginner

.seealso: `PetscDAS`, `PetscDASCreate()`
@*/
PetscErrorCode PetscDASSetFromOptions(PetscDAS das)
{
  PetscBool    flg;
  PetscDASType default_type = PETSCDASETKF;
  char         type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  if (((PetscObject)das)->type_name) default_type = ((PetscObject)das)->type_name;
  PetscObjectOptionsBegin((PetscObject)das);
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-das_type", "PetscDAS type", "PetscDASSetType", PetscDASList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(PetscDASSetType(das, type));
  } else if (!((PetscObject)das)->type_name) {
    PetscCall(PetscDASSetType(das, default_type));
  }
  PetscTryTypeMethod(das, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASSetUp - Sets up the internal data structures for the data assimilation.

  Collective

  Input Parameter:
. das - the `PetscDAS` context

  Level: advanced

.seealso: `PetscDASCreate()`, `PetscDASAssimilate()`, `PetscDASDestroy()`
@*/
PetscErrorCode PetscDASSetUp(PetscDAS das)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  if (das->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(PetscDAS_SetUp, das, 0, 0, 0));
  PetscTryTypeMethod(das, setup);
  das->setupcalled = PETSC_TRUE;
  PetscCall(PetscLogEventEnd(PetscDAS_SetUp, das, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASAssimilate - Perform data assimilation with an observation

  Collective

  Input Parameters:
+ das         - the `PetscDAS` context
- observation - the observation vector

  Level: beginner

.seealso: `PetscDASCreate()`, `PetscDASForecast()`, `PetscDASSetUp()`
@*/
PetscErrorCode PetscDASAssimilate(PetscDAS das, Vec observation)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  if (observation) PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);

  if (observation) {
    PetscCall(PetscObjectReference((PetscObject)observation));
    PetscCall(VecDestroy(&das->observation));
    das->observation = observation;
  }
  PetscCall(PetscDASSetUp(das));

  PetscCall(PetscLogEventBegin(PetscDAS_Assimilate, das, observation, 0, 0));
  PetscUseTypeMethod(das, assimilate, observation);
  PetscCall(PetscLogEventEnd(PetscDAS_Assimilate, das, observation, 0, 0));
  PetscCall(PetscDASViewFromOptions(das, NULL, "-das_view"));
  das->assimilated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASForecast - Propagate the ensemble forward using a model operator

  Collective

  Input Parameters:
+ das            - the `PetscDAS` context
- model_operator - the model operator matrix (can be NULL for nonlinear models)

  Level: beginner

.seealso: `PetscDASAssimilate()`
@*/
PetscErrorCode PetscDASForecast(PetscDAS das, Mat model_operator)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  if (model_operator) PetscValidHeaderSpecific(model_operator, MAT_CLASSID, 2);

  PetscCall(PetscLogEventBegin(PetscDAS_Forecast, das, model_operator, 0, 0));
  PetscTryTypeMethod(das, forecast, model_operator);
  PetscCall(PetscLogEventEnd(PetscDAS_Forecast, das, model_operator, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASReset - Resets a `PetscDAS` context

  Collective

  Input Parameter:
. das - context obtained from `PetscDASCreate()`

  Level: intermediate

.seealso: `PetscDASCreate()`, `PetscDASSetUp()`, `PetscDASDestroy()`
@*/
PetscErrorCode PetscDASReset(PetscDAS das)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  if (das->ops->reset) PetscTryTypeMethod(das, reset);
  if (das->ensemble) {
    for (i = 0; i < das->ensemble_size; i++) PetscCall(VecDestroy(&das->ensemble[i]));
    PetscCall(PetscFree(das->ensemble));
  }
  PetscCall(VecDestroy(&das->observation));
  PetscCall(MatDestroy(&das->obs_operator));
  PetscCall(VecDestroy(&das->obs_error_cov_diag));
  PetscCall(PetscRandomDestroy(&das->random));
  das->setupcalled = PETSC_FALSE;
  das->assimilated = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDASDestroy - Destroys the data assimilation context

  Collective

  Input Parameter:
. das - the `PetscDAS` context

  Level: beginner

.seealso: `PetscDASCreate()`, `PetscDASSetUp()`, `PetscDASReset()`, `PetscDAS`
@*/
PetscErrorCode PetscDASDestroy(PetscDAS *das)
{
  PetscFunctionBegin;
  if (!*das) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*das, PETSCDAS_CLASSID, 1);
  if (--((PetscObject)*das)->refct > 0) {
    *das = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscDASReset(*das));
  PetscTryTypeMethod(*das, destroy);

  PetscCall(PetscHeaderDestroy(das));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDASSetType - Sets the type for the data assimilation method

  Collective

  Input Parameters:
+ das  - the `PetscDAS` context
- type - a known data assimilation method

  Options Database Key:
. -das_type <type> - Sets the type; use -help for a list of available types

  Level: intermediate

.seealso: `PetscDASType`
@*/
PetscErrorCode PetscDASSetType(PetscDAS das, PetscDASType type)
{
  PetscErrorCode (*r)(PetscDAS);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)das, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(PetscDASList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested PetscDAS type %s", type);

  /* Destroy the existing solver information */
  PetscTryTypeMethod(das, destroy);
  das->ops->setup          = NULL;
  das->ops->setfromoptions = NULL;
  das->ops->assimilate     = NULL;
  das->ops->forecast       = NULL;
  das->ops->destroy        = NULL;
  das->ops->reset          = NULL;
  das->ops->view           = NULL;

  /* Call the PetscDASCreate_XXX routine for this particular method */
  das->setupcalled = PETSC_FALSE;
  PetscCall((*r)(das));
  PetscCall(PetscObjectChangeTypeName((PetscObject)das, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASGetType - Gets the current `PetscDASType` being used

  Not Collective

  Input Parameter:
. das - the `PetscDAS` context

  Output Parameter:
. type - the `PetscDASType`

  Level: intermediate

.seealso: `PetscDAS`, `PetscDASType`, `PetscDASSetType()`
@*/
PetscErrorCode PetscDASGetType(PetscDAS das, PetscDASType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)das)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASSetEnsembleSize - Set the ensemble size

  Logically Collective

  Input Parameters:
+ das - the `PetscDAS` context
- m   - the ensemble size

  Level: beginner

.seealso: `PetscDAS`, `PetscDASSetStateSize()`
@*/
PetscErrorCode PetscDASSetEnsembleSize(PetscDAS das, PetscInt m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscValidLogicalCollectiveInt(das, m, 2);
  das->ensemble_size = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASSetStateSize - Set the state vector size

  Logically Collective

  Input Parameters:
+ das - the `PetscDAS` context
- n   - the state vector size

  Level: beginner

.seealso: `PetscDAS`, `PetscDASSetEnsembleSize()`
@*/
PetscErrorCode PetscDASSetStateSize(PetscDAS das, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscValidLogicalCollectiveInt(das, n, 2);
  das->state_size = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASSetObservationOperator - Set the observation operator H

  Collective

  Input Parameters:
+ das - the `PetscDAS` context
- H   - the observation operator matrix

  Level: beginner

.seealso: `PetscDAS`, `PetscDASSetObservationErrorCovariance()`
@*/
PetscErrorCode PetscDASSetObservationOperator(PetscDAS das, Mat H)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)H));
  PetscCall(MatDestroy(&das->obs_operator));
  das->obs_operator = H;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscDASSetObservationErrorCovarianceDiagonal(PetscDAS das, Vec R_diag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscValidHeaderSpecific(R_diag, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)R_diag));
  PetscCall(VecDestroy(&das->obs_error_cov_diag));
  das->obs_error_cov_diag = R_diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASSetEnsemble - Set the ensemble of state vectors

  Collective

  Input Parameters:
+ das      - the `PetscDAS` context
- ensemble - array of state vectors

  Level: beginner

.seealso: `PetscDAS`, `PetscDASGetEnsemble()`, `PetscDASSetEnsembleSize()`
@*/
PetscErrorCode PetscDASSetEnsemble(PetscDAS das, Vec *ensemble)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscAssertPointer(ensemble, 2);
  PetscCheck(das->ensemble_size > 0, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_WRONGSTATE, "Must set ensemble size before setting ensemble");
  if (!das->ensemble) { PetscCall(PetscCalloc1(das->ensemble_size, &das->ensemble)); }
  for (i = 0; i < das->ensemble_size; i++) {
    PetscCall(PetscObjectReference((PetscObject)ensemble[i]));
    PetscCall(VecDestroy(&das->ensemble[i]));
    das->ensemble[i] = ensemble[i];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASGetEnsemble - Get the ensemble of state vectors

  Not Collective

  Input Parameter:
. das - the `PetscDAS` context

  Output Parameter:
. ensemble - array of state vectors

  Level: beginner

.seealso: `PetscDAS`, `PetscDASSetEnsemble()`
@*/
PetscErrorCode PetscDASGetEnsemble(PetscDAS das, Vec **ensemble)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscAssertPointer(ensemble, 2);
  *ensemble = das->ensemble;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASGetEnsembleMean - Get the mean of the ensemble

  Collective

  Input Parameter:
. das - the `PetscDAS` context

  Output Parameter:
. mean - the ensemble mean vector

  Level: beginner

.seealso: `PetscDAS`, `PetscDASGetEnsemble()`
@*/
PetscErrorCode PetscDASGetEnsembleMean(PetscDAS das, Vec *mean)
{
  PetscInt    i;
  PetscScalar scale;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscAssertPointer(mean, 2);
  PetscCheck(das->ensemble, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_WRONGSTATE, "Ensemble not set");
  PetscCheck(das->ensemble_size > 0, PetscObjectComm((PetscObject)das), PETSC_ERR_ARG_WRONGSTATE, "Ensemble size is zero");

  if (!*mean) PetscCall(VecDuplicate(das->ensemble[0], mean));
  PetscCall(VecSet(*mean, 0.0));
  for (i = 0; i < das->ensemble_size; i++) PetscCall(VecAXPY(*mean, 1.0, das->ensemble[i]));
  scale = 1.0 / das->ensemble_size;
  PetscCall(VecScale(*mean, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASSetOptionsPrefix - Sets the prefix used for searching for all
  PetscDAS options in the database.

  Logically Collective

  Input Parameters:
+ das - the `PetscDAS` context
- p   - the prefix string to prepend to all PetscDAS option requests

  Level: advanced

.seealso: `PetscDAS`, `PetscDASSetFromOptions()`, `PetscDASAppendOptionsPrefix()`, `PetscDASGetOptionsPrefix()`
@*/
PetscErrorCode PetscDASSetOptionsPrefix(PetscDAS das, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASAppendOptionsPrefix - Appends to the prefix used for searching for all PetscDAS options in the database.

  Logically Collective

  Input Parameters:
+ das - the `PetscDAS` context
- p   - the prefix string to prepend to all `PetscDAS` option requests

  Level: advanced

.seealso: `PetscDAS`, `PetscDASSetFromOptions()`, `PetscDASSetOptionsPrefix()`, `PetscDASGetOptionsPrefix()`
@*/
PetscErrorCode PetscDASAppendOptionsPrefix(PetscDAS das, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDASGetOptionsPrefix - Gets the prefix used for searching for all
  PetscDAS options in the database

  Not Collective

  Input Parameter:
. das - the `PetscDAS` context

  Output Parameter:
. p - pointer to the prefix string used

  Level: advanced

.seealso: `PetscDAS`, `PetscDASSetFromOptions()`, `PetscDASSetOptionsPrefix()`, `PetscDASAppendOptionsPrefix()`
@*/
PetscErrorCode PetscDASGetOptionsPrefix(PetscDAS das, const char *p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(das, PETSCDAS_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)das, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  VecSetGaussianRandom - Fill a vector with Gaussian random numbers

  Collective

  Input Parameters:
+ v      - the vector to fill
. rctx   - PetscRandom context
. mean   - mean of the Gaussian distribution
- stddev - standard deviation of the Gaussian distribution

  Level: intermediate

  Notes:
  Uses the Box-Muller transform to convert uniform random numbers to Gaussian.

.seealso: `Vec`, `PetscRandom`, `PetscRandomCreate()`
@*/
PetscErrorCode VecSetGaussianRandom(Vec v, PetscRandom rctx, PetscReal mean, PetscReal stddev)
{
  PetscInt     n, i;
  PetscScalar *array;
  PetscReal    u1, u2, z1, z2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(rctx, PETSC_RANDOM_CLASSID, 2);

  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArray(v, &array));

  for (i = 0; i < n; i += 2) {
    PetscCall(PetscRandomGetValueReal(rctx, &u1));
    PetscCall(PetscRandomGetValueReal(rctx, &u2));

    /* Box-Muller transform */
    z1       = PetscSqrtReal(-2.0 * PetscLogReal(u1)) * PetscCosReal(2.0 * PETSC_PI * u2);
    array[i] = mean + stddev * z1;

    if (i + 1 < n) {
      z2           = PetscSqrtReal(-2.0 * PetscLogReal(u1)) * PetscSinReal(2.0 * PETSC_PI * u2);
      array[i + 1] = mean + stddev * z2;
    }
  }

  PetscCall(VecRestoreArray(v, &array));
  PetscFunctionReturn(PETSC_SUCCESS);
}
