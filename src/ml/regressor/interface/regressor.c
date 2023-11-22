#include <petsc/private/regressorimpl.h>

PetscBool         PetscRegressorRegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscRegressorList              = NULL;

PetscClassId PETSCREGRESSOR_CLASSID;

/* Logging support */
PetscLogEvent PetscRegressor_SetUp, PetscRegressor_Fit, PetscRegressor_Predict;

/*@C
   PetscRegressorRegister - Adds a method to the PetscRegressor package.

   Not collective

   Input Parameters:
+  sname - name of a new user-defined regressor
-  function - routine to create method context

   Notes:
   PetscRegressorRegister() may be called multiple times to add several user-defined regressors.

   Sample usage:
.vb
   PetscRegressorRegister("my_regressor",MyRegressorCreate);
.ve

   Then, your regressor can be chosen with the procedural interface via
$     PetscRegressorSetType(regressor,"my_regressor")
   or at runtime via the option
$     -regressor_type my_regressor

   Level: advanced

.seealso: PetscRegressorRegisterAll()
@*/
PetscErrorCode PetscRegressorRegister(const char sname[], PetscErrorCode (*function)(PetscRegressor))
{
  PetscFunctionBegin;
  PetscCall(PetscRegressorInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscRegressorList, sname, function));
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorCreate - Creates a regressor object.

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newregressor - the new regressor object

   Level: beginner

.seealso: PetscRegressorFit(), PetscRegressorPredict(), PetscRegressor
@*/
PetscErrorCode PetscRegressorCreate(MPI_Comm comm, PetscRegressor *newregressor)
{
  PetscRegressor regressor;

  PetscFunctionBegin;
  PetscAssertPointer(newregressor, 2);
  *newregressor = NULL;
  PetscCall(PetscRegressorInitializePackage());

  PetscCall(PetscHeaderCreate(regressor, PETSCREGRESSOR_CLASSID, "PetscRegressor", "Regressor", "PetscRegressor", comm, PetscRegressorDestroy, PetscRegressorView));

  // TODO: Finish setting the various fields of the PetscRegressor private data structure to defaults, etc.
  regressor->setupcalled = PETSC_FALSE;
  regressor->data        = NULL;
  regressor->training    = NULL;
  regressor->target      = NULL;
  regressor->regularizer_weight = 1.0;  // We go ahead and set a default weight here, but some regressor types will have a different default!
  regressor->regularizer_weight_is_set = PETSC_FALSE;

  *newregressor = regressor;
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorSetFromOptions - Sets PetscRegressor options from the options database.

   Collective on PetscRegressor

   Input Parameter:
.  regressor - the PetscRegressor context

   Options Database Keys:
.  -regressor_type <type> - the particular type of regressor to be used; see PetscRegressorType for complete list

   This routine must be called before PetscRegressorSetUp() (or PetscRegressorFit(), which calls
   the former) if the user is to be allowed to set the regressor type.

   Level: beginner

.seealso: PetscRegressor, PetscRegressorCreate()
@*/
PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor regressor)
{
  PetscBool          flg;
  PetscRegressorType default_type = PETSCREGRESSORLINEAR;
  char               type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)regressor);
  if (((PetscObject)regressor)->type_name) { default_type = ((PetscObject)regressor)->type_name; }
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-regressor_type", "PetscRegressor type", "PetscRegressorSetType", PetscRegressorList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(PetscRegressorSetType(regressor, type));
  } else if (!((PetscObject)regressor)->type_name) {
    PetscCall(PetscRegressorSetType(regressor, default_type));
  }
  PetscCall(PetscOptionsReal("-regressor_regularizer_weight", "Weight for the regularizer", "PetscRegressorSetRegularizerWeight", regressor->regularizer_weight, &(regressor->regularizer_weight), &flg));
  if (flg) PetscCall(PetscRegressorSetRegularizerWeight(regressor, regressor->regularizer_weight));
    // The above is a little superfluous, because we have already set regressor->regularizer_weight above, but we also need to set the flag indicating that the user has set the weight!
  /* TODO: Is there code that must be added to handle other options that apply to all PetscRegressor types? */
  if (regressor->ops->setfromoptions) { PetscCall((*regressor->ops->setfromoptions)(PetscOptionsObject, regressor)); }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorSetUp - Sets up the internal data structures for the later use of a regressor.

   Collective on PetscRegressor

   Input Parameters:
.  regressor - the PetscRegressor context

   Notes:
   For basic use of the PetscRegressor solvers the user need not to explicitly call
   PetscRegressorSetUp(), since these actions will automatically occur during
   the call to PetscRegressorFit().  However, if one wishes to control this
   phase separately, PetscRegressorSetUp() should be called after PetscRegressorCreate(),
   PetscRegressorSetUp(), and optional routines of the form PetscRegressorSetXXX(),
   but before PetscRegressorFit().

   Level: advanced

.seealso: PetscRegressorCreate(), PetscRegressorFit(), PetscRegressorDestroy()
@*/
PetscErrorCode PetscRegressorSetUp(PetscRegressor regressor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (regressor->setupcalled) PetscFunctionReturn(0);
  PetscCall(PetscLogEventBegin(PetscRegressor_SetUp, regressor, 0, 0, 0));

  if (!((PetscObject)regressor)->type_name) { PetscCall(PetscRegressorSetType(regressor, PETSCREGRESSORLINEAR)); }

  if (regressor->ops->setup) { PetscCall((*regressor->ops->setup)(regressor)); }

  PetscCall(PetscLogEventEnd(PetscRegressor_SetUp, regressor, 0, 0, 0));
  regressor->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* NOTE: I've decided to make this take X and y, like the Scikit-learn Fit routines do.
 * Am I overlooking some reason that X should be set in a separate function call, a la KSPSetOperators()?. */
/*@
   PetscRegressorFit - Fit, or train, a regressor from a training dataset

   Collective on PetscRegressor

   Input Parameters:
+  regressor - the regressor context
.  X - matrix of training data (of dimension [number of samples] x [number of features])
-  y - vector of target values from the training dataset

   Level: beginner

.seealso: PetscRegressorCreate(), PetscRegressorSetUp(), PetscRegressorDestroy(), PetscRegressorPredict()
@*/
PetscErrorCode PetscRegressorFit(PetscRegressor regressor, Mat X, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (X) PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  if (y) PetscValidHeaderSpecific(y, VEC_CLASSID, 3);

  if (X) {
    PetscCall(PetscObjectReference((PetscObject)X));
    PetscCall(MatDestroy(&regressor->training));
    regressor->training = X;
  }
  if (y) {
    PetscCall(PetscObjectReference((PetscObject)y));
    PetscCall(VecDestroy(&regressor->target));
    regressor->target = y;
  }
  PetscCall(PetscRegressorSetUp(regressor));

  PetscCall(PetscLogEventBegin(PetscRegressor_Fit, regressor, X, y, 0));
  PetscCall((*regressor->ops->fit)(regressor));
  PetscCall(PetscLogEventEnd(PetscRegressor_Fit, regressor, X, y, 0));
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorPredict - Compute predictions (that is, perform inference) using a fitted regression model.

   Collective on PetscRegressorPredict

   Input Parameters:
+  regressor - the regressor context (for which PetscRegressorFit() must have been called)
.  X - data matrix of unlabeled observations
-  y - vector of predicted labels

   Level: beginner

.seealso: PetscRegressorFit(), PetscRegressorDestroy()
@*/
PetscErrorCode PetscRegressorPredict(PetscRegressor regressor, Mat X, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (X) PetscValidHeaderSpecific(X, MAT_CLASSID, 2);
  if (y) PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  // TODO: Add code to check that the Fit step has been completed!
  PetscCall(PetscLogEventBegin(PetscRegressor_Predict, regressor, X, y, 0));
  PetscCall((*regressor->ops->predict)(regressor, X, y));
  PetscCall(PetscLogEventEnd(PetscRegressor_Predict, regressor, X, y, 0));
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorReset - Resets a PetscRegressor context to the setupcalled = 0 state and removes any allocated Vecs and Mats

   Collective on PetscRegressor

   Input Parameter:
.  regressor - context obtained from PetscRegressorCreate()

   Level: intermediate

.seealso: PetscRegressorCreate(), PetscRegressorSetUp(), PetscRegressorFit(), PetscRegressorPredict(), PetscRegressorDestroy()
@*/
PetscErrorCode PetscRegressorReset(PetscRegressor regressor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  if (regressor->ops->reset) { PetscCall((*regressor->ops->reset)(regressor)); }
  PetscCall(MatDestroy(&regressor->training));
  PetscCall(VecDestroy(&regressor->target));
  PetscCall(TaoDestroy(&regressor->tao));
  regressor->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PetscRegressorDestroy - Destroys the regressor context that was created with PetscRegressorCreate().

   Collective on PetscRegressor

   Input Parameter:
.  regressor - the PetscRegressor context

   Level: beginner

.seealso: PetscRegressorCreate(), PetscRegressorSetUp(), PetscRegressorReset(), PetscRegressor
@*/
PetscErrorCode PetscRegressorDestroy(PetscRegressor *regressor)
{
  PetscFunctionBegin;
  if (!*regressor) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*regressor), PETSCREGRESSOR_CLASSID, 1);
  if (--((PetscObject)(*regressor))->refct > 0) {
    *regressor = NULL;
    PetscFunctionReturn(0);
  }

  PetscCall(PetscRegressorReset((*regressor)));
  if ((*regressor)->ops->destroy) PetscCall((*(*regressor)->ops->destroy)(*regressor));

  PetscCall(PetscHeaderDestroy(regressor));
  PetscFunctionReturn(0);
}

/*@C
   PetscRegressorSetType - Sets the type for the regressor.

   Collective on PetscRegressor

   Input Parameters:
+  regressor - the PetscRegressor context
-  type - a known regression method

   Options Database Key:
.  -regressor_type <type> - Sets the type of regressor; use -help for a list of available types

   Notes:
   See "include/petscregressor.h" for available methods (for instance)
.    PETSCREGRESSORLINEAR - Linear regression models (ordinary least squares as well as regularized variants)

   Normally, it is best to use the PetscRegressorSetFromOptions() command and then
   set the PetscRegressor type from the options database rather than by using
   this routine, as this provides maximum flexibility.
   The PetscRegressorSetType() routine is provided for those situations where it
   is necessary to set the nonlinear solver independently of the command
   line or options database.

   Level: intermediate

.seealso: PetscRegressorType
@*/
PetscErrorCode PetscRegressorSetType(PetscRegressor regressor, PetscRegressorType type)
{
  PetscErrorCode (*r)(PetscRegressor);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)regressor, type, &match));
  if (match) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(PetscRegressorList, type, &r));
  if (!r) SETERRQ(PetscObjectComm((PetscObject)regressor), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested PetscRegressor type %s", type);

  /* Destroy the previous private PetscRegressor context */
  if (regressor->ops->destroy) {
    PetscCall((*(regressor)->ops->destroy)(regressor));
    regressor->ops->destroy = NULL;
  }

  /* Reinitialize function pointers in PetscRegressorOps structure */
  regressor->ops->setup          = NULL;
  regressor->ops->setfromoptions = NULL;
  regressor->ops->settraining    = NULL;
  regressor->ops->fit            = NULL;
  regressor->ops->predict        = NULL;
  regressor->ops->destroy        = NULL;
  regressor->ops->reset          = NULL;
  regressor->ops->view           = NULL;

  /* Call the PetscRegressorCreate_XXX routine for this particular regressor */
  regressor->setupcalled = PETSC_FALSE;
  PetscCall((*r)(regressor));
  PetscCall(PetscObjectChangeTypeName((PetscObject)regressor, type));
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorSetRegularizerWeight - Sets the weight to be used for the regularizer for a PetscRegressor context

   Logically Collective

   Input Parameters:
+  regressor - the `PetscRegressor` context
-  weight - the regularizer weight

   Level: beginner

.seealso: `PetscRegressorSetType`
@*/
PetscErrorCode PetscRegressorSetRegularizerWeight(PetscRegressor regressor, PetscReal weight)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  regressor->regularizer_weight = weight;
  regressor->regularizer_weight_is_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscRegressorView(PetscRegressor regressor, PetscViewer viewer)
{
  PetscFunctionBegin;
  // TODO: Complete this when I have a good idea of what bits of the PetscRegressor should be shown!
  PetscFunctionReturn(0);
}

/*@
   PetscRegressorGetTao - Returns the Tao context for a PetscRegressor object.

   Not Collective, but if the PetscRegressor is parallel, then the Tao object is parallel

   Input Parameter:
.  regressor - the regressor context

   Output Parameter:
.  tao - the `Tao` context

   Notes:

   The `Tao` object will be created if it does not yet exist.

   The user can directly manipulate the `TAO` context to set various
   options, etc.  Likewise, the user can then extract and manipulate the
   child contexts such as `KSP` or `TaoLineSearch`as well.

   Depending on the type of the regressor and the options that are set, the regressor may use not use a Tao object.

   Level: beginner

.seealso: PetscRegressorLinearGetKSP()
@*/
PetscErrorCode PetscRegressorGetTao(PetscRegressor regressor, Tao *tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(regressor, PETSCREGRESSOR_CLASSID, 1);
  PetscAssertPointer(tao, 2);
  /* Analogous to how SNESGetKSP() operates, this routine should create the TAO if it doesn't exist.
   * TODO: Follow what SNESGetKSP() does when setting this up. */
  if (!regressor->tao) {
    PetscCall(TaoCreate(PetscObjectComm((PetscObject)regressor), &regressor->tao));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)regressor->tao, (PetscObject)regressor, 1));
    PetscCall(PetscObjectSetOptions((PetscObject)regressor->tao, ((PetscObject)regressor)->options));
  }
  *tao = regressor->tao;
  PetscFunctionReturn(0);
}
