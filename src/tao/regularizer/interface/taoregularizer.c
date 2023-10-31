#include <petsctaoregularizer.h> /*I "petsctaoregularizer.h" I*/
#include <petsc/private/taoregularizerimpl.h>

PetscFunctionList TaoRegularizerList = NULL;

PetscClassId TAOREGULARIZER_CLASSID = 0;

PetscLogEvent TAOREGULARIZER_Apply;
PetscLogEvent TAOREGULARIZER_Eval;

/*@C
   TaoRegularizerViewFromOptions - View a `TaoRegularizer` object based on values in the options database

   Collective

   Input Parameters:
+  A - the `Tao` context
.  obj - Optional object
-  name - command line option

   Level: intermediate

   Note:
   See `PetscObjectViewFromOptions()` for available viewer options

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerView()`, `PetscObjectViewFromOptions()`, `TaoRegularizerCreate()`
@*/
PetscErrorCode TaoRegularizerViewFromOptions(TaoRegularizer A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, TAOREGULARIZER_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegularizerView - Prints information about the `TaoRegularizer`

  Collective

  InputParameters:
+ reg - the `TaoRegularizer` context
- viewer - visualization context

  Options Database Key:
. -tao_reg_view - Calls `TaoRegularizerView()` at the end of each line search

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `PetscViewerASCIIOpen()`, `TaoRegularizerViewFromOptions()`
@*/
PetscErrorCode TaoRegularizerView(TaoRegularizer reg, PetscViewer viewer)
{
  PetscBool          isascii, isstring;
  TaoRegularizerType type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(((PetscObject)reg)->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(reg, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)reg, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(reg, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    /* TODO viewer stuff */
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(TaoRegularizerGetType(reg, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " %-3.3s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegularizerCreate - Creates a `TaoRegularizer` object.

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. newreg - the new `TaoRegularizer` context

   Options Database Key:
.   -tao_reg_type - select which method `Tao` should use

   Level: developer

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerType`, `TaoRegularizerSetType()`, `TaoRegularizerApply()`, `TaoRegularizerDestroy()`
@*/
PetscErrorCode TaoRegularizerCreate(MPI_Comm comm, TaoRegularizer *newreg)
{
  TaoRegularizer reg;

  PetscFunctionBegin;
  PetscAssertPointer(newreg, 2);
  PetscCall(TaoRegularizerInitializePackage());

  PetscCall(PetscHeaderCreate(reg, TAOREGULARIZER_CLASSID, "TaoRegularizer", "Regularizer", "Tao", comm, TaoRegularizerDestroy, TaoRegularizerView));
  reg->scale = 1.0;
  *newreg    = reg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerSetUp - Sets up the internal data structures for the later use
  of a `TaoRegularizer`

  Collective

  Input Parameter:
. reg - the `TaoRegularizer` context

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerCreate()`, `TaoRegularizerApply()`
@*/
PetscErrorCode TaoRegularizerSetUp(TaoRegularizer reg)
{
  const char *default_type = TAOREGULARIZERL2;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  if (reg->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (!((PetscObject)reg)->type_name) PetscCall(TaoRegularizerSetType(reg, default_type));
  PetscTryTypeMethod(reg, setup);
  if (reg->usetaoroutines) {
    PetscCall(TaoIsObjectiveDefined(reg->reg_tao, &flg));
    reg->hasobjective = flg;
    PetscCall(TaoIsGradientDefined(reg->reg_tao, &flg));
    reg->hasgradient = flg;
    PetscCall(TaoIsObjectiveAndGradientDefined(reg->reg_tao, &flg));
    reg->hasobjectiveandgradient = flg;
    /* TODO Hessian */
  } else {
    if (reg->ops->computeobjective) {
      reg->hasobjective = PETSC_TRUE;
    } else {
      reg->hasobjective = PETSC_FALSE;
    }
    if (reg->ops->computegradient) {
      reg->hasgradient = PETSC_TRUE;
    } else {
      reg->hasgradient = PETSC_FALSE;
    }
    if (reg->ops->computeobjectiveandgradient) {
      reg->hasobjectiveandgradient = PETSC_TRUE;
    } else {
      reg->hasobjectiveandgradient = PETSC_FALSE;
    }
  }
  reg->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerDestroy - Destroys the `TaoRegularizer` context that was created with
  `TaoRegularizerCreate()`

  Collective

  Input Parameter:
. reg  - the `TaoRegularizer` context

  Level: developer

.seealse: `TaoRegularizerCreate()`
@*/
PetscErrorCode TaoRegularizerDestroy(TaoRegularizer *reg)
{
  PetscFunctionBegin;
  if (!*reg) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*reg, TAOREGULARIZER_CLASSID, 1);
  if (--((PetscObject)*reg)->refct > 0) {
    *reg = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*reg)->ops->destroy) PetscCall((*(*reg)->ops->destroy)(*reg));
  if ((*reg)->usemonitor) PetscCall(PetscViewerDestroy(&(*reg)->viewer));
  PetscCall(VecDestroy(&(*reg)->workvec));
  PetscCall(PetscHeaderDestroy(reg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoRegularizerSetType - Sets the algorithm used in a line search

   Collective

   Input Parameters:
+  reg  - the `TaoRegularizer` context
-  type - the `TaoRegularizerType` selection

  Options Database Key:
.  -tao_reg_type <type> - select which method Tao should use at runtime

  Level: beginner

.seearego: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerType`, `TaoRegularizerCreate()`, `TaoRegularizerGetType()`,
@*/
PetscErrorCode TaoRegularizerSetType(TaoRegularizer reg, TaoRegularizerType type)
{
  PetscErrorCode (*r)(TaoRegularizer);
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)reg, type, &flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TaoRegularizerList, type, (void (**)(void)) & r));
  PetscCheck(r, PetscObjectComm((PetscObject)reg), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested TaoRegularizer type %s", type);
  PetscTryTypeMethod(reg, destroy);
  reg->scale = 1.0;

  reg->ops->setup          = NULL;
  reg->ops->apply          = NULL;
  reg->ops->view           = NULL;
  reg->ops->setfromoptions = NULL;
  reg->ops->destroy        = NULL;
  reg->setupcalled         = PETSC_FALSE;
  PetscCall((*r)(reg));
  PetscCall(PetscObjectChangeTypeName((PetscObject)reg, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerSetFromOptions - Sets various `TaoRegularizer` parameters from user
  options.

  Collective

  Input Parameter:
. reg - the `TaoRegularizer` context

  Options Database Keys:
+ -tao_reg_type <type> - The type of `TaoRegularizer` (L1,L2,KL,USER)
. -tao_reg_scale <scale> - scalar scale for regularizer
- -tao_reg_view - display line-search results to standard output

  Level: beginner

.seealso: `TaoRegularizer`
@*/
PetscErrorCode TaoRegularizerSetFromOptions(TaoRegularizer reg)
{
  const char *default_type = TAOREGULARIZERL2; //TODO ?? do i need this?
  char        type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer monviewer;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)reg);
  if (((PetscObject)reg)->type_name) default_type = ((PetscObject)reg)->type_name;
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-tao_reg_type", "Tao Line Search type", "TaoRegularizerSetType", TaoRegularizerList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(TaoRegularizerSetType(reg, type));
  } else if (!((PetscObject)reg)->type_name) {
    PetscCall(TaoRegularizerSetType(reg, default_type));
  }

  PetscCall(PetscOptionsReal("-tao_reg_scale", "scale", "", reg->scale, &reg->scale, NULL));
  PetscCall(PetscOptionsString("-tao_reg_monitor", "enable the basic monitor", "TaoRegularizerSetMonitor", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)reg), monfilename, &monviewer));
    reg->viewer     = monviewer;
    reg->usemonitor = PETSC_TRUE;
  }
  PetscTryTypeMethod(reg, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegularizerGetType - Gets the current regularizer algorithm

  Not Collective

  Input Parameter:
. reg - the `TaoRegularizer` context

  Output Parameter:
. type - the line search algorithm in effect

  Level: developer

.seearego: `TaoRegularizer`
@*/
PetscErrorCode TaoRegularizerGetType(TaoRegularizer reg, TaoRegularizerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)reg)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerIsUsingTaoRoutines - Checks whether the regularizer is using
  the standard `Tao` evaluation routines.

  Not Collective

  Input Parameter:
. reg - the `TaoRegularizer` context

  Output Parameter:
. flg - `PETSC_TRUE` if the regularizer is using `Tao` evaluation routines,
        otherwise `PETSC_FALSE`

  Level: developer

.seealso: `TaoRegularizer`
@*/
PetscErrorCode TaoRegularizerIsUsingTaoRoutines(TaoRegularizer reg, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  *flg = reg->usetaoroutines;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegularizerUseTaoRoutines - Informs the `TaoRegularizer` to use the
  objective and gradient evaluation routines from the given `Tao` object. The default.

  Logically Collective

  Input Parameters:
+ reg - the `TaoRegularizer` context
- ts  - the `Tao` context with defined objective/gradient evaluation routines

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerCreate()`
@*/
PetscErrorCode TaoRegularizerUseTaoRoutines(TaoRegularizer reg, Tao ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TAO_CLASSID, 2);
  reg->reg_tao        = ts;
  reg->usetaoroutines = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerComputeObjective - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ reg - the `TaoRegularizer` context
- x - input vector

  Output Parameter:
. f - Objective value at `x`

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerComputeGradient()`, `TaoRegularizerComputeObjectiveAndGradient()`, `TaoRegularizerSetObjectiveRoutine()`
@*/
PetscErrorCode TaoRegularizerComputeObjective(TaoRegularizer reg, Vec x, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscCheckSameComm(reg, 1, x, 2);
  if (reg->usetaoroutines) {
    PetscCall(TaoComputeObjective(reg->reg_tao, x, f));
  } else {
    PetscCheck(reg->ops->computeobjective || reg->ops->computeobjectiveandgradient, PetscObjectComm((PetscObject)reg), PETSC_ERR_ARG_WRONGSTATE, "Regularizer does not have objective function set");
    PetscCall(PetscLogEventBegin(TAOREGULARIZER_Eval, reg, 0, 0, 0));
    if (reg->ops->computeobjective) PetscCallBack("TaoRegularizer callback objective", (*reg->ops->computeobjective)(reg, x, f, reg->userctx_func));
    else {
      /* Unlike Linesearch, we have workvec inside TaoReg, so no need to create dummy vector */
      PetscCallBack("TaoRegularizer callback objective", (*reg->ops->computeobjectiveandgradient)(reg, x, f, reg->workvec, reg->userctx_funcgrad));
    }
    PetscCall(PetscLogEventEnd(TAOREGULARIZER_Eval, reg, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ reg - the `TaoRegularizer` context
- x   - input vector

  Output Parameters:
+ f - Objective value at `x`
- g - Gradient vector at `x`

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerComputeGradient()`, `TaoRegularizerComputeObjectiveAndGradient()`, `TaoRegularizerSetObjectiveRoutine()`
@*/
PetscErrorCode TaoRegularizerComputeObjectiveAndGradient(TaoRegularizer reg, Vec x, PetscReal *f, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscCheckSameComm(reg, 1, x, 2);
  PetscCheckSameComm(reg, 1, g, 4);
  if (reg->usetaoroutines) {
    PetscCall(TaoComputeObjectiveAndGradient(reg->reg_tao, x, f, g));
  } else {
    PetscCall(PetscLogEventBegin(TAOREGULARIZER_Eval, reg, 0, 0, 0));
    if (reg->ops->computeobjectiveandgradient) PetscCallBack("TaoRegularizer callback objective/gradient", (*reg->ops->computeobjectiveandgradient)(reg, x, f, g, reg->userctx_funcgrad));
    else {
      PetscCallBack("TaoRegularizer callback objective", (*reg->ops->computeobjective)(reg, x, f, reg->userctx_func));
      PetscCallBack("TaoRegularizer callback gradient", (*reg->ops->computegradient)(reg, x, g, reg->userctx_grad));
    }
    PetscCall(PetscLogEventEnd(TAOREGULARIZER_Eval, reg, 0, 0, 0));
    PetscCall(PetscInfo(reg, "TaoRegularizer Function evaluation: %14.12e\n", (double)(*f)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerComputeGradient - Computes the gradient of the objective function

  Collective

  Input Parameters:
+ reg - the `TaoRegularizer` context
- x - input vector

  Output Parameter:
. g - gradient vector

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerComputeObjective()`, `TaoRegularizerComputeObjectiveAndGradient()`, `TaoRegularizerSetGradient()`
@*/
PetscErrorCode TaoRegularizerComputeGradient(TaoRegularizer reg, Vec x, Vec g)
{
  PetscReal fdummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCheckSameComm(reg, 1, x, 2);
  PetscCheckSameComm(reg, 1, g, 3);
  if (reg->usetaoroutines) {
    PetscCall(TaoComputeGradient(reg->reg_tao, x, g));
  } else {
    PetscCall(PetscLogEventBegin(TAOREGULARIZER_Eval, reg, 0, 0, 0));
    if (reg->ops->computegradient) PetscCallBack("TaoRegularizer callback gradient", (*reg->ops->computegradient)(reg, x, g, reg->userctx_grad));
    else PetscCallBack("TaoRegularizer callback gradient", (*reg->ops->computeobjectiveandgradient)(reg, x, &fdummy, g, reg->userctx_funcgrad));
    PetscCall(PetscLogEventEnd(TAOREGULARIZER_Eval, reg, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerSetCentralVector - Sets a the central vector of regularizer, for metric.

  Logically Collective

  Input Parameter:
+ reg - the `TaoRegularizer` context
- y   - The initial point of the metric

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerGetCentralVector()`
@*/
PetscErrorCode TaoRegularizerSetCentralVector(TaoRegularizer reg, Vec y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)y));
  PetscCall(VecDestroy(&reg->y));
  reg->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerGetCentralVector - Gets a the central vector of regularizer, for metric.

  Logically Collective

  Input Parameter:
. reg - the `TaoRegularizer` context

  Output Parameter:
. y - The central vector of the regularizer

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerSetCentralVector()`
@*/
PetscErrorCode TaoRegularizerGetCentralVector(TaoRegularizer reg, Vec *y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  if (y) *y = reg->y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerSetScale - Sets scale for the regularizer.
  If this value is not set then 1.0 is assumed.

  Logically Collective

  Input Parameters:
+ reg - the `TaoRegularizer` context
- s   - the scale

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerGetStepLength()`, `TaoRegularizerApply()`
@*/
PetscErrorCode TaoRegularizerSetScale(TaoRegularizer reg, PetscReal s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  PetscValidLogicalCollectiveReal(reg, s, 2);
  reg->scale = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoRegularizerGetScale - Get scale of regularizer.

  Not Collective

  Input Parameter:
. reg - the `TaoRegularizer` context

  Output Parameter:
. s - the current scale

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerApply()`
@*/
PetscErrorCode TaoRegularizerGetScale(TaoRegularizer reg, PetscReal *s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  *s = reg->scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoRegularizerRegister - Adds a regularizer type to the registry

   Not Collective

   Input Parameters:
+  sname - name of a new user-defined regularizer
-  func  - routine to Create method context

   Sample usage:
.vb
   TaoRegularizerRegister("my_regularizer", MyRegularizerCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     TaoRegularizerSetType(reg, "my_regularizer")
   or at runtime via the option
$     -tao_reg_type my_regularizer

   Level: developer

   Note:
   `TaoRegularizerRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`
@*/
PetscErrorCode TaoRegularizerRegister(const char sname[], PetscErrorCode (*func)(TaoRegularizer))
{
  PetscFunctionBegin;
  PetscCall(TaoRegularizerInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoRegularizerList, sname, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoRegularizerAppendOptionsPrefix - Appends to the prefix used for searching
   for all `TaoRegularizer` options in the database.

   Collective

   Input Parameters:
+  ls - the `TaoRegularizer` solver context
-  prefix - the prefix string to prepend to all line search requests

   Level: advanced

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   This is inherited from the `Tao` object so rarely needs to be set

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerSetOptionsPrefix()`, `TaoRegularizerGetOptionsPrefix()`
@*/
PetscErrorCode TaoRegularizerAppendOptionsPrefix(TaoRegularizer reg, const char p[])
{
  return PetscObjectAppendOptionsPrefix((PetscObject)reg, p);
}

/*@C
  TaoRegularizerGetOptionsPrefix - Gets the prefix used for searching for all
  `TaoRegularizer` options in the database

  Not Collective

  Input Parameter:
. reg - the `TaoRegularizer` context

  Output Parameter:
. prefix - pointer to the prefix string used is returned

  Level: advanced

  Fortran Note:
  The user should pass in a string 'prefix' of
  sufficient length to hold the prefix.

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerSetOptionsPrefix()`, `TaoRegularizerAppendOptionsPrefix()`
@*/
PetscErrorCode TaoRegularizerGetOptionsPrefix(TaoRegularizer reg, const char *p[])
{
  return PetscObjectGetOptionsPrefix((PetscObject)reg, p);
}

/*@C
   TaoRegularizerSetOptionsPrefix - Sets the prefix used for searching for all
   `TaoRegularizer` options in the database.

   Logically Collective

   Input Parameters:
+  reg - the `TaoRegularizer` context
-  prefix - the prefix string to prepend to all `reg` option requests

   Level: advanced

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   This is inherited from the `Tao` object so rarely needs to be set

   For example, to distinguish between the runtime options for two
   different line searches, one could call
.vb
      TaoRegularizerSetOptionsPrefix(reg1,"sys1_")
      TaoRegularizerSetOptionsPrefix(reg2,"sys2_")
.ve

   This would enable use of different options for each system, such as
.vb
      -sys1_tao_reg_type regl1
      -sys2_tao_reg_type regl2
.ve

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerAppendOptionsPrefix()`, `TaoRegularizerGetOptionsPrefix()`
@*/
PetscErrorCode TaoRegularizerSetOptionsPrefix(TaoRegularizer reg, const char p[])
{
  return PetscObjectSetOptionsPrefix((PetscObject)reg, p);
}

/*@
   TaoSetRegularizer - Sets an TaoRegularizer to Tao object.
                       This routine needs to be set after
                       TaoSetSolution() has been called.

   Input Parameters:
+  tao - Tao solver context
-  reg - TaoRegularizer context

   Level: advanced

.seealso: `TaoGetRegularizer()`
@*/
PetscErrorCode TaoSetRegularizer(Tao tao, TaoRegularizer reg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (reg) PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)reg));
  PetscCall(TaoRegularizerDestroy(&tao->reg));
  tao->reg = reg;
  PetscCheck(tao->solution, PetscObjectComm((PetscObject)reg), PETSC_ERR_USER, "TaoSetSolution needs to be called first.");
  if (!reg->workvec) { PetscCall(VecDuplicate(tao->solution, &reg->workvec)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TaoGetRegularizer - Gets an TaoRegularizer to Tao object.

   Input Parameters:
.  tao - Tao solver context

   Output Parameter:
.  reg - TaoRegularizer context

   Level: advanced

.seealso: `TaoSetRegularizer()`
@*/
PetscErrorCode TaoGetRegularizer(Tao tao, TaoRegularizer *reg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(reg, 2);
  *reg = tao->reg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegularizerSetObjective- Sets the function evaluation routine for the regularizer

  Logically Collective

  Input Parameters:
+ reg  - the `TaoRegularizer` context
. func - the objective function evaluation routine
- ctx  - the (optional) user-defined context for private data

  Calling sequence of `func`:
+ reg - the regularizer context
. x   - input vector
. f   - function value
- ctx - (optional) user-defined context

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerCreate()`, `TaoRegularizerSetGradient()`, `TaoRegularizerSetObjectiveAndGradient()`, `TaoRegularizerUseTaoRoutines()`
@*/
PetscErrorCode TaoRegularizerSetObjective(TaoRegularizer reg, PetscErrorCode (*func)(TaoRegularizer reg, Vec x, PetscReal *f, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);

  reg->ops->computeobjective = func;
  if (ctx) reg->userctx_func = ctx;
  reg->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegularizerSetGradient - Sets the gradient evaluation routine for the regularizer

  Logically Collective

  Input Parameters:
+ reg  - the `TaoRegularizer` context
. func - the gradient evaluation routine
- ctx  - the (optional) user-defined context for private data

  Calling sequence of `func`:
+ reg - the regularizer object
. x   - input vector
. g   - gradient vector
- ctx - (optional) user-defined context

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoRegularizer`, `TaoRegularizerCreate()`, `TaoRegularizerSetObjective()`, `TaoRegularizerSetObjectiveAndGradient()`, `TaoRegularizerUseTaoRoutines()`
@*/
PetscErrorCode TaoRegularizerSetGradient(TaoRegularizer reg, PetscErrorCode (*func)(TaoRegularizer reg, Vec x, Vec g, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  reg->ops->computegradient = func;
  if (ctx) reg->userctx_grad = ctx;
  reg->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegularizerSetObjectiveAndGradient - Sets a combined objective function and gradient evaluation routine for the function to be optimized for TaoRegularizer

  Logically Collective

  Input Parameters:
+ reg - the `TaoRegularizer` context
. func - the gradient function
- ctx  - [optional] user-defined context for private data for the gradient evaluation
        routine (may be `NULL`)

  Calling sequence of `func`:
+ regg - the regularizer object
. x   - input vector
. f   - objective value (output)
. g   - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  For some optimization methods using a combined function can be more eifficient.

.seealso: [](ch_tao), `TaoRegularizer`, `TaoRegularizerSetObjective()`, `TaoRegularizerSetHessian()`, `TaoRegularizerSetGradient()`, `TaoGetObjectiveAndGradient()`
@*/
PetscErrorCode TaoRegularizerSetObjectiveAndGradient(TaoRegularizer reg, PetscErrorCode (*func)(TaoRegularizer reg, Vec x, PetscReal *f, Vec g, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);
  reg->ops->computeobjectiveandgradient = func;
  if (ctx) reg->userctx_grad = ctx;
  reg->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TODO Hessian */
