#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

PetscClassId TSADAPT_CLASSID;

static PetscFunctionList TSAdaptList;
static PetscBool         TSAdaptPackageInitialized;
static PetscBool         TSAdaptRegisterAllCalled;

PETSC_EXTERN PetscErrorCode TSAdaptCreate_None(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_Basic(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_DSP(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_CFL(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_GLEE(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_History(TSAdapt);

/*@
  TSAdaptRegister -  adds a TSAdapt implementation

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of user-defined adaptivity scheme
- function - routine to create method context

  Level: advanced

  Notes:
  `TSAdaptRegister()` may be called multiple times to add several user-defined families.

  Example Usage:
.vb
   TSAdaptRegister("my_scheme", MySchemeCreate);
.ve

  Then, your scheme can be chosen with the procedural interface via
.vb
  TSAdaptSetType(ts, "my_scheme")
.ve
  or at runtime via the option
.vb
  -ts_adapt_type my_scheme
.ve

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdaptRegisterAll()`
@*/
PetscErrorCode TSAdaptRegister(const char sname[], PetscErrorCode (*function)(TSAdapt))
{
  PetscFunctionBegin;
  PetscCall(TSAdaptInitializePackage());
  PetscCall(PetscFunctionListAdd(&TSAdaptList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptRegisterAll - Registers all of the adaptivity schemes in `TSAdapt`

  Not Collective

  Level: advanced

.seealso: [](ch_ts), `TSAdaptRegisterDestroy()`
@*/
PetscErrorCode TSAdaptRegisterAll(void)
{
  PetscFunctionBegin;
  if (TSAdaptRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  TSAdaptRegisterAllCalled = PETSC_TRUE;
  PetscCall(TSAdaptRegister(TSADAPTNONE, TSAdaptCreate_None));
  PetscCall(TSAdaptRegister(TSADAPTBASIC, TSAdaptCreate_Basic));
  PetscCall(TSAdaptRegister(TSADAPTDSP, TSAdaptCreate_DSP));
  PetscCall(TSAdaptRegister(TSADAPTCFL, TSAdaptCreate_CFL));
  PetscCall(TSAdaptRegister(TSADAPTGLEE, TSAdaptCreate_GLEE));
  PetscCall(TSAdaptRegister(TSADAPTHISTORY, TSAdaptCreate_History));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptFinalizePackage - This function destroys everything in the `TS` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ts), `PetscFinalize()`
@*/
PetscErrorCode TSAdaptFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TSAdaptList));
  TSAdaptPackageInitialized = PETSC_FALSE;
  TSAdaptRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptInitializePackage - This function initializes everything in the `TSAdapt` package. It is
  called from `TSInitializePackage()`.

  Level: developer

.seealso: [](ch_ts), `PetscInitialize()`
@*/
PetscErrorCode TSAdaptInitializePackage(void)
{
  PetscFunctionBegin;
  if (TSAdaptPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TSAdaptPackageInitialized = PETSC_TRUE;
  PetscCall(PetscClassIdRegister("TSAdapt", &TSADAPT_CLASSID));
  PetscCall(TSAdaptRegisterAll());
  PetscCall(PetscRegisterFinalize(TSAdaptFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetType - sets the approach used for the error adapter

  Logicially Collective

  Input Parameters:
+ adapt - the `TS` adapter, most likely obtained with `TSGetAdapt()`
- type  - one of the `TSAdaptType`

  Options Database Key:
. -ts_adapt_type (basic|dsp|none|cfl|glee|history) - to set the adapter type

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSGetAdapt()`, `TSAdaptDestroy()`, `TSAdaptType`, `TSAdaptGetType()`
@*/
PetscErrorCode TSAdaptSetType(TSAdapt adapt, TSAdaptType type)
{
  PetscBool match;
  PetscErrorCode (*r)(TSAdapt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFunctionListFind(TSAdaptList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown TSAdapt type \"%s\" given", type);
  PetscTryTypeMethod(adapt, destroy);
  PetscCall(PetscMemzero(adapt->ops, sizeof(struct _TSAdaptOps)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)adapt, type));
  PetscCall((*r)(adapt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetType - gets the `TS` adapter method type (as a string).

  Not Collective

  Input Parameter:
. adapt - The `TS` adapter, most likely obtained with `TSGetAdapt()`

  Output Parameter:
. type - The name of `TS` adapter method

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptType`, `TSAdaptSetType()`
@*/
PetscErrorCode TSAdaptGetType(TSAdapt adapt, TSAdaptType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)adapt)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetOptionsPrefix - Sets the prefix used for searching for `TSAdapt` options in the options database

  Logically Collective

  Input Parameters:
+ adapt  - the `TSAdapt` context, most likely obtained with `TSGetAdapt()`
- prefix - the prefix to prepend to all option names

  Level: advanced

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSSetOptionsPrefix()`
@*/
PetscErrorCode TSAdaptSetOptionsPrefix(TSAdapt adapt, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)adapt, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptLoad - Loads a TSAdapt that has been stored in binary with `TSAdaptView()`.

  Collective

  Input Parameters:
+ adapt  - the newly loaded `TSAdapt`, this needs to have been created with `TSAdaptCreate()` or
           some related function before a call to `TSAdaptLoad()`.
- viewer - binary file viewer, obtained from `PetscViewerBinaryOpen()` or
           HDF5 file viewer, obtained from `PetscViewerHDF5Open()`

  Level: intermediate

  Note:
  The type is determined by the data in the file, any type set into the `TSAdapt` before this call is ignored.

.seealso: [](ch_ts), `PetscViewerBinaryOpen()`, `TSAdaptView()`, `MatLoad()`, `VecLoad()`, `TSAdapt`
@*/
PetscErrorCode TSAdaptLoad(TSAdapt adapt, PetscViewer viewer)
{
  PetscBool isbinary;
  char      type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCheck(isbinary, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  PetscCall(PetscViewerBinaryRead(viewer, type, 256, NULL, PETSC_CHAR));
  PetscCall(TSAdaptSetType(adapt, type));
  PetscTryTypeMethod(adapt, load, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptView - Prints the `TSAdapt` data structure.

  Collective

  Input Parameters:
+ adapt  - the `TSAdapt` context obtained from `TSGetAdapt()`
- viewer - visualization context

  Options Database Key:
. -ts_view viewer_specification - calls `TSView()` at end of `TSStep()`. See `PetscOptionsCreateViewer()` for the format of `viewer_specification`

  Level: advanced

  Notes:
  This is called by `TSView()` so rarely called directly.

  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file. All other processes send their
  data to the first process to print.

  The user can open an alternative visualization context with
  `PetscViewerASCIIOpen()` - output to a specified file.

  In the debugger you can do call `TSAdaptView`(adapt,0) to display the `TSAdapt`. (The same holds for any PETSc object viewer).

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSView()`, `PetscViewer`, `PetscViewerASCIIOpen()`, `PetscOptionsCreateViewer()`
@*/
PetscErrorCode TSAdaptView(TSAdapt adapt, PetscViewer viewer)
{
  PetscBool isascii, isbinary, isnone, isglee;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)adapt), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(adapt, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)adapt, viewer));
    PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTNONE, &isnone));
    PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTGLEE, &isglee));
    if (!isnone) {
      if (adapt->always_accept) PetscCall(PetscViewerASCIIPrintf(viewer, "  always accepting steps\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  safety factor %g\n", (double)adapt->safety));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  extra safety factor after step rejection %g\n", (double)adapt->reject_safety));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  clip fastest increase %g\n", (double)adapt->clip[1]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  clip fastest decrease %g\n", (double)adapt->clip[0]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum allowed timestep %g\n", (double)adapt->dt_max));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  (relative) minimum allowed timestep %g\n", (double)adapt->dt_min_rel));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  (absolute) minimum allowed timestep %g\n", (double)adapt->dt_min_abs));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum solution absolute value to be ignored %g\n", (double)adapt->ignore_max));
    }
    if (isglee) {
      if (adapt->glee_use_local) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  GLEE uses local error control\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  GLEE uses global error control\n"));
      }
    }
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(adapt, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isbinary) {
    char type[256];

    /* need to save FILE_CLASS_ID for adapt class */
    PetscCall(PetscStrncpy(type, ((PetscObject)adapt)->type_name, 256));
    PetscCall(PetscViewerBinaryWrite(viewer, type, 256, PETSC_CHAR));
  } else PetscTryTypeMethod(adapt, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptReset - Resets a `TSAdapt` context to its defaults

  Collective

  Input Parameter:
. adapt - the `TSAdapt` context obtained from `TSGetAdapt()` or `TSAdaptCreate()`

  Level: developer

.seealso: [](ch_ts), [](sec_ts_error_control), `TSGetAdapt()`, `TSAdapt`, `TSAdaptCreate()`, `TSAdaptDestroy()`
@*/
PetscErrorCode TSAdaptReset(TSAdapt adapt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscTryTypeMethod(adapt, reset);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptDestroy - Destroys a `TSAdapt` context

  Collective

  Input Parameter:
. adapt - the `TSAdapt` context obtained from `TSGetAdapt()` or `TSAdaptCreate()`

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCreate()`, `TSGetAdapt()`
@*/
PetscErrorCode TSAdaptDestroy(TSAdapt *adapt)
{
  PetscFunctionBegin;
  if (!*adapt) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*adapt, TSADAPT_CLASSID, 1);
  if (--((PetscObject)*adapt)->refct > 0) {
    *adapt = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(TSAdaptReset(*adapt));

  PetscTryTypeMethod(*adapt, destroy);
  PetscCall(PetscViewerDestroy(&(*adapt)->monitor));
  PetscCall(PetscHeaderDestroy(adapt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetMonitor - Monitor the choices made by the adaptive controller

  Collective

  Input Parameters:
+ adapt - adaptive controller context
- flg   - `PETSC_TRUE` to active a monitor, `PETSC_FALSE` to disable

  Options Database Key:
. -ts_adapt_monitor - to turn on monitoring

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetMonitor(TSAdapt adapt, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(adapt, flg, 2);
  if (flg) {
    if (!adapt->monitor) PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)adapt), "stdout", &adapt->monitor));
  } else {
    PetscCall(PetscViewerDestroy(&adapt->monitor));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetCheckStage - Set a callback to check convergence for a stage

  Logically Collective

  Input Parameters:
+ adapt - adaptive controller context
- func  - stage check function

  Calling sequence:
+ adapt  - adaptive controller context
. ts     - time stepping context
. t      - current time
. Y      - current solution vector
- accept - pending choice of whether to accept, can be modified by this routine

  Level: advanced

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetCheckStage(TSAdapt adapt, PetscErrorCode (*func)(TSAdapt adapt, TS ts, PetscReal t, Vec Y, PetscBool *accept))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  adapt->checkstage = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetAlwaysAccept - Set whether to always accept steps regardless of
  any error or stability condition not meeting the prescribed goal.

  Logically Collective

  Input Parameters:
+ adapt - time step adaptivity context, usually gotten with `TSGetAdapt()`
- flag  - whether to always accept steps

  Options Database Key:
. -ts_adapt_always_accept - to always accept steps

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetAlwaysAccept(TSAdapt adapt, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(adapt, flag, 2);
  adapt->always_accept = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetSafety - Set safety factors for time step adaptor

  Logically Collective

  Input Parameters:
+ adapt         - adaptive controller context
. safety        - safety factor relative to target error/stability goal
- reject_safety - extra safety factor to apply if the last step was rejected

  Options Database Keys:
+ -ts_adapt_safety safety               - to set safety factor
- -ts_adapt_reject_safety reject_safety - to set reject safety factor

  Level: intermediate

  Note:
  Use `PETSC_CURRENT` to keep the current value for either parameter

  Fortran Note:
  Use `PETSC_CURRENT_REAL`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptGetSafety()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetSafety(TSAdapt adapt, PetscReal safety, PetscReal reject_safety)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, safety, 2);
  PetscValidLogicalCollectiveReal(adapt, reject_safety, 3);
  PetscCheck(safety == (PetscReal)PETSC_CURRENT || safety >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Safety factor %g must be non negative", (double)safety);
  PetscCheck(safety == (PetscReal)PETSC_CURRENT || safety <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Safety factor %g must be less than one", (double)safety);
  PetscCheck(reject_safety == (PetscReal)PETSC_CURRENT || reject_safety >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Reject safety factor %g must be non negative", (double)reject_safety);
  PetscCheck(reject_safety == (PetscReal)PETSC_CURRENT || reject_safety <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Reject safety factor %g must be less than one", (double)reject_safety);
  if (safety != (PetscReal)PETSC_CURRENT) adapt->safety = safety;
  if (reject_safety != (PetscReal)PETSC_CURRENT) adapt->reject_safety = reject_safety;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetSafety - Get safety factors for time step adapter

  Not Collective

  Input Parameter:
. adapt - adaptive controller context

  Output Parameters:
+ safety        - safety factor relative to target error/stability goal
- reject_safety - extra safety factor to apply if the last step was rejected

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptSetSafety()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptGetSafety(TSAdapt adapt, PetscReal *safety, PetscReal *reject_safety)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (safety) PetscAssertPointer(safety, 2);
  if (reject_safety) PetscAssertPointer(reject_safety, 3);
  if (safety) *safety = adapt->safety;
  if (reject_safety) *reject_safety = adapt->reject_safety;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetMaxIgnore - Set error estimation threshold. Solution components below this threshold value will not be considered when computing error norms
  for time step adaptivity (in absolute value). A negative value (default) of the threshold leads to considering all solution components.

  Logically Collective

  Input Parameters:
+ adapt      - adaptive controller context
- max_ignore - threshold for solution components that are ignored during error estimation

  Options Database Key:
. -ts_adapt_max_ignore max_ignore - to set the threshold

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptGetMaxIgnore()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetMaxIgnore(TSAdapt adapt, PetscReal max_ignore)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, max_ignore, 2);
  adapt->ignore_max = max_ignore;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetMaxIgnore - Get error estimation threshold. Solution components below this threshold value will not be considered when computing error norms
  for time step adaptivity (in absolute value).

  Not Collective

  Input Parameter:
. adapt - adaptive controller context

  Output Parameter:
. max_ignore - threshold for solution components that are ignored during error estimation

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptSetMaxIgnore()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptGetMaxIgnore(TSAdapt adapt, PetscReal *max_ignore)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscAssertPointer(max_ignore, 2);
  *max_ignore = adapt->ignore_max;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetClip - Sets the admissible decrease/increase factor in step size in the time step adapter

  Logically collective

  Input Parameters:
+ adapt - adaptive controller context
. low   - admissible decrease factor
- high  - admissible increase factor

  Options Database Key:
. -ts_adapt_clip low,high - to set admissible time step decrease and increase factors

  Level: intermediate

  Note:
  Use `PETSC_CURRENT` to keep the current value for either parameter

  Fortran Note:
  Use `PETSC_CURRENT_REAL`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptChoose()`, `TSAdaptGetClip()`, `TSAdaptSetScaleSolveFailed()`
@*/
PetscErrorCode TSAdaptSetClip(TSAdapt adapt, PetscReal low, PetscReal high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, low, 2);
  PetscValidLogicalCollectiveReal(adapt, high, 3);
  PetscCheck(low == (PetscReal)PETSC_CURRENT || low >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Decrease factor %g must be non negative", (double)low);
  PetscCheck(low == (PetscReal)PETSC_CURRENT || low <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Decrease factor %g must be less than one", (double)low);
  PetscCheck(high == (PetscReal)PETSC_CURRENT || high >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Increase factor %g must be greater than one", (double)high);
  if (low != (PetscReal)PETSC_CURRENT) adapt->clip[0] = low;
  if (high != (PetscReal)PETSC_CURRENT) adapt->clip[1] = high;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetClip - Gets the admissible decrease/increase factor in step size in the time step adapter

  Not Collective

  Input Parameter:
. adapt - adaptive controller context

  Output Parameters:
+ low  - optional, admissible decrease factor
- high - optional, admissible increase factor

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptChoose()`, `TSAdaptSetClip()`, `TSAdaptSetScaleSolveFailed()`
@*/
PetscErrorCode TSAdaptGetClip(TSAdapt adapt, PetscReal *low, PetscReal *high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (low) PetscAssertPointer(low, 2);
  if (high) PetscAssertPointer(high, 3);
  if (low) *low = adapt->clip[0];
  if (high) *high = adapt->clip[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetScaleSolveFailed - Scale step size by this factor if solve fails

  Logically Collective

  Input Parameters:
+ adapt - adaptive controller context
- scale - scale

  Options Database Key:
. -ts_adapt_scale_solve_failed scale - to set scale step by this factor if solve fails

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptChoose()`, `TSAdaptGetScaleSolveFailed()`, `TSAdaptGetClip()`
@*/
PetscErrorCode TSAdaptSetScaleSolveFailed(TSAdapt adapt, PetscReal scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, scale, 2);
  PetscCheck(scale > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Scale factor %g must be positive", (double)scale);
  PetscCheck(scale <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Scale factor %g must be less than one", (double)scale);
  adapt->scale_solve_failed = scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetScaleSolveFailed - Gets the admissible decrease/increase factor in step size

  Not Collective

  Input Parameter:
. adapt - adaptive controller context

  Output Parameter:
. scale - scale factor

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptChoose()`, `TSAdaptSetScaleSolveFailed()`, `TSAdaptSetClip()`
@*/
PetscErrorCode TSAdaptGetScaleSolveFailed(TSAdapt adapt, PetscReal *scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (scale) PetscAssertPointer(scale, 2);
  if (scale) *scale = adapt->scale_solve_failed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// A helper function for consistency checks for dt_min, dt_max
PetscErrorCode TSAdaptCheckStepLimits(TSAdapt adapt, TS ts)
{
  PetscReal   dt_min;
  const char *msg = "Maximum time step %g must be greater than the minimum time step %g (for t = %g)";

  PetscFunctionBegin;
  dt_min = TSMinStepAtTime(adapt->dt_min_rel, adapt->dt_min_abs, ts->ptime); // check for current time t = ts->ptime
  PetscCheck(adapt->dt_max > dt_min, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, msg, (double)adapt->dt_max, (double)dt_min, (double)ts->ptime);

  if (ts->max_time < PETSC_INFINITY && ts->max_time < PETSC_MAX_REAL) {
    dt_min = TSMinStepAtTime(adapt->dt_min_rel, adapt->dt_min_abs, ts->max_time); // check for maximum time t = ts->max_time
    PetscCheck(adapt->dt_max > dt_min, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, msg, (double)adapt->dt_max, (double)dt_min, (double)ts->max_time);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetStepLimits - Set the minimum and maximum step sizes to be considered by the time step controller

  Logically Collective

  Input Parameters:
+ adapt    - time step adaptivity context, usually gotten with `TSGetAdapt()`
. hmin_abs - (absolute) minimum time step
- hmax     - maximum time step

  Options Database Keys:
+ -ts_adapt_dt_min_abs min_abs - set the (absolute) minimum time step
- -ts_adapt_dt_max max         - set the maximum time step

  Level: intermediate

  Note:
  Use `PETSC_CURRENT` to keep the current value for either parameter.

  During the TS run, the actual minimum step size depends on the current time `t`
.vb
  dt_min = Max {dt_min_abs, dt_min_rel * |t|}
.ve
  where `dt_min_abs` and `dt_min_rel` are set by this function and/or `TSAdaptSetMinStep()`.
  The maximum step size `dt_max` is a fixed number.

  Fortran Note:
  Use `PETSC_CURRENT_REAL`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptGetStepLimits()`, `TSAdaptSetMinStep()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetStepLimits(TSAdapt adapt, PetscReal hmin_abs, PetscReal hmax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, hmin_abs, 2);
  PetscValidLogicalCollectiveReal(adapt, hmax, 3);
  PetscCheck(hmin_abs == (PetscReal)PETSC_CURRENT || hmin_abs >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "(Absolute) minimum time step %g must be non negative", (double)hmin_abs);
  PetscCheck(hmax == (PetscReal)PETSC_CURRENT || hmax >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Maximum time step %g must be non negative", (double)hmax);
  if (hmin_abs != (PetscReal)PETSC_CURRENT) adapt->dt_min_abs = hmin_abs;
  if (hmax != (PetscReal)PETSC_CURRENT) adapt->dt_max = hmax;
  hmin_abs = adapt->dt_min_abs;
  hmax     = adapt->dt_max;
  PetscCheck(hmax > hmin_abs, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Maximum time step %g must be greater than the (absolute) minimum time step %g", (double)hmax, (double)hmin_abs);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetStepLimits - Get the minimum and maximum step sizes to be considered by the time step controller

  Not Collective

  Input Parameter:
. adapt - time step adaptivity context, usually gotten with `TSGetAdapt()`

  Output Parameters:
+ hmin_abs - (absolute) minimum time step
- hmax     - maximum time step

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptSetStepLimits()`, `TSAdaptGetMinStep()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptGetStepLimits(TSAdapt adapt, PetscReal *hmin_abs, PetscReal *hmax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (hmin_abs) PetscAssertPointer(hmin_abs, 2);
  if (hmax) PetscAssertPointer(hmax, 3);
  if (hmin_abs) *hmin_abs = adapt->dt_min_abs;
  if (hmax) *hmax = adapt->dt_max;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetMinStep - Set the relative/absolute minimum step sizes for the time step controller

  Logically Collective

  Input Parameters:
+ adapt    - time step adaptivity context, usually gotten with `TSGetAdapt()`
. hmin_rel - (relative) minimum time step
- hmin_abs - (absolute) minimum time step

  Options Database Keys:
+ -ts_adapt_dt_min_rel min_rel - set the (relative) minimum time step
- -ts_adapt_dt_min_abs min_abs - set the (absolute) minimum time step

  Level: intermediate

  Note:
  Use `PETSC_CURRENT` to keep the current value for either parameter.

  During the TS run, the minimum step size `dt_min` considered by the time step controller depends
  on the relative and absolute minimum step sizes set by this function, and the current time `t`
.vb
  dt_min = Max {dt_min_abs, dt_min_rel * |t|}
.ve
  When `t` is far from zero, the relative contribution may become dominant in the definition above,
  ensuring proper work of the floating point arithmetic during the time increment
.vb
  t_next = t + dt
.ve
  When `t` is close to zero, the absolute contribution kicks in.

  Fortran Note:
  Use `PETSC_CURRENT_REAL`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptSetStepLimits()`, `TSAdaptGetMinStep()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetMinStep(TSAdapt adapt, PetscReal hmin_rel, PetscReal hmin_abs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, hmin_rel, 2);
  PetscValidLogicalCollectiveReal(adapt, hmin_abs, 3);
  PetscCheck(hmin_rel == (PetscReal)PETSC_CURRENT || hmin_rel >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "(Relative) minimum time step %g must be non negative", (double)hmin_rel);
  PetscCheck(hmin_abs == (PetscReal)PETSC_CURRENT || hmin_abs >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "(Absolute) minimum time step %g must be non negative", (double)hmin_abs);

  if (hmin_rel != (PetscReal)PETSC_CURRENT) adapt->dt_min_rel = hmin_rel;
  if (hmin_abs != (PetscReal)PETSC_CURRENT) adapt->dt_min_abs = hmin_abs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetMinStep - Get the relative/absolute minimum step sizes for the time step controller

  Not Collective

  Input Parameter:
. adapt - time step adaptivity context, usually gotten with `TSGetAdapt()`

  Output Parameters:
+ hmin_rel - (relative) minimum time step
- hmin_abs - (absolute) minimum time step

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptGetStepLimits()`, `TSAdaptSetMinStep()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptGetMinStep(TSAdapt adapt, PetscReal *hmin_rel, PetscReal *hmin_abs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (hmin_rel) PetscAssertPointer(hmin_rel, 2);
  if (hmin_abs) PetscAssertPointer(hmin_abs, 3);
  if (hmin_rel) *hmin_rel = adapt->dt_min_rel;
  if (hmin_abs) *hmin_abs = adapt->dt_min_abs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptSetFromOptions - Sets various `TSAdapt` parameters from user options.

  Collective

  Input Parameters:
+ adapt              - the `TSAdapt` context
- PetscOptionsObject - object created by `PetscOptionsBegin()`

  Options Database Keys:
+ -ts_adapt_type (basic|dsp|none|cfl|glee|history) - algorithm to use for adaptivity
. -ts_adapt_always_accept (true|false)             - always accept steps regardless of error/stability goals
. -ts_adapt_safety safety                          - safety factor relative to target error/stability goal
. -ts_adapt_reject_safety safety                   - extra safety factor to apply if the last step was rejected
. -ts_adapt_clip low,high                          - admissible time step decrease and increase factors
. -ts_adapt_dt_min_rel min_rel                     - relative minimum timestep to use
. -ts_adapt_dt_min_abs min_abs                     - absolute minimum timestep to use
. -ts_adapt_dt_max max                             - maximum timestep to use
. -ts_adapt_scale_solve_failed scale               - scale timestep by this factor if a solve fails
. -ts_adapt_wnormtype (2|infinity)                 - type of norm for computing error estimates
- -ts_adapt_time_step_increase_delay steps         - number of timesteps to delay increasing the time step after it has been decreased due to failed solver

  Level: advanced

  Note:
  This function is automatically called by `TSSetFromOptions()`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptSetType()`, `TSAdaptSetAlwaysAccept()`, `TSAdaptSetSafety()`,
          `TSAdaptSetClip()`, `TSAdaptSetScaleSolveFailed()`, `TSAdaptSetMinStep()`, `TSAdaptSetStepLimits()`, `TSAdaptSetMonitor()`
@*/
PetscErrorCode TSAdaptSetFromOptions(TSAdapt adapt, PetscOptionItems PetscOptionsObject)
{
  char      type[256] = TSADAPTBASIC;
  PetscReal safety, reject_safety, clip[2], scale, hmin_rel, hmin_abs, hmax;
  PetscBool set, flg;
  PetscInt  two;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  /* This should use PetscOptionsBegin() if/when this becomes an object used outside of TS, but currently this
   * function can only be called from inside TSSetFromOptions()  */
  PetscOptionsHeadBegin(PetscOptionsObject, "TS Adaptivity options");
  PetscCall(PetscOptionsFList("-ts_adapt_type", "Algorithm to use for adaptivity", "TSAdaptSetType", TSAdaptList, ((PetscObject)adapt)->type_name ? ((PetscObject)adapt)->type_name : type, type, sizeof(type), &flg));
  if (flg || !((PetscObject)adapt)->type_name) PetscCall(TSAdaptSetType(adapt, type));

  PetscCall(PetscOptionsBool("-ts_adapt_always_accept", "Always accept the step", "TSAdaptSetAlwaysAccept", adapt->always_accept, &flg, &set));
  if (set) PetscCall(TSAdaptSetAlwaysAccept(adapt, flg));

  safety        = adapt->safety;
  reject_safety = adapt->reject_safety;
  PetscCall(PetscOptionsReal("-ts_adapt_safety", "Safety factor relative to target error/stability goal", "TSAdaptSetSafety", safety, &safety, &set));
  PetscCall(PetscOptionsReal("-ts_adapt_reject_safety", "Extra safety factor to apply if the last step was rejected", "TSAdaptSetSafety", reject_safety, &reject_safety, &flg));
  if (set || flg) PetscCall(TSAdaptSetSafety(adapt, safety, reject_safety));

  two     = 2;
  clip[0] = adapt->clip[0];
  clip[1] = adapt->clip[1];
  PetscCall(PetscOptionsRealArray("-ts_adapt_clip", "Admissible decrease/increase factor in step size", "TSAdaptSetClip", clip, &two, &set));
  PetscCheck(!set || (two == 2), PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Must give exactly two values to -ts_adapt_clip");
  if (set) PetscCall(TSAdaptSetClip(adapt, clip[0], clip[1]));

  hmin_rel = adapt->dt_min_rel;
  hmin_abs = adapt->dt_min_abs;
  hmax     = adapt->dt_max;
  PetscCall(PetscOptionsDeprecated("-ts_adapt_dt_min", "-ts_adapt_dt_min_abs", "3.26", NULL));
  PetscCall(PetscOptionsReal("-ts_adapt_dt_min_rel", "(Relative) minimum time step considered", "TSAdaptSetMinStep", hmin_rel, &hmin_rel, &set));
  PetscCall(PetscOptionsReal("-ts_adapt_dt_min_abs", "(Absolute) minimum time step considered", "TSAdaptSetMinStep", hmin_abs, &hmin_abs, &flg));
  if (set || flg) PetscCall(TSAdaptSetMinStep(adapt, hmin_rel, hmin_abs));

  PetscCall(PetscOptionsReal("-ts_adapt_dt_max", "Maximum time step considered", "TSAdaptSetStepLimits", hmax, &hmax, &flg));
  if (flg) PetscCall(TSAdaptSetStepLimits(adapt, PETSC_CURRENT, hmax));

  PetscCall(PetscOptionsReal("-ts_adapt_max_ignore", "Adaptor ignores (absolute) solution values smaller than this value", "", adapt->ignore_max, &adapt->ignore_max, &set));
  PetscCall(PetscOptionsBool("-ts_adapt_glee_use_local", "GLEE adaptor uses local error estimation for step control", "", adapt->glee_use_local, &adapt->glee_use_local, &set));

  PetscCall(PetscOptionsReal("-ts_adapt_scale_solve_failed", "Scale step by this factor if solve fails", "TSAdaptSetScaleSolveFailed", adapt->scale_solve_failed, &scale, &set));
  if (set) PetscCall(TSAdaptSetScaleSolveFailed(adapt, scale));

  PetscCall(PetscOptionsEnum("-ts_adapt_wnormtype", "Type of norm computed for error estimation", "", NormTypes, (PetscEnum)adapt->wnormtype, (PetscEnum *)&adapt->wnormtype, NULL));
  PetscCheck(adapt->wnormtype == NORM_2 || adapt->wnormtype == NORM_INFINITY, PetscObjectComm((PetscObject)adapt), PETSC_ERR_SUP, "Only 2-norm and infinite norm supported");

  PetscCall(PetscOptionsInt("-ts_adapt_time_step_increase_delay", "Number of timesteps to delay increasing the time step after it has been decreased due to failed solver", "TSAdaptSetTimeStepIncreaseDelay", adapt->timestepjustdecreased_delay, &adapt->timestepjustdecreased_delay, NULL));

  PetscCall(PetscOptionsBool("-ts_adapt_monitor", "Print choices made by adaptive controller", "TSAdaptSetMonitor", adapt->monitor ? PETSC_TRUE : PETSC_FALSE, &flg, &set));
  if (set) PetscCall(TSAdaptSetMonitor(adapt, flg));

  PetscTryTypeMethod(adapt, setfromoptions, PetscOptionsObject);
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptCandidatesClear - clear any previously set candidate schemes

  Logically Collective

  Input Parameter:
. adapt - adaptive controller

  Level: developer

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCreate()`, `TSAdaptCandidateAdd()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptCandidatesClear(TSAdapt adapt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscCall(PetscMemzero(&adapt->candidates, sizeof(adapt->candidates)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptCandidateAdd - add a candidate scheme for the adaptive controller to select from

  Logically Collective; No Fortran Support

  Input Parameters:
+ adapt      - time step adaptivity context, obtained with `TSGetAdapt()` or `TSAdaptCreate()`
. name       - name of the candidate scheme to add
. order      - order of the candidate scheme
. stageorder - stage order of the candidate scheme
. ccfl       - stability coefficient relative to explicit Euler, used for CFL constraints
. cost       - relative measure of the amount of work required for the candidate scheme
- inuse      - indicates that this scheme is the one currently in use, this flag can only be set for one scheme

  Level: developer

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCandidatesClear()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptCandidateAdd(TSAdapt adapt, const char name[], PetscInt order, PetscInt stageorder, PetscReal ccfl, PetscReal cost, PetscBool inuse)
{
  PetscInt c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscCheck(order >= 1, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Classical order %" PetscInt_FMT " must be a positive integer", order);
  if (inuse) {
    PetscCheck(!adapt->candidates.inuse_set, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_WRONGSTATE, "Cannot set the inuse method twice, maybe forgot to call TSAdaptCandidatesClear()");
    adapt->candidates.inuse_set = PETSC_TRUE;
  }
  /* first slot if this is the current scheme, otherwise the next available slot */
  c = inuse ? 0 : !adapt->candidates.inuse_set + adapt->candidates.n;

  adapt->candidates.name[c]       = name;
  adapt->candidates.order[c]      = order;
  adapt->candidates.stageorder[c] = stageorder;
  adapt->candidates.ccfl[c]       = ccfl;
  adapt->candidates.cost[c]       = cost;
  adapt->candidates.n++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptCandidatesGet - Get the list of candidate orders of accuracy and cost

  Not Collective

  Input Parameter:
. adapt - time step adaptivity context

  Output Parameters:
+ n          - number of candidate schemes, always at least 1
. order      - the order of each candidate scheme
. stageorder - the stage order of each candidate scheme
. ccfl       - the CFL coefficient of each scheme
- cost       - the relative cost of each scheme

  Level: developer

  Note:
  The current scheme is always returned in the first slot

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCandidatesClear()`, `TSAdaptCandidateAdd()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptCandidatesGet(TSAdapt adapt, PetscInt *n, const PetscInt **order, const PetscInt **stageorder, const PetscReal **ccfl, const PetscReal **cost)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (n) *n = adapt->candidates.n;
  if (order) *order = adapt->candidates.order;
  if (stageorder) *stageorder = adapt->candidates.stageorder;
  if (ccfl) *ccfl = adapt->candidates.ccfl;
  if (cost) *cost = adapt->candidates.cost;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptChoose - choose which method and step size to use for the next step

  Collective

  Input Parameters:
+ adapt - adaptive controller
. ts    - time stepper
- h     - current step size

  Output Parameters:
+ next_sc - optional, scheme to use for the next step
. next_h  - step size to use for the next step
- accept  - `PETSC_TRUE` to accept the current step, `PETSC_FALSE` to repeat the current step with the new step size

  Level: developer

  Note:
  The input value of parameter accept is retained from the last time step, so it will be `PETSC_FALSE` if the step is
  being retried after an initial rejection.

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCandidatesClear()`, `TSAdaptCandidateAdd()`
@*/
PetscErrorCode TSAdaptChoose(TSAdapt adapt, TS ts, PetscReal h, PetscInt *next_sc, PetscReal *next_h, PetscBool *accept)
{
  PetscInt  ncandidates = adapt->candidates.n;
  PetscInt  scheme      = 0;
  PetscReal wlte        = -1.0;
  PetscReal wltea       = -1.0;
  PetscReal wlter       = -1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  if (next_sc) PetscAssertPointer(next_sc, 4);
  PetscAssertPointer(next_h, 5);
  PetscAssertPointer(accept, 6);
  if (next_sc) *next_sc = 0;

  /* Do not mess with adaptivity while handling events */
  if (ts->event && ts->event->processing) {
    *next_h = h;
    *accept = PETSC_TRUE;
    if (adapt->monitor) {
      PetscCall(PetscViewerASCIIAddTab(adapt->monitor, ((PetscObject)adapt)->tablevel));

      if (ts->event->iterctr == 0) {
        /*
          An event has been found, now finalising the event processing: performing the 1st and 2nd post-event steps.
          Entering this if-branch means both these steps (set to either PETSC_DECIDE or numerical value) are managed
          by the event handler. In this case the 1st post-event step is always accepted, without interference of TSAdapt.
          Note: if the 2nd post-event step is not managed by the event handler (e.g. given 1st = numerical, 2nd = PETSC_DECIDE),
          this if-branch is not entered, and TSAdapt may reject/adjust the proposed 1st post-event step.
        */
        PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "TSAdapt does not interfere, step %3" PetscInt_FMT " accepted. Processing post-event steps: 1-st accepted just now, 2-nd yet to come\n", ts->steps));
      } else PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "TSAdapt does not interfere, step %3" PetscInt_FMT " accepted. Event handling in progress\n", ts->steps));

      PetscCall(PetscViewerASCIISubtractTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscUseTypeMethod(adapt, choose, ts, h, &scheme, next_h, accept, &wlte, &wltea, &wlter);
  PetscCheck(scheme >= 0 && (ncandidates <= 0 || scheme < ncandidates), PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Chosen scheme %" PetscInt_FMT " not in valid range 0..%" PetscInt_FMT, scheme, ncandidates - 1);
  PetscCheck(*next_h >= 0, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Computed step size %g must be positive", (double)*next_h);
  if (next_sc) *next_sc = scheme;

  if (*accept && ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
    // Increase/reduce the *next* step size -- if it ends close to, or overshoots, max_time or evaltimes_i
    PetscReal t = ts->ptime + ts->time_step, h1, tmax, hmax;
    /*
      Logic in using 'next_h_cache':
      => next_h_cache > 0 indicates the step has been cut/adjusted previously to get the current 'h'.
         In this case the overall code design should result in that next_h_cache >= 'h', or they approximately equal.
      => next_h_cache is engaged via the APPLY-CLEAR-SAVE sequence listed below (except the rejected-step case).
      => next_h_cache can be also updated/used by the event handler, see tsevent.c, and in TSSolve(), see ts.c

      APPLY. Nonzero next_h_cache overrides or caps *next_h as follows:
             [---] When *next_h < h, use *next_h (adaptor's proposal) to ensure convergence.
             [USE] When *next_h == h (to within rounding), use next_h_cache.
             [CAP] When *next_h > h, use 2*next_h_cache to cap *next_h proposed by adaptor.
                   E.g. in case the cut step 'h' is very small, the adaptor could have proposed to increase
                   it significantly in *next_h. A time step after a large decrease + large increase
                   may carry less information than the cached step, so use 2*next_h_cache as an extra cap.
      CLEAR. After 'APPLY', next_h_cache is always cleared, whether it has changed *next_h or not.
      SAVE.  The step *next_h may be adjusted by the evaluation time points, or the final point.
             In this case, next_h_cache saves the original version of *next_h.

      If the current step is rejected, the APPLY-CLEAR-SAVE sequence is skipped, nothing happens with next_h_cache.
    */
    if (adapt->next_h_cache > 0) { // engage the cache
      if (PetscIsCloseAtTol(h, *next_h, 10 * PETSC_MACHINE_EPSILON, 0.0)) *next_h = adapt->next_h_cache;
      else if (*next_h > h) *next_h = PetscMin(*next_h, 2 * adapt->next_h_cache); // the factor 2 is somewhat arbitrary and may be reconsidered
    }
    h1                  = *next_h;
    adapt->next_h_cache = 0; // clear the cache
    if (ts->eval_times && ts->eval_times->time_point_idx < ts->eval_times->num_time_points) {
      PetscCheck(ts->eval_times->worktol == 0, PetscObjectComm((PetscObject)adapt), PETSC_ERR_PLIB, "Unexpected state (eval_times->worktol != 0) in TSAdaptChoose()");
      ts->eval_times->worktol = ts->eval_times->reltol * h1 + ts->eval_times->abstol;
    }
    PetscCall(TSEvaluationTimesNext(ts, t, &tmax, &hmax));
    PetscCheck(hmax > 0, PetscObjectComm((PetscObject)adapt), PETSC_ERR_PLIB, "hmax == %g, but should be > 0", (double)hmax);
    PetscCall(TSAdaptCapNextStep(adapt, t + h1, h1, tmax, hmax, next_h));
    if (h1 != *next_h) adapt->next_h_cache = h1; // cache the step size if it is to be changed
  }
  if (adapt->monitor) {
    const char *sc_name = (scheme < ncandidates) ? adapt->candidates.name[scheme] : "";
    PetscCall(PetscViewerASCIIAddTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
    if (wlte < 0) {
      PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "TSAdapt %s %s %" PetscInt_FMT ":%s step %3" PetscInt_FMT " %s t=%-11g+%10.3e dt=%-10.3e\n", ((PetscObject)adapt)->type_name, ((PetscObject)ts)->type_name, scheme, sc_name, ts->steps, *accept ? "accepted" : "rejected",
                                       (double)ts->ptime, (double)h, (double)*next_h));
    } else {
      PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "TSAdapt %s %s %" PetscInt_FMT ":%s step %3" PetscInt_FMT " %s t=%-11g+%10.3e dt=%-10.3e wlte=%5.3g  wltea=%5.3g wlter=%5.3g\n", ((PetscObject)adapt)->type_name, ((PetscObject)ts)->type_name, scheme, sc_name, ts->steps, *accept ? "accepted" : "rejected",
                                       (double)ts->ptime, (double)h, (double)*next_h, (double)wlte, (double)wltea, (double)wlter));
    }
    PetscCall(PetscViewerASCIISubtractTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  A helper function which may adjust (in most cases, cap) the next step 'next_dt', as required by evaltimes/max_time, or leave it intact.
  tplan, dtplan should be consistent,
  tmax, dtmax should be consistent.
  The resulting step *next_dt will be either smaller than dtplan, or approximately equal to it (or exactly equal).
*/
PetscErrorCode TSAdaptCapNextStep(TSAdapt adapt, PetscReal tplan, PetscReal dtplan, PetscReal tmax, PetscReal dtmax, PetscReal *next_dt)
{
  PetscFunctionBegin;
  if (adapt) {
    PetscReal a, b;

    PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
    PetscAssertPointer(next_dt, 6);
    a = 1.0 + adapt->matchstepfac[0];
    b = adapt->matchstepfac[1];
    if (tplan > tmax) *next_dt = dtmax;
    if (tplan < tmax && dtplan * b > dtmax) *next_dt = dtmax / 2;
    if (tplan < tmax && dtplan * a > dtmax) *next_dt = dtmax;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetTimeStepIncreaseDelay - The number of timesteps to wait after a decrease in the timestep due to failed solver
  before increasing the time step.

  Logicially Collective

  Input Parameters:
+ adapt - adaptive controller context
- cnt   - the number of timesteps

  Options Database Key:
. -ts_adapt_time_step_increase_delay cnt - number of steps to delay the increase

  Level: advanced

  Notes:
  This is to prevent an adaptor from bouncing back and forth between two nearby timesteps. The default is 0.

  The successful use of this option is problem dependent

  Developer Notes:
  There is no theory to support this option

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`
@*/
PetscErrorCode TSAdaptSetTimeStepIncreaseDelay(TSAdapt adapt, PetscInt cnt)
{
  PetscFunctionBegin;
  adapt->timestepjustdecreased_delay = cnt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptCheckStage - checks whether to accept a stage, (e.g. reject and change time step size if nonlinear solve fails or solution vector is infeasible)

  Collective

  Input Parameters:
+ adapt - adaptive controller context
. ts    - time stepper
. t     - Current simulation time
- Y     - Current solution vector

  Output Parameter:
. accept - `PETSC_TRUE` to accept the stage, `PETSC_FALSE` to reject

  Level: developer

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`
@*/
PetscErrorCode TSAdaptCheckStage(TSAdapt adapt, TS ts, PetscReal t, Vec Y, PetscBool *accept)
{
  SNESConvergedReason snesreason = SNES_CONVERGED_ITERATING;
  PetscBool           func_accept;
  char                reject_stage_message[128];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscAssertPointer(accept, 5);
  *accept = PETSC_TRUE;

  if (adapt->checkstage) {
    PetscCallBack("TSAdapt callback check stage", (*adapt->checkstage)(adapt, ts, t, Y, accept));
    if (!*accept) {
      PetscCall(PetscStrncpy(reject_stage_message, "rejected by TSAdaptSetCheckStage", sizeof(reject_stage_message)));
      goto reject_stage;
    }
  }

  PetscCall(TSFunctionDomainError(ts, t, Y, &func_accept));
  if (!func_accept) {
    PetscCall(PetscStrncpy(reject_stage_message, "rejected by TSSetFunctionDomainError()", sizeof(reject_stage_message)));
    goto reject_stage;
  }

  if (ts->snes) PetscCall(SNESGetConvergedReason(ts->snes, &snesreason));
  if (snesreason < 0) {
    // SNES_DIVERGED_FUNCTION_DOMAIN should not count against ts->max_snes_failures, see !6581 and commit 6c6709e3a
    if (snesreason != SNES_DIVERGED_FUNCTION_DOMAIN && ++ts->num_snes_failures >= ts->max_snes_failures && ts->max_snes_failures != PETSC_UNLIMITED) {
      ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
      PetscCall(PetscSNPrintf(reject_stage_message, sizeof(reject_stage_message), "nonlinear solve failures %" PetscInt_FMT " greater than current TS allowed, stopping solve", ts->num_snes_failures));
    } else PetscCall(PetscSNPrintf(reject_stage_message, sizeof(reject_stage_message), "SNES solve failure %s", SNESConvergedReasons[snesreason]));
    goto reject_stage;
  }
  PetscFunctionReturn(PETSC_SUCCESS);

reject_stage:
  *accept = PETSC_FALSE;
  PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", %s\n", ts->steps, reject_stage_message));
  if (adapt->monitor) {
    PetscCall(PetscViewerASCIIAddTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "TSAdapt %s step %3" PetscInt_FMT " stage rejected t=%-11g+%10.3e, %s", ((PetscObject)adapt)->type_name, ts->steps, (double)ts->ptime, (double)ts->time_step, reject_stage_message));
    PetscCall(PetscViewerASCIISubtractTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
  }
  if (!ts->reason) {
    PetscReal dt, new_dt;
    PetscCall(TSGetTimeStep(ts, &dt));
    new_dt = dt * adapt->scale_solve_failed;
    PetscCall(TSSetTimeStep(ts, new_dt));
    adapt->timestepjustdecreased += adapt->timestepjustdecreased_delay;
    if (adapt->monitor) PetscCall(PetscViewerASCIIPrintf(adapt->monitor, ", retrying with dt=%-10.3e\n", (double)new_dt));
  } else if (adapt->monitor) {
    PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptCreate - create an adaptive controller context for time stepping

  Collective

  Input Parameter:
. comm - The communicator

  Output Parameter:
. inadapt - new `TSAdapt` object

  Level: developer

  Note:
  `TSAdapt` creation is handled by `TS`, so users should not need to call this function.

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptSetType()`, `TSAdaptDestroy()`
@*/
PetscErrorCode TSAdaptCreate(MPI_Comm comm, TSAdapt *inadapt)
{
  TSAdapt adapt;

  PetscFunctionBegin;
  PetscAssertPointer(inadapt, 2);
  PetscCall(TSAdaptInitializePackage());

  PetscCall(PetscHeaderCreate(adapt, TSADAPT_CLASSID, "TSAdapt", "Time stepping adaptivity", "TS", comm, TSAdaptDestroy, TSAdaptView));
  adapt->always_accept      = PETSC_FALSE;
  adapt->safety             = 0.9;
  adapt->reject_safety      = 0.5;
  adapt->clip[0]            = 0.1;
  adapt->clip[1]            = 10.;
  adapt->dt_min_rel         = 10 * PETSC_MACHINE_EPSILON;
  adapt->dt_min_abs         = PetscSqrtReal(PETSC_REAL_MIN);
  adapt->dt_max             = PETSC_MAX_REAL;
  adapt->ignore_max         = -1.0;
  adapt->glee_use_local     = PETSC_TRUE;
  adapt->scale_solve_failed = 0.25;
  /* these two safety factors are not public, and they are used only in the TS_EXACTFINALTIME_MATCHSTEP case
     to prevent from situations were unreasonably small time steps are taken in order to match the final time */
  adapt->matchstepfac[0]             = 0.01; /* allow 1% step size increase in the last step */
  adapt->matchstepfac[1]             = 2.0;  /* halve last step if it is greater than what remains divided this factor */
  adapt->wnormtype                   = NORM_2;
  adapt->timestepjustdecreased_delay = 0;
  *inadapt                           = adapt;
  PetscFunctionReturn(PETSC_SUCCESS);
}
