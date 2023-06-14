
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/

PetscStageLog _petsc_stageLog = NULL;

PETSC_INTERN PetscErrorCode PetscStageRegLogCreate(PetscStageRegLog *stageLog)
{
  PetscStageRegLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));
  l->num_entries = 0;
  l->max_entries = 8;
  PetscCall(PetscMalloc1(l->max_entries, &l->array));
  *stageLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscStageRegLogEnsureSize(PetscStageRegLog stage_log, int new_size)
{
  PetscStageRegInfo blank_entry;

  PetscFunctionBegin;
  PetscCall(PetscMemzero(&blank_entry, sizeof(blank_entry)));
  PetscCall(PetscLogResizableArrayEnsureSize(stage_log,new_size,blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscStageRegLogInsert(PetscStageRegLog stage_log, const char sname[], int *stage)
{
  PetscStageRegInfo *stage_info;
  PetscFunctionBegin;
  PetscValidCharPointer(sname, 2);
  PetscValidIntPointer(stage, 3);
  for (int s = 0; s < stage_log->num_entries; s++) {
    PetscBool same;

    PetscCall(PetscStrcmp(stage_log->array[s].name, sname, &same));
    PetscCheck(!same, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Duplicate stage name given: %s", sname);
  }
  *stage = stage_log->num_entries;
  PetscCall(PetscStageRegLogEnsureSize(stage_log, stage_log->num_entries + 1));
  stage_info = &(stage_log->array[stage_log->num_entries++]);
  PetscCall(PetscMemzero(stage_info, sizeof(*stage_info)));
  PetscCall(PetscStrallocpy(sname, &stage_info->name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogGetDefaultHandler(PetscStageLog *default_handler)
{
  PetscFunctionBegin;
  PetscValidPointer(default_handler, 1);
  *default_handler = NULL;
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    if (PetscLogHandlers[i] && PetscLogHandlers[i]->id == PETSC_LOG_HANDLER_DEFAULT) {
      *default_handler = (PetscStageLog) PetscLogHandlers[i]->ctx;
    }
  }
  if (*default_handler == NULL) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogGetEventLog(PetscEventRegLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog, 1);
  if (!petsc_log_registry) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *eventLog = petsc_log_registry->events;;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogGetClassLog(PetscClassRegLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 1);
  if (!petsc_log_registry) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *classLog = petsc_log_registry->classes;;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetCurrent - This function returns the stage from the top of the stack.

  Not Collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Output Parameter:
. stage    - The current stage

  Note:
  If no stage is currently active, stage is set to -1.

  Level: developer

  Developer Note:
    Inline since called for EACH `PetscEventLogBeginDefault()` and `PetscEventLogEndDefault()`

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogGetCurrent(PetscStageLog stageLog, int *stage)
{
  PetscBool empty;

  PetscFunctionBegin;
  PetscCall(PetscIntStackEmpty(stageLog->stack, &empty));
  if (empty) {
    *stage = -1;
  } else {
    PetscCall(PetscIntStackTop(stageLog->stack, stage));
  }
  PetscCheck(*stage == stageLog->curStage, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistency in stage log: stage %d should be %d", *stage, stageLog->curStage);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetEventPerfLog - This function returns the `PetscEventPerfLog` for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- stage    - The stage

  Output Parameter:
. eventLog - The `PetscEventPerfLog`

  Level: developer

  Developer Note:
    Inline since called for EACH `PetscEventLogBeginDefault()` and `PetscEventLogEndDefault()`

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogGetEventPerfLog(PetscStageLog stageLog, PetscLogStage stage, PetscEventPerfLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  *eventLog = stageLog->array[stage].eventLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageInfoDestroy - This destroys a `PetscStageInfo` object.

  Not collective

  Input Parameter:
. stageInfo - The `PetscStageInfo`

  Level: developer

.seealso: `PetscStageLogCreate()`
@*/
PetscErrorCode PetscStageInfoDestroy(PetscStageInfo *stageInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(stageInfo->name));
  PetscCall(PetscEventPerfLogDestroy(stageInfo->eventLog));
  PetscCall(PetscClassPerfLogDestroy(stageInfo->classLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogDestroy - This destroys a `PetscStageLog` object.

  Not collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Level: developer

.seealso: `PetscStageLogCreate()`
@*/
PetscErrorCode PetscStageLogDestroy(PetscStageLog stageLog)
{
  int stage;

  PetscFunctionBegin;
  if (!stageLog) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscIntStackDestroy(stageLog->stack));
  for (stage = 0; stage < stageLog->num_entries; stage++) PetscCall(PetscStageInfoDestroy(&stageLog->array[stage]));
  PetscCall(PetscFree(stageLog->array));
  PetscCall(PetscFree(stageLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogRegister - Registers a stage name for logging operations in an application code.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- sname    - the name to associate with that stage

  Output Parameter:
. stage    - The stage index

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscStageLogCreate()`
@*/
PetscErrorCode PetscStageLogRegister(PetscStageLog stageLog, const char sname[], int *stage)
{
  PetscStageInfo *stageInfo;
  int             s;

  PetscFunctionBegin;
  PetscValidCharPointer(sname, 2);
  PetscValidIntPointer(stage, 3);
  /* Check stage already registered */
  for (s = 0; s < stageLog->num_entries; ++s) {
    PetscBool same;

    PetscCall(PetscStrcmp(stageLog->array[s].name, sname, &same));
    PetscCheck(!same, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Duplicate stage name given: %s", sname);
  }
  /* Create new stage */
  s = stageLog->num_entries++;
  if (stageLog->num_entries > stageLog->max_entries) {
    PetscCall(PetscMalloc1(stageLog->max_entries * 2, &stageInfo));
    PetscCall(PetscArraycpy(stageInfo, stageLog->array, stageLog->max_entries));
    PetscCall(PetscFree(stageLog->array));
    stageLog->array = stageInfo;
    stageLog->max_entries *= 2;
  }
  /* Setup new stage info */
  stageInfo = &stageLog->array[s];
  PetscCall(PetscMemzero(stageInfo, sizeof(PetscStageInfo)));
  PetscCall(PetscStrallocpy(sname, &stageInfo->name));
  stageInfo->used             = PETSC_FALSE;
  stageInfo->perfInfo.active  = PETSC_TRUE;
  stageInfo->perfInfo.visible = PETSC_TRUE;
  PetscCall(PetscEventPerfLogCreate(&stageInfo->eventLog));
  PetscCall(PetscClassPerfLogCreate(&stageInfo->classLog));
  *stage = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogPush - This function pushes a stage on the stack.

  Not Collective

  Input Parameters:
+ stageLog   - The `PetscStageLog`
- stage - The stage to log

  Options Database Key:
. -log_view - Activates logging

  Usage:
  If the option -log_view is used to run the program containing the
  following code, then 2 sets of summary data will be printed during
  `PetscFinalize()`.
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]
      PetscStageLogPush(stageLog,1);
      [stage 1 of code]
      PetscStageLogPop(stageLog);
      PetscBarrier(...);
      [more stage 0 of code]
      PetscFinalize();
.ve

  Note;
  Use `PetscLogStageRegister()` to register a stage. All previous stages are
  accumulating time and flops, but events will only be logged in this stage.

  Level: developer

.seealso: `PetscStageLogPop()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogPush(PetscStageLog stageLog, int stage)
{
  int       curStage = 0;
  PetscLogDouble *timer_old = NULL;
  PetscLogDouble *timer_new = NULL;
  PetscBool empty;

  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);

  /* Record flops/time of previous stage */
  PetscCall(PetscIntStackEmpty(stageLog->stack, &empty));
  if (!empty) {
    PetscCall(PetscIntStackTop(stageLog->stack, &curStage));
    if (stageLog->array[curStage].perfInfo.active) {
      timer_old = &stageLog->array[curStage].perfInfo.time;
      stageLog->array[curStage].perfInfo.flops += petsc_TotalFlops;
      stageLog->array[curStage].perfInfo.numMessages += petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
      stageLog->array[curStage].perfInfo.messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
      stageLog->array[curStage].perfInfo.numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
    }
  }
  /* Activate the stage */
  PetscCall(PetscIntStackPush(stageLog->stack, stage));

  stageLog->array[stage].used = PETSC_TRUE;
  stageLog->array[stage].perfInfo.count++;
  stageLog->curStage = stage;
  /* Subtract current quantities so that we obtain the difference when we pop */
  if (stageLog->array[stage].perfInfo.active) {
    timer_new = &stageLog->array[stage].perfInfo.time;
    stageLog->array[stage].perfInfo.flops -= petsc_TotalFlops;
    stageLog->array[stage].perfInfo.numMessages -= petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
    stageLog->array[stage].perfInfo.messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
    stageLog->array[stage].perfInfo.numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  }
  PetscCall(PetscTimeAddSubtract(timer_old, timer_new));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogPop - This function pops a stage from the stack.

  Not Collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Usage:
  If the option -log_view is used to run the program containing the
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]
      PetscStageLogPush(stageLog,1);
      [stage 1 of code]
      PetscStageLogPop(stageLog);
      PetscBarrier(...);
      [more stage 0 of code]
      PetscFinalize();
.ve

  Note:
  Use `PetscStageLogRegister()` to register a stage.

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogPop(PetscStageLog stageLog)
{
  int       curStage;
  PetscBool empty;

  PetscFunctionBegin;
  /* Record flops/time of current stage */
  PetscCall(PetscIntStackPop(stageLog->stack, &curStage));
  if (stageLog->array[curStage].perfInfo.active) {
    PetscCall(PetscTimeAdd(&stageLog->array[curStage].perfInfo.time));
    stageLog->array[curStage].perfInfo.flops += petsc_TotalFlops;
    stageLog->array[curStage].perfInfo.numMessages += petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
    stageLog->array[curStage].perfInfo.messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
    stageLog->array[curStage].perfInfo.numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  }
  PetscCall(PetscIntStackEmpty(stageLog->stack, &empty));
  if (!empty) {
    /* Subtract current quantities so that we obtain the difference when we pop */
    PetscCall(PetscIntStackTop(stageLog->stack, &curStage));
    if (stageLog->array[curStage].perfInfo.active) {
      PetscCall(PetscTimeSubtract(&stageLog->array[curStage].perfInfo.time));
      stageLog->array[curStage].perfInfo.flops -= petsc_TotalFlops;
      stageLog->array[curStage].perfInfo.numMessages -= petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
      stageLog->array[curStage].perfInfo.messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
      stageLog->array[curStage].perfInfo.numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
    }
    stageLog->curStage = curStage;
  } else stageLog->curStage = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetClassRegLog - This function returns the PetscClassRegLog for the given stage.

  Not Collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Output Parameter:
. classLog - The `PetscClassRegLog`

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogGetClassRegLog(PetscStageLog stageLog, PetscClassRegLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 2);
  PetscCall(PetscLogGetClassLog(classLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetEventRegLog - This function returns the `PetscEventRegLog`.

  Not Collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Output Parameter:
. eventLog - The `PetscEventRegLog`

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogGetEventRegLog(PetscStageLog stageLog, PetscEventRegLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog, 2);
  PetscCall(PetscLogGetEventLog(eventLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetClassPerfLog - This function returns the `PetscClassPerfLog` for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- stage    - The stage

  Output Parameter:
. classLog - The `PetscClassPerfLog`

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog stageLog, int stage, PetscClassPerfLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  *classLog = stageLog->array[stage].classLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogSetActive - This function determines whether events will be logged during this state.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
. stage    - The stage to log
- isActive - The activity flag, `PETSC_TRUE` for logging, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Level: developer

.seealso: `PetscStageLogGetActive()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogSetActive(PetscStageLog stageLog, int stage, PetscBool isActive)
{
  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  stageLog->array[stage].perfInfo.active = isActive;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetActive - This function returns whether events will be logged suring this stage.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- stage    - The stage to log

  Output Parameter:
. isActive - The activity flag, `PETSC_TRUE` for logging, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Level: developer

.seealso: `PetscStageLogSetActive()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogGetActive(PetscStageLog stageLog, int stage, PetscBool *isActive)
{
  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  PetscValidBoolPointer(isActive, 3);
  *isActive = stageLog->array[stage].perfInfo.active;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogSetVisible - This function determines whether a stage is printed during `PetscLogView()`

  Not Collective

  Input Parameters:
+ stageLog  - The `PetscStageLog`
. stage     - The stage to log
- isVisible - The visibility flag, `PETSC_TRUE` for printing, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

.seealso: `PetscStageLogGetVisible()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogSetVisible(PetscStageLog stageLog, int stage, PetscBool isVisible)
{
  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  stageLog->array[stage].perfInfo.visible = isVisible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetVisible - This function returns whether a stage is printed during `PetscLogView()`

  Not Collective

  Input Parameters:
+ stageLog  - The `PetscStageLog`
- stage     - The stage to log

  Output Parameter:
. isVisible - The visibility flag, `PETSC_TRUE` for printing, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

.seealso: `PetscStageLogSetVisible()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogGetVisible(PetscStageLog stageLog, int stage, PetscBool *isVisible)
{
  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  PetscValidBoolPointer(isVisible, 3);
  *isVisible = stageLog->array[stage].perfInfo.visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetStage - This function returns the stage id given the stage name.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- name     - The stage name

  Output Parameter:
. stage    - The stage id, or -1 if it does not exist

  Level: developer

.seealso: `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscStageLogGetStage(PetscStageLog stageLog, const char name[], PetscLogStage *stage)
{
  PetscBool match;
  int       s;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 2);
  PetscValidIntPointer(stage, 3);
  *stage = -1;
  for (s = 0; s < stageLog->num_entries; s++) {
    PetscCall(PetscStrcasecmp(stageLog->array[s].name, name, &match));
    if (match) {
      *stage = s;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogCreate - This creates a `PetscStageLog` object.

  Not collective

  Output Parameter:
. stageLog - The `PetscStageLog`

  Level: developer

.seealso: `PetscStageLogCreate()`
@*/
PetscErrorCode PetscStageLogCreate(PetscStageLog *stageLog)
{
  PetscStageLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));

  l->num_entries = 0;
  l->max_entries = 10;
  l->curStage  = -1;

  PetscCall(PetscIntStackCreate(&l->stack));
  PetscCall(PetscMalloc1(l->max_entries, &l->array));

  *stageLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}
