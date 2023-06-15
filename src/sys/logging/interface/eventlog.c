
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/
#include <petscdevice.h>
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  #include <../src/sys/perfstubs/timer.h>
#endif
#include <../src/sys/logging/impls/default/logdefault.h>

PetscBool PetscLogSyncOn = PETSC_FALSE;
PetscBool PetscLogMemory = PETSC_FALSE;
#if defined(PETSC_HAVE_DEVICE)
PetscBool PetscLogGpuTraffic = PETSC_FALSE;
#endif

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
/* Note: these functions do not have prototypes in a public directory, so they are considered "internal" and not exported. */

/*@C
  PetscEventRegLogCreate - This creates a `PetscEventRegLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventRegLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventRegLogDestroy()`, `PetscStageLogCreate()`
@*/
PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *eventLog)
{
  PetscEventRegLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));
  l->num_entries = 0;
  l->max_entries = 128; 
  PetscCall(PetscMalloc1(l->max_entries, &l->array));
  *eventLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventRegLogEnsureSize(PetscEventRegLog event_log, int new_size)
{
  PetscEventRegInfo blank_entry;

  PetscFunctionBegin;
  PetscCall(PetscMemzero(&blank_entry, sizeof(blank_entry)));
  PetscCall(PetscLogResizableArrayEnsureSize(event_log,new_size,blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventRegLogDestroy - This destroys a `PetscEventRegLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventRegLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventRegLogCreate()`
@*/
PetscErrorCode PetscEventRegLogDestroy(PetscEventRegLog eventLog)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->num_entries; e++) PetscCall(PetscFree(eventLog->array[e].name));
  PetscCall(PetscFree(eventLog->array));
  PetscCall(PetscFree(eventLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscStageRegLogDestroy(PetscStageRegLog stageLog)
{
  PetscFunctionBegin;
  for (int s = 0; s < stageLog->num_entries; s++) PetscCall(PetscFree(stageLog->array[s].name));
  PetscCall(PetscFree(stageLog->array));
  PetscCall(PetscFree(stageLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogCreate - This creates a `PetscEventPerfLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventPerfLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDestroy()`, `PetscStageLogCreate()`
@*/
PetscErrorCode PetscEventPerfLogCreate(PetscEventPerfLog *eventLog)
{
  PetscEventPerfLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));
  l->num_entries = 0;
  l->max_entries = 100;
  PetscCall(PetscCalloc1(l->max_entries, &l->array));
  *eventLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDestroy - This destroys a `PetscEventPerfLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventPerfLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogCreate()`
@*/
PetscErrorCode PetscEventPerfLogDestroy(PetscEventPerfLog eventLog)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(eventLog->array));
  PetscCall(PetscFree(eventLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ General Functions -------------------------------------------------*/
/*@C
  PetscEventPerfInfoClear - This clears a `PetscEventPerfInfo` object.

  Not collective

  Input Parameter:
. eventInfo - The `PetscEventPerfInfo`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogCreate()`
@*/
PetscErrorCode PetscEventPerfInfoClear(PetscEventPerfInfo *eventInfo)
{
  PetscFunctionBegin;
  eventInfo->id            = -1;
  eventInfo->active        = PETSC_TRUE;
  eventInfo->visible       = PETSC_TRUE;
  eventInfo->depth         = 0;
  eventInfo->count         = 0;
  eventInfo->flops         = 0.0;
  eventInfo->flops2        = 0.0;
  eventInfo->flopsTmp      = 0.0;
  eventInfo->time          = 0.0;
  eventInfo->time2         = 0.0;
  eventInfo->timeTmp       = 0.0;
  eventInfo->syncTime      = 0.0;
  eventInfo->dof[0]        = -1.0;
  eventInfo->dof[1]        = -1.0;
  eventInfo->dof[2]        = -1.0;
  eventInfo->dof[3]        = -1.0;
  eventInfo->dof[4]        = -1.0;
  eventInfo->dof[5]        = -1.0;
  eventInfo->dof[6]        = -1.0;
  eventInfo->dof[7]        = -1.0;
  eventInfo->errors[0]     = -1.0;
  eventInfo->errors[1]     = -1.0;
  eventInfo->errors[2]     = -1.0;
  eventInfo->errors[3]     = -1.0;
  eventInfo->errors[4]     = -1.0;
  eventInfo->errors[5]     = -1.0;
  eventInfo->errors[6]     = -1.0;
  eventInfo->errors[7]     = -1.0;
  eventInfo->numMessages   = 0.0;
  eventInfo->messageLength = 0.0;
  eventInfo->numReductions = 0.0;
#if defined(PETSC_HAVE_DEVICE)
  eventInfo->CpuToGpuCount = 0.0;
  eventInfo->GpuToCpuCount = 0.0;
  eventInfo->CpuToGpuSize  = 0.0;
  eventInfo->GpuToCpuSize  = 0.0;
  eventInfo->GpuFlops      = 0.0;
  eventInfo->GpuTime       = 0.0;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfInfoCopy - Copy the activity and visibility data in eventInfo to outInfo

  Not collective

  Input Parameter:
. eventInfo - The input `PetscEventPerfInfo`

  Output Parameter:
. outInfo   - The output `PetscEventPerfInfo`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfInfoClear()`
@*/
PetscErrorCode PetscEventPerfInfoCopy(const PetscEventPerfInfo *eventInfo, PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->id      = eventInfo->id;
  outInfo->active  = eventInfo->active;
  outInfo->visible = eventInfo->visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfInfoAdd - Add data in eventInfo to outInfo

  Not collective

  Input Parameter:
. eventInfo - The input `PetscEventPerfInfo`

  Output Parameter:
. outInfo   - The output `PetscEventPerfInfo`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfInfoClear()`
@*/
PetscErrorCode PetscEventPerfInfoAdd(const PetscEventPerfInfo *eventInfo, PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->count += eventInfo->count;
  outInfo->time += eventInfo->time;
  outInfo->time2 += eventInfo->time2;
  outInfo->flops += eventInfo->flops;
  outInfo->flops2 += eventInfo->flops2;
  outInfo->numMessages += eventInfo->numMessages;
  outInfo->messageLength += eventInfo->messageLength;
  outInfo->numReductions += eventInfo->numReductions;
#if defined(PETSC_HAVE_DEVICE)
  outInfo->CpuToGpuCount += eventInfo->CpuToGpuCount;
  outInfo->GpuToCpuCount += eventInfo->GpuToCpuCount;
  outInfo->CpuToGpuSize += eventInfo->CpuToGpuSize;
  outInfo->GpuToCpuSize += eventInfo->GpuToCpuSize;
  outInfo->GpuFlops += eventInfo->GpuFlops;
  outInfo->GpuTime += eventInfo->GpuTime;
#endif
  outInfo->memIncrease += eventInfo->memIncrease;
  outInfo->mallocSpace += eventInfo->mallocSpace;
  outInfo->mallocIncreaseEvent += eventInfo->mallocIncreaseEvent;
  outInfo->mallocIncrease += eventInfo->mallocIncrease;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogEnsureSize - This ensures that a `PetscEventPerfLog` is at least of a certain size.

  Not collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- size     - The size

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogCreate()`
@*/
PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog eventLog, int size)
{
  PetscEventPerfInfo blank_entry;

  PetscFunctionBegin;
  PetscCall(PetscEventPerfInfoClear(&blank_entry));
  PetscCall(PetscLogResizableArrayEnsureSize(eventLog,size,blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPE)
  #include <mpe.h>
PETSC_INTERN PetscErrorCode PetscLogMPEGetRGBColor(const char *[]);
PetscErrorCode              PetscLogEventBeginMPE(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(MPE_Log_event(petsc_stageLog->eventLog->array[event].mpe_id_begin, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndMPE(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(MPE_Log_event(petsc_stageLog->eventLog->array[event].mpe_id_end, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
/*@C
  PetscEventRegLogRegister - Registers an event for logging operations in an application code.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventLog`
. ename    - The name associated with the event
- classid   - The classid associated to the class for this event

  Output Parameter:
. event    - The event

  Example of Usage:
.vb
      int USER_EVENT;
      PetscLogDouble user_event_flops;
      PetscLogEventRegister("User event name",0,&USER_EVENT);
      PetscLogEventBegin(USER_EVENT,0,0,0,0);
         [code segment to monitor]
         PetscLogFlops(user_event_flops);
      PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

  Level: developer

  Notes:
  PETSc can gather data for use with the utilities Jumpshot
  (part of the MPICH distribution).  If PETSc has been compiled
  with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
  MPICH), the user can employ another command line option, -log_mpe,
  to create a logfile, "mpe.log", which can be visualized
  Jumpshot.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogFlops()`,
          `PetscEventLogActivate()`, `PetscEventLogDeactivate()`
@*/
PetscErrorCode PetscEventRegLogRegister(PetscEventRegLog eventLog, const char ename[], PetscClassId classid, PetscLogEvent *event)
{
  PetscEventRegInfo *eventInfo;
  int                e = eventLog->num_entries;

  PetscFunctionBegin;
  PetscValidCharPointer(ename, 2);
  PetscValidIntPointer(event, 4);
  /* Should check classid I think */
  PetscCall(PetscEventRegLogEnsureSize(eventLog, e+1));
  eventInfo = &eventLog->array[e];
  eventLog->num_entries++;

  PetscCall(PetscStrallocpy(ename, &(eventInfo->name)));
  eventInfo->classid = classid;
  eventInfo->collective = PETSC_TRUE;
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS) PetscStackCallExternalVoid("ps_timer_create_", eventInfo->timer = ps_timer_create_(eventInfo->name));
#endif
#if defined(PETSC_HAVE_MPE)
  if (PetscLogPLB == PetscLogEventBeginMPE) {
    const char *color;
    PetscMPIInt rank;
    int         beginID, endID;

    beginID = MPE_Log_get_event_number();
    endID   = MPE_Log_get_event_number();

    eventInfo->mpe_id_begin = beginID;
    eventInfo->mpe_id_end   = endID;

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    if (rank == 0) {
      PetscCall(PetscLogMPEGetRGBColor(&color));
      MPE_Describe_state(beginID, endID, eventInfo->name, (char *)color);
    }
  }
#endif
  *event = e;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*---------------------------------------------- Activation Functions -----------------------------------------------*/
/*@C
  PetscEventPerfLogActivate - Indicates that a particular event should be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogDeactivatePop()`, `PetscEventPerfLogDeactivatePush()`
@*/
PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].active = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDeactivate - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

   Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivatePop()`, `PetscEventPerfLogDeactivatePush()`
@*/
PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].active = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDeactivatePush - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivatePush(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogDeactivatePop(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivatePop()`
@*/
PetscErrorCode PetscEventPerfLogDeactivatePush(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].depth++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDeactivatePop - Indicates that a particular event should  be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivatePush(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogDeactivatePop(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivatePush()`
@*/
PetscErrorCode PetscEventPerfLogDeactivatePop(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].depth--;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogActivateClass - Activates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The `PetscEventPerfLog`
. eventRegLog - The `PetscEventRegLog`
- classid      - The class id, for example `MAT_CLASSID`, `SNES_CLASSID`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivateClass()`, `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivate()`
@*/
PetscErrorCode PetscEventPerfLogActivateClass(PetscEventPerfLog eventLog, PetscEventRegLog eventRegLog, PetscClassId classid)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->num_entries; e++) {
    int c = eventRegLog->array[e].classid;
    if (c == classid) eventLog->array[e].active = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDeactivateClass - Deactivates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The `PetscEventPerfLog`
. eventRegLog - The `PetscEventRegLog`
- classid - The class id, for example `MAT_CLASSID`, `SNES_CLASSID`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivateClass()`, `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogActivate()`
@*/
PetscErrorCode PetscEventPerfLogDeactivateClass(PetscEventPerfLog eventLog, PetscEventRegLog eventRegLog, PetscClassId classid)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->num_entries; e++) {
    int c = eventRegLog->array[e].classid;
    if (c == classid) eventLog->array[e].active = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
/*@C
  PetscEventRegLogGetEvent - This function returns the event id given the event name.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventRegLog`
- name     - The stage name

  Output Parameter:
. event    - The event id, or -1 if not found

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventRegLogRegister()`
@*/
PetscErrorCode PetscEventRegLogGetEvent(PetscEventRegLog eventLog, const char name[], PetscLogEvent *event)
{
  PetscBool match;
  int       e;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 2);
  PetscValidIntPointer(event, 3);
  *event = -1;
  for (e = 0; e < eventLog->num_entries; e++) {
    PetscCall(PetscStrcasecmp(eventLog->array[e].name, name, &match));
    if (match) {
      *event = e;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogSetVisible - This function determines whether an event is printed during `PetscLogView()`

  Not Collective

  Input Parameters:
+ eventLog  - The `PetscEventPerfLog`
. event     - The event to log
- isVisible - The visibility flag, `PETSC_TRUE` for printing, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogGetVisible()`, `PetscEventRegLogRegister()`, `PetscStageLogGetEventLog()`
@*/
PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog eventLog, PetscLogEvent event, PetscBool isVisible)
{
  PetscFunctionBegin;
  eventLog->array[event].visible = isVisible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogGetVisible - This function returns whether an event is printed during `PetscLogView()`

  Not Collective

  Input Parameters:
+ eventLog  - The `PetscEventPerfLog`
- event     - The event id to log

  Output Parameter:
. isVisible - The visibility flag, `PETSC_TRUE` for printing, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogSetVisible()`, `PetscEventRegLogRegister()`, `PetscStageLogGetEventLog()`
@*/
PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog eventLog, PetscLogEvent event, PetscBool *isVisible)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(isVisible, 3);
  *isVisible = eventLog->array[event].visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogEventGetPerfInfo - Return the performance information about the given event in the given stage

  Input Parameters:
+ stage - The stage number or `PETSC_DETERMINE` for the current stage
- event - The event number

  Output Parameter:
. info - This structure is filled with the performance information

  Level: Intermediate

  Note:
  This is a low level routine used by the logging functions in PETSc
@*/
PetscErrorCode PetscLogEventGetPerfInfo(PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo *info)
{
  PetscStageLog stage_log;
  PetscEventPerfInfo *event_info;

  PetscFunctionBegin;
  PetscValidPointer(info, 3);
  PetscCall(PetscLogGetDefaultHandler(&stage_log));
  if (stage < 0) PetscCall(PetscLogStageGetCurrent(&stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfInfo(stage_log, stage, event, &event_info));
  *info = *event_info;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogEventSetDof - Set the nth number of degrees of freedom of a numerical problem associated with this event

  Not Collective

  Input Parameters:
+ event - The event id to log
. n     - The dof index, in [0, 8)
- dof   - The number of dofs

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Note:
  This is to enable logging of convergence

.seealso: `PetscLogEventSetError()`, `PetscEventRegLogRegister()`, `PetscStageLogGetEventLog()`
@*/
PetscErrorCode PetscLogEventSetDof(PetscLogEvent event, PetscInt n, PetscLogDouble dof)
{
  PetscStageLog     stageLog;
  PetscEventPerfInfo *event_info;
  int               stage;

  PetscFunctionBegin;
  PetscCheck(!(n < 0) && !(n > 7), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %" PetscInt_FMT " is not in [0, 8)", n);
  PetscCall(PetscLogGetDefaultHandler(&stageLog));
  PetscCall(PetscLogStageGetCurrent(&stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfInfo(stageLog, stage, event, &event_info));
  event_info->dof[n] = dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogEventSetError - Set the nth error associated with a numerical problem associated with this event

  Not Collective

  Input Parameters:
+ event - The event id to log
. n     - The error index, in [0, 8)
- error - The error

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Notes:
  This is to enable logging of convergence, and enable users to interpret the errors as they wish. For example,
  as different norms, or as errors for different fields

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscLogEventSetDof()`, `PetscEventRegLogRegister()`, `PetscStageLogGetEventLog()`
@*/
PetscErrorCode PetscLogEventSetError(PetscLogEvent event, PetscInt n, PetscLogDouble error)
{
  PetscStageLog     stageLog;
  PetscEventPerfInfo *event_info;
  int               stage;

  PetscFunctionBegin;
  PetscCheck(!(n < 0) && !(n > 7), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %" PetscInt_FMT " is not in [0, 8)", n);
  PetscCall(PetscLogGetDefaultHandler(&stageLog));
  PetscCall(PetscLogStageGetCurrent(&stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfInfo(stageLog, stage, event, &event_info));
  event_info->errors[n] = error;
  PetscFunctionReturn(PETSC_SUCCESS);
}
