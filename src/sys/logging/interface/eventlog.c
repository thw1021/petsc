
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

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
/* Note: these functions do not have prototypes in a public directory, so they are considered "internal" and not exported. */

/*
  PetscEventRegLogCreate - This creates a `PetscEventRegLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventRegLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventRegLogDestroy()`
*/
PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *eventLog)
{
  PetscEventRegInfo blank_entry;

  PetscFunctionBegin;
  PetscCall(PetscMemzero(&blank_entry, sizeof(blank_entry)));
  PetscCall(PetscLogResizableArrayCreate(eventLog, 128, blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventRegLogEnsureSize(PetscEventRegLog event_log, int new_size)
{
  PetscFunctionBegin;
  PetscCall(PetscLogResizableArrayEnsureSize(event_log,new_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscEventRegLogDestroy - This destroys a `PetscEventRegLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventRegLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventRegLogCreate()`
*/
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

PETSC_INTERN PetscErrorCode PetscEventRegLogSetCollective(PetscEventRegLog event_log, PetscLogEvent event, PetscBool is_collective)
{
  PetscFunctionBegin;
  event_log->array[event].collective = is_collective;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
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
*/
PetscErrorCode PetscEventPerfInfoCopy(const PetscEventPerfInfo *eventInfo, PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->id      = eventInfo->id;
  outInfo->active  = eventInfo->active;
  outInfo->visible = eventInfo->visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscEventPerfInfoTic(PetscEventPerfInfo *eventInfo, PetscLogDouble time, PetscBool logMemory, int event)
{
  PetscFunctionBegin;
  eventInfo->timeTmp = -time;
  eventInfo->flopsTmp = -petsc_TotalFlops_th;
  eventInfo->numMessages -= petsc_irecv_ct_th + petsc_isend_ct_th + petsc_recv_ct_th + petsc_send_ct_th;
  eventInfo->messageLength -= petsc_irecv_len_th + petsc_isend_len_th + petsc_recv_len_th + petsc_send_len_th;
  eventInfo->numReductions -= petsc_allreduce_ct_th + petsc_gather_ct_th + petsc_scatter_ct_th;
#if defined(PETSC_HAVE_DEVICE)
  eventInfo->CpuToGpuCount -= petsc_ctog_ct_th;
  eventInfo->GpuToCpuCount -= petsc_gtoc_ct_th;
  eventInfo->CpuToGpuSize -= petsc_ctog_sz_th;
  eventInfo->GpuToCpuSize -= petsc_gtoc_sz_th;
  eventInfo->GpuFlops -= petsc_gflops_th;
  eventInfo->GpuTime -= petsc_gtime;
#endif
  if (logMemory) {
    PetscLogDouble usage;
    PetscCall(PetscMemoryGetCurrentUsage(&usage));
    eventInfo->memIncrease -= usage;
    PetscCall(PetscMallocGetCurrentUsage(&usage));
    eventInfo->mallocSpace -= usage;
    PetscCall(PetscMallocGetMaximumUsage(&usage));
    eventInfo->mallocIncrease -= usage;
    PetscCall(PetscMallocPushMaximumUsage(event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscEventPerfInfoToc(PetscEventPerfInfo *eventInfo, PetscLogDouble time, PetscBool logMemory, int event)
{
  PetscFunctionBegin;
  eventInfo->timeTmp += time;
  eventInfo->flopsTmp += petsc_TotalFlops_th;
  eventInfo->time += eventInfo->timeTmp;
  eventInfo->time2 += eventInfo->timeTmp * eventInfo->timeTmp;
  eventInfo->flops += eventInfo->flopsTmp;
  eventInfo->flops2 += eventInfo->flopsTmp * eventInfo->flopsTmp;
  eventInfo->numMessages += petsc_irecv_ct_th + petsc_isend_ct_th + petsc_recv_ct_th + petsc_send_ct_th;
  eventInfo->messageLength += petsc_irecv_len_th + petsc_isend_len_th + petsc_recv_len + petsc_send_len_th;
  eventInfo->numReductions += petsc_allreduce_ct_th + petsc_gather_ct_th + petsc_scatter_ct_th;
#if defined(PETSC_HAVE_DEVICE)
  eventInfo->CpuToGpuCount += petsc_ctog_ct_th;
  eventInfo->GpuToCpuCount += petsc_gtoc_ct_th;
  eventInfo->CpuToGpuSize += petsc_ctog_sz_th;
  eventInfo->GpuToCpuSize += petsc_gtoc_sz_th;
  eventInfo->GpuFlops += petsc_gflops_th;
  eventInfo->GpuTime += petsc_gtime;
#endif
  if (logMemory) {
    PetscLogDouble usage, musage;
    PetscCall(PetscMemoryGetCurrentUsage(&usage)); /* the comments below match the column labels printed in PetscLogView_Default() */
    eventInfo->memIncrease += usage;               /* RMI */
    PetscCall(PetscMallocGetCurrentUsage(&usage));
    eventInfo->mallocSpace += usage; /* Malloc */
    PetscCall(PetscMallocPopMaximumUsage((int)event, &musage));
    eventInfo->mallocIncreaseEvent = PetscMax(musage - usage, eventInfo->mallocIncreaseEvent); /* EMalloc */
    PetscCall(PetscMallocGetMaximumUsage(&usage));
    eventInfo->mallocIncrease += usage; /* MMalloc */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
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
*/
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

/*
  PetscEventPerfLogEnsureSize - This ensures that a `PetscEventPerfLog` is at least of a certain size.

  Not collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- size     - The size

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogCreate()`
*/
PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog eventLog, int size)
{
  PetscFunctionBegin;
  PetscCall(PetscLogResizableArrayEnsureSize(eventLog,size));
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
/*
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
*/
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
/*
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
*/
PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].active = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
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
*/
PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].active = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
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
*/
PetscErrorCode PetscEventPerfLogDeactivatePush(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].depth++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
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
*/
PetscErrorCode PetscEventPerfLogDeactivatePop(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->array[event].depth--;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
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
*/
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

/*
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
*/
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
/*
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
*/
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

/*
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

.seealso: `PetscEventPerfLogGetVisible()`, `PetscEventRegLogRegister()`
*/
PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog eventLog, PetscLogEvent event, PetscBool isVisible)
{
  PetscFunctionBegin;
  eventLog->array[event].visible = isVisible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
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

.seealso: `PetscEventPerfLogSetVisible()`, `PetscEventRegLogRegister()`
*/
PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog eventLog, PetscLogEvent event, PetscBool *isVisible)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(isVisible, 3);
  *isVisible = eventLog->array[event].visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

