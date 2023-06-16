#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  #include <../src/sys/perfstubs/timer.h>
#endif
#include "logdefault.h"

/* Action and object logging variables */
PetscActionLog petsc_actions    = NULL;
PetscObjectLog petsc_objects    = NULL;
PetscBool petsc_logActions = PETSC_FALSE;
PetscBool petsc_logObjects = PETSC_FALSE;
int       petsc_numObjectsDestroyed = 0;


/*@C
  PetscLogDefaultBegin - Turns on logging of objects and events using the default logging functions `PetscLogEventBeginDefault()` and `PetscLogEventEndDefault()`. This logs flop
  rates and object creation and should not slow programs down too much.
  This routine may be called more than once.

  Logically Collective over `PETSC_COMM_WORLD`

  Options Database Key:
. -log_view [viewertype:filename:viewerformat] - Prints summary of flop and timing information to the
                  screen (for code configured with --with-log=1 (which is the default))

  Usage:
.vb
      PetscInitialize(...);
      PetscLogDefaultBegin();
       ... code ...
      PetscLogView(viewer); or PetscLogDump();
      PetscFinalize();
.ve

  Level: advanced

  Note:
  `PetscLogView()` or `PetscLogDump()` actually cause the printing of
  the logging information.

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogView()`, `PetscLogTraceBegin()`
@*/
PetscErrorCode PetscLogDefaultBegin(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogAllBegin - Turns on extensive logging of objects and events. Logs
  all events. This creates large log files and slows the program down.

  Logically Collective on `PETSC_COMM_WORLD`

  Options Database Key:
. -log_all - Prints extensive log information

  Usage:
.vb
     PetscInitialize(...);
     PetscLogAllBegin();
     ... code ...
     PetscLogDump(filename);
     PetscFinalize();
.ve

  Level: advanced

  Note:
  A related routine is `PetscLogDefaultBegin()` (with the options key -log_view), which is
  intended for production runs since it logs only flop rates and object
  creation (and shouldn't significantly slow the programs).

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogDefaultBegin()`, `PetscLogTraceBegin()`
@*/
PetscErrorCode PetscLogAllBegin(void)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogTraceBegin - Activates trace logging.  Every time a PETSc event
  begins or ends, the event name is printed.

  Logically Collective on `PETSC_COMM_WORLD`

  Input Parameter:
. file - The file to print trace in (e.g. stdout)

  Options Database Key:
. -log_trace [filename] - Activates `PetscLogTraceBegin()`

  Level: intermediate

  Notes:
  `PetscLogTraceBegin()` prints the processor number, the execution time (sec),
  then "Event begin:" or "Event end:" followed by the event name.

  `PetscLogTraceBegin()` allows tracing of all PETSc calls, which is useful
  to determine where a program is hanging without running in the
  debugger.  Can be used in conjunction with the -info option.

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogView()`, `PetscLogDefaultBegin()`
@*/
PetscErrorCode PetscLogTraceBegin(FILE *file)
{
  PetscStageLog default_handler;
  PetscFunctionBegin;
  PetscCall(PetscLogGetDefaultHandler(&default_handler));
  default_handler->petsc_tracefile = file;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGetDefaultHandler(PetscStageLog *default_handler)
{
  PetscFunctionBegin;
  PetscValidPointer(default_handler, 1);
  *default_handler = NULL;
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    if (PetscLogHandlers[i] && PetscLogHandlers[i]->impl->type == PETSC_LOG_HANDLER_DEFAULT) {
      *default_handler = (PetscStageLog) PetscLogHandlers[i]->ctx;
    }
  }
  if (*default_handler == NULL) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogGetEventPerfLog(PetscStageLog stageLog, PetscLogStage stage, PetscEventPerfLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  *eventLog = stageLog->array[stage].eventLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscStageLog stage_log, PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo **event_info)
{
  PetscEventPerfLog event_log;

  PetscFunctionBegin;
  PetscCall(PetscStageLogGetEventPerfLog(stage_log, stage, &event_log));
  *event_info = &event_log->array[stage];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageInfoDestroy(PetscStageInfo *stageInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(stageInfo->name));
  PetscCall(PetscEventPerfLogDestroy(stageInfo->eventLog));
  PetscCall(PetscClassPerfLogDestroy(stageInfo->classLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogDestroy(PetscStageLog stageLog)
{
  int stage;

  PetscFunctionBegin;
  if (!stageLog) PetscFunctionReturn(PETSC_SUCCESS);
  for (stage = 0; stage < stageLog->num_entries; stage++) PetscCall(PetscStageInfoDestroy(&stageLog->array[stage]));
  PetscCall(PetscFree(stageLog->array));
  PetscCall(PetscFree(stageLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog stageLog, int stage, PetscClassPerfLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  *classLog = stageLog->array[stage].classLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogCreate(PetscStageLog *stageLog)
{
  PetscStageLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));

  l->num_entries = 0;
  l->max_entries = 10;

  PetscCall(PetscCalloc1(l->max_entries, &l->array));

  *stageLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogObjCreateDefault(PetscObject obj, void *ctx)
{
  PetscStageLog     stage_log = (PetscStageLog) ctx;
  PetscLogStage     stage;
  PetscLogRegistry  registry;
  PetscClassPerfLog classPerfLog;
  int               oclass = 0;

  PetscFunctionBegin;
  /* Record stage info */
  PetscCall(PetscLogGetRegistry(&registry));
  PetscCall(PetscLogStageGetCurrent(&stage));
  PetscCall(PetscStageLogGetClassPerfLog(stage_log, stage, &classPerfLog));
  PetscCall(PetscClassRegLogGetClass(registry->classes, obj->classid, &oclass));
  classPerfLog->array[oclass].creations++;
  /* Dynamically enlarge logging structures */

  /* Record the creation action */
  if (petsc_logActions) {
    PetscInt  new_num_actions = ++petsc_actions->num_entries;
    Action new_action;
    PetscCall(PetscTime(&new_action.time));
    new_action.time -= petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_CREATE;
    new_action.classid = obj->classid;
    new_action.id1     = obj->id;
    new_action.id2     = -1;
    new_action.id3     = -1;
    new_action.flops   = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogResizableArrayEnsureSize(petsc_actions,new_num_actions,new_action));
  }
  /* Record the object */
  if (petsc_logObjects) {
    Object new_object;
    PetscInt new_num_objects = ++petsc_objects->num_entries;

    new_object.parent = -1;
    new_object.obj    = obj;

    PetscCall(PetscMemzero(new_object.name, sizeof(new_object.name)));
    PetscCall(PetscMemzero(new_object.info, sizeof(new_object.info)));
    PetscCall(PetscLogResizableArrayEnsureSize(petsc_objects, new_num_objects, new_object));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogObjDestroyDefault(PetscLogState state, PetscObject obj, void *ctx)
{
  PetscLogRegistry  registry;
  PetscStageLog     stageLog = (PetscStageLog) ctx;
  PetscLogStage     stage;
  PetscClassRegLog  classRegLog;
  PetscClassPerfLog classPerfLog;
  int               oclass = 0;

  PetscFunctionBegin;
  /* Record stage info */
  PetscCall(PetscSpinlockLock(&stageLog->lock));
  registry = state->registry;
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  if (stage != -1) {
    /* That can happen if the log summary is output before some things are destroyed */
    classRegLog = registry->classes;
    PetscCall(PetscStageLogGetClassPerfLog(stageLog, stage, &classPerfLog));
    PetscCall(PetscClassRegLogGetClass(classRegLog, obj->classid, &oclass));
    classPerfLog->array[oclass].destructions++;
  }
  /* Cannot Credit all ancestors with your memory because they may have already been destroyed*/
  petsc_numObjectsDestroyed++;
  /* Dynamically enlarge logging structures */
  /* Record the destruction action */
  if (petsc_logActions) {
    PetscInt new_num_actions = ++petsc_actions->num_entries;
    Action new_action;
    PetscCall(PetscTime(&new_action.time));
    new_action.time -= petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_DESTROY;
    new_action.classid = obj->classid;
    new_action.id1     = obj->id;
    new_action.id2     = -1;
    new_action.id3     = -1;
    new_action.flops   = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogResizableArrayEnsureSize(petsc_actions, new_num_actions, new_action));
  }
  if (petsc_logObjects) {
    if (obj->name) PetscCall(PetscStrncpy(petsc_objects->array[obj->id].name, obj->name, 64));
    petsc_objects->array[obj->id].obj = NULL;
  }
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventSynchronizeDefault(PetscLogState state, PetscLogEvent event, MPI_Comm comm, void *ctx)
{
  PetscStageLog     stage_log = (PetscStageLog) ctx;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscLogDouble    time = 0.0;

  PetscFunctionBegin;
  if (!PetscLogSyncOn || comm == MPI_COMM_NULL) PetscFunctionReturn(PETSC_SUCCESS);
  if (!state->registry->events->array[event].collective) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stage_log, stage, &eventLog));
  PetscCall(PetscEventPerfLogEnsureSize(eventLog, event + 1));
  if (eventLog->array[event].depth > 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscTimeSubtract(&time));
  PetscCallMPI(MPI_Barrier(comm));
  PetscCall(PetscTimeAdd(&time));
  eventLog->array[event].syncTime += time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_CUDA)
  #include <nvToolsExt.h>
#endif
#if defined(PETSC_HAVE_THREADSAFETY)
static PetscErrorCode PetscLogGetStageEventPerfInfo_threaded(PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo **eventInfo)
{
  PetscHashIJKKey     key;
  PetscEventPerfInfo *leventInfo;

  PetscFunctionBegin;
  key.i = PetscLogGetTid();
  key.j = stage;
  key.k = event;
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  PetscCall(PetscHMapEventGet(eventInfoMap_th, key, &leventInfo));
  if (!leventInfo) {
    PetscCall(PetscNew(&leventInfo));
    PetscCall(PetscHMapEventSet(eventInfoMap_th, key, leventInfo));
  }
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  *eventInfo = leventInfo;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode PetscLogEventBeginDefault(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscStageLog       stageLog;
  PetscEventPerfLog   eventLog  = NULL;
  PetscEventPerfInfo *eventInfo = NULL;
  int                 stage;

  PetscFunctionBegin;
  /* Synchronization */
  PetscCall(PetscLogEventSynchronize(event, PetscObjectComm(o1)));
  PetscCall(PetscLogGetDefaultHandler(&stageLog));
  PetscCall(PetscLogStageGetCurrent(&stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
  PetscCall(PetscEventPerfLogEnsureSize(eventLog, stage+1));
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscLogGetStageEventPerfInfo_threaded(stage, event, &eventInfo));
  if (eventInfo->depth == 0) {
    PetscCall(PetscEventPerfInfoClear(eventInfo));
    PetscCall(PetscEventPerfInfoCopy(eventLog->array + event, eventInfo));
  }
#else
  eventInfo = eventLog->array + event;
#endif
  /* Check for double counting */
  eventInfo->depth++;
  if (eventInfo->depth > 1) PetscFunctionReturn(PETSC_SUCCESS);
#if defined(PETSC_HAVE_CUDA)
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {
    PetscEventRegLog eventRegLog;
    PetscCall(PetscStageLogGetEventRegLog(stageLog, &eventRegLog));
    nvtxRangePushA(eventRegLog->array[event].name);
  }
#endif
  /* Log the performance info */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS) {
    PetscEventRegLog regLog = NULL;
    regLog = stageLog->registry->events;
    if (regLog->array[event].timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(regLog->array[event].timer));
  }
#endif
  eventInfo->count++;
  eventInfo->timeTmp = 0.0;
  PetscCall(PetscTimeSubtract(&eventInfo->timeTmp));
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
  if (PetscLogMemory) {
    PetscLogDouble usage;
    PetscCall(PetscMemoryGetCurrentUsage(&usage));
    eventInfo->memIncrease -= usage;
    PetscCall(PetscMallocGetCurrentUsage(&usage));
    eventInfo->mallocSpace -= usage;
    PetscCall(PetscMallocGetMaximumUsage(&usage));
    eventInfo->mallocIncrease -= usage;
    PetscCall(PetscMallocPushMaximumUsage((int)event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndDefault(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscStageLog       stageLog;
  PetscEventPerfLog   eventLog  = NULL;
  PetscEventPerfInfo *eventInfo = NULL;
  int                 stage;

  PetscFunctionBegin;
  PetscCall(PetscLogGetDefaultHandler(&stageLog));
  PetscCall(PetscLogStageGetCurrent(&stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
  PetscCall(PetscEventPerfLogEnsureSize(eventLog, event+1));
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscLogGetStageEventPerfInfo_threaded(stage, event, &eventInfo));
#else
  eventInfo = eventLog->array + event;
#endif
  /* Check for double counting */
  eventInfo->depth--;
  if (eventInfo->depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  else PetscCheck(eventInfo->depth == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");

    /* Log performance info */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS) {
    PetscEventRegLog regLog = state->registry->events;
    if (regLog->array[event].timer != NULL) PetscStackCallExternalVoid("ps_timer_stop_", ps_timer_stop_(regLog->array[event].timer));
  }
#endif
  PetscCall(PetscTimeAdd(&eventInfo->timeTmp));
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
  if (PetscLogMemory) {
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
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  PetscCall(PetscEventPerfInfoAdd(eventInfo, eventLog->array + event));
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
#endif
#if defined(PETSC_HAVE_CUDA)
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA)) nvtxRangePop();
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventBeginComplete(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscStageLog     stageLog = (PetscStageLog) ctx;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    curTime;
  int               stage;

  PetscFunctionBegin;
  /* Record the event */
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  eventRegLog = state->registry->events;
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventPerfLog));
  PetscCall(PetscEventPerfLogEnsureSize(eventPerfLog, event + 1));
  PetscCall(PetscTime(&curTime));
  if (petsc_logActions) {
    PetscInt new_num_actions = ++petsc_actions->num_entries;
    Action new_action;
    
    new_action.time    = curTime - petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_BEGIN;
    new_action.event   = event;
    new_action.classid = eventRegLog->array[event].classid;
    new_action.id1 = o1 ? o1->id : -1;
    new_action.id2 = o2 ? o2->id : -1;
    new_action.id3 = o3 ? o3->id : -1;
    new_action.flops = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogResizableArrayEnsureSize(petsc_actions, new_num_actions, new_action));
  }
  /* Check for double counting */
  eventPerfLog->array[event].depth++;
  if (eventPerfLog->array[event].depth > 1) PetscFunctionReturn(PETSC_SUCCESS);
  /* Log the performance info */
  eventPerfLog->array[event].count++;
  eventPerfLog->array[event].time -= curTime;
  eventPerfLog->array[event].flops -= petsc_TotalFlops;
  eventPerfLog->array[event].numMessages -= petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
  eventPerfLog->array[event].messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventPerfLog->array[event].numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndComplete(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscStageLog     stageLog = (PetscStageLog) ctx;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    curTime;
  int               stage;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  /* Record the event */
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  eventRegLog = state->registry->events;
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventPerfLog));
  PetscCall(PetscEventPerfLogEnsureSize(eventPerfLog, event + 1));
  PetscCall(PetscTime(&curTime));
  if (petsc_logActions) {
    PetscInt new_num_actions = ++petsc_actions->num_entries;
    Action new_action;

    new_action.time    = curTime - petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_END;
    new_action.event   = event;
    new_action.classid = eventRegLog->array[event].classid;
    new_action.id1 = o1 ? o1->id : -1;
    new_action.id2 = o2 ? o2->id : -2;
    new_action.id3 = o3 ? o3->id : -3;
    new_action.flops = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogResizableArrayEnsureSize(petsc_actions, new_num_actions, new_action));
  }
  /* Check for double counting */
  eventPerfLog->array[event].depth--;
  if (eventPerfLog->array[event].depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  else PetscCheck(eventPerfLog->array[event].depth >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");
  /* Log the performance info */
  eventPerfLog->array[event].count++;
  eventPerfLog->array[event].time += curTime;
  eventPerfLog->array[event].flops += petsc_TotalFlops;
  eventPerfLog->array[event].numMessages += petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
  eventPerfLog->array[event].messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventPerfLog->array[event].numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventBeginTrace(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscStageLog     stageLog = (PetscStageLog) ctx;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  PetscMPIInt       rank;
  int               stage;

  PetscFunctionBegin;
  if (!stageLog->petsc_tracetime) PetscCall(PetscTime(&stageLog->petsc_tracetime));

  stageLog->petsc_tracelevel++;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscLogGetDefaultHandler(&stageLog));
  PetscCall(PetscLogStageGetCurrent(&stage));
  eventRegLog = state->registry->events;
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventPerfLog));
  /* Check for double counting */
  eventPerfLog->array[event].depth++;
  if (eventPerfLog->array[event].depth > 1) PetscFunctionReturn(PETSC_SUCCESS);
  /* Log performance info */
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, stageLog->petsc_tracefile, "%s[%d] %g Event begin: %s\n", stageLog->petsc_tracespace, rank, cur_time - stageLog->petsc_tracetime, eventRegLog->array[event].name));
  PetscCall(PetscStrncpy(stageLog->petsc_tracespace, stageLog->petsc_traceblanks, 2 * stageLog->petsc_tracelevel));

  stageLog->petsc_tracespace[2 * stageLog->petsc_tracelevel] = 0;

  PetscCall(PetscFFlush(stageLog->petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndTrace(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscStageLog     stageLog = (PetscStageLog) ctx;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  int               stage;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  stageLog->petsc_tracelevel--;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscLogGetDefaultHandler(&stageLog));
  PetscCall(PetscLogStageGetCurrent(&stage));
  eventRegLog = state->registry->events;
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventPerfLog));
  /* Check for double counting */
  eventPerfLog->array[event].depth--;
  if (eventPerfLog->array[event].depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  else PetscCheck(eventPerfLog->array[event].depth >= 0 && stageLog->petsc_tracelevel >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");

  /* Log performance info */
  if (stageLog->petsc_tracelevel) PetscCall(PetscStrncpy(stageLog->petsc_tracespace, stageLog->petsc_traceblanks, 2 * stageLog->petsc_tracelevel));
  stageLog->petsc_tracespace[2 * stageLog->petsc_tracelevel] = 0;
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, stageLog->petsc_tracefile, "%s[%d] %g Event end: %s\n", stageLog->petsc_tracespace, rank, cur_time - stageLog->petsc_tracetime, eventRegLog->array[event].name));
  PetscCall(PetscFFlush(stageLog->petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}
