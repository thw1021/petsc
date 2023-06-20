#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  #include <../src/sys/perfstubs/timer.h>
#endif
#include "logdefault.h"

static PetscErrorCode PetscLogObjCreateDefault(PetscLogState, PetscObject, void *);
static PetscErrorCode PetscLogObjDestroyDefault(PetscLogState, PetscObject, void *);
static PetscErrorCode PetscLogEventSynchronizeDefault(PetscLogState, PetscLogEvent, MPI_Comm, void *);
PETSC_INTERN PetscErrorCode PetscLogEventBeginDefault(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_INTERN PetscErrorCode PetscLogEventEndDefault(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
static PetscErrorCode PetscLogEventBeginTrace(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
static PetscErrorCode PetscLogEventEndTrace(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
static PetscErrorCode PetscLogStagePush_Default(PetscLogState, PetscLogStage, void *);
static PetscErrorCode PetscLogStagePop_Default(PetscLogState, PetscLogStage, void *);

static PetscErrorCode PetscLogHandlerCreateContext_Default(PetscStageLog *stage_log_p)
{
  PetscStageLog stage_log;

  PetscFunctionBegin;
  PetscCall(PetscLogResizableArrayCreate(stage_log_p, 8));
  stage_log = *stage_log_p;
  PetscCall(PetscLogResizableArrayCreate(&(stage_log->petsc_actions), 64));
  PetscCall(PetscLogResizableArrayCreate(&(stage_log->petsc_objects), 64));

  {
    PetscBool opt;

    PetscCall(PetscOptionsHasName(NULL, NULL, "-log_exclude_actions", &opt));
    if (opt) stage_log->petsc_logActions = PETSC_FALSE;
    PetscCall(PetscOptionsHasName(NULL, NULL, "-log_exclude_objects", &opt));
    if (opt) stage_log->petsc_logObjects = PETSC_FALSE;
  }
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  PetscStackCallExternalVoid("ps_initialize_", ps_initialize_());
#endif
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscHMapEventCreate(&stage_log->eventInfoMap_th));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroyContext_Default(void *ctx)
{
  PetscStageLog stageLog = (PetscStageLog) ctx;

  PetscFunctionBegin;
  for (int s = 0; s < stageLog->num_entries; s++) {
    PetscStageInfo *stage = &stageLog->array[s];

    PetscCall(PetscFree(stage->eventLog->array));
    PetscCall(PetscFree(stage->eventLog));
    PetscCall(PetscFree(stage->classLog->array));
    PetscCall(PetscFree(stage->classLog));
  }
  PetscCall(PetscFree(stageLog->array));
  PetscCall(PetscFree(stageLog->petsc_actions->array));
  PetscCall(PetscFree(stageLog->petsc_actions));
  PetscCall(PetscFree(stageLog->petsc_objects->array));
  PetscCall(PetscFree(stageLog->petsc_objects));
#if defined(PETSC_HAVE_THREADSAFETY)
  if (stageLog->eventInfoMap_th) {
    PetscEventPerfInfo **array;
    PetscInt             n, off = 0;

    PetscCall(PetscHMapEventGetSize(stageLog->eventInfoMap_th, &n));
    PetscCall(PetscMalloc1(n, &array));
    PetscCall(PetscHMapEventGetVals(stageLog->eventInfoMap_th, &off, array));
    for (PetscInt i = 0; i < n; i++) PetscCall(PetscFree(array[i]));
    PetscCall(PetscFree(array));
    PetscCall(PetscHMapEventDestroy(&stageLog->eventInfoMap_th));
  }
#endif
  PetscCall(PetscFree(stageLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Default(PetscLogHandler *handler_p)
{
  PetscLogHandler     handler;
  PetscFunctionBegin;
  PetscCall(PetscNew(handler_p));
  handler = *handler_p;
  PetscCall(PetscNew(&handler->impl));
  PetscCall(PetscLogHandlerCreateContext_Default((PetscStageLog *) &handler->ctx));
  handler->object_create = PetscLogObjCreateDefault;
  handler->object_destroy = PetscLogObjDestroyDefault;
  handler->event_sync = PetscLogEventSynchronizeDefault;
  handler->event_begin = PetscLogEventBeginDefault;
  handler->event_end = PetscLogEventEndDefault;
  handler->impl->event_deactivate_pop = NULL;
  handler->impl->event_deactivate_push = NULL;
  handler->impl->stage_push = PetscLogStagePush_Default;
  handler->impl->stage_pop = PetscLogStagePop_Default;
  handler->impl->destroy = PetscLogHandlerDestroyContext_Default;
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
  PetscLogHandler handler;
  int i_free = -1;

  PetscFunctionBegin;
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i];
    if (h) {
      if (h->impl->type == PETSC_LOG_HANDLER_DEFAULT) {
        // Default handler has already been created
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    } else if (i_free < 0) i_free = i;
  }
  PetscCheck(i_free >= 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "Too many log handlers already running, cannot begin default log handler");
  PetscCall(PetscLogHandlerCreate_Default(&handler));
  PetscLogHandlers[i_free] = handler;
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

  PetscFunctionBegin;
  PetscCall(PetscLogDefaultBegin());
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i];

    if (h && h->impl->type == PETSC_LOG_HANDLER_DEFAULT) {
      PetscStageLog default_handler = (PetscStageLog) h->ctx;

      h->event_begin = PetscLogEventBeginTrace;
      h->event_end = PetscLogEventEndTrace;
      default_handler->petsc_tracefile = file;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
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

static PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog stageLog, int stage, PetscClassPerfLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= stageLog->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->num_entries);
  *classLog = stageLog->array[stage].classLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogObjCreateDefault(PetscLogState state, PetscObject obj, void *ctx)
{
  PetscStageLog     stage_log = (PetscStageLog) ctx;
  PetscLogStage     stage;
  PetscLogRegistry  registry = state->registry;
  PetscClassPerfLog classPerfLog;
  int               oclass = 0;

  PetscFunctionBegin;
  /* Record stage info */
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscStageLogGetClassPerfLog(stage_log, stage, &classPerfLog));
  PetscCall(PetscClassRegLogGetClass(registry->classes, obj->classid, &oclass));
  classPerfLog->array[oclass].creations++;
  /* Dynamically enlarge logging structures */

  /* Record the creation action */
  if (stage_log->petsc_logActions) {
    PetscInt  new_num_actions = ++stage_log->petsc_actions->num_entries;
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
    PetscCall(PetscLogResizableArrayEnsureSize(stage_log->petsc_actions,new_num_actions,new_action));
  }
  /* Record the object */
  if (stage_log->petsc_logObjects) {
    Object new_object;
    PetscInt new_num_objects = ++stage_log->petsc_objects->num_entries;

    new_object.parent = -1;
    new_object.obj    = obj;

    PetscCall(PetscMemzero(new_object.name, sizeof(new_object.name)));
    PetscCall(PetscMemzero(new_object.info, sizeof(new_object.info)));
    PetscCall(PetscLogResizableArrayEnsureSize(stage_log->petsc_objects, new_num_objects, new_object));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogObjDestroyDefault(PetscLogState state, PetscObject obj, void *ctx)
{
  PetscLogRegistry  registry = state->registry;
  PetscStageLog     stage_log = (PetscStageLog) ctx;
  PetscLogStage     stage;
  PetscClassRegLog  classRegLog;
  PetscClassPerfLog classPerfLog;
  int               oclass = 0;

  PetscFunctionBegin;
  /* Record stage info */
  PetscCall(PetscSpinlockLock(&stageLog->lock));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  if (stage != -1) {
    /* That can happen if the log summary is output before some things are destroyed */
    classRegLog = registry->classes;
    PetscCall(PetscStageLogGetClassPerfLog(stage_log, stage, &classPerfLog));
    PetscCall(PetscClassRegLogGetClass(classRegLog, obj->classid, &oclass));
    classPerfLog->array[oclass].destructions++;
  }
  /* Cannot Credit all ancestors with your memory because they may have already been destroyed*/
  stage_log->petsc_numObjectsDestroyed++;
  /* Dynamically enlarge logging structures */
  /* Record the destruction action */
  if (stage_log->petsc_logActions) {
    PetscInt new_num_actions = ++stage_log->petsc_actions->num_entries;
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
    PetscCall(PetscLogResizableArrayEnsureSize(stage_log->petsc_actions, new_num_actions, new_action));
  }
  if (stage_log->petsc_logObjects) {
    if (obj->name) PetscCall(PetscStrncpy(stage_log->petsc_objects->array[obj->id].name, obj->name, 64));
    stage_log->petsc_objects->array[obj->id].obj = NULL;
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
  if (!(stage_log->PetscLogSyncOn) || comm == MPI_COMM_NULL) PetscFunctionReturn(PETSC_SUCCESS);
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
static PetscErrorCode PetscLogGetStageEventPerfInfo_threaded(PetscStageLog stageLog, PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo **eventInfo)
{
  PetscHashIJKKey     key;
  PetscEventPerfInfo *leventInfo;

  PetscFunctionBegin;
  key.i = PetscLogGetTid();
  key.j = stage;
  key.k = event;
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  PetscCall(PetscHMapEventGet(stageLog->eventInfoMap_th, key, &leventInfo));
  if (!leventInfo) {
    PetscCall(PetscNew(&leventInfo));
    PetscCall(PetscHMapEventSet(stageLog->eventInfoMap_th, key, leventInfo));
  }
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  *eventInfo = leventInfo;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode PetscLogEventBeginDefault(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscStageLog       stageLog = (PetscStageLog) ctx;
  PetscEventPerfLog   eventLog  = NULL;
  PetscEventPerfInfo *eventInfo = NULL;
  PetscLogDouble      time;
  int                 stage;

  PetscFunctionBegin;
  /* Synchronization */
  PetscCall(PetscLogEventSynchronizeDefault(state, event, PetscObjectComm(o1), ctx));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
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
    nvtxRangePushA(state->registry->events->array[event].name);
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
  PetscCall(PetscTime(&time));
  PetscCall(PetscEventPerfInfoTic(eventInfo, time, stageLog->PetscLogMemory, (int) event));
  if (stageLog->petsc_logActions) {
    PetscLogDouble curTime;
    PetscInt new_num_actions = ++stageLog->petsc_actions->num_entries;
    Action new_action;

    PetscCall(PetscTime(&curTime));
    
    new_action.time    = curTime - petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_BEGIN;
    new_action.event   = event;
    new_action.classid = state->registry->events->array[event].classid;
    new_action.id1 = o1 ? o1->id : -1;
    new_action.id2 = o2 ? o2->id : -1;
    new_action.id3 = o3 ? o3->id : -1;
    new_action.flops = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogResizableArrayEnsureSize(stageLog->petsc_actions, new_num_actions, new_action));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndDefault(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscStageLog       stageLog = (PetscStageLog) ctx;
  PetscEventPerfLog   eventLog  = NULL;
  PetscEventPerfInfo *eventInfo = NULL;
  PetscLogDouble      time;
  int                 stage;

  PetscFunctionBegin;
  if (stageLog->petsc_logActions) {
    PetscLogDouble    curTime;
    PetscInt new_num_actions = ++stageLog->petsc_actions->num_entries;
    Action new_action;

    PetscCall(PetscTime(&curTime));
    new_action.time    = curTime - petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_END;
    new_action.event   = event;
    new_action.classid = state->registry->events->array[event].classid;
    new_action.id1 = o1 ? o1->id : -1;
    new_action.id2 = o2 ? o2->id : -2;
    new_action.id3 = o3 ? o3->id : -3;
    new_action.flops = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogResizableArrayEnsureSize(stageLog->petsc_actions, new_num_actions, new_action));
  }
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
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
  PetscCall(PetscTime(&time));
  PetscCall(PetscEventPerfInfoToc(eventInfo, time, stageLog->PetscLogMemory, (int) event));
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

static PetscErrorCode PetscLogEventBeginTrace(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
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
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
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

static PetscErrorCode PetscLogEventEndTrace(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
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
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
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

static PetscErrorCode PetscLogStagePush_Default(PetscLogState state, PetscLogStage new_stage, void *ctx)
{
  PetscStageLog stageLog = (PetscStageLog) ctx;
  PetscLogDouble time;
  int       curStage = state->current_stage;

  PetscFunctionBegin;
  if (state->registry->stages->num_entries > stageLog->num_entries) {
    PetscStageInfo empty_stage;
    PetscStageInfo *new_stage_info;
    PetscInt old_num_entries = stageLog->num_entries;

    PetscCall(PetscMemzero(&empty_stage, sizeof(empty_stage)));
    PetscCall(PetscLogResizableArrayEnsureSize(stageLog, state->registry->stages->num_entries, empty_stage));
    for (PetscInt s = old_num_entries; s < stageLog->num_entries; s++) {
      new_stage_info = &stageLog->array[s];
      PetscCall(PetscLogResizableArrayCreate(&(new_stage_info->classLog), state->registry->classes->max_entries));
      PetscCall(PetscLogResizableArrayCreate(&(new_stage_info->eventLog), state->registry->events->max_entries));
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
      if (perfstubs_initialized == PERFSTUBS_SUCCESS) PetscStackCallExternalVoid("ps_timer_create_", new_stage_info->timer = ps_timer_create_(state->registry->stages->array[s].name));
#endif
    }
  }
  PetscCall(PetscTime(&time));

  /* Record flops/time of previous stage */
  if (curStage >= 0) {
    if (PetscBTLookup(state->active, curStage)) {
      PetscCall(PetscEventPerfInfoToc(&stageLog->array[curStage].perfInfo, time, stageLog->PetscLogMemory, (int) -(curStage + 2)));
    }
  }
  stageLog->array[new_stage].used = PETSC_TRUE;
  stageLog->array[new_stage].perfInfo.count++;
  /* Subtract current quantities so that we obtain the difference when we pop */
  if (PetscBTLookup(state->active, new_stage)) {
    PetscCall(PetscEventPerfInfoTic(&stageLog->array[new_stage].perfInfo, time, stageLog->PetscLogMemory, (int) -(new_stage + 2)));
  }
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS && stageLog->array[new_stage].timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(stageLog->array[new_stage].timer));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePop_Default(PetscLogState state, PetscLogStage old_stage, void *ctx)
{
  PetscStageLog stageLog = (PetscStageLog) ctx;
  PetscInt curStage = state->current_stage;
  PetscLogDouble time;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS && stageLog->array[old_stage].timer != NULL) PetscStackCallExternalVoid("ps_timer_stop_", ps_timer_stop_(stageLog->array[old_stage].timer));
#endif
  PetscCall(PetscTime(&time));
  if (PetscBTLookup(state->active, old_stage)) {
    PetscCall(PetscEventPerfInfoToc(&stageLog->array[old_stage].perfInfo, time, stageLog->PetscLogMemory, (int) -(old_stage + 2)));
  }
  if (curStage >= 0) {
    if (PetscBTLookup(state->active, curStage)) {
      PetscCall(PetscEventPerfInfoTic(&stageLog->array[curStage].perfInfo, time, stageLog->PetscLogMemory, (int) -(curStage + 2)));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
