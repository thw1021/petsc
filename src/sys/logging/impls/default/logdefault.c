#include <petscviewer.h>
#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  #include <../src/sys/perfstubs/timer.h>
#endif
#include <petscconfiginfo.h>
#include <petscmachineinfo.h>
#include "logdefault.h"

typedef struct _PetscStageInfo {
  PetscBool          used;     /* The stage was pushed on this processor */
  PetscEventPerfInfo perfInfo; /* The stage performance information */
  PetscEventPerfLog  eventLog; /* The event information for this stage */
  PetscClassPerfLog  classLog; /* The class information for this stage */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer; /* Associated external tool timer for this stage */
#endif

} PetscStageInfo;

/* The structure for action logging */
typedef enum {
  PETSC_LOG_ACTION_CREATE,
  PETSC_LOG_ACTION_DESTROY,
  PETSC_LOG_ACTION_BEGIN,
  PETSC_LOG_ACTION_END,
} PetscLogActionType;

typedef struct _Action {
  PetscLogActionType action;        /* The type of execution */
  PetscLogEvent      event;         /* The event number */
  PetscClassId       classid;       /* The event class id */
  PetscLogDouble     time;          /* The time of occurrence */
  PetscLogDouble     flops;         /* The cumulative flops */
  PetscLogDouble     mem;           /* The current memory usage */
  PetscLogDouble     maxmem;        /* The maximum memory usage */
  int                id1, id2, id3; /* The ids of associated objects */
} Action;

/* The structure for object logging */
typedef struct _Object {
  PetscObject    obj;      /* The associated PetscObject */
  int            parent;   /* The parent id */
  PetscLogDouble mem;      /* The memory associated with the object */
  char           name[64]; /* The object name */
  char           info[64]; /* The information string */
} Object;

PETSC_LOG_RESIZABLE_ARRAY(Action, PetscActionLog)
PETSC_LOG_RESIZABLE_ARRAY(Object, PetscObjectLog)

#if defined(PETSC_HAVE_THREADSAFETY)
  /* Map from (threadid,stage,event) to perfInfo data struct */
  #include <petsc/private/hashmap.h>

typedef struct _PetscHashIJKKey {
  PetscInt i, j, k;
} PetscHashIJKKey;

  #define PetscHashIJKKeyHash(key)     PetscHashCombine(PetscHashInt((key).i), PetscHashCombine(PetscHashInt((key).j), PetscHashInt((key).k)))
  #define PetscHashIJKKeyEqual(k1, k2) (((k1).i == (k2).i) ? (((k1).j == (k2).j) ? ((k1).k == (k2).k) : 0) : 0)

PETSC_HASH_MAP(HMapEvent, PetscHashIJKKey, PetscEventPerfInfo *, PetscHashIJKKeyHash, PetscHashIJKKeyEqual, NULL)
#endif

typedef struct _n_PetscLogHandler_Default *PetscLogHandler_Default;
struct _n_PetscLogHandler_Default {
  int              max_entries;
  int              num_entries;
  PetscStageInfo   _default;
  PetscStageInfo  *array;
  PetscLogRegistry registry;
  PetscSpinlock    lock;
  PetscActionLog petsc_actions;
  PetscObjectLog petsc_objects;
  PetscBool petsc_logActions;
  PetscBool petsc_logObjects;
  int       petsc_numObjectsDestroyed;
  FILE          *petsc_tracefile;
  int            petsc_tracelevel;
  const char    *petsc_traceblanks;
  char           petsc_tracespace[128];
  PetscLogDouble petsc_tracetime;
  PetscBool      PetscLogSyncOn;
  PetscBool      PetscLogMemory;
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscHMapEvent eventInfoMap_th = NULL;
#endif
};

/* --- PetscClassPerfLog --- */

static PetscErrorCode PetscClassPerfInfoClear(PetscClassPerfInfo *classInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(classInfo, sizeof(*classInfo)));
  classInfo->id = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscClassPerfLogCreate(PetscClassPerfLog *classLog)
{
  PetscClassPerfInfo blank_entry;

  PetscFunctionBegin;
  PetscCall(PetscClassPerfInfoClear(&blank_entry));
  PetscCall(PetscLogResizableArrayCreate(classLog, 128, blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscClassPerfLogDestroy(PetscClassPerfLog classLog)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(classLog->array));
  PetscCall(PetscFree(classLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PETSC_UNUSED PetscErrorCode PetscClassPerfLogEnsureSize(PetscClassPerfLog classLog, int size)
{
  PetscFunctionBegin;
  PetscCall(PetscLogResizableArrayEnsureSize(classLog,size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- PetscEventPerfLog --- */

static PetscErrorCode PetscEventPerfInfoClear(PetscEventPerfInfo *eventInfo)
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

static PetscErrorCode PetscEventPerfLogCreate(PetscEventPerfLog *eventLog)
{
  PetscEventPerfInfo blank_entry;

  PetscFunctionBegin;
  PetscCall(PetscEventPerfInfoClear(&blank_entry));
  PetscCall(PetscLogResizableArrayCreate(eventLog, 128, blank_entry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventPerfLogDestroy(PetscEventPerfLog eventLog)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(eventLog->array));
  PetscCall(PetscFree(eventLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- PetscLogHandler_Default --- */

static PetscErrorCode PetscLogObjectCreate_Default(PetscLogState, PetscObject, void *);
static PetscErrorCode PetscLogObjectDestroy_Default(PetscLogState, PetscObject, void *);
static PetscErrorCode PetscLogEventSynchronize_Default(PetscLogState, PetscLogEvent, MPI_Comm, void *);
static PetscErrorCode PetscLogStagePush_Default(PetscLogState, PetscLogStage, void *);
static PetscErrorCode PetscLogStagePop_Default(PetscLogState, PetscLogStage, void *);
static PetscErrorCode PetscLogEventBegin_Default(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
static PetscErrorCode PetscLogEventEnd_Default(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
static PetscErrorCode PetscLogEventBegin_Trace(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
static PetscErrorCode PetscLogEventEnd_Trace(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
static PetscErrorCode PetscLogEventDeactivatePush_Default(PetscLogState, PetscLogEvent, void *);
static PetscErrorCode PetscLogEventDeactivatePop_Default(PetscLogState, PetscLogEvent, void *);

static PetscErrorCode PetscLogHandlerCreateContext_Default(PetscLogHandler_Default *default_handler_p)
{
  PetscStageInfo blank_stage;
  Action        blank_action;
  Object        blank_object;
  PetscLogHandler_Default default_handler;

  PetscFunctionBegin;
  PetscCall(PetscMemzero(&blank_stage, sizeof(blank_stage)));
  PetscCall(PetscMemzero(&blank_action, sizeof(blank_action)));
  PetscCall(PetscMemzero(&blank_object, sizeof(blank_object)));
  PetscCall(PetscLogResizableArrayCreate(default_handler_p, 8, blank_stage));
  default_handler = *default_handler_p;
  PetscCall(PetscLogResizableArrayCreate(&(default_handler->petsc_actions), 64, blank_action));
  PetscCall(PetscLogResizableArrayCreate(&(default_handler->petsc_objects), 64, blank_object));

  {
    PetscBool opt;

    PetscCall(PetscOptionsHasName(NULL, NULL, "-log_exclude_actions", &opt));
    if (opt) default_handler->petsc_logActions = PETSC_FALSE;
    PetscCall(PetscOptionsHasName(NULL, NULL, "-log_exclude_objects", &opt));
    if (opt) default_handler->petsc_logObjects = PETSC_FALSE;
  }
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  PetscStackCallExternalVoid("ps_initialize_", ps_initialize_());
#endif
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscHMapEventCreate(&default_handler->eventInfoMap_th));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroyContext_Default(void *ctx)
{
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) ctx;

  PetscFunctionBegin;
  for (int s = 0; s < default_handler->num_entries; s++) {
    PetscStageInfo *stage = &default_handler->array[s];

    PetscCall(PetscClassPerfLogDestroy(stage->classLog));
    PetscCall(PetscEventPerfLogDestroy(stage->eventLog));
  }
  PetscCall(PetscFree(default_handler->array));
  PetscCall(PetscFree(default_handler->petsc_actions->array));
  PetscCall(PetscFree(default_handler->petsc_actions));
  PetscCall(PetscFree(default_handler->petsc_objects->array));
  PetscCall(PetscFree(default_handler->petsc_objects));
#if defined(PETSC_HAVE_THREADSAFETY)
  if (default_handler->eventInfoMap_th) {
    PetscEventPerfInfo **array;
    PetscInt             n, off = 0;

    PetscCall(PetscHMapEventGetSize(default_handler->eventInfoMap_th, &n));
    PetscCall(PetscMalloc1(n, &array));
    PetscCall(PetscHMapEventGetVals(default_handler->eventInfoMap_th, &off, array));
    for (PetscInt i = 0; i < n; i++) PetscCall(PetscFree(array[i]));
    PetscCall(PetscFree(array));
    PetscCall(PetscHMapEventDestroy(&default_handler->eventInfoMap_th));
  }
#endif
  PetscCall(PetscFree(default_handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Default(PetscLogHandler *handler_p)
{
  PetscLogHandler     handler;
  PetscFunctionBegin;
  PetscCall(PetscNew(handler_p));
  handler = *handler_p;
  PetscCall(PetscNew(&handler->impl));
  PetscCall(PetscLogHandlerCreateContext_Default((PetscLogHandler_Default *) &handler->ctx));
  handler->object_create = PetscLogObjectCreate_Default;
  handler->object_destroy = PetscLogObjectDestroy_Default;
  handler->event_sync = PetscLogEventSynchronize_Default;
  handler->event_begin = PetscLogEventBegin_Default;
  handler->event_end = PetscLogEventEnd_Default;
  handler->impl->event_deactivate_pop = PetscLogEventDeactivatePush_Default;
  handler->impl->event_deactivate_push = PetscLogEventDeactivatePop_Default;
  handler->impl->stage_push = PetscLogStagePush_Default;
  handler->impl->stage_pop = PetscLogStagePop_Default;
  handler->impl->destroy = PetscLogHandlerDestroyContext_Default;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetTrace(PetscLogHandler handler, FILE *file)
{
  PetscLogHandler_Default   default_handler = (PetscLogHandler_Default) handler->ctx;

  PetscFunctionBegin;
  default_handler = (PetscLogHandler_Default) handler->ctx;
  handler->event_begin = PetscLogEventBegin_Trace;
  handler->event_end = PetscLogEventEnd_Trace;
  default_handler->petsc_tracefile = file;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDefaultGetEventPerfLog(PetscLogHandler_Default default_handler, PetscLogStage stage, PetscEventPerfLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= default_handler->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, default_handler->num_entries);
  *eventLog = default_handler->array[stage].eventLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscLogHandler handler, PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo **event_info)
{
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) handler->ctx;
  PetscEventPerfLog event_log;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerDefaultGetEventPerfLog(default_handler, stage, &event_log));
  *event_info = &event_log->array[stage];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDefaultGetClassPerfLog(PetscLogHandler_Default default_handler, int stage, PetscClassPerfLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= default_handler->num_entries), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, default_handler->num_entries);
  *classLog = default_handler->array[stage].classLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogObjectCreate_Default(PetscLogState state, PetscObject obj, void *ctx)
{
  PetscLogHandler_Default     default_handler = (PetscLogHandler_Default) ctx;
  PetscLogStage     stage;
  PetscLogRegistry  registry = state->registry;
  PetscClassPerfLog classPerfLog;
  int               oclass = 0;

  PetscFunctionBegin;
  /* Record stage info */
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscLogHandlerDefaultGetClassPerfLog(default_handler, stage, &classPerfLog));
  PetscCall(PetscClassRegLogGetClass(registry->classes, obj->classid, &oclass));
  classPerfLog->array[oclass].creations++;
  /* Dynamically enlarge logging structures */

  /* Record the creation action */
  if (default_handler->petsc_logActions) {
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
    PetscCall(PetscLogResizableArrayPush(default_handler->petsc_actions, new_action));
  }
  /* Record the object */
  if (default_handler->petsc_logObjects) {
    Object new_object;

    new_object.parent = -1;
    new_object.obj    = obj;

    PetscCall(PetscMemzero(new_object.name, sizeof(new_object.name)));
    PetscCall(PetscMemzero(new_object.info, sizeof(new_object.info)));
    PetscCall(PetscLogResizableArrayPush(default_handler->petsc_objects, new_object));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogObjectDestroy_Default(PetscLogState state, PetscObject obj, void *ctx)
{
  PetscLogRegistry  registry = state->registry;
  PetscLogHandler_Default     default_handler = (PetscLogHandler_Default) ctx;
  PetscLogStage     stage;
  PetscClassRegLog  classRegLog;
  PetscClassPerfLog classPerfLog;
  int               oclass = 0;

  PetscFunctionBegin;
  /* Record stage info */
  PetscCall(PetscSpinlockLock(&default_handler->lock));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  if (stage != -1) {
    /* That can happen if the log summary is output before some things are destroyed */
    classRegLog = registry->classes;
    PetscCall(PetscLogHandlerDefaultGetClassPerfLog(default_handler, stage, &classPerfLog));
    PetscCall(PetscClassRegLogGetClass(classRegLog, obj->classid, &oclass));
    classPerfLog->array[oclass].destructions++;
  }
  /* Cannot Credit all ancestors with your memory because they may have already been destroyed*/
  default_handler->petsc_numObjectsDestroyed++;
  /* Dynamically enlarge logging structures */
  /* Record the destruction action */
  if (default_handler->petsc_logActions) {
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
    PetscCall(PetscLogResizableArrayPush(default_handler->petsc_actions, new_action));
  }
  if (default_handler->petsc_logObjects) {
    if (obj->name) PetscCall(PetscStrncpy(default_handler->petsc_objects->array[obj->id].name, obj->name, 64));
    default_handler->petsc_objects->array[obj->id].obj = NULL;
  }
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventSynchronize_Default(PetscLogState state, PetscLogEvent event, MPI_Comm comm, void *ctx)
{
  PetscLogHandler_Default     default_handler = (PetscLogHandler_Default) ctx;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscLogDouble    time = 0.0;

  PetscFunctionBegin;
  if (!(default_handler->PetscLogSyncOn) || comm == MPI_COMM_NULL) PetscFunctionReturn(PETSC_SUCCESS);
  if (!state->registry->events->array[event].collective) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfLog(default_handler, stage, &eventLog));
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
static PetscErrorCode PetscLogGetStageEventPerfInfo_threaded(PetscLogHandlerDefault default_handler, PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo **eventInfo)
{
  PetscHashIJKKey     key;
  PetscEventPerfInfo *leventInfo;

  PetscFunctionBegin;
  key.i = PetscLogGetTid();
  key.j = stage;
  key.k = event;
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  PetscCall(PetscHMapEventGet(default_handler->eventInfoMap_th, key, &leventInfo));
  if (!leventInfo) {
    PetscCall(PetscNew(&leventInfo));
    PetscCall(PetscHMapEventSet(default_handler->eventInfoMap_th, key, leventInfo));
  }
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  *eventInfo = leventInfo;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode PetscLogEventBegin_Default(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscLogHandler_Default       default_handler = (PetscLogHandler_Default) ctx;
  PetscEventPerfLog   eventLog  = NULL;
  PetscEventPerfInfo *eventInfo = NULL;
  PetscLogDouble      time;
  int                 stage;

  PetscFunctionBegin;
  /* Synchronization */
  PetscCall(PetscLogEventSynchronize_Default(state, event, PetscObjectComm(o1), ctx));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfLog(default_handler, stage, &eventLog));
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
    regLog = default_handler->registry->events;
    if (regLog->array[event].timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(regLog->array[event].timer));
  }
#endif
  eventInfo->count++;
  PetscCall(PetscTime(&time));
  PetscCall(PetscEventPerfInfoTic(eventInfo, time, default_handler->PetscLogMemory, (int) event));
  if (default_handler->petsc_logActions) {
    PetscLogDouble curTime;
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
    PetscCall(PetscLogResizableArrayPush(default_handler->petsc_actions, new_action));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventEnd_Default(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscLogHandler_Default       default_handler = (PetscLogHandler_Default) ctx;
  PetscEventPerfLog   eventLog  = NULL;
  PetscEventPerfInfo *eventInfo = NULL;
  PetscLogDouble      time;
  int                 stage;

  PetscFunctionBegin;
  if (default_handler->petsc_logActions) {
    PetscLogDouble    curTime;
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
    PetscCall(PetscLogResizableArrayPush(default_handler->petsc_actions, new_action));
  }
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfLog(default_handler, stage, &eventLog));
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
  PetscCall(PetscEventPerfInfoToc(eventInfo, time, default_handler->PetscLogMemory, (int) event));
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

static PetscErrorCode PetscLogEventBegin_Trace(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscLogHandler_Default     default_handler = (PetscLogHandler_Default) ctx;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  PetscMPIInt       rank;
  int               stage;

  PetscFunctionBegin;
  if (!default_handler->petsc_tracetime) PetscCall(PetscTime(&default_handler->petsc_tracetime));
  default_handler->petsc_tracelevel++;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  eventRegLog = state->registry->events;
  PetscCall(PetscLogHandlerDefaultGetEventPerfLog(default_handler, stage, &eventPerfLog));
  /* Check for double counting */
  eventPerfLog->array[event].depth++;
  if (eventPerfLog->array[event].depth > 1) PetscFunctionReturn(PETSC_SUCCESS);
  /* Log performance info */
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, default_handler->petsc_tracefile, "%s[%d] %g Event begin: %s\n", default_handler->petsc_tracespace, rank, cur_time - default_handler->petsc_tracetime, eventRegLog->array[event].name));
  PetscCall(PetscStrncpy(default_handler->petsc_tracespace, default_handler->petsc_traceblanks, 2 * default_handler->petsc_tracelevel));
  default_handler->petsc_tracespace[2 * default_handler->petsc_tracelevel] = 0;
  PetscCall(PetscFFlush(default_handler->petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventEnd_Trace(PetscLogState state, PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscLogHandler_Default     default_handler = (PetscLogHandler_Default) ctx;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  int               stage;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  default_handler->petsc_tracelevel--;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  eventRegLog = state->registry->events;
  PetscCall(PetscLogHandlerDefaultGetEventPerfLog(default_handler, stage, &eventPerfLog));
  /* Check for double counting */
  eventPerfLog->array[event].depth--;
  if (eventPerfLog->array[event].depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  else PetscCheck(eventPerfLog->array[event].depth >= 0 && default_handler->petsc_tracelevel >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");

  /* Log performance info */
  if (default_handler->petsc_tracelevel) PetscCall(PetscStrncpy(default_handler->petsc_tracespace, default_handler->petsc_traceblanks, 2 * default_handler->petsc_tracelevel));
  default_handler->petsc_tracespace[2 * default_handler->petsc_tracelevel] = 0;
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, default_handler->petsc_tracefile, "%s[%d] %g Event end: %s\n", default_handler->petsc_tracespace, rank, cur_time - default_handler->petsc_tracetime, eventRegLog->array[event].name));
  PetscCall(PetscFFlush(default_handler->petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventDeactivatePush_Default(PetscLogState state, PetscLogEvent event, void *ctx)
{
  PetscLogStage stage;
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscEventPerfLogEnsureSize(default_handler->array[stage].eventLog, event + 1));
  default_handler->array[stage].eventLog->array[event].depth++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventDeactivatePop_Default(PetscLogState state, PetscLogEvent event, void *ctx)
{
  PetscLogStage stage;
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscEventPerfLogEnsureSize(default_handler->array[stage].eventLog, event + 1));
  default_handler->array[stage].eventLog->array[event].depth--;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePush_Default(PetscLogState state, PetscLogStage new_stage, void *ctx)
{
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) ctx;
  PetscLogDouble time;
  int       curStage = state->current_stage;

  PetscFunctionBegin;
  if (state->registry->stages->num_entries > default_handler->num_entries) {
    PetscStageInfo empty_stage;
    PetscStageInfo *new_stage_info;
    PetscInt old_num_entries = default_handler->num_entries;

    PetscCall(PetscMemzero(&empty_stage, sizeof(empty_stage)));
    PetscCall(PetscLogResizableArrayEnsureSize(default_handler, state->registry->stages->num_entries));
    for (PetscInt s = old_num_entries; s < default_handler->num_entries; s++) {
      new_stage_info = &default_handler->array[s];
      PetscCall(PetscClassPerfLogCreate(&(new_stage_info->classLog)));
      PetscCall(PetscEventPerfLogCreate(&(new_stage_info->eventLog)));
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
      if (perfstubs_initialized == PERFSTUBS_SUCCESS) PetscStackCallExternalVoid("ps_timer_create_", new_stage_info->timer = ps_timer_create_(state->registry->stages->array[s].name));
#endif
    }
  }
  PetscCall(PetscTime(&time));

  /* Record flops/time of previous stage */
  if (curStage >= 0) {
    if (PetscBTLookup(state->active, curStage)) {
      PetscCall(PetscEventPerfInfoToc(&default_handler->array[curStage].perfInfo, time, default_handler->PetscLogMemory, (int) -(curStage + 2)));
    }
  }
  default_handler->array[new_stage].used = PETSC_TRUE;
  default_handler->array[new_stage].perfInfo.count++;
  /* Subtract current quantities so that we obtain the difference when we pop */
  if (PetscBTLookup(state->active, new_stage)) {
    PetscCall(PetscEventPerfInfoTic(&default_handler->array[new_stage].perfInfo, time, default_handler->PetscLogMemory, (int) -(new_stage + 2)));
  }
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS && default_handler->array[new_stage].timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(default_handler->array[new_stage].timer));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePop_Default(PetscLogState state, PetscLogStage old_stage, void *ctx)
{
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) ctx;
  PetscInt curStage = state->current_stage;
  PetscLogDouble time;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS && default_handler->array[old_stage].timer != NULL) PetscStackCallExternalVoid("ps_timer_stop_", ps_timer_stop_(default_handler->array[old_stage].timer));
#endif
  PetscCall(PetscTime(&time));
  if (PetscBTLookup(state->active, old_stage)) {
    PetscCall(PetscEventPerfInfoToc(&default_handler->array[old_stage].perfInfo, time, default_handler->PetscLogMemory, (int) -(old_stage + 2)));
  }
  if (curStage >= 0) {
    if (PetscBTLookup(state->active, curStage)) {
      PetscCall(PetscEventPerfInfoTic(&default_handler->array[curStage].perfInfo, time, default_handler->PetscLogMemory, (int) -(curStage + 2)));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogActions(PetscLogHandler handler, PetscBool flag)
{
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) handler->ctx;

  PetscFunctionBegin;
  default_handler->petsc_logActions = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogObjects(PetscLogHandler handler, PetscBool flag)
{
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) handler->ctx;

  PetscFunctionBegin;
  default_handler->petsc_logObjects = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultLogObjectState(PetscLogHandler handler, PetscObject obj, const char format[], va_list Argp)
{
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) handler->ctx;
  size_t  fullLength;

  PetscFunctionBegin;
  if (default_handler->petsc_logObjects) {
    PetscCall(PetscVSNPrintf(default_handler->petsc_objects->array[obj->id].info, 64, format, &fullLength, Argp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetNumObjects(PetscLogHandler handler, PetscInt *num_objects)
{
  PetscLogHandler_Default default_handler = (PetscLogHandler_Default) handler->ctx;

  PetscFunctionBegin;
  *num_objects = default_handler->petsc_objects->num_entries;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogDump - Dumps logs of objects to a file. This file is intended to
  be read by bin/petscview. This program no longer exists.

  Collective on `PETSC_COMM_WORLD`

  Input Parameter:
. name - an optional file name

  Usage:
.vb
     PetscInitialize(...);
     PetscLogDefaultBegin(); or PetscLogAllBegin();
     ... code ...
     PetscLogDump(filename);
     PetscFinalize();
.ve

  Level: advanced

  Note:
  The default file name is Log.<rank> where <rank> is the MPI process rank. If no name is specified,
  this file will be used.

.seealso: [](ch_profiling), `PetscLogDefaultBegin()`, `PetscLogAllBegin()`, `PetscLogView()`
@*/
PetscErrorCode PetscLogDump_Default(PetscLogHandler handler, const char sname[])
{
  PetscLogHandler_Default       default_handler = (PetscLogHandler_Default) handler->ctx;
  PetscEventPerfInfo *eventInfo;
  FILE               *fd;
  char                file[PETSC_MAX_PATH_LEN], fname[PETSC_MAX_PATH_LEN];
  PetscLogDouble      flops, _TotalTime;
  PetscMPIInt         rank;
  int                 curStage;
  PetscLogState       state;
  PetscLogEvent       event;

  PetscFunctionBegin;
  /* Calculate the total elapsed time */
  PetscCall(PetscTime(&_TotalTime));
  _TotalTime -= petsc_BaseTime;
  /* Open log file */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscSNPrintf(file, PETSC_STATIC_ARRAY_LENGTH(file), "%s.%d", sname && sname[0] ? sname : "Log", rank));
  PetscCall(PetscFixFilename(file, fname));
  PetscCall(PetscFOpen(PETSC_COMM_SELF, fname, "w", &fd));
  PetscCheck(!(rank == 0) || !(!fd), PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open file: %s", fname);
  /* Output totals */
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Total Flop %14e %16.8e\n", petsc_TotalFlops, _TotalTime));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Clock Resolution %g\n", 0.0));
  /* Output actions */
  if (default_handler->petsc_logActions) {
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Actions accomplished %d\n", default_handler->petsc_actions->num_entries));
    for (int a = 0; a < default_handler->petsc_actions->num_entries; a++) {
      Action *action = &default_handler->petsc_actions->array[a];
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "%g %d %d %d %d %d %d %g %g %g\n", action->time, action->action, (int)action->event, (int)action->classid, action->id1,
                             action->id2, action->id3, action->flops, action->mem, action->maxmem));
    }
  }
  /* Output objects */
  if (default_handler->petsc_logObjects) {
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Objects created %d destroyed %d\n", default_handler->petsc_objects->num_entries, default_handler->petsc_numObjectsDestroyed));
    for (int o = 0; o < default_handler->petsc_objects->num_entries; o++) {
      Object *object = &default_handler->petsc_objects->array[o];
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Parent ID: %d Memory: %d\n", object->parent, (int)object->mem));
      if (!object->name[0]) {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "No Name\n"));
      } else {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Name: %s\n", object->name));
      }
      if (object->info[0] != 0) {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "No Info\n"));
      } else {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Info: %s\n", object->info));
      }
    }
  }
  /* Output events */
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Event log:\n"));
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateGetCurrentStage(state, &curStage));
  eventInfo = default_handler->array[curStage].eventLog->array;
  for (event = 0; event < default_handler->array[curStage].eventLog->num_entries; event++) {
    if (eventInfo[event].time != 0.0) flops = eventInfo[event].flops / eventInfo[event].time;
    else flops = 0.0;
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "%d %16d %16g %16g %16g\n", event, eventInfo[event].count, eventInfo[event].flops, eventInfo[event].time, flops));
  }
  PetscCall(PetscFClose(PETSC_COMM_SELF, fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/*
  PetscLogView_Detailed - Each process prints the times for its own events

*/
static PetscErrorCode PetscLogView_Detailed(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogState      state;
  PetscLogHandler_Default      default_handler = (PetscLogHandler_Default) handler->ctx;
  PetscLogDouble     locTotalTime, numRed, maxMem;
  int                numStages, numEvents;
  MPI_Comm           comm = PetscObjectComm((PetscObject)viewer);
  PetscMPIInt        rank, size;
  PetscLogGlobalNames        global_stages, global_events;
  PetscEventPerfInfo zero_info;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Must preserve reduction count before we go on */
  numRed = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  /* Get the total elapsed time */
  PetscCall(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;
  PetscCall(PetscViewerASCIIPrintf(viewer, "size = %d\n", size));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalTimes = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalMessages = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalMessageLens = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalReductions = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalFlop = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalObjects = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalMemory = {}\n"));
  PetscCall(PetscLogRegistryCreateGlobalStageNames(comm, state->registry, &global_stages));
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, state->registry, &global_events));
  numStages = global_stages->count;
  numEvents = global_events->count;
  PetscCall(PetscMemzero(&zero_info, sizeof(zero_info)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Stages = {}\n"));
  for (PetscInt stage = 0; stage < numStages; stage++) {
    PetscInt stage_id = global_stages->global_to_local[stage];
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stages[\"%s\"] = {}\n", global_stages->names[stage]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stages[\"%s\"][\"summary\"] = {}\n", global_stages->names[stage]));
    for (PetscInt event = 0; event < numEvents; event++) {
      PetscInt            event_id  = global_events->global_to_local[event];
      PetscEventPerfInfo *eventInfo = &zero_info;
      PetscBool           is_zero   = PETSC_FALSE;

      if (event_id >= 0 && stage_id >= 0 && event_id < default_handler->array[stage_id].eventLog->num_entries) { eventInfo = &default_handler->array[stage_id].eventLog->array[event_id]; }
      PetscCall(PetscMemcmp(eventInfo, &zero_info, sizeof(zero_info), &is_zero));
      PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &is_zero, 1, MPIU_BOOL, MPI_LAND, comm));
      if (!is_zero) { PetscCall(PetscViewerASCIIPrintf(viewer, "Stages[\"%s\"][\"%s\"] = {}\n", global_stages->names[stage], global_events->names[event])); }
    }
  }
  PetscCall(PetscMallocGetMaximumUsage(&maxMem));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalTimes[%d] = %g\n", rank, locTotalTime));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalMessages[%d] = %g\n", rank, (petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct)));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalMessageLens[%d] = %g\n", rank, (petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len)));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalReductions[%d] = %g\n", rank, numRed));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalFlop[%d] = %g\n", rank, petsc_TotalFlops));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalObjects[%d] = %d\n", rank, default_handler->petsc_objects->num_entries));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalMemory[%d] = %g\n", rank, maxMem));
  PetscCall(PetscViewerFlush(viewer));
  for (PetscInt stage = 0; stage < numStages; stage++) {
    PetscInt            stage_id  = global_stages->global_to_local[stage];
    PetscEventPerfInfo *stageInfo = (stage_id >= 0) ? &default_handler->array[stage_id].perfInfo : &zero_info;
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Stages[\"%s\"][\"summary\"][%d] = {\"time\" : %g, \"numMessages\" : %g, \"messageLength\" : %g, \"numReductions\" : %g, \"flop\" : %g}\n", global_stages->names[stage], rank, stageInfo->time,
                                                 stageInfo->numMessages, stageInfo->messageLength, stageInfo->numReductions, stageInfo->flops));
    for (PetscInt event = 0; event < numEvents; event++) {
      PetscInt            event_id  = global_events->global_to_local[event];
      PetscEventPerfInfo *eventInfo = &zero_info;
      PetscBool           is_zero   = PETSC_FALSE;

      if (event_id >= 0 && stage_id >= 0 && event_id < default_handler->array[stage_id].eventLog->num_entries) { eventInfo = &default_handler->array[stage_id].eventLog->array[event_id]; }
      PetscCall(PetscMemcmp(eventInfo, &zero_info, sizeof(zero_info), &is_zero));
      PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &is_zero, 1, MPIU_BOOL, MPI_LAND, comm));
      if (!is_zero) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Stages[\"%s\"][\"%s\"][%d] = {\"count\" : %d, \"time\" : %g, \"syncTime\" : %g, \"numMessages\" : %g, \"messageLength\" : %g, \"numReductions\" : %g, \"flop\" : %g", global_stages->names[stage],
                                                     global_events->names[event], rank, eventInfo->count, eventInfo->time, eventInfo->syncTime, eventInfo->numMessages, eventInfo->messageLength, eventInfo->numReductions, eventInfo->flops));
        if (eventInfo->dof[0] >= 0.) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ", \"dof\" : ["));
          for (PetscInt d = 0; d < 8; ++d) {
            if (d > 0) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ", "));
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%g", eventInfo->dof[d]));
          }
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "]"));
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ", \"error\" : ["));
          for (PetscInt e = 0; e < 8; ++e) {
            if (e > 0) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ", "));
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%g", eventInfo->errors[e]));
          }
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "]"));
        }
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "}\n"));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));
  PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscLogView_CSV - Each process prints the times for its own events in Comma-Separated Value Format
*/
static PetscErrorCode PetscLogView_CSV(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogState      state;
  PetscLogHandler_Default      default_handler = (PetscLogHandler_Default) handler->ctx;
  PetscLogDouble     locTotalTime, maxMem;
  int                numStages, numEvents, stage, event;
  MPI_Comm           comm = PetscObjectComm((PetscObject)viewer);
  PetscMPIInt        rank, size;
  PetscLogGlobalNames        global_stages, global_events;
  PetscEventPerfInfo zero_info;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Must preserve reduction count before we go on */
  /* Get the total elapsed time */
  PetscCall(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;
  PetscCallMPI(MPI_Allreduce(&default_handler->num_entries, &numStages, 1, MPI_INT, MPI_MAX, comm));
  PetscCall(PetscMallocGetMaximumUsage(&maxMem));
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogRegistryCreateGlobalStageNames(comm, state->registry, &global_stages));
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, state->registry, &global_events));
  numStages = global_stages->count;
  numEvents = global_events->count;
  PetscCall(PetscMemzero(&zero_info, sizeof(zero_info)));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Stage Name,Event Name,Rank,Count,Time,Num Messages,Message Length,Num Reductions,FLOP,dof0,dof1,dof2,dof3,dof4,dof5,dof6,dof7,e0,e1,e2,e3,e4,e5,e6,e7,%d\n", size));
  PetscCall(PetscViewerFlush(viewer));
  for (stage = 0; stage < numStages; stage++) {
    PetscInt            stage_id  = global_stages->global_to_local[stage];
    PetscEventPerfInfo *stageInfo = (stage_id >= 0) ? &default_handler->array[stage_id].perfInfo : &zero_info;

    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%s,summary,%d,1,%g,%g,%g,%g,%g\n", global_stages->names[stage], rank, stageInfo->time, stageInfo->numMessages, stageInfo->messageLength, stageInfo->numReductions, stageInfo->flops));
    PetscCallMPI(MPI_Allreduce(&default_handler->array[stage].eventLog->num_entries, &numEvents, 1, MPI_INT, MPI_MAX, comm));
    for (event = 0; event < numEvents; event++) {
      PetscInt            event_id  = global_events->global_to_local[event];
      PetscEventPerfInfo *eventInfo = &zero_info;
      PetscBool           is_zero   = PETSC_FALSE;

      if (event_id >= 0 && stage_id >= 0 && event_id < default_handler->array[stage_id].eventLog->num_entries) { eventInfo = &default_handler->array[stage_id].eventLog->array[event_id]; }
      PetscCall(PetscMemcmp(eventInfo, &zero_info, sizeof(zero_info), &is_zero));
      PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &is_zero, 1, MPIU_BOOL, MPI_LAND, comm));
      if (!is_zero) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%s,%s,%d,%d,%g,%g,%g,%g,%g", global_stages->names[stage], global_events->names[event], rank, eventInfo->count, eventInfo->time, eventInfo->numMessages, eventInfo->messageLength, eventInfo->numReductions,
                                                     eventInfo->flops));
        if (eventInfo->dof[0] >= 0.) {
          PetscInt d, e;

          for (d = 0; d < 8; ++d) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ",%g", eventInfo->dof[d]));
          for (e = 0; e < 8; ++e) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ",%g", eventInfo->errors[e]));
        }
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogViewWarnSync(MPI_Comm comm, FILE *fd)
{
  PetscFunctionBegin;
  if (!PetscLogSyncOn) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFPrintf(comm, fd, "\n\n"));
  PetscCall(PetscFPrintf(comm, fd, "      ##########################################################\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                       WARNING!!!                       #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   This program was run with logging synchronization.   #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   This option provides more meaningful imbalance       #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   figures at the expense of slowing things down and    #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   providing a distorted view of the overall runtime.   #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      ##########################################################\n\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogViewWarnDebugging(MPI_Comm comm, FILE *fd)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(PetscFPrintf(comm, fd, "\n\n"));
    PetscCall(PetscFPrintf(comm, fd, "      ##########################################################\n"));
    PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
    PetscCall(PetscFPrintf(comm, fd, "      #                       WARNING!!!                       #\n"));
    PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
    PetscCall(PetscFPrintf(comm, fd, "      #   This code was compiled with a debugging option.      #\n"));
    PetscCall(PetscFPrintf(comm, fd, "      #   To get timing results run ./configure                #\n"));
    PetscCall(PetscFPrintf(comm, fd, "      #   using --with-debugging=no, the performance will      #\n"));
    PetscCall(PetscFPrintf(comm, fd, "      #   be generally two or three times faster.              #\n"));
    PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
    PetscCall(PetscFPrintf(comm, fd, "      ##########################################################\n\n\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogViewWarnNoGpuAwareMpi(MPI_Comm comm, FILE *fd)
{
  #if defined(PETSC_HAVE_DEVICE)
  PetscMPIInt size;
  PetscBool   deviceInitialized = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  for (int i = PETSC_DEVICE_HOST + 1; i < PETSC_DEVICE_MAX; ++i) {
    const PetscDeviceType dtype = PetscDeviceTypeCast(i);
    if (PetscDeviceInitialized(dtype)) { /* a non-host device was initialized */
      deviceInitialized = PETSC_TRUE;
      break;
    }
  }
  /* the last condition says petsc is configured with device but it is a pure CPU run, so don't print misleading warnings */
  if (use_gpu_aware_mpi || size == 1 || !deviceInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFPrintf(comm, fd, "\n\n"));
  PetscCall(PetscFPrintf(comm, fd, "      ##########################################################\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                       WARNING!!!                       #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   This code was compiled with GPU support and you've   #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   created PETSc/GPU objects, but you intentionally     #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   used -use_gpu_aware_mpi 0, requiring PETSc to copy   #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   additional data between the GPU and CPU. To obtain   #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   meaningful timing results on multi-rank runs, use    #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   GPU-aware MPI instead.                               #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      ##########################################################\n\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
  #else
  return PETSC_SUCCESS;
  #endif
}

static PetscErrorCode PetscLogViewWarnGpuTime(MPI_Comm comm, FILE *fd)
{
  #if defined(PETSC_HAVE_DEVICE)

  PetscFunctionBegin;
  if (!PetscLogGpuTimeFlag || petsc_gflops == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFPrintf(comm, fd, "\n\n"));
  PetscCall(PetscFPrintf(comm, fd, "      ##########################################################\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                       WARNING!!!                       #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   This code was run with -log_view_gpu_time            #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   This provides accurate timing within the GPU kernels #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   but can slow down the entire computation by a        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   measurable amount. For fastest runs we recommend     #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #   not using this option.                               #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      #                                                        #\n"));
  PetscCall(PetscFPrintf(comm, fd, "      ##########################################################\n\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
  #else
  return PETSC_SUCCESS;
  #endif
}

static PetscErrorCode PetscLogView_Default_Info(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogState       state;
  FILE               *fd;
  PetscLogHandler_Default       default_handler = (PetscLogHandler_Default) handler;
  PetscClassRegLog    class_log;
  PetscEventRegLog    event_log;
  PetscStageInfo     *stageInfo = NULL;
  PetscClassPerfInfo *classInfo;
  char                arch[128], hostname[128], username[128], pname[PETSC_MAX_PATH_LEN], date[128];
  PetscLogDouble      locTotalTime, TotalTime, TotalFlops;
  PetscLogDouble      numMessages, messageLength, avgMessLen, numReductions;
  PetscLogDouble      stageTime, flops, flopr, mem, mess, messLen, red;
  PetscLogDouble      fracTime, fracFlops, fracMessages, fracLength, fracReductions, fracMess, fracMessLen, fracRed;
  PetscLogDouble      fracStageTime, fracStageFlops, fracStageMess, fracStageMessLen, fracStageRed;
  PetscLogDouble      min, max, tot, ratio, avg, x, y;
  PetscLogDouble      minf, maxf, totf, ratf, mint, maxt, tott, ratt, ratC, totm, totml, totr, mal, malmax, emalmax;
  #if defined(PETSC_HAVE_DEVICE)
  PetscLogEvent  KSP_Solve, SNES_Solve, TS_Step, TAO_Solve; /* These need to be fixed to be some events registered with certain objects */
  PetscLogDouble cct, gct, csz, gsz, gmaxt, gflops, gflopr, fracgflops;
  #endif
  PetscMPIInt   minC, maxC;
  PetscMPIInt   size, rank;
  PetscBool    *localStageUsed, *stageUsed;
  PetscBool    *localStageVisible, *stageVisible;
  int           numStages, numEvents;
  int           stage, oclass;
  PetscLogEvent event;
  char          version[256];
  MPI_Comm      comm;
  #if defined(PETSC_HAVE_DEVICE)
  PetscLogEvent eventid;
  PetscInt64    nas = 0x7FF0000000000002;
  #endif
  PetscLogGlobalNames        global_stages, global_events;
  PetscEventPerfInfo zero_info;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(PetscViewerASCIIGetPointer(viewer, &fd));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Get the total elapsed time */
  PetscCall(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;

  PetscCall(PetscFPrintf(comm, fd, "****************************************************************************************************************************************************************\n"));
  PetscCall(PetscFPrintf(comm, fd, "***                                WIDEN YOUR WINDOW TO 160 CHARACTERS.  Use 'enscript -r -fCourier9' to print this document                                 ***\n"));
  PetscCall(PetscFPrintf(comm, fd, "****************************************************************************************************************************************************************\n"));
  PetscCall(PetscFPrintf(comm, fd, "\n------------------------------------------------------------------ PETSc Performance Summary: ------------------------------------------------------------------\n\n"));
  PetscCall(PetscLogViewWarnSync(comm, fd));
  PetscCall(PetscLogViewWarnDebugging(comm, fd));
  PetscCall(PetscLogViewWarnNoGpuAwareMpi(comm, fd));
  PetscCall(PetscLogViewWarnGpuTime(comm, fd));
  PetscCall(PetscGetArchType(arch, sizeof(arch)));
  PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
  PetscCall(PetscGetUserName(username, sizeof(username)));
  PetscCall(PetscGetProgramName(pname, sizeof(pname)));
  PetscCall(PetscGetDate(date, sizeof(date)));
  PetscCall(PetscGetVersion(version, sizeof(version)));
  if (size == 1) {
    PetscCall(PetscFPrintf(comm, fd, "%s on a %s named %s with %d processor, by %s %s\n", pname, arch, hostname, size, username, date));
  } else {
    PetscCall(PetscFPrintf(comm, fd, "%s on a %s named %s with %d processors, by %s %s\n", pname, arch, hostname, size, username, date));
  }
  #if defined(PETSC_HAVE_OPENMP)
  PetscCall(PetscFPrintf(comm, fd, "Using %" PetscInt_FMT " OpenMP threads\n", PetscNumOMPThreads));
  #endif
  PetscCall(PetscFPrintf(comm, fd, "Using %s\n", version));

  /* Must preserve reduction count before we go on */
  red = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;

  /* Calculate summary information */
  PetscCall(PetscFPrintf(comm, fd, "\n                         Max       Max/Min     Avg       Total\n"));
  /*   Time */
  PetscCall(MPIU_Allreduce(&locTotalTime, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(&locTotalTime, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(MPIU_Allreduce(&locTotalTime, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscFPrintf(comm, fd, "Time (sec):           %5.3e   %7.3f   %5.3e\n", max, ratio, avg));
  TotalTime = tot;
  /*   Objects */
  avg = (PetscLogDouble)default_handler->petsc_objects->num_entries;
  PetscCall(MPIU_Allreduce(&avg, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(&avg, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(MPIU_Allreduce(&avg, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscFPrintf(comm, fd, "Objects:              %5.3e   %7.3f   %5.3e\n", max, ratio, avg));
  /*   Flops */
  PetscCall(MPIU_Allreduce(&petsc_TotalFlops, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(&petsc_TotalFlops, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(MPIU_Allreduce(&petsc_TotalFlops, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscFPrintf(comm, fd, "Flops:                %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  TotalFlops = tot;
  /*   Flops/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = petsc_TotalFlops / locTotalTime;
  else flops = 0.0;
  PetscCall(MPIU_Allreduce(&flops, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(&flops, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(MPIU_Allreduce(&flops, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscFPrintf(comm, fd, "Flops/sec:            %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  /*   Memory */
  PetscCall(PetscMallocGetMaximumUsage(&mem));
  if (mem > 0.0) {
    PetscCall(MPIU_Allreduce(&mem, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
    PetscCall(MPIU_Allreduce(&mem, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
    PetscCall(MPIU_Allreduce(&mem, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    avg = tot / ((PetscLogDouble)size);
    if (min != 0.0) ratio = max / min;
    else ratio = 0.0;
    PetscCall(PetscFPrintf(comm, fd, "Memory (bytes):       %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  }
  /*   Messages */
  mess = 0.5 * (petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  PetscCall(MPIU_Allreduce(&mess, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(&mess, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(MPIU_Allreduce(&mess, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscFPrintf(comm, fd, "MPI Msg Count:        %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  numMessages = tot;
  /*   Message Lengths */
  mess = 0.5 * (petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
  PetscCall(MPIU_Allreduce(&mess, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(&mess, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(MPIU_Allreduce(&mess, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  if (numMessages != 0) avg = tot / numMessages;
  else avg = 0.0;
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscFPrintf(comm, fd, "MPI Msg Len (bytes):  %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  messageLength = tot;
  /*   Reductions */
  PetscCall(MPIU_Allreduce(&red, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(&red, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(MPIU_Allreduce(&red, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscFPrintf(comm, fd, "MPI Reductions:       %5.3e   %7.3f\n", max, ratio));
  numReductions = red; /* wrong because uses count from process zero */
  PetscCall(PetscFPrintf(comm, fd, "\nFlop counting convention: 1 flop = 1 real number operation of type (multiply/divide/add/subtract)\n"));
  PetscCall(PetscFPrintf(comm, fd, "                            e.g., VecAXPY() for real vectors of length N --> 2N flops\n"));
  PetscCall(PetscFPrintf(comm, fd, "                            and VecAXPY() for complex vectors of length N --> 8N flops\n"));

  PetscCall(PetscLogGetEventLog(&event_log));
  PetscCall(PetscLogGetClassLog(&class_log));
  PetscCall(PetscLogRegistryCreateGlobalStageNames(comm, state->registry, &global_stages));
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, state->registry, &global_events));
  numStages = global_stages->count;
  numEvents = global_events->count;
  PetscCall(PetscMemzero(&zero_info, sizeof(zero_info)));
  PetscCall(PetscMalloc1(numStages, &localStageUsed));
  PetscCall(PetscMalloc1(numStages, &stageUsed));
  PetscCall(PetscMalloc1(numStages, &localStageVisible));
  PetscCall(PetscMalloc1(numStages, &stageVisible));
  if (numStages > 0) {
    stageInfo = default_handler->array;
    for (stage = 0; stage < numStages; stage++) {
      PetscInt stage_id = global_stages->global_to_local[stage];
      if (stage_id >= 0) {
        localStageUsed[stage]    = stageInfo[stage_id].used;
        localStageVisible[stage] = stageInfo[stage_id].perfInfo.visible;
      } else {
        localStageUsed[stage]    = PETSC_FALSE;
        localStageVisible[stage] = PETSC_TRUE;
      }
    }
    PetscCall(MPIU_Allreduce(localStageUsed, stageUsed, numStages, MPIU_BOOL, MPI_LOR, comm));
    PetscCall(MPIU_Allreduce(localStageVisible, stageVisible, numStages, MPIU_BOOL, MPI_LAND, comm));
    for (stage = 0; stage < numStages; stage++) {
      if (stageUsed[stage]) {
        PetscCall(PetscFPrintf(comm, fd, "\nSummary of Stages:   ----- Time ------  ----- Flop ------  --- Messages ---  -- Message Lengths --  -- Reductions --\n"));
        PetscCall(PetscFPrintf(comm, fd, "                        Avg     %%Total     Avg     %%Total    Count   %%Total     Avg         %%Total    Count   %%Total\n"));
        break;
      }
    }
    for (stage = 0; stage < numStages; stage++) {
      PetscInt            stage_id = global_stages->global_to_local[stage];
      PetscEventPerfInfo *stage_info;

      if (!stageUsed[stage]) continue;
      stage_info = localStageUsed[stage] ? &stageInfo[stage_id].perfInfo : &zero_info;
      PetscCall(MPIU_Allreduce(&stage_info->time, &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCall(MPIU_Allreduce(&stage_info->flops, &flops, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCall(MPIU_Allreduce(&stage_info->numMessages, &mess, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCall(MPIU_Allreduce(&stage_info->messageLength, &messLen, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCall(MPIU_Allreduce(&stage_info->numReductions, &red, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      mess *= 0.5;
      messLen *= 0.5;
      red /= size;
      if (TotalTime != 0.0) fracTime = stageTime / TotalTime;
      else fracTime = 0.0;
      if (TotalFlops != 0.0) fracFlops = flops / TotalFlops;
      else fracFlops = 0.0;
      /* Talk to Barry if (stageTime     != 0.0) flops          = (size*flops)/stageTime; else flops          = 0.0; */
      if (numMessages != 0.0) fracMessages = mess / numMessages;
      else fracMessages = 0.0;
      if (mess != 0.0) avgMessLen = messLen / mess;
      else avgMessLen = 0.0;
      if (messageLength != 0.0) fracLength = messLen / messageLength;
      else fracLength = 0.0;
      if (numReductions != 0.0) fracReductions = red / numReductions;
      else fracReductions = 0.0;
      PetscCall(PetscFPrintf(comm, fd, "%2d: %15s: %6.4e %5.1f%%  %6.4e %5.1f%%  %5.3e %5.1f%%  %5.3e      %5.1f%%  %5.3e %5.1f%%\n", stage, global_stages->names[stage], stageTime / size, 100.0 * fracTime, flops, 100.0 * fracFlops, mess, 100.0 * fracMessages, avgMessLen, 100.0 * fracLength, red, 100.0 * fracReductions));
    }
  }

  PetscCall(PetscFPrintf(comm, fd, "\n------------------------------------------------------------------------------------------------------------------------\n"));
  PetscCall(PetscFPrintf(comm, fd, "See the 'Profiling' chapter of the users' manual for details on interpreting output.\n"));
  PetscCall(PetscFPrintf(comm, fd, "Phase summary info:\n"));
  PetscCall(PetscFPrintf(comm, fd, "   Count: number of times phase was executed\n"));
  PetscCall(PetscFPrintf(comm, fd, "   Time and Flop: Max - maximum over all processors\n"));
  PetscCall(PetscFPrintf(comm, fd, "                  Ratio - ratio of maximum to minimum over all processors\n"));
  PetscCall(PetscFPrintf(comm, fd, "   Mess: number of messages sent\n"));
  PetscCall(PetscFPrintf(comm, fd, "   AvgLen: average message length (bytes)\n"));
  PetscCall(PetscFPrintf(comm, fd, "   Reduct: number of global reductions\n"));
  PetscCall(PetscFPrintf(comm, fd, "   Global: entire computation\n"));
  PetscCall(PetscFPrintf(comm, fd, "   Stage: stages of a computation. Set stages with PetscLogStagePush() and PetscLogStagePop().\n"));
  PetscCall(PetscFPrintf(comm, fd, "      %%T - percent time in this phase         %%F - percent flop in this phase\n"));
  PetscCall(PetscFPrintf(comm, fd, "      %%M - percent messages in this phase     %%L - percent message lengths in this phase\n"));
  PetscCall(PetscFPrintf(comm, fd, "      %%R - percent reductions in this phase\n"));
  PetscCall(PetscFPrintf(comm, fd, "   Total Mflop/s: 10e-6 * (sum of flop over all processors)/(max time over all processors)\n"));
  if (PetscLogMemory) {
    PetscCall(PetscFPrintf(comm, fd, "   Memory usage is summed over all MPI processes, it is given in mega-bytes\n"));
    PetscCall(PetscFPrintf(comm, fd, "   Malloc Mbytes: Memory allocated and kept during event (sum over all calls to event). May be negative\n"));
    PetscCall(PetscFPrintf(comm, fd, "   EMalloc Mbytes: extra memory allocated during event and then freed (maximum over all calls to events). Never negative\n"));
    PetscCall(PetscFPrintf(comm, fd, "   MMalloc Mbytes: Increase in high water mark of allocated memory (sum over all calls to event). Never negative\n"));
    PetscCall(PetscFPrintf(comm, fd, "   RMI Mbytes: Increase in resident memory (sum over all calls to event)\n"));
  }
  #if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscFPrintf(comm, fd, "   GPU Mflop/s: 10e-6 * (sum of flop on GPU over all processors)/(max GPU time over all processors)\n"));
  PetscCall(PetscFPrintf(comm, fd, "   CpuToGpu Count: total number of CPU to GPU copies per processor\n"));
  PetscCall(PetscFPrintf(comm, fd, "   CpuToGpu Size (Mbytes): 10e-6 * (total size of CPU to GPU copies per processor)\n"));
  PetscCall(PetscFPrintf(comm, fd, "   GpuToCpu Count: total number of GPU to CPU copies per processor\n"));
  PetscCall(PetscFPrintf(comm, fd, "   GpuToCpu Size (Mbytes): 10e-6 * (total size of GPU to CPU copies per processor)\n"));
  PetscCall(PetscFPrintf(comm, fd, "   GPU %%F: percent flops on GPU in this event\n"));
  #endif
  PetscCall(PetscFPrintf(comm, fd, "------------------------------------------------------------------------------------------------------------------------\n"));

  PetscCall(PetscLogViewWarnDebugging(comm, fd));

  /* Report events */
  PetscCall(PetscFPrintf(comm, fd, "Event                Count      Time (sec)     Flop                              --- Global ---  --- Stage ----  Total"));
  if (PetscLogMemory) PetscCall(PetscFPrintf(comm, fd, "  Malloc EMalloc MMalloc RMI"));
  #if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscFPrintf(comm, fd, "   GPU    - CpuToGpu -   - GpuToCpu - GPU"));
  #endif
  PetscCall(PetscFPrintf(comm, fd, "\n"));
  PetscCall(PetscFPrintf(comm, fd, "                   Max Ratio  Max     Ratio   Max  Ratio  Mess   AvgLen  Reduct  %%T %%F %%M %%L %%R  %%T %%F %%M %%L %%R Mflop/s"));
  if (PetscLogMemory) PetscCall(PetscFPrintf(comm, fd, " Mbytes Mbytes Mbytes Mbytes"));
  #if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscFPrintf(comm, fd, " Mflop/s Count   Size   Count   Size  %%F"));
  #endif
  PetscCall(PetscFPrintf(comm, fd, "\n"));
  PetscCall(PetscFPrintf(comm, fd, "------------------------------------------------------------------------------------------------------------------------"));
  if (PetscLogMemory) PetscCall(PetscFPrintf(comm, fd, "-----------------------------"));
  #if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscFPrintf(comm, fd, "---------------------------------------"));
  #endif
  PetscCall(PetscFPrintf(comm, fd, "\n"));

  #if defined(PETSC_HAVE_DEVICE)
  /* this indirect way of accessing these values is needed when PETSc is build with multiple libraries since the symbols are not in libpetscsys */
  PetscCall(PetscEventRegLogGetEvent(default_handler->eventLog, "TAOSolve", &TAO_Solve));
  PetscCall(PetscEventRegLogGetEvent(default_handler->eventLog, "TSStep", &TS_Step));
  PetscCall(PetscEventRegLogGetEvent(default_handler->eventLog, "SNESSolve", &SNES_Solve));
  PetscCall(PetscEventRegLogGetEvent(default_handler->eventLog, "KSPSolve", &KSP_Solve));
  #endif

  /* Problem: The stage name will not show up unless the stage executed on proc 1 */
  for (stage = 0; stage < numStages; stage++) {
    PetscInt            stage_id = global_stages->global_to_local[stage];
    PetscEventPerfInfo *stage_info;

    if (!stageVisible[stage]) continue;
    PetscCall(PetscFPrintf(comm, fd, "\n--- Event Stage %d: %s\n\n", stage, global_stages->names[stage]));
    stage_info = localStageUsed[stage] ? &stageInfo[stage_id].perfInfo : &zero_info;
    PetscCall(MPIU_Allreduce(&stage_info->time, &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscCall(MPIU_Allreduce(&stage_info->flops, &flops, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscCall(MPIU_Allreduce(&stage_info->numMessages, &mess, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscCall(MPIU_Allreduce(&stage_info->messageLength, &messLen, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscCall(MPIU_Allreduce(&stage_info->numReductions, &red, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    mess *= 0.5;
    messLen *= 0.5;
    red /= size;

    for (event = 0; event < numEvents; event++) {
      PetscInt event_id = global_events->global_to_local[event];

      PetscEventPerfInfo *event_info = &zero_info;
      PetscBool           is_zero    = PETSC_FALSE;

      if (event_id >= 0 && stage_id >= 0 && event_id < default_handler->array[stage_id].eventLog->num_entries) { event_info = &default_handler->array[stage_id].eventLog->array[event_id]; }
      PetscCall(PetscMemcmp(event_info, &zero_info, sizeof(zero_info), &is_zero));
      PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &is_zero, 1, MPIU_BOOL, MPI_LAND, comm));
      if (!is_zero) {
        flopr = event_info->flops;
        PetscCall(MPIU_Allreduce(&flopr, &minf, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
        PetscCall(MPIU_Allreduce(&flopr, &maxf, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
        PetscCall(MPIU_Allreduce(&event_info->flops, &totf, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&event_info->time, &mint, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
        PetscCall(MPIU_Allreduce(&event_info->time, &maxt, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
        PetscCall(MPIU_Allreduce(&event_info->time, &tott, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&event_info->numMessages, &totm, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&event_info->messageLength, &totml, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&event_info->numReductions, &totr, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&event_info->count, &minC, 1, MPI_INT, MPI_MIN, comm));
        PetscCall(MPIU_Allreduce(&event_info->count, &maxC, 1, MPI_INT, MPI_MAX, comm));
        if (PetscLogMemory) {
          PetscCall(MPIU_Allreduce(&event_info->memIncrease, &mem, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
          PetscCall(MPIU_Allreduce(&event_info->mallocSpace, &mal, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
          PetscCall(MPIU_Allreduce(&event_info->mallocIncrease, &malmax, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
          PetscCall(MPIU_Allreduce(&event_info->mallocIncreaseEvent, &emalmax, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        }
  #if defined(PETSC_HAVE_DEVICE)
        PetscCall(MPIU_Allreduce(&eventInfo->CpuToGpuCount, &cct, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&eventInfo->GpuToCpuCount, &gct, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&eventInfo->CpuToGpuSize, &csz, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&eventInfo->GpuToCpuSize, &gsz, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&eventInfo->GpuFlops, &gflops, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCall(MPIU_Allreduce(&eventInfo->GpuTime, &gmaxt, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  #endif
        if (mint < 0.0) {
          PetscCall(
            PetscFPrintf(comm, fd, "WARNING!!! Minimum time %g over all processors for %s is negative! This happens\n on some machines whose times cannot handle too rapid calls.!\n artificially changing minimum to zero.\n", mint, global_events->names[event]));
          mint = 0;
        }
        PetscCheck(minf >= 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Minimum flop %g over all processors for %s is negative! Not possible!", minf, global_events->names[event]);
        /* Put NaN into the time for all events that may not be time accurately since they may happen asynchronously on the GPU */
  #if defined(PETSC_HAVE_DEVICE)
        if (!PetscLogGpuTimeFlag && petsc_gflops > 0) {
          memcpy(&gmaxt, &nas, sizeof(PetscLogDouble));
          PetscCall(PetscEventRegLogGetEvent(default_handler->eventLog, name, &eventid));
          if (eventid != SNES_Solve && eventid != KSP_Solve && eventid != TS_Step && eventid != TAO_Solve) {
            memcpy(&mint, &nas, sizeof(PetscLogDouble));
            memcpy(&maxt, &nas, sizeof(PetscLogDouble));
          }
        }
  #endif
        totm *= 0.5;
        totml *= 0.5;
        totr /= size;

        if (maxC != 0) {
          if (minC != 0) ratC = ((PetscLogDouble)maxC) / minC;
          else ratC = 0.0;
          if (mint != 0.0) ratt = maxt / mint;
          else ratt = 0.0;
          if (minf != 0.0) ratf = maxf / minf;
          else ratf = 0.0;
          if (TotalTime != 0.0) fracTime = tott / TotalTime;
          else fracTime = 0.0;
          if (TotalFlops != 0.0) fracFlops = totf / TotalFlops;
          else fracFlops = 0.0;
          if (stageTime != 0.0) fracStageTime = tott / stageTime;
          else fracStageTime = 0.0;
          if (flops != 0.0) fracStageFlops = totf / flops;
          else fracStageFlops = 0.0;
          if (numMessages != 0.0) fracMess = totm / numMessages;
          else fracMess = 0.0;
          if (messageLength != 0.0) fracMessLen = totml / messageLength;
          else fracMessLen = 0.0;
          if (numReductions != 0.0) fracRed = totr / numReductions;
          else fracRed = 0.0;
          if (mess != 0.0) fracStageMess = totm / mess;
          else fracStageMess = 0.0;
          if (messLen != 0.0) fracStageMessLen = totml / messLen;
          else fracStageMessLen = 0.0;
          if (red != 0.0) fracStageRed = totr / red;
          else fracStageRed = 0.0;
          if (totm != 0.0) totml /= totm;
          else totml = 0.0;
          if (maxt != 0.0) flopr = totf / maxt;
          else flopr = 0.0;
          if (fracStageTime > 1.0 || fracStageFlops > 1.0 || fracStageMess > 1.0 || fracStageMessLen > 1.0 || fracStageRed > 1.0)
            PetscCall(PetscFPrintf(comm, fd, "%-16s %7d %3.1f %5.4e %3.1f %3.2e %3.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f Multiple stages %5.0f", global_events->names[event], maxC, ratC, maxt, ratt, maxf, ratf, totm, totml, totr, 100.0 * fracTime, 100.0 * fracFlops, 100.0 * fracMess, 100.0 * fracMessLen, 100.0 * fracRed, PetscAbs(flopr) / 1.0e6));
          else
            PetscCall(PetscFPrintf(comm, fd, "%-16s %7d %3.1f %5.4e %3.1f %3.2e %3.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f %3.0f %2.0f %2.0f %2.0f %2.0f %5.0f", global_events->names[event], maxC, ratC, maxt, ratt, maxf, ratf, totm, totml, totr, 100.0 * fracTime, 100.0 * fracFlops, 100.0 * fracMess, 100.0 * fracMessLen, 100.0 * fracRed, 100.0 * fracStageTime, 100.0 * fracStageFlops, 100.0 * fracStageMess, 100.0 * fracStageMessLen, 100.0 * fracStageRed, PetscAbs(flopr) / 1.0e6));
          if (PetscLogMemory) PetscCall(PetscFPrintf(comm, fd, " %5.0f   %5.0f   %5.0f   %5.0f", mal / 1.0e6, emalmax / 1.0e6, malmax / 1.0e6, mem / 1.0e6));
  #if defined(PETSC_HAVE_DEVICE)
          if (totf != 0.0) fracgflops = gflops / totf;
          else fracgflops = 0.0;
          if (gmaxt != 0.0) gflopr = gflops / gmaxt;
          else gflopr = 0.0;
          PetscCall(PetscFPrintf(comm, fd, "   %5.0f   %4.0f %3.2e %4.0f %3.2e % 2.0f", PetscAbs(gflopr) / 1.0e6, cct / size, csz / (1.0e6 * size), gct / size, gsz / (1.0e6 * size), 100.0 * fracgflops));
  #endif
          PetscCall(PetscFPrintf(comm, fd, "\n"));
        }
      }
    }
  }

  /* Memory usage and object creation */
  PetscCall(PetscFPrintf(comm, fd, "------------------------------------------------------------------------------------------------------------------------"));
  if (PetscLogMemory) PetscCall(PetscFPrintf(comm, fd, "-----------------------------"));
  #if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscFPrintf(comm, fd, "---------------------------------------"));
  #endif
  PetscCall(PetscFPrintf(comm, fd, "\n"));
  PetscCall(PetscFPrintf(comm, fd, "\n"));

  /* Right now, only stages on the first processor are reported here, meaning only objects associated with
     the global communicator, or MPI_COMM_SELF for proc 1. We really should report global stats and then
     stats for stages local to processor sets.
  */
  /* We should figure out the longest object name here (now 20 characters) */
  PetscCall(PetscFPrintf(comm, fd, "Object Type          Creations   Destructions. Reports information only for process 0.\n"));
  for (stage = 0; stage < numStages; stage++) {
    PetscCall(PetscFPrintf(comm, fd, "\n--- Event Stage %d: %s\n\n", stage, global_stages->names[stage]));
    if (localStageUsed[stage]) {
      classInfo = default_handler->array[stage].classLog->array;
      for (oclass = 0; oclass < default_handler->array[stage].classLog->num_entries; oclass++) {
        if ((classInfo[oclass].creations > 0) || (classInfo[oclass].destructions > 0)) {
          PetscCall(PetscFPrintf(comm, fd, "%20s %5d          %5d\n", class_log->array[oclass].name, classInfo[oclass].creations, classInfo[oclass].destructions));
        }
      }
    }
  }

  PetscCall(PetscFree(localStageUsed));
  PetscCall(PetscFree(stageUsed));
  PetscCall(PetscFree(localStageVisible));
  PetscCall(PetscFree(stageVisible));
  PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));

  /* Information unrelated to this particular run */
  PetscCall(PetscFPrintf(comm, fd, "========================================================================================================================\n"));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&x));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscFPrintf(comm, fd, "Average time to get PetscTime(): %g\n", (y - x) / 10.0));
  /* MPI information */
  if (size > 1) {
    MPI_Status  status;
    PetscMPIInt tag;
    MPI_Comm    newcomm;

    PetscCallMPI(MPI_Barrier(comm));
    PetscCall(PetscTime(&x));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCall(PetscTime(&y));
    PetscCall(PetscFPrintf(comm, fd, "Average time for MPI_Barrier(): %g\n", (y - x) / 5.0));
    PetscCall(PetscCommDuplicate(comm, &newcomm, &tag));
    PetscCallMPI(MPI_Barrier(comm));
    if (rank) {
      PetscCallMPI(MPI_Recv(NULL, 0, MPI_INT, rank - 1, tag, newcomm, &status));
      PetscCallMPI(MPI_Send(NULL, 0, MPI_INT, (rank + 1) % size, tag, newcomm));
    } else {
      PetscCall(PetscTime(&x));
      PetscCallMPI(MPI_Send(NULL, 0, MPI_INT, 1, tag, newcomm));
      PetscCallMPI(MPI_Recv(NULL, 0, MPI_INT, size - 1, tag, newcomm, &status));
      PetscCall(PetscTime(&y));
      PetscCall(PetscFPrintf(comm, fd, "Average time for zero size MPI_Send(): %g\n", (y - x) / size));
    }
    PetscCall(PetscCommDestroy(&newcomm));
  }
  PetscCall(PetscOptionsView(NULL, viewer));

  /* Machine and compile information */
  #if defined(PETSC_USE_FORTRAN_KERNELS)
  PetscCall(PetscFPrintf(comm, fd, "Compiled with FORTRAN kernels\n"));
  #else
  PetscCall(PetscFPrintf(comm, fd, "Compiled without FORTRAN kernels\n"));
  #endif
  #if defined(PETSC_USE_64BIT_INDICES)
  PetscCall(PetscFPrintf(comm, fd, "Compiled with 64-bit PetscInt\n"));
  #elif defined(PETSC_USE___FLOAT128)
  PetscCall(PetscFPrintf(comm, fd, "Compiled with 32-bit PetscInt\n"));
  #endif
  #if defined(PETSC_USE_REAL_SINGLE)
  PetscCall(PetscFPrintf(comm, fd, "Compiled with single precision PetscScalar and PetscReal\n"));
  #elif defined(PETSC_USE___FLOAT128)
  PetscCall(PetscFPrintf(comm, fd, "Compiled with 128 bit precision PetscScalar and PetscReal\n"));
  #endif
  #if defined(PETSC_USE_REAL_MAT_SINGLE)
  PetscCall(PetscFPrintf(comm, fd, "Compiled with single precision matrices\n"));
  #else
  PetscCall(PetscFPrintf(comm, fd, "Compiled with full precision matrices (default)\n"));
  #endif
  PetscCall(PetscFPrintf(comm, fd, "sizeof(short) %d sizeof(int) %d sizeof(long) %d sizeof(void*) %d sizeof(PetscScalar) %d sizeof(PetscInt) %d\n", (int)sizeof(short), (int)sizeof(int), (int)sizeof(long), (int)sizeof(void *), (int)sizeof(PetscScalar), (int)sizeof(PetscInt)));

  PetscCall(PetscFPrintf(comm, fd, "Configure options: %s", petscconfigureoptions));
  PetscCall(PetscFPrintf(comm, fd, "%s", petscmachineinfo));
  PetscCall(PetscFPrintf(comm, fd, "%s", petsccompilerinfo));
  PetscCall(PetscFPrintf(comm, fd, "%s", petsccompilerflagsinfo));
  PetscCall(PetscFPrintf(comm, fd, "%s", petsclinkerinfo));

  /* Cleanup */
  PetscCall(PetscFPrintf(comm, fd, "\n"));
  PetscCall(PetscLogViewWarnNoGpuAwareMpi(comm, fd));
  PetscCall(PetscLogViewWarnDebugging(comm, fd));
  PetscCall(PetscFPTrapPop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogView_Default(PetscLogHandler handler, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_ASCII_INFO) {
    PetscCall(PetscLogView_Default_Info(handler, viewer));
  } else if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscCall(PetscLogView_Detailed(handler, viewer));
  } else if (format == PETSC_VIEWER_ASCII_CSV) {
    PetscCall(PetscLogView_CSV(handler, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

