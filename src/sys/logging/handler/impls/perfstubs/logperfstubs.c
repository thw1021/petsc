#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>
#include <../src/sys/perfstubs/timer.h>

typedef struct _n_PetscEventPS {
  void *timer;
  int depth;
} PetscEventPS;

PETSC_LOG_RESIZABLE_ARRAY(PSArray, PetscEventPS, void *, NULL, NULL, NULL);

typedef struct _n_PetscLogHandler_Perfstubs *PetscLogHandler_Perfstubs;

struct _n_PetscLogHandler_Perfstubs {
  PetscLogPSArray events;
  PetscLogPSArray stages;
};

static PetscErrorCode PetscLogHandlerContextCreate_Perfstubs(PetscLogHandler_Perfstubs *ps_p)
{
  PetscLogHandler_Perfstubs ps;

  PetscFunctionBegin;
  PetscCall(PetscNew(ps_p));
  ps = *ps_p;
  PetscCall(PetscLogPSArrayCreate(128, &ps->events));
  PetscCall(PetscLogPSArrayCreate(8, &ps->stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_Perfstubs(PetscLogHandler h)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs) h->ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogPSArrayDestroy(&ps->events));
  PetscCall(PetscLogPSArrayDestroy(&ps->stages));
  PetscCall(PetscFree(ps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerPSUpdateEvents(PetscLogHandler h)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs) h->ctx;
  PetscLogState state;
  PetscInt      num_events, num_events_old;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetNumEvents(state, &num_events));
  PetscCall(PetscLogPSArrayGetSize(ps->events, &num_events_old, NULL));
  for (PetscInt i = num_events_old; i < num_events; i++) {
    PetscLogEventInfo event_info;
    PetscEventPS      ps_event;

    PetscCall(PetscLogStateEventGetInfo(state, (PetscLogEvent)i, &event_info));
    PetscStackCallExternalVoid("ps_timer_create_", ps_event.timer = ps_timer_create_(event_info.name));
    ps_event.depth = 0;
    PetscCall(PetscLogPSArrayPush(ps->events, ps_event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerPSUpdateStages(PetscLogHandler h)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs) h->ctx;
  PetscLogState state;
  PetscInt      num_stages, num_stages_old;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetNumStages(state, &num_stages));
  PetscCall(PetscLogPSArrayGetSize(ps->stages, &num_stages_old, NULL));
  for (PetscInt i = num_stages_old; i < num_stages; i++) {
    PetscLogStageInfo stage_info;
    PetscEventPS      ps_stage;

    PetscCall(PetscLogStateStageGetInfo(state, (PetscLogStage)i, &stage_info));
    PetscStackCallExternalVoid("ps_timer_create_", ps_stage.timer = ps_timer_create_(stage_info.name));
    ps_stage.depth = 0;
    PetscCall(PetscLogPSArrayPush(ps->stages, ps_stage));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventBegin_Perfstubs(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs) handler->ctx;
  PetscEventPS     ps_event = {NULL, 0};

  PetscFunctionBegin;
  if (event >= ps->events->num_entries) PetscCall(PetscLogHandlerPSUpdateEvents(handler));
  PetscCall(PetscLogPSArrayGet(ps->events, event, &ps_event));
  ps_event.depth++;
  PetscCall(PetscLogPSArraySet(ps->events, event, ps_event));
  if (ps_event.depth == 1 && ps_event.timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(ps_event.timer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_Perfstubs(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs) handler->ctx;
  PetscEventPS     ps_event = {NULL, 0};

  PetscFunctionBegin;
  if (event >= ps->events->num_entries) PetscCall(PetscLogHandlerPSUpdateEvents(handler));
  PetscCall(PetscLogPSArrayGet(ps->events, event, &ps_event));
  ps_event.depth--;
  PetscCall(PetscLogPSArraySet(ps->events, event, ps_event));
  if (ps_event.depth == 0 && ps_event.timer != NULL) PetscStackCallExternalVoid("ps_timer_stop_", ps_timer_stop_(ps_event.timer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePush_Perfstubs(PetscLogHandler handler, PetscLogStage stage)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs) handler->ctx;
  PetscEventPS     ps_event = {NULL, 0};

  PetscFunctionBegin;
  if (stage >= ps->stages->num_entries) PetscCall(PetscLogHandlerPSUpdateStages(handler));
  PetscCall(PetscLogPSArrayGet(ps->stages, stage, &ps_event));
  if (ps_event.timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(ps_event.timer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePop_Perfstubs(PetscLogHandler handler, PetscLogStage stage)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs) handler->ctx;
  PetscEventPS     ps_event = {NULL, 0};

  PetscFunctionBegin;
  if (stage >= ps->stages->num_entries) PetscCall(PetscLogHandlerPSUpdateStages(handler));
  PetscCall(PetscLogPSArrayGet(ps->stages, stage, &ps_event));
  if (ps_event.timer != NULL) PetscStackCallExternalVoid("ps_timer_stop_", ps_timer_stop_(ps_event.timer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Perfstubs(MPI_Comm comm, PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  if (perfstubs_initialized == PERFSTUBS_UNKNOWN) PetscStackCallExternalVoid("ps_initialize_", ps_initialize_());
  PetscCall(PetscLogHandlerCreate(comm, handler_p));
  handler              = *handler_p;
  PetscCall(PetscLogHandlerContextCreate_Perfstubs((PetscLogHandler_Perfstubs *) &handler->ctx));
  handler->type        = PETSC_LOG_HANDLER_PERFSTUBS;
  handler->destroy     = PetscLogHandlerDestroy_Perfstubs;
  handler->eventBegin  = PetscLogHandlerEventBegin_Perfstubs;
  handler->eventEnd    = PetscLogHandlerEventEnd_Perfstubs;
  handler->stagePush   = PetscLogHandlerStagePush_Perfstubs;
  handler->stagePop    = PetscLogHandlerStagePop_Perfstubs;
  PetscFunctionReturn(PETSC_SUCCESS);
}
