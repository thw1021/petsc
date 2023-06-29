
#include <petsc/private/logimpl.h> /*I "petsclog.h" I*/

/*S
   PetscLogState - Interface for the shared state information used by `PetscLogHandler`s.  It holds
   a registry of events (`PetscLogStateEventRegister()`), stages (`PetscLogStateStageRegiser()`), and
   classes (`PetscLogStateClassRegister()`).  It keeps track of when the use has activated
   events (`PetscLogStateEventSetActive()`) and stages (`PetscLogStateStageSetActive()`).  It
   also keeps a stack of running stages (`PetscLogStateStagePush()`, `PetscLogStateStagePop()`).

   Most users will not need to reference a `PetscLogState` directly: global logging routines
   like `PetscLogEventRegister()`  and `PetscLogStagePush()` implicitly manipulate PETSc's global
   logging state, `petsc_log_state`.

   Level: developer

.seealso: [](ch_profiling), `PetscLogStateCreate()`, `PetscLogStateDestroy()`
S*/

/*@
  PetscLogStateCreate - Create a logging state.

  Not collective

  Output Parameters:
. state - a `PetscLogState`

  Level: developer

  Note:

  Most users will not need to create a `PetscLogState`.  The global state `petsc_log_state`
  is created in `PetscInitialize()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateDestroy()`
@*/
PetscErrorCode PetscLogStateCreate(PetscLogState *state)
{
  PetscInt      num_entries, max_events, max_stages;
  PetscLogState s;

  PetscFunctionBegin;
  PetscCall(PetscNew(state));
  s = *state;
  PetscCall(PetscLogRegistryCreate(&s->registry));
  PetscCall(PetscIntStackCreate(&s->stage_stack));
  PetscCall(PetscLogRegistryGetNumEvents(s->registry, NULL, &max_events));
  PetscCall(PetscLogRegistryGetNumStages(s->registry, NULL, &max_stages));

  s->bt_num_events = max_events + 1; // one extra column for default stage activity
  s->bt_num_stages = max_stages;
  num_entries          = s->bt_num_events * s->bt_num_stages;
  PetscCall(PetscBTCreate(num_entries, &s->active));
  s->current_stage = -1;
  s->refct         = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateDestroy - Destroy a logging state.

  Not collective

  Input Parameters:
. state - a `PetscLogState`

  Level: developer

  Note:

  Most users will not need to destroy a `PetscLogState`.  The global state `petsc_log_state`
  is destroyed in `PetscFinalize()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateCreate()`
@*/
PetscErrorCode PetscLogStateDestroy(PetscLogState *state)
{
  PetscLogState s;
  PetscFunctionBegin;
  s = *state;
  *state = NULL;
  if (s == NULL || --(s->refct) > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogRegistryDestroy(s->registry));
  PetscCall(PetscIntStackDestroy(s->stage_stack));
  PetscCall(PetscBTDestroy(&s->active));
  PetscCall(PetscFree(s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStagePush - Start a new logging stage.

  Not collective

  Input Parameters:
- state - a `PetscLogState`
+ stage - a registered `PetscLogStage`

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogStagePush()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`, `PetscLogStateStagePop()`, `PetscLogStateGetCurrentStage()`
@*/
PetscErrorCode PetscLogStateStagePush(PetscLogState state, PetscLogStage stage)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscInt num_stages;
    PetscCall(PetscLogRegistryGetNumStages(state->registry, &num_stages, NULL));
    PetscCheck(stage >= 0 && stage < num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d not in [0,%d)", stage, (int)num_stages);
  }
  PetscCall(PetscIntStackPush(state->stage_stack, stage));
  state->current_stage = stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStagePop - End a running logging stage.

  Not collective

  Input Parameter:
. state - a `PetscLogState`

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogStagePush()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`, `PetscLogStateStagePush()`, `PetscLogStateGetCurrentStage()`
@*/
PetscErrorCode PetscLogStateStagePop(PetscLogState state)
{
  int       curStage;
  PetscBool empty;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(state->stage_stack, &curStage));
  PetscCall(PetscIntStackEmpty(state->stage_stack, &empty));
  if (!empty) {
    PetscCall(PetscIntStackTop(state->stage_stack, &state->current_stage));
  } else state->current_stage = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetCurrentStage - Get the last stage that was started

  Not collective

  Input Parameter:
. state - a `PetscLogState`

  Output Parameter:
. current - the last `PetscLogStage` started with `PetscLogStateStagePop()`

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogGetCurrentStage()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`, `PetscLogStateStagePush()`, `PetscLogStateStagePop()`
@*/
PetscErrorCode PetscLogStateGetCurrentStage(PetscLogState state, PetscLogStage *current)
{
  PetscFunctionBegin;
  *current = state->current_stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStateResize(PetscLogState state)
{
  PetscBT  active_new;
  PetscInt new_num_events;
  PetscInt new_num_stages;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, NULL, &new_num_events));
  new_num_events++;
  PetscCall(PetscLogRegistryGetNumStages(state->registry, NULL, &new_num_stages));

  if (state->bt_num_events == new_num_events && state->bt_num_stages == new_num_stages) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck((new_num_stages % PETSC_BITS_PER_BYTE) == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "new number of stages must be multiple of %d", PETSC_BITS_PER_BYTE);
  PetscCall(PetscBTCreate(new_num_events * new_num_stages, &active_new));
  if (new_num_stages == state->bt_num_stages) {
    // single memcpy
    size_t num_chars = (state->bt_num_stages * state->bt_num_events) / PETSC_BITS_PER_BYTE;

    PetscCall(PetscMemcpy(active_new, state->active, num_chars));
  } else {
    size_t num_chars_old = state->bt_num_stages / PETSC_BITS_PER_BYTE;
    size_t num_chars_new = new_num_stages / PETSC_BITS_PER_BYTE;

    for (PetscInt i = 0; i < state->bt_num_events; i++) { PetscCall(PetscMemcpy(&active_new[i * num_chars_new], &(state->active[i * num_chars_old]), num_chars_old)); }
  }
  PetscCall(PetscBTDestroy(&state->active));
  state->active        = active_new;
  state->bt_num_events = new_num_events;
  state->bt_num_stages = new_num_stages;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStageRegister - Register a new stage with a logging state

  Not collective

  Input parameters:
+ state - a `PetscLogState`
- sname - a unique name

  Output parameter:
. stage - the identifier for the registered stage

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogStageRegister()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStagePush()`, `PetscLogStateStagePop()`
@*/
PetscErrorCode PetscLogStateStageRegister(PetscLogState state, const char sname[], PetscLogStage *stage)
{
  PetscInt s;
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryStageRegister(state->registry, sname, stage));
  PetscCall(PetscLogStateResize(state));
  s = *stage;
  PetscCall(PetscBTSet(state->active, s)); // stages are by default active
  for (PetscInt e = 1; e < state->bt_num_events; e++) {
    // copy "Main Stage" activities
    if (PetscBTLookup(state->active, 0 + e * state->bt_num_stages)) {
      PetscCall(PetscBTSet(state->active, s + e * state->bt_num_stages));
    } else {
      PetscCall(PetscBTClear(state->active, s + e * state->bt_num_stages));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventRegister - Register a new event with a logging state

  Not collective

  Input parameters:
+ state - a `PetscLogState`
. sname - a unique name
- id - the `PetscClassId` for the type of object most closely associated with this event

  Output parameter:
. event - the identifier for the registered event

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogEventRegister()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStageRegister()`
@*/
PetscErrorCode PetscLogStateEventRegister(PetscLogState state, const char sname[], PetscClassId id, PetscLogEvent *event)
{
  PetscInt e;

  PetscFunctionBegin;
  *event = PETSC_DECIDE;
  PetscCall(PetscLogRegistryGetEventFromName(state->registry, sname, event));
  if (*event > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogRegistryEventRegister(state->registry, sname, id, event));
  PetscCall(PetscLogStateResize(state));
  e = *event;
  for (PetscInt s = 0; s < state->bt_num_stages; s++) PetscCall(PetscBTSet(state->active, s + (e + 1) * state->bt_num_stages)); // events are by default active
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStageSetActive - Mark a stage as active or inactive.

  Not collective

  Input parameters:
+ state - a `PetscLogState`
. stage - a registered `PetscLogStage`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return `PETSC_FALSE` for all events

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogStageSetActive()`

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateEventSetActive()`
@*/
PetscErrorCode PetscLogStateStageSetActive(PetscLogState state, PetscLogStage stage, PetscBool isActive)
{
  PetscFunctionBegin;
  PetscCheck(stage >= 0 && stage < state->bt_num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, state->bt_num_stages);
  if (isActive) {
    for (PetscInt e = 0; e < state->bt_num_events; e++) { PetscCall(PetscBTSet(state->active, stage + e * state->bt_num_stages)); }
  } else {
    for (PetscInt e = 0; e < state->bt_num_events; e++) { PetscCall(PetscBTClear(state->active, stage + e * state->bt_num_stages)); }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStageGetActive - Check if a logging stage is active or inactive.

  Not collective

  Input parameters:
+ state - a `PetscLogState`
. stage - a registered `PetscLogStage`

  Output parameter:
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return `PETSC_FALSE` for all events

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogStageGetActive()`. 

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStageSetActive()`, `PetscLogHandler`, `PetscLogHandlerStart()`, `PetscLogHandlerEventBegin()`, `PetscLogHandlerEventEnd()`
@*/
PetscErrorCode PetscLogStateStageGetActive(PetscLogState state, PetscLogStage stage, PetscBool *isActive)
{
  PetscFunctionBegin;
  *isActive = PetscBTLookup(state->active, stage) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventSetActive - Set a logging event as active or inactive during a logging stage.

  Not collective

  Input parameters:
+ state - a `PetscLogState`
. stage - a registered `PetscLogStage`, or `PETSC_DEFAULT` for the current stage
. event - a registered `PetscLogEvent`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return `PETSC_FALSE` for this stage and this event

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogEventSetActive()`. 

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`, `PetscLogStateGetCurrentStage()`, `PetscLogEventSetActiveAll()`
@*/
PetscErrorCode PetscLogStateEventSetActive(PetscLogState state, PetscLogStage stage, PetscLogEvent event, PetscBool isActive)
{
  PetscFunctionBegin;
  stage = (stage < 0) ? state->current_stage : stage;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  PetscCheck(stage >= 0 && stage < state->bt_num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", event, state->bt_num_stages);
  PetscCall((isActive ? PetscBTSet : PetscBTClear)(state->active, state->current_stage + (event + 1) * state->bt_num_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventSetActiveAll - Set logging event as active or inactive for all logging stages

  Not collective

  Input parameters:
+ state - a `PetscLogState`
. stage - a registered `PetscLogStage`, or `PETSC_DEFAULT` for the current stage
. event - a registered `PetscLogEvent`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return `PETSC_FALSE` for all stages and this event

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogEventSetActiveAll()`. 

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`
@*/
PetscErrorCode PetscLogStateEventSetActiveAll(PetscLogState state, PetscLogEvent event, PetscBool isActive)
{
  PetscFunctionBegin;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  for (int stage = 0; stage < state->bt_num_stages; stage++) {
    PetscCall((isActive ? PetscBTSet : PetscBTClear)(state->active, state->current_stage + (event + 1) * state->bt_num_stages));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventSetActive - Set a logging event as active or inactive during a logging stage.

  Not collective

  Input parameters:
+ state - a `PetscLogState`
. classid - a `PetscClassId`
. event - a registered `PetscLogEvent`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return `PETSC_FALSE` for this stage and this event

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogEventSetActive()`. 

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`, `PetscLogStateGetCurrentStage()`
@*/
PetscErrorCode PetscLogStateClassSetActiveAll(PetscLogState state, PetscClassId classid, PetscBool isActive)
{
  PetscInt num_events, num_stages;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, &num_events, NULL));
  PetscCall(PetscLogRegistryGetNumStages(state->registry, &num_stages, NULL));
  for (PetscLogEvent e = 0; e < num_events; e++) {
    PetscLogEventInfo event_info;
    PetscCall(PetscLogRegistryEventGetInfo(state->registry, e, &event_info));
    if (event_info.classid == classid) {
      for (PetscLogStage s = 0; s < num_stages; s++) { PetscCall(PetscBTSet(state->active, s + (e + 1) * state->bt_num_stages)); }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateEventExcludeClass(PetscLogState state, PetscClassId classid)
{
  PetscInt num_events, num_stages;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, &num_events, NULL));
  PetscCall(PetscLogRegistryGetNumStages(state->registry, &num_stages, NULL));
  for (PetscLogEvent e = 0; e < num_events; e++) {
    PetscLogEventInfo event_info;
    PetscCall(PetscLogRegistryEventGetInfo(state->registry, e, &event_info));
    if (event_info.classid == classid) {
      for (PetscLogStage s = 0; s < num_stages; s++) { PetscCall(PetscBTClear(state->active, s + (e + 1) * state->bt_num_stages)); }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateEventActivateClass(PetscLogState state, PetscClassId classid)
{
  PetscLogStage stage = state->current_stage;
  PetscInt      num_events;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, &num_events, NULL));
  for (PetscLogEvent e = 0; e < num_events; e++) {
    PetscLogEventInfo event_info;
    PetscCall(PetscLogRegistryEventGetInfo(state->registry, e, &event_info));
    if (event_info.classid == classid) PetscCall(PetscBTSet(state->active, stage + (e + 1) * state->bt_num_stages));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateEventDeactivateClass(PetscLogState state, PetscClassId classid)
{
  PetscLogStage stage = state->current_stage;
  PetscInt      num_events;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, &num_events, NULL));
  for (PetscLogEvent e = 0; e < num_events; e++) {
    PetscLogEventInfo event_info;
    PetscCall(PetscLogRegistryEventGetInfo(state->registry, e, &event_info));
    if (event_info.classid == classid) PetscCall(PetscBTClear(state->active, stage + (e + 1) * state->bt_num_stages));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventGetActive - Check if a logging event is active or inactive during a logging stage.

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
. stage - a registered `PetscLogStage`, or `PETSC_DEFAULT` for the current stage
. event - a registered `PetscLogEvent`

  Return Paramter:
- isActive - `PETSC_TRUE` or `PETSC_FALSE`

  Level: developer

  Note:

  This is called for `petsc_log_state` in `PetscLogEventGetActive()`, where it has significance
  for what information is sent to log handlers.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`, `PetscLogStateGetCurrentStage()`, `PetscLogHandler()`
@*/
PetscErrorCode PetscLogStateEventGetActive(PetscLogState state, PetscLogStage stage, PetscLogEvent event, PetscBool *isActive)
{
  PetscFunctionBegin;
  stage = (stage < 0) ? state->current_stage : stage;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  PetscCheck(stage >= 0 && stage < state->bt_num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", event, state->bt_num_stages);
  *isActive = PetscLogStateStageEventActive(state, stage, event);
  PetscFunctionReturn(PETSC_SUCCESS);
}

