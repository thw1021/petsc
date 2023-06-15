
#include <petsc/private/logimpl.h> /*I "petsclog.h" I*/

PETSC_INTERN PetscErrorCode PetscLogStateCreate(PetscLogState *state_p)
{
  PetscInt      num_active;
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscNew(state_p));
  state = *state_p;
  PetscCall(PetscLogRegistryCreate(&state->registry));

  PetscCall(PetscIntStackCreate(&state->stage_stack));
  state->bt_num_events = state->registry->events->max_entries + 1; // one extra column for default stage activity
  state->bt_num_stages = state->registry->stages->max_entries;
  num_active = state->bt_num_events * state->bt_num_stages;
  PetscCall(PetscBTCreate(num_active, &state->active));
  for (PetscInt i = 0; i < num_active; i++) PetscCall(PetscBTSet(state->active, i));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateDestroy(PetscLogState state)
{
  PetscFunctionBegin;
  PetscCall(PetscSpinlockDestroy(&state->lock));
  PetscCall(PetscLogRegistryDestroy(state->registry));
  PetscCall(PetscIntStackDestroy(state->stage_stack));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateStagePush(PetscLogState state, PetscLogStage stage)
{
  PetscFunctionBegin;
  /* Activate the stage */
  PetscCall(PetscLogStateLock(state));
  PetscCall(PetscIntStackPush(state->stage_stack, stage));
  state->current_stage = stage;
  PetscCall(PetscLogStateUnlock(state));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateStagePop(PetscLogState state)
{
  int       curStage;
  PetscBool empty;

  PetscFunctionBegin;
  /* Record flops/time of current stage */
  PetscCall(PetscLogStateLock(state));
  PetscCall(PetscIntStackPop(state->stage_stack, &curStage));
  PetscCall(PetscIntStackEmpty(state->stage_stack, &empty));
  if (!empty) {
    /* Subtract current quantities so that we obtain the difference when we pop */
    PetscCall(PetscIntStackTop(state->stage_stack, &state->current_stage));
  } else state->current_stage = -1;
  PetscCall(PetscLogStateUnlock(state));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateGetCurrentStage(PetscLogState state, PetscLogStage *current)
{
  PetscFunctionBegin;
  *current = state->current_stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscBTResize(PetscBT *bt_p, PetscInt old_size, PetscInt new_size)
{
  PetscFunctionBegin;
  if (new_size > old_size) {
    PetscBT bt_new;
    size_t old_num_chars = PetscMax(0, old_size);

    PetscCall(PetscBTCreate(new_size, &bt_new));
    if (old_size > 0) old_num_chars = (old_size - 1) / PETSC_BITS_PER_BYTE + 1;
    PetscCall(PetscMemcpy(bt_new, *bt_p, old_num_chars));
    PetscCall(PetscBTDestroy(bt_p));
    *bt_p = bt_new;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateLock(PetscLogState state)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryLock(state->registry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateUnlock(PetscLogState state)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryUnlock(state->registry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStateResize(PetscLogState state)
{
  PetscBT  active_new;
  PetscInt new_num_events = state->registry->events->num_entries + 1;
  PetscInt new_num_stages = state->registry->stages->num_entries;

  PetscFunctionBegin;
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

    for (PetscInt i = 0; i < state->bt_num_events; i++) {
      PetscCall(PetscMemcpy(&active_new[i * num_chars_new], &(state->active[i * num_chars_old]), num_chars_old));
    }
  }
  PetscCall(PetscBTDestroy(&state->active));
  state->active = active_new;
  state->bt_num_events = new_num_events;
  state->bt_num_stages = new_num_stages;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateStageRegister(PetscLogState state, const char sname[], PetscLogStage *stage)
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

PETSC_INTERN PetscErrorCode PetscLogStateEventRegister(PetscLogState state, const char sname[], PetscClassId id, PetscLogEvent *event)
{
  PetscInt e;

  PetscFunctionBegin;
  *event = PETSC_DECIDE;
  PetscCall(PetscLogRegistryGetEvent(state->registry, sname, event));
  if (*event > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogRegistryEventRegister(state->registry, sname, id, event));
  PetscCall(PetscLogStateResize(state));
  e = *event;
  for (PetscInt s = 0; s < state->bt_num_stages; s++) PetscCall(PetscBTSet(state->active, s + (e + 1) * state->bt_num_stages)); // events are by default active
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateStageSetActive(PetscLogState state, PetscLogStage stage, PetscBool isActive)
{
  PetscFunctionBegin;
  PetscCheck(stage >= 0 && stage < state->bt_num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, state->bt_num_stages);
  if (isActive) {
    for (PetscInt e = 0; e < state->bt_num_events; e++) {
      PetscCall(PetscBTSet(state->active, stage + e * state->bt_num_stages));
    }
  } else {
    for (PetscInt e = 0; e < state->bt_num_events; e++) {
      PetscCall(PetscBTClear(state->active, stage + e * state->bt_num_stages));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateStageGetActive(PetscLogState state, PetscLogStage stage, PetscBool *isActive)
{
  PetscFunctionBegin;
  *isActive = PetscBTLookup(state->active, stage) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateEventActivateAll(PetscLogState state, PetscLogEvent event)
{
  PetscFunctionBegin;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  for (PetscInt s = 0; s < state->bt_num_stages; s++) PetscCall(PetscBTSet(state->active, s + (event+1) * state->bt_num_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateEventDeactivateAll(PetscLogState state, PetscLogEvent event, PetscBool isActive)
{
  PetscFunctionBegin;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  for (PetscInt s = 0; s < state->bt_num_stages; s++) PetscCall(PetscBTClear(state->active, s + (event+1) * state->bt_num_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateEventActivate(PetscLogState state, PetscLogEvent event)
{
  PetscFunctionBegin;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  PetscCall(PetscBTSet(state->active, state->current_stage + (event+1) * state->bt_num_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateEventDeactivate(PetscLogState state, PetscLogEvent event)
{
  PetscFunctionBegin;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  PetscCall(PetscBTClear(state->active, state->current_stage + (event+1) * state->bt_num_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

