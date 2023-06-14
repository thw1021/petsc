
#include <petsc/private/logimpl.h> /*I "petsclog.h" I*/

PETSC_INTERN PetscErrorCode PetscLogStateCreate(PetscInt num_stages, PetscInt num_events, PetscLogState *state_p)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscNew(state_p));
  state = *state_p;
  state->bt_num_stages = num_stages;
  state->bt_num_events = num_events;

  PetscCall(PetscIntStackCreate(&state->stage_stack));
  PetscCall(PetscBTCreate(state->bt_num_stages * state->bt_num_events, &state->inactive));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateDestroy(PetscLogState state)
{
  PetscFunctionBegin;

  PetscCall(PetscIntStackDestroy(state->stage_stack));
  PetscCall(PetscBTDestroy(&state->inactive));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateStagePush(PetscLogState state, PetscLogStage stage)
{
  PetscFunctionBegin;
  /* Activate the stage */
  PetscCall(PetscIntStackPush(state->stage_stack, stage));
  state->current_stage = stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateStagePop(PetscLogState state)
{
  int       curStage;
  PetscBool empty;

  PetscFunctionBegin;
  /* Record flops/time of current stage */
  PetscCall(PetscIntStackPop(state->stage_stack, &curStage));
  PetscCall(PetscIntStackEmpty(state->stage_stack, &empty));
  if (!empty) {
    /* Subtract current quantities so that we obtain the difference when we pop */
    PetscCall(PetscIntStackTop(state->stage_stack, &state->current_stage));
  } else state->current_stage = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateGetCurrentStage(PetscLogState state, PetscLogStage *current)
{
  PetscFunctionBegin;
  *current = state->current_stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStateBTResize(PetscBT *bt_p, PetscInt old_num_stages, PetscInt old_num_events, PetscInt new_num_stages, PetscInt new_num_events)
{
  PetscBT bt_new;

  PetscFunctionBegin;
  if (old_num_stages == new_num_stages && old_num_events == new_num_events) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(new_num_stages >= old_num_stages && new_num_events >= old_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "New PetscBT sizes must not be smaller than old PetscBT sizes");
  PetscCall(PetscBTCreate(new_num_stages * new_num_events, &bt_new));
  if (new_num_stages == old_num_stages) {
    size_t old_num_chars = 0;
    size_t old_actual_size = (size_t) PetscMax(0, old_num_events * old_num_stages);
    if (old_actual_size) old_num_chars = (old_actual_size - 1) / PETSC_BITS_PER_BYTE + 1;
    PetscCall(PetscMemcpy(bt_new, *bt_p, old_num_chars));
  } else {
    size_t old_chars_per_event = 0;
    size_t new_chars_per_event = 0;
    PetscCheck(((old_num_stages % PETSC_BITS_PER_BYTE) == 0) && ((new_num_stages % PETSC_BITS_PER_BYTE) == 0), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Number of stages must be multiples of %d\n", PETSC_BITS_PER_BYTE);
    if (old_num_stages) old_chars_per_event = (old_num_stages - 1) / PETSC_BITS_PER_BYTE + 1;
    if (new_num_stages) new_chars_per_event = (new_num_stages - 1) / PETSC_BITS_PER_BYTE + 1;
    for (PetscInt i = 0; i < old_num_events; i++) PetscCall(PetscMemcpy(&((char *) bt_new)[i * new_chars_per_event], &((char *) *bt_p)[i * old_chars_per_event], old_chars_per_event));
  }
  PetscCall(PetscBTDestroy(bt_p));
  *bt_p = bt_new;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStateEnsureSize(PetscLogState state, PetscInt new_num_stages, PetscInt new_num_events)
{
  PetscFunctionBegin;
  if (new_num_stages != state->bt_num_stages || new_num_events != state->bt_num_events) {
    PetscCall(PetscLogStateBTResize(&(state->inactive), state->bt_num_stages, new_num_stages, state->bt_num_events, new_num_events));
    state->bt_num_stages = new_num_stages;
    state->bt_num_events = new_num_events;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
