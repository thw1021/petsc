
#include "lognested.h"

static PetscErrorCode PetscLogEventBegin_Nested(PetscLogState state, PetscLogEvent e, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) ctx;
  NestedIdPair key;
  NestedId     nested_id;
  PetscHashIter iter;
  PetscLogEvent nested_event;
  PetscBool missing;

  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = NestedIdFromEvent(e);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    // register a new nested event

    // TODO: nested name and copy from registry
    PetscCall(PetscLogStateEventRegister(nested->state, "", 0, &nested_event));
    nested_id = NestedIdFromEvent(nested_event);
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, &nested_id));
    nested_event = NestedIdToEvent(nested_id);
  }
  PetscCall(nested->handler->event_begin(nested->state, nested_event, t, o1, o2, o3, o4, nested->handler->ctx));
  PetscCall(PetscIntStackPush(nested->stack, nested_id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogEventEnd_Nested(PetscLogState state, PetscLogEvent e, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, void *ctx)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) ctx;
  NestedId nested_id;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(nested->stack, &nested_id));
  if (PetscDefined(USE_DEBUG)) {
    NestedIdPair key;
    NestedId     val;

    PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
    key.leaf = NestedIdFromEvent(e);
    PetscCall(PetscNestedHashGet(nested->pair_map, key, &val));
    PetscCheck(val == nested_id, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging events appear to be unnested, nested logging cannot be used");
  }
  PetscCall(nested->handler->event_end(nested->state, NestedIdToEvent(nested_id), t, o1, o2, o3, o4, nested->handler->ctx));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePush_Nested(PetscLogState state, PetscLogStage stage, void *ctx)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) ctx;
  NestedIdPair key;
  NestedId     nested_id;
  PetscHashIter iter;
  PetscLogStage nested_stage;
  PetscBool missing;

  PetscFunctionBegin;
  PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
  key.leaf = NestedIdFromStage(stage);
  PetscCall(PetscNestedHashPut(nested->pair_map, key, &iter, &missing));
  if (missing) {
    // register a new nested event

    // TODO: nested name and copy from registry
    PetscCall(PetscLogStateStageRegister(nested->state, "", &nested_stage));
    nested_id = NestedIdFromStage(nested_stage);
  } else {
    PetscCall(PetscNestedHashIterGet(nested->pair_map, iter, &nested_id));
    nested_stage = NestedIdToStage(nested_id);
  }
  PetscCall(nested->handler->impl->stage_push(nested->state, nested_stage, nested->handler->ctx));
  PetscCall(PetscLogStateStagePush(nested->state, nested_stage));
  PetscCall(PetscIntStackPush(nested->stack, nested_id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStagePop_Nested(PetscLogState state, PetscLogStage stage, void *ctx)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) ctx;
  PetscLogStage nested_stage;
  NestedId nested_id;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(nested->stack, &nested_id));
  if (PetscDefined(USE_DEBUG)) {
    NestedIdPair key;
    NestedId     val;

    PetscCall(PetscIntStackTop(nested->stack, &(key.root)));
    key.leaf = NestedIdFromStage(stage);
    PetscCall(PetscNestedHashGet(nested->pair_map, key, &val));
    PetscCheck(val == nested_id, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging stages appear to be unnested, nested logging cannot be used");
  }
  nested_stage = nested->state->current_stage;
  PetscCall(PetscLogStateStagePop(nested->state));
  PetscCall(nested->handler->impl->stage_pop(nested->state, nested_stage, nested->handler->ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerContextCreate_Nested(PetscLogHandler_Nested *nested_p)
{
  PetscLogHandler_Nested nested;

  PetscFunctionBegin;
  PetscCall(PetscNew(nested_p));
  nested = *nested_p;
  PetscCall(PetscLogStateCreate(&nested->state));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerContextDestroy_Nested(void *ctx)
{
  PetscLogHandler_Nested nested = (PetscLogHandler_Nested) ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogStateDestroy(nested->state));
  PetscCall(PetscFree(nested));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerCreate_Nested(PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscNew(handler_p));
  handler = *handler_p;
  PetscCall(PetscNew(&handler->impl));
  handler->impl->type = PETSC_LOG_HANDLER_NESTED;
  PetscCall(PetscLogHandlerContextCreate_Nested((PetscLogHandler_Nested *) &handler->ctx));
  handler->impl->destroy = PetscLogHandlerContextDestroy_Nested;
  handler->event_begin = PetscLogEventBegin_Nested;
  handler->event_end = PetscLogEventEnd_Nested;
  handler->impl->stage_push = PetscLogStagePush_Nested;
  handler->impl->stage_pop = PetscLogStagePop_Nested;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogNestedBegin(void)
{
  int i_free = -1;
  
  PetscFunctionBegin;
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i];
    if (h) {
     if (h->impl->type == PETSC_LOG_HANDLER_NESTED) PetscFunctionReturn(PETSC_SUCCESS);
    } else if (i_free < 0) {
      i_free = i;
    }
  }
  PetscCheck(i_free >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Too many log handlers, cannot created nested handler");
  PetscCall(PetscLogHandlerCreate_Nested(&PetscLogHandlers[i_free]));
  PetscFunctionReturn(PETSC_SUCCESS);
}
