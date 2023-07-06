
#include <petscviewer.h>
#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>
#include "loghandler.h"

/*@
  PetscLogHandlerCreate - Create a log handler for profiling events and stages

  Collective

  Input Parameter:
. comm - the communicator for synchronizing and viewing events with this handler

  Output Parameter:
. handler - the `PetscLogHandler`

  Level: developer

  Note:
  This does not put the handler in use in PETSc's global logging system: use `PetscLogHandlerStart()` after creation.

  See `PetscLogHandler` for example usage.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerSetContext()`, `PetscLogHandlerSetOperation()`, `PetscLogHandlerStart()`, `PetscLogHandlerStop()`
@*/
PetscErrorCode PetscLogHandlerCreate(MPI_Comm comm, PetscLogHandler *handler)
{
  PetscLogHandler h;

  PetscFunctionBegin;
  PetscCall(PetscNew(handler));
  h = *handler;
  PetscCall(PetscCommDuplicate(comm, &h->comm, NULL));
  h->refct++;
  h->type = PETSC_LOG_HANDLER_USER;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerDestroy - Destroy a `PetscLogHandler`

  Logically collective

  Input Parameter:
. handler - handler to be destroyed

  Level: developer

  Note: `PetscLogHandler` is reference counted.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`
@*/
PetscErrorCode PetscLogHandlerDestroy(PetscLogHandler *handler)
{
  PetscLogHandler h;

  PetscFunctionBegin;
  PetscValidPointer(handler, 1);
  h        = *handler;
  *handler = NULL;
  if (h == NULL || --h->refct > 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (h->destroy) PetscCall((*((h)->destroy))(h));
  PetscCall(PetscLogStateDestroy(&h->state));
  PetscCall(PetscCommDestroy(&h->comm));
  PetscCall(PetscFree(h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerSetContext - Set the context available to a `PetscLogHandler`'s callback operations

  Logically collective

  Input Parameters:
+ handler - the `PetscLogHandler`
- ctx - Arbitrary pointer to data that will be used in callbacks

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`, `PetscLogHandlerGetContext()`
@*/
PetscErrorCode PetscLogHandlerSetContext(PetscLogHandler handler, void *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(handler, 1);
  if (ctx) PetscValidPointer(ctx, 2);
  handler->ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerGetContext - Get a `PetscLogHandler`'s context

  Not collective

  Input Parameter:
. handler - the `PetscLogHandler`

  Output Parameter:
. ctx - Arbitrary pointer to data that will be used in callbacks

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`, `PetscLogHandlerSetContext()`
@*/
PetscErrorCode PetscLogHandlerGetContext(PetscLogHandler handler, void *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(handler, 1);
  PetscValidPointer(ctx, 2);
  *((void **)ctx) = handler->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerSetOperation - Set the callback a `PetscLogHandler` uses for a profiling event

  Logically collective

  Input Parameters:
+ handler - the `PetscLogHandler`
. op - the type of the operation
- f - the callback function, which is cast to `(void (*) (void))` but should have the signature of the corresponding function (see `PetscLogHandlerOpType` for a list)

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`, `PetscLogHandlerSetContext()`, `PetscLogHandlerGetContext()`, `PetscLogHandlerOpType`, `PescLogHandlerGetOperation()`
@*/
PetscErrorCode PetscLogHandlerSetOperation(PetscLogHandler h, PetscLogHandlerOpType type, void (*f)(void))
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
#define PETSC_LOG_HANDLER_SET_OP_CASE(NAME, name, Type, h, f) \
  case PETSC_LOG_HANDLER_OP_##NAME: \
    (h)->name = (PetscLog##Type##Fn)f; \
    break;
  switch (type) {
    PETSC_LOG_HANDLER_SET_OP_CASE(DESTROY, destroy, Destroy, h, f)
    PETSC_LOG_HANDLER_SET_OP_CASE(EVENT_BEGIN, eventBegin, Event, h, f)
    PETSC_LOG_HANDLER_SET_OP_CASE(EVENT_END, eventEnd, Event, h, f)
    PETSC_LOG_HANDLER_SET_OP_CASE(EVENT_SYNC, eventSync, EventSync, h, f)
    PETSC_LOG_HANDLER_SET_OP_CASE(OBJECT_CREATE, objectCreate, Object, h, f)
    PETSC_LOG_HANDLER_SET_OP_CASE(OBJECT_DESTROY, objectDestroy, Object, h, f)
    PETSC_LOG_HANDLER_SET_OP_CASE(STAGE_PUSH, stagePush, Stage, h, f)
    PETSC_LOG_HANDLER_SET_OP_CASE(STAGE_POP, stagePop, Stage, h, f)
    PETSC_LOG_HANDLER_SET_OP_CASE(VIEW, view, View, h, f)
  }
#undef PETSC_LOG_HANDLER_SET_OP_CASE

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerGetOperation - Get the callback a `PetscLogHandler` uses for a profiling event

  Logically collective

  Input Parameters:
+ handler - the `PetscLogHandler`
- op - the type of the operation

  Output Parameter:
. f - the callback function, which is cast to `(void (*) (void))` but should have the signature of the corresponding function (see `PetscLogHandlerOpType` for a list)

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`, `PetscLogHandlerSetContext()`, `PetscLogHandlerGetContext()`, `PetscLogHandlerOpType`, `PetscLogHandlerSetOperation()`
@*/
PetscErrorCode PetscLogHandlerGetOperation(PetscLogHandler h, PetscLogHandlerOpType type, void (**f)(void))
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(f, 3);
#define PETSC_LOG_HANDLER_GET_OP_CASE(NAME, name, Type, h, f) \
  case PETSC_LOG_HANDLER_OP_##NAME: \
    *f = (void (*)(void))(h)->name; \
    break;
  switch (type) {
    PETSC_LOG_HANDLER_GET_OP_CASE(DESTROY, destroy, Destroy, h, f)
    PETSC_LOG_HANDLER_GET_OP_CASE(EVENT_BEGIN, eventBegin, Event, h, f)
    PETSC_LOG_HANDLER_GET_OP_CASE(EVENT_END, eventEnd, Event, h, f)
    PETSC_LOG_HANDLER_GET_OP_CASE(EVENT_SYNC, eventSync, EventSync, h, f)
    PETSC_LOG_HANDLER_GET_OP_CASE(OBJECT_CREATE, objectCreate, Object, h, f)
    PETSC_LOG_HANDLER_GET_OP_CASE(OBJECT_DESTROY, objectDestroy, Object, h, f)
    PETSC_LOG_HANDLER_GET_OP_CASE(STAGE_PUSH, stagePush, Stage, h, f)
    PETSC_LOG_HANDLER_GET_OP_CASE(STAGE_POP, stagePop, Stage, h, f)
    PETSC_LOG_HANDLER_GET_OP_CASE(VIEW, view, View, h, f)
  }
#undef PETSC_LOG_HANDLER_GET_OP_CASE

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerSetState - Set the logging state that provides the stream of events and stages for a log handler.

  Logically collective

  Input Parameters:
+ h - the `PetscLogHandler`
- state - the `PetscLogState`

  Level: developer

  Note:
  Most users well not need to set a state explicitly: the global logging state (`PetscLogGetState()`) is set when calling `PetscLogHandlerStart()`

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogState`, `PetscLogEventBegin()`, `PetscLogHandlerStart()`
@*/
PetscErrorCode PetscLogHandlerSetState(PetscLogHandler h, PetscLogState state)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  if (state) {
    PetscValidPointer(state, 2);
    state->refct++;
  }
  PetscCall(PetscLogStateDestroy(&h->state));
  h->state = state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerGetState - Get the logging state that provides the stream of events and stages for a log handler.

  Logically collective

  Input Parameter:
. h - the `PetscLogHandler`

  Output Parameter:
. state - the `PetscLogState`

  Level: developer

  Note:
  For a log handler started with `PetscLogHandlerStart()`, this will be the PETSc global logging state (`PetscLogGetState()`)

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogState`, `PetscLogEventBegin()`, `PetscLogHandlerStart()`
@*/
PetscErrorCode PetscLogHandlerGetState(PetscLogHandler h, PetscLogState *state)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  *state = h->state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventBegin - Record the beginning of an event in a log handler

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. e - a registered `PetscLogEvent`
- o1, o2, o3, o4 - `PetscObject`s associated with the event (each may be `NULL`)

  Level: developer

  Note:
  Most users will use `PetscLogEventBegin()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventSync()`, `PetscLogHandlerEventEnd()`, `PetscLogHandlerEventSync()`
@*/
PetscErrorCode PetscLogHandlerEventBegin(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscLogHandlerTry(h, eventBegin, (h, e, o1, o2, o3, o4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventEnd - Record the end of an event in a log handler

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. e - a registered `PetscLogEvent`
- o1, o2, o3, o4 - `PetscObject`s associated with the event (each may be `NULL`)

  Level: developer

  Note:
  Most users will use `PetscLogEventEnd()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventSync()`, `PetscLogHandlerEventBegin()`, `PetscLogHandlerEventSync()`
@*/
PetscErrorCode PetscLogHandlerEventEnd(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscLogHandlerTry(h, eventEnd, (h, e, o1, o2, o3, o4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventSync - Synchronize a logging event

  Collective over comm

  Input Arguments:
+ h - the `PetscLogHandler`
. e - a registered `PetscLogEvent`
- comm - the communicator over which to synchronize `e`

  Level: developer

  Note:
  Most users will use `PetscLogEventSync()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventSync()`, `PetscLogHandlerEventBegin()`, `PetscLogHandlerEventEnd()`
@*/
PetscErrorCode PetscLogHandlerEventSync(PetscLogHandler h, PetscLogEvent e, MPI_Comm comm)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (comm == MPI_COMM_NULL || size == 1) PetscFunctionReturn(PETSC_SUCCESS); // nothing to sync
  if (PetscDefined(USE_DEBUG)) {
    PetscMPIInt h_comm_world, compare;
    PetscCallMPI(MPI_Comm_compare(h->comm, PETSC_COMM_WORLD, &h_comm_world));
    PetscCallMPI(MPI_Comm_compare(h->comm, comm, &compare));
    // only synchronze if h->comm and comm have the same processes or h->comm is PETSC_COMM_WORLD
    PetscCheck(h_comm_world != MPI_UNEQUAL || compare != MPI_UNEQUAL, comm, PETSC_ERR_SUP, "PetscLogHandlerSync does not support arbitrary mismatched communicators");
  }
  PetscLogHandlerTry(h, eventSync, (h, e, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerObjectCreate - Record the creation of an object in a log handler.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
- obj - a newly created `PetscObject`

  Level: developer

  Notes:
  Most users will use `PetscLogObjectCreate()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogObjectCreate()`, `PetscLogObjectDestroy()`, `PetscLogHandlerObjectDestroy()`
@*/
PetscErrorCode PetscLogHandlerObjectCreate(PetscLogHandler h, PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscLogHandlerTry(h, objectCreate, (h, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerObjectDestroy - Record the destruction of an object in a log handler.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
- obj - a newly created `PetscObject`

  Level: developer

  Notes:
  Most users will use `PetscLogObjectDestroy()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogObjectCreate()`, `PetscLogObjectDestroy()`, `PetscLogHandlerObjectCreate()`
@*/
PetscErrorCode PetscLogHandlerObjectDestroy(PetscLogHandler h, PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscLogHandlerTry(h, objectDestroy, (h, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStagePush - Begin a new logging stage in a log handler.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
- stage - a registered `PetscLogStage`

  Level: developer

  Notes:
  Most users will use `PetscLogStagePush()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`.

  This function is called right before the stage is pushed for the handler's `PetscLogState`, so `PetscLogStateGetCurrentStage()`
  can be used to see what the previous stage was.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogHandlerStagePop()`
@*/
PetscErrorCode PetscLogHandlerStagePush(PetscLogHandler h, PetscLogStage stage)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscLogHandlerTry(h, stagePush, (h, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStagePop - End the current logging stage in a log handler.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
- stage - a registered `PetscLogStage`

  Level: developer

  Notes:
  Most users will use `PetscLogStagePop()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`.

  This function is called right after the stage is popped for the handler's `PetscLogState`, so `PetscLogStateGetCurrentStage()`
  can be used to see what the next stage will be.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogHandlerStagePush()`
@*/
PetscErrorCode PetscLogHandlerStagePop(PetscLogHandler h, PetscLogStage stage)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscLogHandlerTry(h, stagePop, (h, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerView - View the data recorded in a log handler.

  Collective

  Input Parameters:
+ h - the `PetscLogHandler`
- viewer - the `PetscViewer`

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogView()`
@*/
PetscErrorCode PetscLogHandlerView(PetscLogHandler h, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 3);
  PetscLogHandlerTry(h, view, (h, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
