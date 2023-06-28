
#include <petscviewer.h>
#include <petsc/private/loghandlerimpl.h> /*I "petscsys.h" I*/
#include "loghandler.h"

/*@
  PetscLogHandlerCreate - Create a log handler for profiling events and stages

  Collective

  Input Parameter:
. comm - the communicator for synchronizing and viewing events with this handler

  Output Parameter:
. handler - the `PetscLogHandler`

  Usage:
.vb
      #include <petscsys.h>
      UserCtx *user_context;
      PetscErrorCode UserEventBegin(PetscLogHandler, PetscLogState, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
      PetscErrorCode UserEventEnd(PetscLogHandler, PetscLogState, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);

      PetscLogHandlerCreate(comm, &handler);
      PetscLogHandlerSetContext(handler, (void *) user_context);
      PetscLogHandlerSetOperation(handler, PETSC_LOG_HANDLER_OP_EVENT_BEGIN, (void (*)(void)) UserEventBegin);
      PetscLogHandlerSetOperation(handler, PETSC_LOG_HANDLER_OP_EVENT_END, (void (*)(void)) UserEventEnd);
      PetscLogAddHandler(handler);

      // ... code that handler will profile
      
      PetscLogDropHandler(handler);
      PetscLogHandlerView(handler, viewer);
.ve

  Level: developer

  Note:

  This does not put the handler in use in PETSc's global logging system: use `PetscLogAddHandler()` after creation

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerSetContext()`, `PetscLogHandlerSetOperation()`, `PetscLogAddHandler()`, `PetscLogDropHandler()`
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

  Note: `PetscLogHandler` is reference counted

.seealso: [](ch_profiling), `PetscLogHandlerCreate()`
@*/
PetscErrorCode PetscLogHandlerDestroy(PetscLogHandler *handler)
{
  PetscLogHandler h;

  PetscFunctionBegin;
  PetscValidPointer(handler, 1);
  h = *handler;
  *handler = NULL;
  if (--h->refct > 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (h->Destroy) PetscCall((*((h)->Destroy))(h));
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

  Usage:
.vb
      #include <petscsys.h>
      typedef struct _UserCtx UserCtx;

      PetscErrorCode UserEventBegin(PetscLogHandler handler, PetscLogState state, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
      {
        UserCtx *user_context;
        
        PetscFunctionBegin;
        PetscCall(PetscLogHandlerGetContext(handler, (void *) &user_context));
        PetscFunctionReturn(PETSC_SUCCESS);
      }

      int main() {
        UserCtx         ctx;
        PetscLogHandler handler;

        // ...
        PetscLogHandlerCreate(comm, &handler);
        PetscLogHandlerSetContext(handler, (void *) &ctx));
        PetscLogHandlerSetOperation(handler, PETSC_LOG_HANDLER_OP_EVENT_BEGIN, (void (*)(void)) UserEventBegin);
        // ...
      }
.ve

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandlerCreate()`, `PetscLogHandlerGetContext()`
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

.seealso: [](ch_profiling), `PetscLogHandlerCreate()`, `PetscLogHandlerSetContext()`
@*/
PetscErrorCode PetscLogHandlerGetContext(PetscLogHandler handler, void *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(handler, 1);
  PetscValidPointer(ctx, 2);
  *((void **) ctx) = handler->ctx;
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

.seealso: [](ch_profiling), `PetscLogHandlerCreate()`, `PetscLogHandlerSetContext()`, `PetscLogHandlerGetContext()`, `PetscLogHandlerOpType`, `PescLogHandlerGetOperation()`
@*/
PetscErrorCode PetscLogHandlerSetOperation(PetscLogHandler h, PetscLogHandlerOpType type, void (*f)(void))
{
  PetscValidPointer(h,1);
  if (f) PetscValidPointer(f,3);
  PetscFunctionBegin;
#define PETSC_LOG_HANDLER_SET_OP_CASE(NAME,Name,Type,h,f) \
  case PETSC_LOG_HANDLER_OP_##NAME: (h)->Name = (_PetscLog##Type##Fn) f; break;
  switch (type) {
  PETSC_LOG_HANDLER_SET_OP_CASE(DESTROY,Destroy,Destroy,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(EVENT_BEGIN,EventBegin,Event,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(EVENT_END,EventEnd,Event,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(EVENT_SYNC,EventSync,EventSync,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(EVENT_IGNORE_PUSH,EventIgnorePush,EventIgnore,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(EVENT_IGNORE_POP,EventIgnorePop,EventIgnore,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(EVENTS_PAUSE,EventsPause,Pause,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(EVENTS_UNPAUSE,EventsUnpause,Pause,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(OBJECT_CREATE,ObjectCreate,Object,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(OBJECT_DESTROY,ObjectDestroy,Object,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(STAGE_PUSH,StagePush,Stage,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(STAGE_POP,StagePop,Stage,h,f)
  PETSC_LOG_HANDLER_SET_OP_CASE(VIEW,View,View,h,f)
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

.seealso: [](ch_profiling), `PetscLogHandlerCreate()`, `PetscLogHandlerSetContext()`, `PetscLogHandlerGetContext()`, `PetscLogHandlerOpType`, `PetscLogHandlerSetOperation()`
@*/
PetscErrorCode PetscLogHandlerGetOperation(PetscLogHandler h, PetscLogHandlerOpType type, void (**f)(void))
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(f, 3);
#define PETSC_LOG_HANDLER_GET_OP_CASE(NAME,Name,Type,h,f) \
  case PETSC_LOG_HANDLER_OP_##NAME: *f = (void (*)(void)) (h)->Name; break;
  switch (type) {
  PETSC_LOG_HANDLER_GET_OP_CASE(DESTROY,Destroy,Destroy,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(EVENT_BEGIN,EventBegin,Event,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(EVENT_END,EventEnd,Event,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(EVENT_SYNC,EventSync,EventSync,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(EVENT_IGNORE_PUSH,EventIgnorePush,EventActivity,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(EVENT_IGNORE_POP,EventIgnorePop,EventActivity,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(EVENTS_PAUSE,EventsPause,Pause,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(EVENTS_UNPAUSE,EventsUnpause,Pause,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(OBJECT_CREATE,ObjectCreate,Object,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(OBJECT_DESTROY,ObjectDestroy,Object,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(STAGE_PUSH,StagePush,Stage,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(STAGE_POP,StagePop,Stage,h,f)
  PETSC_LOG_HANDLER_GET_OP_CASE(VIEW,View,View,h,f)
  }
#undef PETSC_LOG_HANDLER_GET_OP_CASE
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventBegin - Record the beginning of an event in a log handler
  
  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the `PetscLogState`
. e - a registered `PetscLogEvent`
- o1, o2, o3, o4 - `PetscObject`s associated with the event (each may be `NULL`)

  Level: developer

  Note:

  Most users will use `PetscLogEventBegin()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`

.seealso: [](ch_profiling), `PetscLogEventBegin()`, `PetscLogHandler`, `PetscLogHandlerRegister()`, `PetscLogState`
@*/
PetscErrorCode PetscLogHandlerEventBegin(PetscLogHandler h, PetscLogState state, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscLogHandlerTry(h,EventBegin,state,e,o1,o2,o3,o4);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventEnd - Record the end of an event in a log handler
  
  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the current profiling state
. e - a registered `PetscLogEvent`
- o1, o2, o3, o4 - `PetscObject`s associated with the event (each may be `NULL`)

  Level: developer

  Note:

  Most users will use `PetscLogEventEnd()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`

.seealso: [](ch_profiling), `PetscLogEventBegin()`, `PetscLogHandler`, `PetscLogHandlerRegister()`, `PetscLogState`
@*/
PetscErrorCode PetscLogHandlerEventEnd(PetscLogHandler h, PetscLogState state, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscLogHandlerTry(h,EventEnd,state,e,o1,o2,o3,o4);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventSync - Synchronize a logging event
  
  Collective over comm

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the `PetscLogState`
. e - a registered `PetscLogEvent`
- comm - the communicator over which to synchronize `e`

  Level: developer

  Note:

  Most users will use `PetscLogEventSync()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`

.seealso: [](ch_profiling), `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogHandler`, `PetscLogHandlerRegister()`, `PetscLogState`
@*/
PetscErrorCode PetscLogHandlerEventSync(PetscLogHandler h, PetscLogState state, PetscLogEvent e, MPI_Comm comm)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscCall(MPI_Comm_size(comm, &size));
  if (comm == MPI_COMM_NULL || size == 1) PetscFunctionReturn(PETSC_SUCCESS); // nothing to sync
  if (PetscDefined(USE_DEBUG)) {
    PetscMPIInt h_comm_world, compare;
    PetscCallMPI(MPI_Comm_compare(h->comm, PETSC_COMM_WORLD, &h_comm_world));
    PetscCallMPI(MPI_Comm_compare(h->comm, comm, &compare));
    // only synchronze if h->comm and comm have the same processes or h->comm is PETSC_COMM_WORLD
    PetscCheck(h_comm_world != MPI_UNEQUAL || compare != MPI_UNEQUAL, comm, PETSC_ERR_SUP, "PetscLogHandlerSync does not support arbitrary mismatched communicators");
  }
  PetscLogHandlerTry(h,EventSync,state,e,comm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventIgnorePush - Tell a log handler to ignore an event for a given stage.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the `PetscLogState`
. stage - a registered `PetscLogStage`, or `PETSC_DEFAULT` for the state's current stage
- e - a registered `PetscLogEvent`

  Level: developer

  Notes:

  Most users will use `PetscLogEventIgnorePush()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`.

.seealso: [](ch_profiling), `PetscLogHandlerEventIgnorePop()`, `PetscLogEventIgnorePush()`, `PetscLogEventIgnorePop()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerEventIgnorePush(PetscLogHandler h, PetscLogState state, PetscLogStage stage, PetscLogEvent e)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscLogHandlerTry(h,EventIgnorePush,state,stage,e);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventIgnorePop - Tell a log handler to stop ignoring an event for a given stage.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the `PetscLogState`
. stage - a registered `PetscLogStage`, or `PETSC_DEFAULT` for the state's current stage
- e - a registered `PetscLogEvent`

  Level: developer

  Notes:

  Most users will use `PetscLogEventIgnorePop()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`.

.seealso: [](ch_profiling), `PetscLogHandlerEventIgnorePop()`, `PetscLogEventIgnorePush()`, `PetscLogEventIgnorePop()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerEventIgnorePop(PetscLogHandler h, PetscLogState state, PetscLogStage stage, PetscLogEvent e)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscLogHandlerTry(h,EventIgnorePop,state,stage,e);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventsPause - Tell a log handler to pause running events.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
- state - the `PetscLogState`

  Level: developer

  Notes:

  Most users will use `PetscLogEventsPause()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`.

.seealso: [](ch_profiling), `PetscLogHandlerEventsUnpause()`, `PetscLogEventsPause()`, `PetscLogEventsUnpause()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerEventsPause(PetscLogHandler h, PetscLogState state)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscLogHandlerTry(h,EventsPause,state);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventsUnpause - Tell a log handler to unpause paused events.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
- state - the `PetscLogState`

  Level: developer

  Notes:

  Most users will use `PetscLogEventsUnpause()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`.

.seealso: [](ch_profiling), `PetscLogHandlerEventsUnpause()`, `PetscLogEventsPause()`, `PetscLogEventsUnpause()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerEventsUnpause(PetscLogHandler h, PetscLogState state)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscLogHandlerTry(h,EventsUnpause,state);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerObjectCreate - Record the creation of an object in a log handler.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the `PetscLogState`
- obj - a newly created `PetscObject`

  Level: developer

  Notes:

  Most users will use `PetscLogObjectCreate()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`.

.seealso: [](ch_profiling), `PetscLogHandlerObjectDestroy()`, `PetscLogObjectCreate()`, `PetscLogObjectDestroy()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerObjectCreate(PetscLogHandler h, PetscLogState state, PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscLogHandlerTry(h,ObjectCreate,state,obj);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerObjectDestroy - Record the destruction of an object in a log handler.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the `PetscLogState`
- obj - a newly created `PetscObject`

  Level: developer

  Notes:

  Most users will use `PetscLogObjectDestroy()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`.

.seealso: [](ch_profiling), `PetscLogHandlerObjectDestroy()`, `PetscLogObjectCreate()`, `PetscLogObjectDestroy()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerObjectDestroy(PetscLogHandler h, PetscLogState state, PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscLogHandlerTry(h,ObjectDestroy,state,obj);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStagePush - Begin a new logging stage in a log handler.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the `PetscLogState`
- stage - a registered `PetscLogStage`

  Level: developer

  Notes:

  Most users will use `PetscLogStagePush()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`.

.seealso: [](ch_profiling), `PetscLogHandlerStagePop()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerStagePush(PetscLogHandler h, PetscLogState state, PetscLogStage stage)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscLogHandlerTry(h,StagePush,state,stage);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStagePop - End the current logging stage in a log handler.

  Not collective

  Input Arguments:
+ h - the `PetscLogHandler`
. state - the `PetscLogState`
- stage - a registered `PetscLogStage`

  Level: developer

  Notes:

  Most users will use `PetscLogStagePop()`, which will call this function for all handlers registered with `PetscLogHandlerRegister()`.

.seealso: [](ch_profiling), `PetscLogHandlerStagePop()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerStagePop(PetscLogHandler h, PetscLogState state, PetscLogStage stage)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidPointer(state, 2);
  PetscLogHandlerTry(h,StagePop,state,stage);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerView - View the data recorded in a log handler.

  Collective

  Input Parameters:
+ h - the `PetscLogHandler`
- viewer - the `PetscViewer`

  Level: developer

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogHandler`
@*/
PetscErrorCode PetscLogHandlerView(PetscLogHandler h, PetscLogState state, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 3);
  PetscLogHandlerTry(h,View,state,viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

