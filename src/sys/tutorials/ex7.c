const char help[] = "How to create a log handler using the PetscLogHandler interface";

#include <petscsys.h>
#include <petsc/private/hashmapi.h> // use PetscHMapI: a PetscInt -> PetscInt hashmap

// Log handlers that use the PetscLogHandler interface get their information
// from the PetscLogState available to each handler and the user-defined
// context pointer.  Compare this example to src/sys/tutorials/ex6.c.

// A logging event can be started multiple times before it stops: for example,
// a linear solve may involve a subsolver, so PetscLogEventBegin() can
// be called for the event KSP_Solve multiple times before a call to
// PetscLogEventEnd().  The user defined handler in this example shows
// how many times and event is running.

typedef struct _HandlerCtx *HandlerCtx;

struct _HandlerCtx {
  PetscHMapI running;
};

static PetscErrorCode HandlerCtxCreate(HandlerCtx *ctx_p)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(ctx_p));
  ctx = *ctx_p;
  PetscCall(PetscHMapICreate(&ctx->running));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HandlerCtxDestroy(HandlerCtx *ctx_p)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  ctx    = *ctx_p;
  *ctx_p = NULL;
  PetscCall(PetscHMapIDestroy(&ctx->running));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// All of the callbacks of PetscLogHandler are optional: this example only implements:
//
// - PetscLogHandlerEventBegin()
// - PetscLogHandlerEventEnd()
// - PetscLogHandlerDestroy()

#define PrintData(format_string, ...) \
  do { \
    PetscMPIInt    rank; \
    PetscLogDouble time; \
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank)); \
    PetscCall(PetscTime(&time)); \
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d:%-7g:%-30s] " format_string, rank, time, PETSC_FUNCTION_NAME, __VA_ARGS__)); \
  } while (0)

static PetscErrorCode PetscLogHandlerEventBegin_User(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  HandlerCtx        ctx;
  PetscInt          count;
  PetscLogState     state;
  PetscLogEventInfo event_info;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetContext(h, (void *)&ctx));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscHMapIGetWithDefault(ctx->running, (PetscInt)e, 0, &count));
  count += 1;
  PetscCall(PetscLogStateEventGetInfo(state, e, &event_info));
  PrintData("Event \"%s\" started: now running %" PetscInt_FMT " times\n", event_info.name, count);
  PetscCall(PetscHMapISet(ctx->running, (PetscInt)e, count));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_User(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  HandlerCtx        ctx;
  PetscInt          count;
  PetscLogState     state;
  PetscLogEventInfo event_info;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetContext(h, (void *)&ctx));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscHMapIGetWithDefault(ctx->running, (PetscInt)e, 0, &count));
  count -= 1;
  PetscCall(PetscLogStateEventGetInfo(state, e, &event_info));
  PrintData("Event \"%s\" stopped: now running %" PetscInt_FMT " times\n", event_info.name, count);
  PetscCall(PetscHMapISet(ctx->running, (PetscInt)e, count));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_User(PetscLogHandler h)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetContext(h, (void *)&ctx));
  PetscCall(HandlerCtxDestroy(&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create

static PetscErrorCode PetscLogHandlerCreateUser(MPI_Comm comm, PetscLogHandler *handler_p)
{
  HandlerCtx      ctx;
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(comm, handler_p));
  handler = *handler_p;
  PetscCall(HandlerCtxCreate(&ctx));
  PetscCall(PetscLogHandlerSetContext(handler, (void *)ctx));
  PetscCall(PetscLogHandlerSetOperation(handler, PETSC_LOG_HANDLER_OP_EVENT_BEGIN, (void(*))PetscLogHandlerEventBegin_User));
  PetscCall(PetscLogHandlerSetOperation(handler, PETSC_LOG_HANDLER_OP_EVENT_END, (void(*))PetscLogHandlerEventEnd_User));
  PetscCall(PetscLogHandlerSetOperation(handler, PETSC_LOG_HANDLER_OP_DESTROY, (void(*))PetscLogHandlerDestroy_User));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscClassId    user_classid;
  PetscLogEvent   event_1, event_2;
  PetscLogHandler h;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscLogHandlerCreateUser(PETSC_COMM_WORLD, &h));
  PetscCall(PetscLogHandlerStart(h));

  PetscCall(PetscClassIdRegister("User class", &user_classid));
  PetscCall(PetscLogEventRegister("Event 1", user_classid, &event_1));
  PetscCall(PetscLogEventRegister("Event 2", user_classid, &event_2));

  PetscCall(PetscLogEventBegin(event_1, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventBegin(event_2, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventBegin(event_1, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventEnd(event_1, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventEnd(event_2, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventEnd(event_1, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogHandlerStop(h));
  PetscCall(PetscLogHandlerDestroy(&h));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: defined(PETSC_USE_LOG)
    suffix: 0

TEST*/
