#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>

typedef struct _n_PetscLogHandler_Trace *PetscLogHandler_Trace;
struct _n_PetscLogHandler_Trace {
  FILE          *petsc_tracefile;
  int            petsc_tracelevel;
  char           petsc_tracespace[128];
  PetscLogDouble petsc_tracetime;
};

static PetscErrorCode PetscLogHandlerEventBegin_Trace(PetscLogHandler h, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Trace tr = (PetscLogHandler_Trace)h->ctx;
  PetscLogEventInfo     event_info;
  PetscLogDouble        cur_time;
  PetscMPIInt           rank;
  PetscLogState         state;
  PetscLogStage         stage;

  PetscFunctionBegin;
  if (!tr->petsc_tracetime) PetscCall(PetscTime(&tr->petsc_tracetime));
  tr->petsc_tracelevel++;
  PetscCallMPI(MPI_Comm_rank(h->comm, &rank));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  /* Log performance info */
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, tr->petsc_tracefile, "%s[%d] %g Event begin: %s\n", tr->petsc_tracespace, rank, cur_time - tr->petsc_tracetime, event_info.name));
  for (size_t i = 0; i < PetscMin(sizeof(tr->petsc_tracespace), 2 * tr->petsc_tracelevel); i++) tr->petsc_tracespace[i] = ' ';
  tr->petsc_tracespace[PetscMin(127, 2 * tr->petsc_tracelevel)] = '\0';
  PetscCall(PetscFFlush(tr->petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_Trace(PetscLogHandler h, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Trace tr = (PetscLogHandler_Trace)h->ctx;
  PetscLogEventInfo     event_info;
  PetscLogDouble        cur_time;
  PetscLogState         state;
  PetscLogStage         stage;
  PetscMPIInt           rank;

  PetscFunctionBegin;
  tr->petsc_tracelevel--;
  PetscCallMPI(MPI_Comm_rank(h->comm, &rank));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  /* Log performance info */
  for (size_t i = 0; i < PetscMin(sizeof(tr->petsc_tracespace), 2 * tr->petsc_tracelevel); i++) tr->petsc_tracespace[i] = ' ';
  tr->petsc_tracespace[PetscMin(127, 2 * tr->petsc_tracelevel)] = '\0';
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, tr->petsc_tracefile, "%s[%d] %g Event end: %s\n", tr->petsc_tracespace, rank, cur_time - tr->petsc_tracetime, event_info.name));
  PetscCall(PetscFFlush(tr->petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Trace(MPI_Comm comm, PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(comm, handler_p));
  handler             = *handler_p;
  handler->ctx        = NULL;
  handler->type       = PETSC_LOG_HANDLER_TRACE;
  handler->eventBegin = PetscLogHandlerEventBegin_Trace;
  handler->eventEnd   = PetscLogHandlerEventEnd_Trace;
  PetscFunctionReturn(PETSC_SUCCESS);
}
