#include <petsc/private/loghandlerimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#if defined(PETSC_USE_LOG) && defined(PETSC_HAVE_MPE)
  #include <mpe.h>

static PetscErrorCode PetscLogHandlerEventBegin_MPE(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogState     state;
  PetscLogEventInfo event_info;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(handler, &state));
  PetscCall(PetscLogRegistryEventGetInfo(state->registry, event, &event_info));
  PetscCall(MPE_Log_event(event_info.mpe_id_begin, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_MPE(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogState     state;
  PetscLogEventInfo event_info;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(handler, &state));
  PetscCall(PetscLogRegistryEventGetInfo(state->registry, event, &event_info));
  PetscCall(MPE_Log_event(event_info.mpe_id_end, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_MPE(MPI_Comm comm, PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(comm, handler_p));
  handler              = *handler_p;
  handler->ctx         = NULL;
  handler->type        = PETSC_LOG_HANDLER_MPE;
  handler->EventBegin  = PetscLogHandlerEventBegin_MPE;
  handler->EventEnd    = PetscLogHandlerEventEnd_MPE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
