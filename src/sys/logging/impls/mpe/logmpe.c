#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#if defined(PETSC_USE_LOG) && defined(PETSC_HAVE_MPE)
#include <mpe.h>

static PetscErrorCode PetscLogHandlerEventBegin_MPE(PetscLogHandler handler, PetscLogState state, PetscLogEvent event, int i, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscEventRegInfo event_info;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryEventGetInfo(state->registry, event, &event_info));
  PetscCall(MPE_Log_event(event_info.mpe_id_begin, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_MPE(PetscLogHandler handler, PetscLogState state, PetscLogEvent event, int i, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscEventRegInfo event_info;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryEventGetInfo(state->registry, event, &event_info));
  PetscCall(MPE_Log_event(event_info.mpe_id_end, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_MPE(PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscNew(handler_p));
  handler = *handler_p;
  PetscCall(PetscNew(&handler->impl));
  handler->impl->ctx = NULL;
  handler->impl->type = PETSC_LOG_HANDLER_MPE;
  handler->event_begin = PetscLogHandlerEventBegin_MPE;
  handler->event_end = PetscLogHandlerEventEnd_MPE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
