#include <petsc/private/loghandlerimpl.h>

typedef struct _n_PetscLogHandler_Legacy *PetscLogHandler_Legacy;
struct _n_PetscLogHandler_Legacy {
  PetscErrorCode (*PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
  PetscErrorCode (*PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
  PetscErrorCode (*PetscLogPHC)(PetscObject);
  PetscErrorCode (*PetscLogPHD)(PetscObject);
};

static PetscErrorCode PetscLogHandlerEventBegin_Legacy(PetscLogHandler handler, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->ctx;

  return (*(legacy->PetscLogPLB))(e, 0, o1, o2, o3, o4);
}

static PetscErrorCode PetscLogHandlerEventEnd_Legacy(PetscLogHandler handler, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->ctx;

  return (*(legacy->PetscLogPLE))(e, 0, o1, o2, o3, o4);
}

static PetscErrorCode PetscLogHandlerObjectCreate_Legacy(PetscLogHandler handler, PetscObject o)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->ctx;

  return (*(legacy->PetscLogPHC))(o);
}

static PetscErrorCode PetscLogHandlerObjectDestroy_Legacy(PetscLogHandler handler, PetscObject o)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->ctx;

  return (*(legacy->PetscLogPHD))(o);
}

static PetscErrorCode PetcLogHandlerDestroy_Legacy(PetscLogHandler handler)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(handler->ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Legacy(PetscErrorCode (*PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject), PetscErrorCode (*PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject), PetscErrorCode (*PetscLogPHC)(PetscObject), PetscErrorCode (*PetscPHD)(PetscObject), PetscLogHandler *handler_p)
{
  PetscLogHandler        handler;
  PetscLogHandler_Legacy legacy;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(PETSC_COMM_WORLD, handler_p));
  handler = *handler_p;
  PetscCall(PetscNew(&legacy));
  handler->ctx           = (void *)legacy;
  handler->eventBegin    = PetscLogPLB ? PetscLogHandlerEventBegin_Legacy : NULL;
  handler->eventEnd      = PetscLogPLE ? PetscLogHandlerEventEnd_Legacy : NULL;
  handler->objectCreate  = PetscLogPLE ? PetscLogHandlerObjectCreate_Legacy : NULL;
  handler->objectDestroy = PetscLogPLE ? PetscLogHandlerObjectDestroy_Legacy : NULL;
  handler->destroy       = PetcLogHandlerDestroy_Legacy;

  PetscFunctionReturn(PETSC_SUCCESS);
}
