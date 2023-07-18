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
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->data;

  return (*(legacy->PetscLogPLB))(e, 0, o1, o2, o3, o4);
}

static PetscErrorCode PetscLogHandlerEventEnd_Legacy(PetscLogHandler handler, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->data;

  return (*(legacy->PetscLogPLE))(e, 0, o1, o2, o3, o4);
}

static PetscErrorCode PetscLogHandlerObjectCreate_Legacy(PetscLogHandler handler, PetscObject o)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->data;

  return (*(legacy->PetscLogPHC))(o);
}

static PetscErrorCode PetscLogHandlerObjectDestroy_Legacy(PetscLogHandler handler, PetscObject o)
{
  PetscLogHandler_Legacy legacy = (PetscLogHandler_Legacy)handler->data;

  return (*(legacy->PetscLogPHD))(o);
}

static PetscErrorCode PetcLogHandlerDestroy_Legacy(PetscLogHandler handler)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(handler->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSC_LOG_HANDLER_LEGACY - PETSC_LOG_HANDLER_LEGACY = "legacy" -  A
  `PetscLogHandler` that can be constructed from the callbacks used in
  `PetscLogSet()`.  A log handler of this type is created and started by
  `PetscLogLegacyCallbacksBegin()`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreateLegacy()`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Legacy(PetscLogHandler handler)
{
  PetscLogHandler_Legacy legacy;

  PetscFunctionBegin;
  PetscCall(PetscNew(&legacy));
  handler->data         = (void *)legacy;
  handler->ops->destroy = PetcLogHandlerDestroy_Legacy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerCreateLegacy - Create a `PetscLogHandler` from callbacks matching PETSc's legacy log handler callbacks

  Collective

  Input Parameters:
+ comm        - an MPI communicator
. PetscLogPLB - a function
. PetscLogPLE - a function
. PetscLogPHC - a function
- PetscLogPHD - a function

  Output Parameter:
. handler - a `PetscLogHandler`

  Level: developer

.seealso: [](ch_profiling)
@*/
PetscErrorCode PetscLogHandlerCreateLegacy(MPI_Comm comm, PetscErrorCode (*PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject), PetscErrorCode (*PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject), PetscErrorCode (*PetscLogPHC)(PetscObject), PetscErrorCode (*PetscLogPHD)(PetscObject), PetscLogHandler *handler)
{
  PetscLogHandler_Legacy legacy;

  PetscLogHandler h;
  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreate(comm, handler));
  h = *handler;
  PetscCall(PetscLogHandlerSetType(h, PETSC_LOG_HANDLER_LEGACY));
  legacy = (PetscLogHandler_Legacy)h->data;

  legacy->PetscLogPLB = PetscLogPLB;
  legacy->PetscLogPLE = PetscLogPLE;
  legacy->PetscLogPHC = PetscLogPHC;
  legacy->PetscLogPHD = PetscLogPHD;

  h->ops->eventbegin    = PetscLogPLB ? PetscLogHandlerEventBegin_Legacy : NULL;
  h->ops->eventend      = PetscLogPLE ? PetscLogHandlerEventEnd_Legacy : NULL;
  h->ops->objectcreate  = PetscLogPHC ? PetscLogHandlerObjectCreate_Legacy : NULL;
  h->ops->objectdestroy = PetscLogPHD ? PetscLogHandlerObjectDestroy_Legacy : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
