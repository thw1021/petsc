#ifndef PETSCLOGHANDLERIMPL_H
#define PETSCLOGHANDLERIMPL_H

#include <petsc/private/petscimpl.h>

typedef PetscErrorCode (*_PetscLogEventActivityFn)(PetscLogHandler, PetscLogState, PetscLogEvent);
typedef PetscErrorCode (*_PetscLogStageFn)(PetscLogHandler, PetscLogState, PetscLogStage);
typedef PetscErrorCode (*_PetscLogObjectFn)(PetscLogHandler, PetscLogState, PetscObject);
typedef PetscErrorCode (*_PetscLogPauseFn)(PetscLogHandler, PetscLogState);
typedef PetscErrorCode (*_PetscLogViewFn)(PetscLogHandler, PetscViewer);
typedef PetscErrorCode (*_PetscLogDestroyFn)(PetscLogHandler);
typedef PetscErrorCode (*_PetscLogEventFn)(PetscLogHandler, PetscLogState, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
typedef PetscErrorCode (*_PetscLogEventSyncFn)(PetscLogHandler, PetscLogState, PetscLogEvent, MPI_Comm);
typedef PetscErrorCode (*_PetscLogObjectFn)(PetscLogHandler, PetscLogState, PetscObject);

typedef enum {
  PETSC_LOG_HANDLER_DEFAULT,
  PETSC_LOG_HANDLER_NESTED,
#if defined(PETSC_HAVE_MPE)
  PETSC_LOG_HANDLER_MPE,
#endif
  PETSC_LOG_HANDLER_USER
} PetscLogHandlerType;

struct _n_PetscLogHandler {
  MPI_Comm                 comm;
  void                    *ctx;
  int                      refct;
  PetscLogHandlerType      type;
  _PetscLogDestroyFn       Destroy;
  _PetscLogEventFn         EventBegin;
  _PetscLogEventFn         EventEnd;
  _PetscLogEventSyncFn     EventSync;
  _PetscLogEventActivityFn EventDeactivatePush;
  _PetscLogEventActivityFn EventDeactivatePop;
  _PetscLogPauseFn         EventsPause;
  _PetscLogPauseFn         EventsUnpause;
  _PetscLogObjectFn        ObjectCreate;
  _PetscLogObjectFn        ObjectDestroy;
  _PetscLogStageFn         StagePush;
  _PetscLogStageFn         StagePop;
  _PetscLogViewFn          View;
};

#endif /* #define PETSCLOGHANDLERIMPL_H */
