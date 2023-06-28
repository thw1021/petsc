#ifndef PETSCLOGHANDLERIMPL_H
#define PETSCLOGHANDLERIMPL_H

#include <petsc/private/petscimpl.h>

typedef PetscErrorCode (*_PetscLogDestroyFn)(PetscLogHandler);
typedef PetscErrorCode (*_PetscLogEventFn)(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
typedef PetscErrorCode (*_PetscLogEventSyncFn)(PetscLogHandler, PetscLogEvent, MPI_Comm);
typedef PetscErrorCode (*_PetscLogObjectFn)(PetscLogHandler, PetscObject);
typedef PetscErrorCode (*_PetscLogStageFn)(PetscLogHandler, PetscLogStage);
typedef PetscErrorCode (*_PetscLogViewFn)(PetscLogHandler, PetscViewer);

typedef enum {
  _PETSC_LOG_HANDLER_DEFAULT,
  _PETSC_LOG_HANDLER_TRACE,
  _PETSC_LOG_HANDLER_NESTED,
#if defined(PETSC_HAVE_MPE)
  _PETSC_LOG_HANDLER_MPE,
#endif
  _PETSC_LOG_HANDLER_USER
} _PetscLogHandlerType;

struct _n_PetscLogHandler {
  MPI_Comm               comm;
  void                  *ctx;
  PetscLogState          state;
  int                    refct;
  _PetscLogHandlerType    type;
  _PetscLogDestroyFn     Destroy;
  _PetscLogEventFn       EventBegin;
  _PetscLogEventFn       EventEnd;
  _PetscLogEventSyncFn   EventSync;
  _PetscLogObjectFn      ObjectCreate;
  _PetscLogObjectFn      ObjectDestroy;
  _PetscLogStageFn       StagePush;
  _PetscLogStageFn       StagePop;
  _PetscLogViewFn        View;
};

#endif /* #define PETSCLOGHANDLERIMPL_H */
