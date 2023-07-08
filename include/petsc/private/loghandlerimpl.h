#ifndef PETSCLOGHANDLERIMPL_H
#define PETSCLOGHANDLERIMPL_H

#include <petsc/private/petscimpl.h>

typedef PetscErrorCode (*PetscLogDestroyFn)(PetscLogHandler);
typedef PetscErrorCode (*PetscLogStageFn)(PetscLogHandler, PetscLogStage);
typedef PetscErrorCode (*PetscLogViewFn)(PetscLogHandler, PetscViewer);

typedef enum {
  PETSC_LOG_HANDLER_DEFAULT,
  PETSC_LOG_HANDLER_TRACE,
  PETSC_LOG_HANDLER_NESTED,
  PETSC_LOG_HANDLER_USER,
  PETSC_LOG_HANDLER_PERFSTUBS,
  PETSC_LOG_HANDLER_MPE,
} PetscLogHandlerType;

struct _n_PetscLogHandler {
  MPI_Comm            comm;
  void               *ctx;
  PetscLogState       state;
  int                 refct;
  PetscLogHandlerType type;
  PetscLogDestroyFn   destroy;
  PetscLogEventFn     eventBegin;
  PetscLogEventFn     eventEnd;
  PetscLogEventSyncFn eventSync;
  PetscLogObjectFn    objectCreate;
  PetscLogObjectFn    objectDestroy;
  PetscLogStageFn     stagePush;
  PetscLogStageFn     stagePop;
  PetscLogViewFn      view;
};

#endif /* #define PETSCLOGHANDLERIMPL_H */
