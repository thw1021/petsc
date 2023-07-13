#ifndef PETSCLOGHANDLERIMPL_H
#define PETSCLOGHANDLERIMPL_H

#include <petsc/private/petscimpl.h>

typedef enum {
  PETSC_LOG_HANDLER_USER,
  PETSC_LOG_HANDLER_DEFAULT,
  PETSC_LOG_HANDLER_NESTED,
  PETSC_LOG_HANDLER_TRACE
} PetscLogHandlerType;

struct _n_PetscLogHandler {
  MPI_Comm            comm;
  void               *ctx;
  PetscLogState       state;
  int                 refct;
  PetscLogHandlerType type;
  PetscErrorCode (*destroy)(PetscLogHandler);
  PetscErrorCode (*eventBegin)(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
  PetscErrorCode (*eventEnd)(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
  PetscErrorCode (*eventSync)(PetscLogHandler, PetscLogEvent, MPI_Comm);
  PetscErrorCode (*objectCreate)(PetscLogHandler, PetscObject);
  PetscErrorCode (*objectDestroy)(PetscLogHandler, PetscObject);
  PetscErrorCode (*stagePush)(PetscLogHandler, PetscLogStage);
  PetscErrorCode (*stagePop)(PetscLogHandler, PetscLogStage);
  PetscErrorCode (*view)(PetscLogHandler, PetscViewer);
};

#endif /* #define PETSCLOGHANDLERIMPL_H */
