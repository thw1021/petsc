#ifndef PETSCLOGOBJECTS_H
#define PETSCLOGOBJECTS_H
#include <petscsys.h>

PETSC_EXTERN PetscErrorCode PetscLogStateCreate(PetscLogState *);
PETSC_EXTERN PetscErrorCode PetscLogStateDestroy(PetscLogState *);
PETSC_EXTERN PetscErrorCode PetscLogStateGetRegistry(PetscLogState, PetscLogRegistry *);

PETSC_EXTERN PetscErrorCode PetscLogStateClassSetActive(PetscLogState, PetscLogStage, PetscClassId, PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogStateClassSetActiveAll(PetscLogState, PetscClassId, PetscBool);

PETSC_EXTERN PetscErrorCode PetscLogStateStageRegister(PetscLogState, const char[], PetscLogStage *);
PETSC_EXTERN PetscErrorCode PetscLogStateStagePush(PetscLogState, PetscLogStage);
PETSC_EXTERN PetscErrorCode PetscLogStateStagePop(PetscLogState);
PETSC_EXTERN PetscErrorCode PetscLogStateStageSetActive(PetscLogState, PetscLogStage, PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogStateStageGetActive(PetscLogState, PetscLogStage, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscLogStateGetCurrentStage(PetscLogState, PetscLogStage *);

PETSC_EXTERN PetscErrorCode PetscLogStateEventRegister(PetscLogState, const char[], PetscClassId, PetscLogEvent *);
PETSC_EXTERN PetscErrorCode PetscLogStateEventSetActive(PetscLogState, PetscLogStage, PetscLogEvent, PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogStateEventSetActiveAll(PetscLogState, PetscLogEvent, PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogStateEventGetActive(PetscLogState, PetscLogStage, PetscLogEvent, PetscBool *);

PETSC_EXTERN PetscErrorCode PetscLogHandlerCreate(MPI_Comm, PetscLogHandler *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerDestroy(PetscLogHandler *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerSetContext(PetscLogHandler, void *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerGetContext(PetscLogHandler, void *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerSetOperation(PetscLogHandler, PetscLogHandlerOpType, void (*)(void));
PETSC_EXTERN PetscErrorCode PetscLogHandlerGetOperation(PetscLogHandler, PetscLogHandlerOpType, void (**)(void));
PETSC_EXTERN PetscErrorCode PetscLogHandlerSetState(PetscLogHandler, PetscLogState);
PETSC_EXTERN PetscErrorCode PetscLogHandlerGetState(PetscLogHandler, PetscLogState *);
PETSC_EXTERN PetscErrorCode PetscLogHandlerEventBegin(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogHandlerEventEnd(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogHandlerEventSync(PetscLogHandler, PetscLogEvent, MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscLogHandlerObjectCreate(PetscLogHandler, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogHandlerObjectDestroy(PetscLogHandler, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogHandlerStagePush(PetscLogHandler, PetscLogStage);
PETSC_EXTERN PetscErrorCode PetscLogHandlerStagePop(PetscLogHandler, PetscLogStage);
PETSC_EXTERN PetscErrorCode PetscLogHandlerView(PetscLogHandler, PetscViewer);

#endif /* #define PETSCLOGOBJECTS_H */
