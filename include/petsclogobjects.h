#ifndef PETSCLOGOBJECTS_H
#define PETSCLOGOBJECTS_H
#include <petscsys.h>

typedef struct {
  char        *name;       /* The name of this event */
  PetscClassId classid;    /* The class the event is associated with */
  PetscBool    collective; /* Flag this event as collective */
  PetscBool    visible;    /* The flag to print info in summary */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer; /* Associated external tool timer for this event */
#endif
#if defined(PETSC_HAVE_MPE)
  int mpe_id_begin; /* MPE IDs that define the event */
  int mpe_id_end;
#endif
} PetscLogEventInfo;

typedef struct {
  char        *name;    /* The class name */
  PetscClassId classid; /* The integer identifying this class */
} PetscLogClassInfo;

typedef struct _PetscLogStageInfo {
  char     *name;    /* The stage name */
  PetscBool visible; /* The flag to print info in summary */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer; /* Associated external tool timer for this event */
#endif
} PetscLogStageInfo;

PETSC_EXTERN PetscErrorCode PetscLogStateCreate(PetscLogState *);
PETSC_EXTERN PetscErrorCode PetscLogStateDestroy(PetscLogState *);
PETSC_EXTERN PetscErrorCode PetscLogStateGetRegistry(PetscLogState, PetscLogRegistry *);

PETSC_EXTERN PetscErrorCode PetscLogStateClassRegister(PetscLogState, const char[], PetscLogStage *);
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

PETSC_EXTERN PetscErrorCode PetscLogStateGetEventFromName(PetscLogState, const char[], PetscLogEvent *);
PETSC_EXTERN PetscErrorCode PetscLogStateGetStageFromName(PetscLogState, const char[], PetscLogStage *);
PETSC_EXTERN PetscErrorCode PetscLogStateGetClassFromClassId(PetscLogState, PetscClassId, PetscLogClass *);
PETSC_EXTERN PetscErrorCode PetscLogStateGetNumEvents(PetscLogState, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscLogStateGetNumStages(PetscLogState, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscLogStateGetNumClasses(PetscLogState, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscLogStateEventGetInfo(PetscLogState, PetscLogEvent, PetscLogEventInfo *);
PETSC_EXTERN PetscErrorCode PetscLogStateStageGetInfo(PetscLogState, PetscLogStage, PetscLogStageInfo *);
PETSC_EXTERN PetscErrorCode PetscLogStateClassGetInfo(PetscLogState, PetscLogClass, PetscLogClassInfo *);

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
