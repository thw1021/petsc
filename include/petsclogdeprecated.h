#if !defined(PETSCLOGDEPRECATED_H)
  #define PETSCLOGDEPRECATED_H

  #include <petscsystypes.h>
  #include <petsclogtypes.h>
  #include <petscconf.h>

/* These data structures are no longer used by any non-deprecated PETSc interface functions */

typedef struct {
  char        *name;
  PetscClassId classid;
} PetscClassRegInfo;

typedef struct _n_PetscClassRegLog *PetscClassRegLog;
struct _n_PetscClassRegLog {
  int                numClasses;
  int                maxClasses;
  PetscClassRegInfo *classInfo;
};

typedef struct {
  PetscClassId   id;
  int            creations;
  int            destructions;
  PetscLogDouble mem;
  PetscLogDouble descMem;
} PetscClassPerfInfo;

typedef struct _n_PetscClassPerfLog *PetscClassPerfLog;
struct _n_PetscClassPerfLog {
  int                 numClasses;
  int                 maxClasses;
  PetscClassPerfInfo *classInfo;
};

typedef struct {
  char        *name;
  PetscClassId classid;
  PetscBool    collective;
  #if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer;
  #endif
  #if defined(PETSC_HAVE_MPE)
  int mpe_id_begin;
  int mpe_id_end;
  #endif
} PetscEventRegInfo;

typedef struct _n_PetscEventRegLog *PetscEventRegLog;
struct _n_PetscEventRegLog {
  int                numEvents;
  int                maxEvents;
  PetscEventRegInfo *eventInfo; /* The registration information for each event */
};

typedef struct _n_PetscEventPerfLog *PetscEventPerfLog;
struct _n_PetscEventPerfLog {
  int                 numEvents;
  int                 maxEvents;
  PetscEventPerfInfo *eventInfo;
};

typedef struct _PetscStageInfo {
  char              *name;
  PetscBool          used;
  PetscEventPerfInfo perfInfo;
  PetscClassPerfLog  classLog;
  #if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer;
  #endif
} PetscStageInfo;

typedef struct _n_PetscStageLog *PetscStageLog;
struct _n_PetscStageLog {
  int              numStages;
  int              maxStages;
  PetscIntStack    stack;
  int              curStage;
  PetscStageInfo  *stageInfo;
  PetscEventRegLog eventLog;
  PetscClassRegLog classLog;
};

PETSC_DEPRECATED_OBJECT("Use PetscLog interface functions (since version 3.20)") PETSC_UNUSED static PetscStageLog petsc_stageLog = NULL;

  #define PETSC_DEPRECATED_LOG(c) PETSC_DEPRECATED_FUNCTION("Petsc" #c " is unused by PETSc (since version 3.20)") PETSC_UNUSED static inline

/*@C
  PetscLogGetStageLog - Deprecated.

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling), `PetscLogEventGetPerfInfo()`
@*/
PETSC_DEPRECATED_LOG(StageLog) PetscErrorCode PetscLogGetStageLog(PetscStageLog *s)
{
  *s = NULL;
  return PETSC_SUCCESS;
}

/*@C
  PetscStageLogGetCurrent - Deprecated

  Level: deprecated

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_LOG(StageLog) PetscErrorCode PetscStageLogGetCurrent(PetscStageLog a, int *b)
{
  (void)a;
  *b = -1;
  return PETSC_SUCCESS;
}

/*@C
  PetscStageLogGetEventPerfLog - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_LOG(StageLog) PetscErrorCode PetscStageLogGetEventPerfLog(PetscStageLog a, int b, PetscEventPerfLog *c)
{
  (void)a;
  (void)b;
  *c = NULL;
  return PETSC_SUCCESS;
}

  #undef PETSC_DEPRECATED_LOG

PETSC_DEPRECATED_OBJECT("User PetscLogLegacyCallbacksBegin() (since version 3.20)") PETSC_UNUSED static PetscErrorCode (*PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = NULL;
PETSC_DEPRECATED_OBJECT("User PetscLogLegacyCallbacksBegin() (since version 3.20)") PETSC_UNUSED static PetscErrorCode (*PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = NULL;
PETSC_DEPRECATED_OBJECT("User PetscLogLegacyCallbacksBegin() (since version 3.20)") PETSC_UNUSED static PetscErrorCode (*PetscLogPHC)(PetscObject)                                                            = NULL;
PETSC_DEPRECATED_OBJECT("User PetscLogLegacyCallbacksBegin() (since version 3.20)") PETSC_UNUSED static PetscErrorCode (*PetscLogPHD)(PetscObject)                                                            = NULL;

PETSC_DEPRECATED_FUNCTION("Use PetscLogEventsPause() (since version 3.20)") static inline PetscErrorCode PetscLogPushCurrentEvent_Internal(PetscLogEvent e)
{
  (void)e;
  return PETSC_SUCCESS;
}

PETSC_DEPRECATED_FUNCTION("Use PetscLogEventsUnpause() (since version 3.20)") static inline PetscErrorCode PetscLogPopCurrentEvent_Internal(void)
{
  return PETSC_SUCCESS;
}

#endif /* define PETSCLOGDEPRECATED_H */
