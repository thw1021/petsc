#if !defined(PETSCLOGDEPRECATED_H)
#define PETSCLOGDEPRECATED_H

#include <petscsystypes.h>
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

PETSC_DEPRECATED_FUNCTION("Use PetscLog interface functions (since version 3.20)") static inline PetscErrorCode PetscLogGetStageLog(PetscStageLog *s)
{
  *s = NULL;
  return PETSC_SUCCESS;
}

PETSC_DEPRECATED_FUNCTION("Use PetscLog interface functions (since version 3.20)") static inline PetscErrorCode PetscStageLogGetCurrent(PetscStageLog p, int *s)
{
  *s = -1;
  return PETSC_SUCCESS;
}

PETSC_DEPRECATED_FUNCTION("Use PetscLog interface functions (since version 3.20)") static inline PetscErrorCode PetscStageLogGetEventPerfLog(PetscStageLog s, int i, PetscEventPerfLog *p)
{
  *p = NULL;
  return PETSC_SUCCESS;
}

PETSC_DEPRECATED_OBJECT("Create a PetscLogHandler object (since version 3.20)") PETSC_UNUSED static PetscErrorCode (*PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = NULL;
PETSC_DEPRECATED_OBJECT("Create a PetscLogHandler object (since version 3.20)") PETSC_UNUSED static PetscErrorCode (*PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = NULL;
PETSC_DEPRECATED_OBJECT("Create a PetscLogHandler object (since version 3.20)") PETSC_UNUSED static PetscErrorCode (*PetscLogPHC)(PetscObject);
PETSC_DEPRECATED_OBJECT("Create a PetscLogHandler object (since version 3.20)") PETSC_UNUSED static PetscErrorCode (*PetscLogPHD)(PetscObject);

PETSC_DEPRECATED_FUNCTION("Create a PetscLogHandler object (since version 3.20)") static inline PetscErrorCode PetscLogSet(PetscErrorCode (*a)(int, int, PetscObject, PetscObject, PetscObject, PetscObject), PetscErrorCode (*b)(int, int, PetscObject, PetscObject, PetscObject, PetscObject))
{
  (void)a;
  (void)b;
  return PETSC_SUCCESS;
}

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
