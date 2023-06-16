#if !defined(PETSCLOGDEFAULT_H)
#define PETSCLOGDEFAULT_H

#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/

typedef struct _PetscStageInfo {
  PetscBool          used;     /* The stage was pushed on this processor */
  PetscEventPerfInfo perfInfo; /* The stage performance information */
  PetscEventPerfLog  eventLog; /* The event information for this stage */
  PetscClassPerfLog  classLog; /* The class information for this stage */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer; /* Associated external tool timer for this stage */
#endif

} PetscStageInfo;

/* The structure for action logging */
typedef enum {
  PETSC_LOG_ACTION_CREATE,
  PETSC_LOG_ACTION_DESTROY,
  PETSC_LOG_ACTION_BEGIN,
  PETSC_LOG_ACTION_END,
} PetscLogActionType;

typedef struct _Action {
  PetscLogActionType action;        /* The type of execution */
  PetscLogEvent      event;         /* The event number */
  PetscClassId       classid;       /* The event class id */
  PetscLogDouble     time;          /* The time of occurrence */
  PetscLogDouble     flops;         /* The cumulative flops */
  PetscLogDouble     mem;           /* The current memory usage */
  PetscLogDouble     maxmem;        /* The maximum memory usage */
  int                id1, id2, id3; /* The ids of associated objects */
} Action;

/* The structure for object logging */
typedef struct _Object {
  PetscObject    obj;      /* The associated PetscObject */
  int            parent;   /* The parent id */
  PetscLogDouble mem;      /* The memory associated with the object */
  char           name[64]; /* The object name */
  char           info[64]; /* The information string */
} Object;

PETSC_LOG_RESIZABLE_ARRAY(Action, PetscActionLog)
PETSC_LOG_RESIZABLE_ARRAY(Object, PetscObjectLog)

typedef struct _n_PetscStageLog *PetscStageLog;
struct _n_PetscStageLog {
  int              max_entries;
  int              num_entries;
  PetscStageInfo  *array;
  PetscLogRegistry registry;
  PetscSpinlock    lock;
  PetscActionLog petsc_actions;
  PetscObjectLog petsc_objects;
  PetscBool petsc_logActions;
  PetscBool petsc_logObjects;
  int       petsc_numObjectsDestroyed;
  FILE          *petsc_tracefile;
  int            petsc_tracelevel;
  const char    *petsc_traceblanks;
  char           petsc_tracespace[128];
  PetscLogDouble petsc_tracetime;
  PetscBool      PetscLogSyncOn;
  PetscBool      PetscLogMemory;
};

PETSC_INTERN PetscErrorCode PetscLogGetDefaultHandler(PetscStageLog *);

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscStageLog, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **);

PETSC_INTERN PetscErrorCode PetscStageLogDuplicate(PetscStageLog, PetscStageLog *);

PETSC_INTERN PetscErrorCode PetscLogEventBeginDefault(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_INTERN PetscErrorCode PetscLogEventEndDefault(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_INTERN PetscErrorCode PetscLogEventBeginComplete(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_INTERN PetscErrorCode PetscLogEventEndComplete(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_INTERN PetscErrorCode PetscLogEventBeginTrace(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_INTERN PetscErrorCode PetscLogEventEndTrace(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);


/* Action and object logging variables */


#endif // #define PETSCLOGDEFAULT_H
