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

#if defined(PETSC_HAVE_THREADSAFETY)
  /* Map from (threadid,stage,event) to perfInfo data struct */
  #include <petsc/private/hashmap.h>

typedef struct _PetscHashIJKKey {
  PetscInt i, j, k;
} PetscHashIJKKey;

  #define PetscHashIJKKeyHash(key)     PetscHashCombine(PetscHashInt((key).i), PetscHashCombine(PetscHashInt((key).j), PetscHashInt((key).k)))
  #define PetscHashIJKKeyEqual(k1, k2) (((k1).i == (k2).i) ? (((k1).j == (k2).j) ? ((k1).k == (k2).k) : 0) : 0)

PETSC_HASH_MAP(HMapEvent, PetscHashIJKKey, PetscEventPerfInfo *, PetscHashIJKKeyHash, PetscHashIJKKeyEqual, NULL)
#endif

typedef struct _n_PetscStageLog *PetscStageLog;
struct _n_PetscStageLog {
  int              max_entries;
  int              num_entries;
  PetscStageInfo   _default;
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
#if defined(PETSC_HAVE_THREADSAFETYE)
  PetscHMapEvent eventInfoMap_th = NULL;
#endif

};

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Default(PetscLogHandler *);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscLogHandler, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **);
PETSC_INTERN PetscErrorCode PetscStageLogDuplicate(PetscStageLog, PetscStageLog *);
PETSC_INTERN PetscErrorCode PetscLogEventBeginDefault(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_INTERN PetscErrorCode PetscLogEventEndDefault(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogActions(PetscLogHandler, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogObjects(PetscLogHandler, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogDefaultHandlerLogObjectState(PetscLogHandler, PetscObject, const char [], va_list);


/* Action and object logging variables */


#endif // #define PETSCLOGDEFAULT_H
