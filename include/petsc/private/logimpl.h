#ifndef PETSC_LOGIMPL_H
#define PETSC_LOGIMPL_H

#include <petsc/private/petscimpl.h>

/* --- PetscLogHandlerImpl: things a log handlers must do that don't need to be exposed --- */

typedef PetscErrorCode (*PetscLogEventActivityFn)(PetscLogState, PetscLogEvent, void *);
typedef PetscErrorCode (*PetscLogStageFn)(PetscLogState, PetscLogStage, void *);
typedef PetscErrorCode (*PetscLogObjectFn)(PetscLogState, PetscObject, void *);
typedef PetscErrorCode (*PetscLogViewFn)(PetscViewer, void *);
typedef PetscErrorCode (*PetscLogDestroyFn)(void *);

typedef enum {PETSC_LOG_HANDLER_DEFAULT, PETSC_LOG_HANDLER_NESTED} PetscLogHandlerType;

struct _n_PetscLogHandlerImpl {
  PetscLogHandlerType     type;
  PetscLogStageFn         stage_push;
  PetscLogStageFn         stage_pop;
  PetscLogEventActivityFn event_deactivate_push;
  PetscLogEventActivityFn event_deactivate_pop;
  PetscLogViewFn          view;
  PetscLogDestroyFn       destroy;
  void *ctx;
};

PETSC_INTERN PetscErrorCode PetscLogHandlerDestroy(PetscLogHandler *);

/* --- Macros for resizable arrays that show up frequently in the implementation of logging --- */

#define PETSC_LOG_RESIZABLE_ARRAY(entrytype,containertype) \
  typedef struct _n_##containertype *containertype; \
  struct _n_##containertype { \
    int max_entries; \
    int num_entries; \
    entrytype _default; \
    entrytype *array; \
  }; \

#define PetscLogResizableArrayEnsureSize(ra,new_size) \
  PetscMacroReturnStandard( \
    if ((new_size) > ra->max_entries) { \
      int new_max_entries = 2; \
      int rem_size = PetscMax(0,(new_size) - 1); \
      char *new_array; \
      char **old_array = (char **) &((ra)->array); \
      while (rem_size >>= 1) new_max_entries *= 2; \
      PetscCall(PetscMalloc(new_max_entries * sizeof(*((ra)->array)), &new_array)); \
      PetscCall(PetscMemcpy(new_array, (ra)->array, sizeof(*((ra)->array)) * (ra)->num_entries)); \
      PetscCall(PetscFree((ra)->array)); \
      *old_array = new_array; \
      (ra)->max_entries = new_max_entries; \
    } \
    for (int i = (ra)->num_entries; i < (new_size); i++) (ra)->array[i] = ((ra)->_default); \
    (ra)->num_entries = (new_size); \
  )

#define PetscLogResizableArrayPush(ra,new_elem) \
  PetscMacroReturnStandard( \
    PetscInt new_size = ++(ra)->num_entries; \
    PetscCall(PetscLogResizableArrayEnsureSize((ra),new_size)); \
    (ra)->array[new_size-1] = new_elem; \
  )

#define PetscLogResizableArrayCreate(ra_p,max_init,_def) \
  PetscMacroReturnStandard( \
    PetscCall(PetscNew(ra_p)); \
    (*(ra_p))->num_entries = 0; \
    (*(ra_p))->max_entries = (max_init); \
    (*(ra_p))->_default = (_def); \
    PetscCall(PetscMalloc1((max_init), &((*(ra_p))->array))); \
  )

/* --- PetscEventPerfInfo (declared in petsclog.h) --- */

PETSC_EXTERN PetscErrorCode PetscEventPerfInfoCopy(const PetscEventPerfInfo *, PetscEventPerfInfo *);
PETSC_INTERN PetscErrorCode PetscEventPerfInfoTic(PetscEventPerfInfo *, PetscLogDouble, PetscBool, int);
PETSC_INTERN PetscErrorCode PetscEventPerfInfoToc(PetscEventPerfInfo *, PetscLogDouble, PetscBool, int);

/* --- Registration info types that are not part of the public API --- */

/* --- PetscEventRegInfo --- */
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
} PetscEventRegInfo;

typedef struct {
  char        *name;    /* The class name */
  PetscClassId classid; /* The integer identifying this class */
} PetscClassRegInfo;

typedef struct _PetscStageRegInfo {
  char              *name;     /* The stage name */
  PetscBool          visible;  /* The flag to print info in summary */
} PetscStageRegInfo;

typedef struct {
  PetscClassId   id;           /* The integer identifying this class */
  int            creations;    /* The number of objects of this class created */
  int            destructions; /* The number of objects of this class destroyed */
  PetscLogDouble mem;          /* The total memory allocated by objects of this class; this is completely wrong and should possibly be removed */
  PetscLogDouble descMem;      /* The total memory allocated by descendents of these objects; this is completely wrong and should possibly be removed */
} PetscClassPerfInfo;

/* --- resizable arrays of the info types --- */

/* --- PetscEventRegLog --- */
PETSC_LOG_RESIZABLE_ARRAY(PetscEventRegInfo,PetscEventRegLog)
PETSC_INTERN PetscErrorCode PetscEventRegLogRegister(PetscEventRegLog, const char[], PetscClassId, PetscLogEvent *);
PETSC_INTERN PetscErrorCode PetscEventRegLogSetCollective(PetscEventRegLog, PetscLogEvent, PetscBool);


/* --- PetscClassRegLog --- */
PETSC_LOG_RESIZABLE_ARRAY(PetscClassRegInfo,PetscClassRegLog)

/* --- PetscClassRegLog --- */
PETSC_LOG_RESIZABLE_ARRAY(PetscClassPerfInfo,PetscClassPerfLog)

/* --- PetscEventPerfLog --- */
PETSC_LOG_RESIZABLE_ARRAY(PetscEventPerfInfo,PetscEventPerfLog)
PETSC_INTERN PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog, int);

/* --- PetscStageRegxLog --- */
PETSC_LOG_RESIZABLE_ARRAY(PetscStageRegInfo,PetscStageRegLog)
PETSC_INTERN PetscErrorCode PetscStageRegLogInsert(PetscStageRegLog, const char[], int *);
PETSC_INTERN PetscErrorCode PetscStageRegLogSetVisible(PetscStageRegLog, PetscLogStage, PetscBool);
PETSC_INTERN PetscErrorCode PetscStageRegLogGetVisible(PetscStageRegLog, PetscLogStage, PetscBool *);
PETSC_INTERN PetscErrorCode PetscStageRegLogGetId(PetscStageRegLog, const char[], PetscLogStage *);

/* --- the registry: information about registered things ---

   Log handler instances should not change the registry: it is shared
   data that should be useful to more than one type of logging

 */

struct _n_PetscLogRegistry {
  PetscEventRegLog events;
  PetscClassRegLog classes;
  PetscStageRegLog stages;
  PetscSpinlock    lock;
};

PETSC_INTERN PetscErrorCode PetscLogGetRegistry(PetscLogRegistry *);
PETSC_INTERN PetscErrorCode PetscLogRegistryCreate(PetscLogRegistry *);
PETSC_INTERN PetscErrorCode PetscLogRegistryDestroy(PetscLogRegistry);
PETSC_INTERN PetscErrorCode PetscLogRegistryStageRegister(PetscLogRegistry,const char[],PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryEventRegister(PetscLogRegistry,const char[],PetscClassId,PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetClassLog(PetscLogRegistry, PetscClassRegLog *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetEventLog(PetscLogRegistry, PetscEventRegLog *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetEvent(PetscLogRegistry, const char[], PetscLogEvent *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetStageLog(PetscLogRegistry, PetscStageRegLog *);
PETSC_INTERN PetscErrorCode PetscLogRegistryLock(PetscLogRegistry);
PETSC_INTERN PetscErrorCode PetscLogRegistryUnlock(PetscLogRegistry);

/* --- globally synchronized registry information --- */

typedef struct _n_PetscLogGlobalNames *PetscLogGlobalNames;
struct _n_PetscLogGlobalNames {
  MPI_Comm     comm;
  PetscInt     count;
  const char **names;
  PetscInt    *global_to_local;
  PetscInt    *local_to_global;
};

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesCreate(MPI_Comm, PetscInt, const char **, PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesDestroy(PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogRegistryCreateGlobalStageNames(MPI_Comm, PetscLogRegistry, PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogRegistryCreateGlobalEventNames(MPI_Comm, PetscLogRegistry, PetscLogGlobalNames *);

/* --- methods for PetscLogState --- */

PETSC_INTERN PetscErrorCode PetscLogGetState(PetscLogState *);
PETSC_INTERN PetscErrorCode PetscLogStateCreate(PetscLogState *);
PETSC_INTERN PetscErrorCode PetscLogStateGetRegistry(PetscLogState, PetscLogRegistry *);
PETSC_INTERN PetscErrorCode PetscLogStateDestroy(PetscLogState);
PETSC_INTERN PetscErrorCode PetscLogStateStagePush(PetscLogState,PetscLogStage);
PETSC_INTERN PetscErrorCode PetscLogStateStagePop(PetscLogState);
PETSC_INTERN PetscErrorCode PetscLogStateStageSetActive(PetscLogState, PetscLogStage, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogStateStageGetActive(PetscLogState, PetscLogStage, PetscBool *);
PETSC_INTERN PetscErrorCode PetscLogStateGetCurrentStage(PetscLogState, PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogStateStageRegister(PetscLogState, const char[], PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogStateEventRegister(PetscLogState, const char[], PetscClassId, PetscLogEvent *);
PETSC_INTERN PetscErrorCode PetscLogStateLock(PetscLogState);
PETSC_INTERN PetscErrorCode PetscLogStateUnlock(PetscLogState);
PETSC_INTERN PetscErrorCode PetscLogStateEventIncludeClass(PetscLogState, PetscClassId);
PETSC_INTERN PetscErrorCode PetscLogStateEventExcludeClass(PetscLogState, PetscClassId);
PETSC_INTERN PetscErrorCode PetscLogStateEventActivateClass(PetscLogState, PetscClassId);
PETSC_INTERN PetscErrorCode PetscLogStateEventDeactivateClass(PetscLogState, PetscClassId);
PETSC_INTERN PetscErrorCode PetscLogStateEventActivate(PetscLogState, PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscLogStateEventDeactivate(PetscLogState, PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscLogStateEventActivateAll(PetscLogState, PetscLogEvent);

/* --- A simple stack --- */

struct _n_PetscIntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

PETSC_EXTERN PetscErrorCode PetscIntStackCreate(PetscIntStack *);
PETSC_EXTERN PetscErrorCode PetscIntStackDestroy(PetscIntStack);
PETSC_EXTERN PetscErrorCode PetscIntStackPush(PetscIntStack, int);
PETSC_EXTERN PetscErrorCode PetscIntStackPop(PetscIntStack, int *);
PETSC_EXTERN PetscErrorCode PetscIntStackTop(PetscIntStack, int *);
PETSC_EXTERN PetscErrorCode PetscIntStackEmpty(PetscIntStack, PetscBool *);

/* --- Thread-safety internals --- */

#if defined(PETSC_HAVE_THREADSAFETY)
  #if defined(__cplusplus)
    #define PETSC_TLS thread_local
  #else
    #define PETSC_TLS _Thread_local
  #endif
  #define PETSC_INTERN_TLS extern PETSC_TLS PETSC_VISIBILITY_INTERNAL

/* Access PETSc internal thread id */
PETSC_INTERN PetscInt PetscLogGetTid(void);
#else
  #define PETSC_TLS
  #define PETSC_INTERN_TLS PETSC_INTERN
#endif

#ifdef PETSC_USE_LOG


/* Query functions */
PETSC_EXTERN PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool *);
/* Activaton functions */
PETSC_EXTERN PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDeactivatePush(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDeactivatePop(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogActivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);
PETSC_EXTERN PetscErrorCode PetscEventPerfLogDeactivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);

/* Logging functions */
/* Creation and destruction functions */
PETSC_EXTERN PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *);
/* Registration functions */
PETSC_EXTERN PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog, const char[], PetscClassId);
/* Query functions */
PETSC_EXTERN PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog, PetscClassId, int *);

PETSC_EXTERN PetscErrorCode PetscEventRegLogGetEvent(PetscEventRegLog, const char[], PetscLogEvent *);

PETSC_EXTERN PetscErrorCode PetscLogGetEventLog(PetscEventRegLog *);
PETSC_EXTERN PetscErrorCode PetscLogGetClassLog(PetscClassRegLog *);

PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscLogHandler, PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogView_Default(PetscLogHandler, PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogDump_Default(PetscLogHandler, const char []);
PETSC_INTERN PetscErrorCode PetscLogNestedEnd(void);

  #if defined(PETSC_HAVE_DEVICE)
PETSC_EXTERN PetscBool PetscLogGpuTimeFlag;
  #endif
#endif /* PETSC_USE_LOG */

#define PETSC_LOG_VIEW_FROM_OPTIONS_MAX 4
#endif /* PETSC_LOGIMPL_H */
