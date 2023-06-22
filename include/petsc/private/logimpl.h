#ifndef PETSC_LOGIMPL_H
#define PETSC_LOGIMPL_H

#include <petsc/private/petscimpl.h>

/* --- PetscLogHandlerImpl: things a log handlers must do that don't need to be exposed --- */

typedef PetscErrorCode (*PetscLogEventActivityFn)(PetscLogHandler, PetscLogState, PetscLogEvent);
typedef PetscErrorCode (*PetscLogStageFn)(PetscLogHandler, PetscLogState, PetscLogStage);
typedef PetscErrorCode (*PetscLogObjectFn)(PetscLogHandler, PetscLogState, PetscObject);
typedef PetscErrorCode (*PetscLogViewFn)(PetscLogHandler, PetscViewer);
typedef PetscErrorCode (*PetscLogDestroyFn)(PetscLogHandler);

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

#define _PETSC_LOG_RESIZABLE_ARRAY(Container,Entry,Key,Constructor,Destructor,Equal) \
  typedef struct _n_PetscLog##Container *PetscLog##Container; \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Create(int,PetscLog##Container*); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Destroy(PetscLog##Container*); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Recapacity(PetscLog##Container,int); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Resize(PetscLog##Container,int); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Push(PetscLog##Container,Entry); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Find(PetscLog##Container,Key,int *); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##GetNumEntries(PetscLog##Container,PetscInt*,PetscInt*); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Get(PetscLog##Container,PetscInt,Entry*); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##GetRef(PetscLog##Container,PetscInt,Entry**); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Set(PetscLog##Container,PetscInt,Entry); \
  struct _n_PetscLog##Container { \
    int num_entries; \
    int max_entries; \
    Entry *array; \
  }; \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Create(int max_init, PetscLog##Container *a_p) \
  { \
    PetscLog##Container a; \
    PetscErrorCode (*constructor)(Entry *) = Constructor; \
    PetscFunctionBegin; \
    PetscCall(PetscNew(a_p)); \
    a = *a_p; \
    a->num_entries = 0; \
    a->max_entries = max_init; \
    if (constructor) { \
      PetscCall(PetscMalloc1(max_init, &(a->array))); \
    } else { \
      PetscCall(PetscCalloc1(max_init, &(a->array))); \
    } \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Destroy(PetscLog##Container *a_p) \
  { \
    PetscLog##Container a; \
    PetscErrorCode (*destructor)(Entry *) = Destructor; \
    PetscFunctionBegin; \
    a = *a_p; \
    *a_p = NULL; \
    if (destructor) { \
      for (int i = 0; i < a->num_entries; i++) { \
        PetscCall((*destructor)(&(a->array[i]))); \
      } \
    } \
    PetscCall(PetscFree(a->array)); \
    PetscCall(PetscFree(a)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Recapacity(PetscLog##Container a, int new_size) \
  { \
    PetscErrorCode (*constructor)(Entry *) = Constructor; \
    PetscFunctionBegin; \
    if (new_size > a->max_entries) { \
      int new_max_entries = 2; \
      int rem_size = PetscMax(0,new_size - 1); \
      Entry *new_array; \
      while (rem_size >>= 1) new_max_entries *= 2; \
      if (constructor) { \
        PetscCall(PetscMalloc1(new_max_entries, &new_array)); \
      } else { \
        PetscCall(PetscCalloc1(new_max_entries, &new_array)); \
      } \
      PetscCall(PetscArraycpy(new_array, a->array, a->num_entries)); \
      PetscCall(PetscFree(a->array)); \
      a->array = new_array; \
      a->max_entries = new_max_entries; \
    } \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Resize(PetscLog##Container a, int new_size) \
  { \
    PetscErrorCode (*constructor)(Entry *) = Constructor; \
    PetscFunctionBegin; \
    PetscCall(PetscLog##Container##Recapacity(a, new_size)); \
    if (constructor) for (int i = a->num_entries; i < new_size; i++) PetscCall((*constructor)(&(a->array[i]))); \
    a->num_entries = PetscMax(a->num_entries, new_size); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Push(PetscLog##Container a, Entry new_entry) \
  { \
    PetscFunctionBegin; \
    PetscCall(PetscLog##Container##Recapacity(a, a->num_entries + 1)); \
    a->array[a->num_entries++] = new_entry; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Find(PetscLog##Container a, Key key, int *idx_p) \
  { \
    PetscErrorCode (*equal)(Entry *,Key,PetscBool *) = Equal; \
    PetscFunctionBegin; \
    *idx_p = -1; \
    if (equal) { \
      for (int i = 0; i < a->num_entries; i++) { \
        PetscBool is_equal; \
        PetscCall((*equal)(&(a->array[i]), key, &is_equal)); \
        if (is_equal) { \
          *idx_p = i; \
          break; \
        } \
      } \
    } \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##GetNumEntries(PetscLog##Container a, PetscInt *num_entries, PetscInt *max_entries) \
  { \
    PetscFunctionBegin; \
    if (num_entries) *num_entries = a->num_entries; \
    if (max_entries) *max_entries = a->max_entries; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Get(PetscLog##Container a, PetscInt i, Entry *entry) \
  { \
    PetscFunctionBegin; \
    PetscCheck(i >= 0 && i < a->num_entries, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %d is not in range [0,%d)", (int) i, a->num_entries); \
    *entry = a->array[i]; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##GetRef(PetscLog##Container a, PetscInt i, Entry **entry) \
  { \
    PetscFunctionBegin; \
    PetscCheck(i >= 0 && i < a->num_entries, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %d is not in range [0,%d)", (int) i, a->num_entries); \
    *entry = &a->array[i]; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Set(PetscLog##Container a, PetscInt i, Entry entry) \
  { \
    PetscFunctionBegin; \
    PetscCheck(i >= 0 && i < a->num_entries, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %d is not in range [0,%d)", (int) i, a->num_entries); \
    a->array[i] = entry; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

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

#define PetscLogResizableArrayPush_Internal(ra,new_elem) \
  PetscInt new_size = ++(ra)->num_entries; \
  PetscCall(PetscLogResizableArrayEnsureSize((ra),new_size)); \
  (ra)->array[new_size-1] = new_elem;

#define PetscLogResizableArrayPush(ra,new_elem) \
  PetscMacroReturnStandard(PetscLogResizableArrayPush_Internal((ra),(new_elem)))

#define PetscLogResizableArrayCreate(ra_p,max_init,_def) \
  PetscMacroReturnStandard( \
    PetscCall(PetscNew(ra_p)); \
    (*(ra_p))->num_entries = 0; \
    (*(ra_p))->max_entries = (max_init); \
    (*(ra_p))->_default = (_def); \
    PetscCall(PetscMalloc1((max_init), &((*(ra_p))->array))); \
  )

/* --- Registration info types that are not part of the public API, but handlers need to know --- */

/* --- PetscEventRegInfo --- */
typedef struct {
  char        *name;       /* The name of this event */
  PetscClassId classid;    /* The class the event is associated with */
  PetscBool    collective; /* Flag this event as collective */
  PetscBool    visible;    /* The flag to print info in summary */
#if 0
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer; /* Associated external tool timer for this event */
#endif
#if defined(PETSC_HAVE_MPE)
  int mpe_id_begin; /* MPE IDs that define the event */
  int mpe_id_end;
#endif
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

/* --- the registry: information about registered things ---

   Log handler instances should not change the registry: it is shared
   data that should be useful to more than one type of logging

 */

typedef int PetscLogClass;

PETSC_INTERN PetscErrorCode PetscLogGetRegistry(PetscLogRegistry *);
PETSC_INTERN PetscErrorCode PetscLogRegistryCreate(PetscLogRegistry *);
PETSC_INTERN PetscErrorCode PetscLogRegistryDestroy(PetscLogRegistry);
PETSC_INTERN PetscErrorCode PetscLogRegistryLock(PetscLogRegistry);
PETSC_INTERN PetscErrorCode PetscLogRegistryUnlock(PetscLogRegistry);
PETSC_INTERN PetscErrorCode PetscLogRegistryStageRegister(PetscLogRegistry,const char[],PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryEventRegister(PetscLogRegistry,const char[],PetscClassId,PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryClassRegister(PetscLogRegistry,const char[],PetscClassId,PetscLogClass *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetEventFromName(PetscLogRegistry, const char[], PetscLogEvent *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetStageFromName(PetscLogRegistry, const char[], PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetClassFromClassId(PetscLogRegistry, PetscClassId, PetscLogClass *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumEvents(PetscLogRegistry, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumStages(PetscLogRegistry, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumClasses(PetscLogRegistry, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogRegistryEventGetInfo(PetscLogRegistry, PetscLogEvent, PetscEventRegInfo *);
PETSC_INTERN PetscErrorCode PetscLogRegistryStageGetInfo(PetscLogRegistry, PetscLogStage, PetscStageRegInfo *);
PETSC_INTERN PetscErrorCode PetscLogRegistryClassGetInfo(PetscLogRegistry, PetscLogClass, PetscClassRegInfo *);
PETSC_INTERN PetscErrorCode PetscLogRegistryEventSetInfo(PetscLogRegistry, PetscLogEvent, PetscEventRegInfo);
PETSC_INTERN PetscErrorCode PetscLogRegistryStageSetInfo(PetscLogRegistry, PetscLogStage, PetscStageRegInfo);
PETSC_INTERN PetscErrorCode PetscLogRegistryClassSetInfo(PetscLogRegistry, PetscLogClass, PetscClassRegInfo);

/* --- globally synchronized registry information --- */

typedef struct _n_PetscLogGlobalNames *PetscLogGlobalNames;

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesCreate(MPI_Comm, PetscInt, const char **, PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesDestroy(PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGetSize(PetscLogGlobalNames, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGlobalGetName(PetscLogGlobalNames, PetscInt, const char **);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGlobalGetLocal(PetscLogGlobalNames, PetscInt, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesLocalGetGlobal(PetscLogGlobalNames, PetscInt, PetscInt *);
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
PETSC_INTERN PetscErrorCode PetscLogStateEventSetActiveAll(PetscLogState, PetscLogEvent, PetscBool);

/* --- A simple stack --- */

struct _n_PetscIntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

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
/* Registration functions */
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
