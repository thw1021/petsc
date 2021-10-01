#ifndef PETSC_LOGIMPL_H
#define PETSC_LOGIMPL_H

#include <petsc/private/petscimpl.h>
#include <petsctime.h>

/* A simple stack */
struct _n_PetscIntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

/* The structure for action logging */
#define CREATE      0
#define DESTROY     1
#define ACTIONBEGIN 2
#define ACTIONEND   3
typedef struct _Action {
  int            action;        /* The type of execution */
  PetscLogEvent  event;         /* The event number */
  PetscClassId   classid;        /* The event class id */
  PetscLogDouble time;          /* The time of occurence */
  PetscLogDouble flops;         /* The cumlative flops */
  PetscLogDouble mem;           /* The current memory usage */
  PetscLogDouble maxmem;        /* The maximum memory usage */
  int            id1, id2, id3; /* The ids of associated objects */
} Action;

/* The structure for object logging */
typedef struct _Object {
  PetscObject    obj;      /* The associated PetscObject */
  int            parent;   /* The parent id */
  PetscLogDouble mem;      /* The memory associated with the object */
  char           name[64]; /* The object name */
  char           info[64]; /* The information string */
} Object;

/* Action and object logging variables */
PETSC_INTERN Action    *petsc_actions;
PETSC_INTERN Object    *petsc_objects;
PETSC_INTERN PetscBool  petsc_logActions;
PETSC_INTERN PetscBool  petsc_logObjects;
PETSC_INTERN int        petsc_numActions;
PETSC_INTERN int        petsc_maxActions;
PETSC_INTERN int        petsc_numObjects;
PETSC_INTERN int        petsc_maxObjects;
PETSC_INTERN int        petsc_numObjectsDestroyed;

PETSC_INTERN FILE          *petsc_tracefile;
PETSC_INTERN int            petsc_tracelevel;
PETSC_INTERN const char    *petsc_traceblanks;
PETSC_INTERN char           petsc_tracespace[128];
PETSC_INTERN PetscLogDouble petsc_tracetime;

#ifdef PETSC_USE_LOG

PETSC_INTERN PetscErrorCode PetscIntStackCreate(PetscIntStack *);
PETSC_INTERN PetscErrorCode PetscIntStackDestroy(PetscIntStack);
PETSC_INTERN PetscErrorCode PetscIntStackPush(PetscIntStack, int);
PETSC_INTERN PetscErrorCode PetscIntStackPop(PetscIntStack, int *);
PETSC_INTERN PetscErrorCode PetscIntStackTop(PetscIntStack, int *);
PETSC_INTERN PetscErrorCode PetscIntStackEmpty(PetscIntStack, PetscBool  *);

/* Creation and destruction functions */
PETSC_INTERN PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *);
PETSC_INTERN PetscErrorCode PetscEventRegLogDestroy(PetscEventRegLog);
PETSC_INTERN PetscErrorCode PetscEventPerfLogCreate(PetscEventPerfLog *);
PETSC_INTERN PetscErrorCode PetscEventPerfLogDestroy(PetscEventPerfLog);
/* General functions */
PETSC_INTERN PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog, int);
PETSC_INTERN PetscErrorCode PetscEventPerfInfoClear(PetscEventPerfInfo *);
PETSC_INTERN PetscErrorCode PetscEventPerfInfoCopy(PetscEventPerfInfo *, PetscEventPerfInfo *);
/* Registration functions */
PETSC_INTERN PetscErrorCode PetscEventRegLogRegister(PetscEventRegLog, const char [], PetscClassId, PetscLogEvent *);
/* Query functions */
PETSC_INTERN PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool);
PETSC_INTERN PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool  *);
/* Activaton functions */
PETSC_INTERN PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog, PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog, PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscEventPerfLogDeactivatePush(PetscEventPerfLog,PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscEventPerfLogDeactivatePop(PetscEventPerfLog,PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscEventPerfLogActivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);
PETSC_INTERN PetscErrorCode PetscEventPerfLogDeactivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);

/* Logging functions */
PETSC_INTERN PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_INTERN PetscErrorCode PetscLogEventEndDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_INTERN PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_INTERN PetscErrorCode PetscLogEventEndComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_INTERN PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_INTERN PetscErrorCode PetscLogEventEndTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);

/* Creation and destruction functions */
PETSC_INTERN PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *);
PETSC_INTERN PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog);
PETSC_INTERN PetscErrorCode PetscClassPerfLogCreate(PetscClassPerfLog *);
PETSC_INTERN PetscErrorCode PetscClassPerfLogDestroy(PetscClassPerfLog);
PETSC_INTERN PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *);
/* General functions */
PETSC_INTERN PetscErrorCode PetscClassPerfLogEnsureSize(PetscClassPerfLog, int);
PETSC_INTERN PetscErrorCode PetscClassPerfInfoClear(PetscClassPerfInfo *);
/* Registration functions */
PETSC_INTERN PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog, const char [], PetscClassId);
/* Query functions */
PETSC_INTERN PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog, PetscClassId, int *);
/* Logging functions */
PETSC_INTERN PetscErrorCode PetscLogObjCreateDefault(PetscObject);
PETSC_INTERN PetscErrorCode PetscLogObjDestroyDefault(PetscObject);

/* Creation and destruction functions */
PETSC_INTERN PetscErrorCode PetscStageLogCreate(PetscStageLog *);
PETSC_INTERN PetscErrorCode PetscStageLogDestroy(PetscStageLog);
/* Registration functions */
PETSC_INTERN PetscErrorCode PetscStageLogRegister(PetscStageLog, const char [], int *);
/* Runtime functions */
PETSC_INTERN PetscErrorCode PetscStageLogPush(PetscStageLog, int);
PETSC_INTERN PetscErrorCode PetscStageLogPop(PetscStageLog);
PETSC_INTERN PetscErrorCode PetscStageLogSetActive(PetscStageLog, int, PetscBool);
PETSC_INTERN PetscErrorCode PetscStageLogGetActive(PetscStageLog, int, PetscBool  *);
PETSC_INTERN PetscErrorCode PetscStageLogSetVisible(PetscStageLog, int, PetscBool);
PETSC_INTERN PetscErrorCode PetscStageLogGetVisible(PetscStageLog, int, PetscBool  *);
PETSC_INTERN PetscErrorCode PetscStageLogGetStage(PetscStageLog, const char [], PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscStageLogGetClassRegLog(PetscStageLog, PetscClassRegLog *);
PETSC_INTERN PetscErrorCode PetscStageLogGetEventRegLog(PetscStageLog, PetscEventRegLog *);
PETSC_INTERN PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog, int, PetscClassPerfLog *);

PETSC_INTERN PetscErrorCode PetscEventRegLogGetEvent(PetscEventRegLog, const char [], PetscLogEvent *);

PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogNestedEnd(void);
PETSC_INTERN PetscErrorCode PetscLogView_Flamegraph(PetscViewer);
#endif /* PETSC_USE_LOG */
#endif /* PETSC_LOGIMPL_H */
