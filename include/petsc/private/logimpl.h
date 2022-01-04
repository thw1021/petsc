#ifndef PETSC_LOGIMPL_H
#define PETSC_LOGIMPL_H

#include <petsc/private/petscimpl.h>
#include <petsclog.h>

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
PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogNestedEnd(void);
PETSC_INTERN PetscErrorCode PetscLogView_Flamegraph(PetscViewer);
#endif /* PETSC_USE_LOG */

#endif /* PETSC_LOGIMPL_H */
