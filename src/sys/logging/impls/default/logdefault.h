#if !defined(PETSCLOGDEFAULT_H)
#define PETSCLOGDEFAULT_H

#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/

typedef struct _PetscStageInfo {
  char              *name;     /* The stage name */
  PetscBool          used;     /* The stage was pushed on this processor */
  PetscBool          active;
  PetscEventPerfInfo perfInfo; /* The stage performance information */
  PetscEventPerfLog  eventLog; /* The event information for this stage */
  PetscClassPerfLog  classLog; /* The class information for this stage */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer; /* Associated external tool timer for this stage */
#endif

} PetscStageInfo;

typedef struct _n_PetscStageLog *PetscStageLog;
struct _n_PetscStageLog {
  int              max_entries;
  int              num_entries;
  PetscStageInfo  *array;
  PetscLogRegistry registry;
  PetscSpinlock    lock;
};

PETSC_INTERN PetscErrorCode PetscLogGetDefaultHandler(PetscStageLog *);

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscStageLog, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **);

PETSC_INTERN PetscErrorCode PetscStageLogDuplicate(PetscStageLog, PetscStageLog *);

PETSC_EXTERN PetscErrorCode PetscLogEventBeginDefault(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_EXTERN PetscErrorCode PetscLogEventEndDefault(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_EXTERN PetscErrorCode PetscLogEventBeginComplete(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_EXTERN PetscErrorCode PetscLogEventEndComplete(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_EXTERN PetscErrorCode PetscLogEventBeginTrace(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);
PETSC_EXTERN PetscErrorCode PetscLogEventEndTrace(PetscLogState, PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject, void *);

#endif // #define PETSCLOGDEFAULT_H
