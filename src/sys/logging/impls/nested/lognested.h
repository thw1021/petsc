#if !defined(PETSC_LOGNESTED_H)
#define PETSC_LOGNESTED_H

#include <petsc/private/logimpl.h>
#include <../src/sys/logging/impls/default/logdefault.h>

typedef enum {PETSC_NESTED_AWAKE, PETSC_NESTED_STAGE, PETSC_NESTED_EVENT} PetscNestedObjectType;

typedef PetscLogEvent NestedEventId;
typedef struct {
  PetscNestedObjectType type;
  NestedEventId  nstEvent;         // event-code for this nested event, argument 'event' in PetscLogEventStartNested
  PetscLogEvent  lastDftEvent;     // last default event activated under this nested event
  int            nParents;         // number of 'dftParents': the default timer which was the dftParentActive when this nested timer was activated
  PetscLogEvent *dftParentsSorted; // The default timers which were the dftParentActive when this nested event was started
  PetscLogEvent *dftEvents;        // The default timers which represent the different 'instances' of this nested event
  PetscLogEvent *dftParents;       // The default timers which were the dftParentActive when this nested event was started
  PetscLogEvent *dftEventsSorted;  // The default timers which represent the different 'instances' of this nested event
} PetscNestedEvent;

PETSC_LOG_RESIZABLE_ARRAY(PetscNestedEvent,PetscNestedEventLog)
PETSC_LOG_RESIZABLE_ARRAY(NestedEventId,NestedEventMap)

typedef struct _n_PetscLogHandler_Nested *PetscLogHandler_Nested;
struct _n_PetscLogHandler_Nested {
  PetscLogState nested_state;
  PetscStageLog nested_handler;
  PetscNestedEventLog nested_events;
  PetscLogDouble threshold_time;
  NestedEventMap nested_stage_to_root_stage;
  NestedEventMap nested_event_to_root_event;
};

#endif // #define PETSC_LOGNESTED_H
