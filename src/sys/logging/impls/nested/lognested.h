#if !defined(PETSC_LOGNESTED_H)
#define PETSC_LOGNESTED_H

#include <petsc/private/logimpl.h>
#include <../src/sys/logging/impls/default/logdefault.h>
#include <petsc/private/hashmap.h>

typedef int NestedId;

typedef enum {PETSC_NESTED_NULL=-1, PETSC_NESTED_STAGE, PETSC_NESTED_EVENT} NestedIdType;

typedef NestedIdType PetscNestedObjectType;
typedef NestedId     NestedEventId;

static inline NestedIdType  NestedIdToType(NestedId id) {return id < -1 ? PETSC_NESTED_STAGE : id == -1 ? PETSC_NESTED_NULL : PETSC_NESTED_EVENT;}
static inline NestedId      NestedIdFromStage(PetscLogStage stage) {return -(stage+2);}
static inline PetscLogStage NestedIdToStage(NestedId id) {return -(id+2);}
static inline NestedId      NestedIdFromEvent(PetscLogEvent event) {return event;}
static inline PetscLogEvent NestedIdToEvent(NestedId id) {return id;}

typedef struct _n_NestedIdPair NestedIdPair;
struct _n_NestedIdPair {
  NestedId root;
  NestedId leaf;
};

#define NestedIdPairHash(key) PetscHashCombine(PetscHash_UInt32((PetscHash32_t)((key).root)),PetscHash_UInt32((PetscHash32_t)((key).leaf)))
#define NestedIdPairEqual(k1,k2) (((k1).root == (k2).root) && ((k1).leaf == (k2).leaf))

PETSC_HASH_MAP(NestedHash, NestedIdPair, NestedId, NestedIdPairHash, NestedIdPairEqual, -1);

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
  PetscLogState       state;
  PetscLogHandler     handler;
  PetscNestedHash     pair_map;
  PetscIntStack       stack; // stack of nested ids
  PetscClassId        nested_stage_id;
  PetscLogDouble      threshold, threshold_time;
};

typedef struct {
  const char *name;
  PetscInt id;
  PetscInt parent;
  PetscInt num_descendants;
} PetscNestedEventNode;

typedef struct {
  MPI_Comm            comm;
  PetscLogGlobalNames global_events;
  PetscNestedEventNode *nodes;
  PetscEventPerfInfo   *perf;
} PetscNestedEventTree;

typedef enum {
  PETSC_LOG_NESTED_XML,
  PETSC_LOG_NESTED_FLAMEGRAPH
} PetscLogNestedType;

PETSC_INTERN PetscErrorCode PetscLogView_Nested_XML(PetscLogHandler_Nested, PetscNestedEventTree *, PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogView_Nested_Flamegraph(PetscLogHandler_Nested, PetscNestedEventTree *, PetscViewer);

#endif // #define PETSC_LOGNESTED_H
