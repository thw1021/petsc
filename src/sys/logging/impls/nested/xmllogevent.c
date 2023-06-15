/*************************************************************************************
 *    M A R I T I M E  R E S E A R C H  I N S T I T U T E  N E T H E R L A N D S     *
 *************************************************************************************
 *    authors: Bas van 't Hof, Koos Huijssen, Christiaan M. Klaij                    *
 *************************************************************************************
 *    content: Support for nested PetscTimers                                        *
 *************************************************************************************/
#include <petsclog.h> /*I "petsclog.h" I*/
#include <petsc/private/logimpl.h>
#include <petsctime.h>
#include <petscviewer.h>
#include <../src/sys/logging/impls/nested/xmlviewer.h>
#include <../src/sys/logging/interface/plog.h>

#if defined(PETSC_USE_LOG)

/*
 * Support for nested PetscTimers
 *
 * PetscTimers keep track of a lot of useful information: Wall clock times,
 * message passing statistics, flop counts.  Information about the nested structure
 * of the timers is lost. Example:
 *
 * 7:30   Start: awake
 * 7:30      Start: morning routine
 * 7:40         Start: eat
 * 7:49         Done:  eat
 * 7:43      Done:  morning routine
 * 8:15      Start: work
 * 12:15        Start: eat
 * 12:45        Done:  eat
 * 16:00     Done:  work
 * 16:30     Start: evening routine
 * 18:30        Start: eat
 * 19:15        Done:  eat
 * 22:00     Done:  evening routine
 * 22:00  Done:  awake
 *
 * Petsc timers provide the following timer results:
 *
 *    awake:              1 call    14:30 hours
 *    morning routine:    1 call     0:13 hours
 *    eat:                3 calls    1:24 hours
 *    work:               1 call     7:45 hours
 *    evening routine     1 call     5:30 hours
 *
 * Nested timers can be used to get the following table:
 *
 *   [1 call]: awake                14:30 hours
 *   [1 call]:    morning routine         0:13 hours         ( 2 % of awake)
 *   [1 call]:       eat                       0:09 hours         (69 % of morning routine)
 *                   rest (morning routine)    0:04 hours         (31 % of morning routine)
 *   [1 call]:    work                    7:45 hours         (53 % of awake)
 *   [1 call]:       eat                       0:30 hours         ( 6 % of work)
 *                   rest (work)               7:15 hours         (94 % of work)
 *   [1 call]:    evening routine         5:30 hours         (38 % of awake)
 *   [1 call]:       eat                       0:45 hours         (14 % of evening routine)
 *                   rest (evening routine)    4:45 hours         (86 % of morning routine)
 *
 * In default logging, events and stages both accumulate performance statistics, the
 * only difference is that in PetscLogView_Default() the stages are used to aggregate
 * event timers separately.  In nested logging, events and stages are treated almost identically.
 *
 */

/*
 * Data structures for keeping track of nested timers:
 *
 *   nestedEvents: information about the timers that have actually been activated
 *   dftParentActive: if a timer is started now, it is part of (nested inside) the dftParentActive
 *
 * The Default-timers are used to time the nested timers. Every nested timer corresponds to
 * (one or more) default timers, where one of the default timers has the same event-id as the
 * nested one.
 *
 * Because of the risk of confusion between nested timer ids and default timer ids, we
 * introduce a typedef for nested events (NestedEventId) and use the existing type PetscLogEvent
 * only for default events. Also, all nested event variables are prepended with 'nst', and
 * default timers with 'dft'.
 */

  #define DFT_ID_AWAKE    -1
  #define MAINSTAGE_EVENT -2

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

static PetscLogEvent     dftParentActive             = DFT_ID_AWAKE;
static int               nNestedEvents               = 0;
static int               nNestedEventsAllocated      = 0;
static PetscNestedEvent *nestedEvents                = NULL;
static PetscLogDouble    thresholdTime               = 0.01; /* initial value was 0.1 */
static size_t            num_stages_allocated        = 0;
static size_t            num_events_allocated        = 0;
static NestedEventId    *nested_stage_to_root_stage  = NULL;
static NestedEventId    *nested_event_to_root_event  = NULL;

static PetscErrorCode DefaultStageToNestedStage(NestedEventId id, NestedEventId *root_id)
{
  PetscLogRegistry registry;

  PetscFunctionBegin;
  PetscCall(PetscLogGetRegistry(&registry));
  if (!nested_stage_to_root_stage) {
    PetscCall(PetscMalloc1(registry->stages->num_entries, &nested_stage_to_root_stage));
    for (int i = 0; i < registry->stages->num_entries; i++) nested_stage_to_root_stage[i] = i;
    num_stages_allocated = registry->stages->num_entries;
  }
  if (id < num_stages_allocated) {
    *root_id = nested_stage_to_root_stage[id];
  } else {
    *root_id = id;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DefaultStageSetNestedStage(NestedEventId id, NestedEventId root_id)
{
  PetscFunctionBegin;
  if (id >= num_stages_allocated) {
    size_t         new_num_stages_allocated = (id + 1) * 2;
    NestedEventId *new_table;

    PetscCall(PetscMalloc1(new_num_stages_allocated, &new_table));
    PetscCall(PetscArraycpy(new_table, nested_stage_to_root_stage, num_stages_allocated));
    PetscCall(PetscFree(nested_stage_to_root_stage));
    for (int i = num_stages_allocated; i < new_num_stages_allocated; i++) new_table[i] = i;
    nested_stage_to_root_stage = new_table;
    num_stages_allocated        = new_num_stages_allocated;
  }
  nested_stage_to_root_stage[id] = root_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DefaultEventToNestedEvent(NestedEventId id, NestedEventId *root_id)
{
  PetscEventRegLog eventLog;

  PetscFunctionBegin;
  PetscCall(PetscLogGetEventLog(&eventLog));
  if (!nested_event_to_root_event) {
    PetscCall(PetscMalloc1(eventLog->num_entries, &nested_event_to_root_event));
    for (int i = 0; i < eventLog->num_entries; i++) nested_event_to_root_event[i] = i;
    num_events_allocated = eventLog->num_entries;
  }
  if (id < num_events_allocated) {
    *root_id = nested_event_to_root_event[id];
  } else {
    *root_id = id;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DefaultEventSetNestedEvent(NestedEventId id, NestedEventId root_id)
{
  PetscFunctionBegin;
  if (id >= num_events_allocated) {
    size_t         new_num_events_allocated = (id + 1) * 2;
    NestedEventId *new_table;

    PetscCall(PetscMalloc1(new_num_events_allocated, &new_table));
    PetscCall(PetscArraycpy(new_table, nested_event_to_root_event, num_events_allocated));
    PetscCall(PetscFree(nested_event_to_root_event));
    for (int i = num_events_allocated; i < new_num_events_allocated; i++) new_table[i] = i;
    nested_event_to_root_event = new_table;
    num_events_allocated        = new_num_events_allocated;
  }
  nested_event_to_root_event[id] = root_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #define THRESHOLD (thresholdTime / 100.0 + 1e-12)

static PetscErrorCode       PetscLogEventBeginNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4);
static PetscErrorCode       PetscLogEventEndNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4);
//static PetscErrorCode       PetscLogStageBeginHandler_Nested(PetscStageLog);
//static PetscErrorCode       PetscLogStageEndHandler_Nested(PetscStageLog);
PETSC_INTERN PetscErrorCode PetscLogView_Nested(PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogView_Flamegraph(PetscViewer);
static PetscClassId         LogNestedEvent = -1;

/*@C
  PetscLogNestedBegin - Turns on nested logging of objects and events. This logs flop
  rates and object creation and should not slow programs down too much.

  Logically Collective over `PETSC_COMM_WORLD`

  Options Database Keys:
. -log_view :filename.xml:ascii_xml - Prints an XML summary of flop and timing information to the file

  Usage:
.vb
      PetscInitialize(...);
      PetscLogNestedBegin();
       ... code ...
      PetscLogView(viewer);
      PetscFinalize();
.ve

  Level: advanced

.seealso: `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogView()`, `PetscLogTraceBegin()`, `PetscLogDefaultBegin()`
@*/
PetscErrorCode PetscLogNestedBegin(void)
{
  PetscFunctionBegin;
  PetscCheck(!nestedEvents, PETSC_COMM_SELF, PETSC_ERR_COR, "nestedEvents already allocated");

  nNestedEventsAllocated = 10;
  PetscCall(PetscMalloc1(nNestedEventsAllocated, &nestedEvents));
  dftParentActive = DFT_ID_AWAKE;
  nNestedEvents   = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Delete the data structures for the nested timers */
PetscErrorCode PetscLogNestedEnd(void)
{
  int i;

  PetscFunctionBegin;
  if (!nestedEvents) PetscFunctionReturn(PETSC_SUCCESS);
  for (i = 0; i < nNestedEvents; i++) PetscCall(PetscFree4(nestedEvents[i].dftParentsSorted, nestedEvents[i].dftEventsSorted, nestedEvents[i].dftParents, nestedEvents[i].dftEvents));
  PetscCall(PetscFree(nestedEvents));
  nestedEvents           = NULL;
  nNestedEvents          = 0;
  nNestedEventsAllocated = 0;
  num_stages_allocated   = 0;
  num_events_allocated   = 0;
  PetscCall(PetscFree(nested_stage_to_root_stage));
  PetscCall(PetscFree(nested_event_to_root_event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 UTILITIES: FIND STUFF IN SORTED ARRAYS

    dftIndex - index to be found
    dftArray - sorted array of PetscLogEvent-ids
    narray - dimension of dftArray
    entry - entry in the array where dftIndex may be found;

     if dftArray[entry] != dftIndex, then dftIndex is not part of dftArray
     In that case, the dftIndex can be inserted at this entry.
*/
static PetscErrorCode PetscLogEventFindDefaultTimer(PetscLogEvent dftIndex, const PetscLogEvent *dftArray, int narray, int *entry)
{
  PetscFunctionBegin;
  if (narray == 0 || dftIndex <= dftArray[0]) {
    *entry = 0;
  } else if (dftIndex > dftArray[narray - 1]) {
    *entry = narray;
  } else {
    int ihigh = narray - 1, ilow = 0;
    while (ihigh > ilow) {
      const int imiddle = (ihigh + ilow) / 2;
      if (dftArray[imiddle] > dftIndex) {
        ihigh = imiddle;
      } else if (dftArray[imiddle] < dftIndex) {
        ilow = imiddle + 1;
      } else {
        ihigh = imiddle;
        ilow  = imiddle;
      }
    }
    *entry = ihigh;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Utility: find the nested event with given identification

    nstEvent - Nested event to be found
    entry - entry in the nestedEvents where nstEvent may be found;

    if nestedEvents[entry].nstEvent != nstEvent, then index is not part of iarray
*/
static PetscErrorCode PetscLogEventFindNestedTimer(NestedEventId nstEvent, int *entry)
{
  PetscFunctionBegin;
  if (nNestedEvents == 0 || nstEvent <= nestedEvents[0].nstEvent) {
    *entry = 0;
  } else if (nstEvent > nestedEvents[nNestedEvents - 1].nstEvent) {
    *entry = nNestedEvents;
  } else {
    int ihigh = nNestedEvents - 1, ilow = 0;
    while (ihigh > ilow) {
      const int imiddle = (ihigh + ilow) / 2;
      if (nestedEvents[imiddle].nstEvent > nstEvent) {
        ihigh = imiddle;
      } else if (nestedEvents[imiddle].nstEvent < nstEvent) {
        ilow = imiddle + 1;
      } else {
        ihigh = imiddle;
        ilow  = imiddle;
      }
    }
    *entry = ihigh;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStagePush_Internal(PetscLogStage);
PETSC_INTERN PetscErrorCode PetscLogStagePop_Internal();

/******************************************************************************************/
/* Start a nested event or stage */
static PetscErrorCode PetscLogEventBeginNested_Internal(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, PetscBool is_event)
{
  int           entry, pentry, tentry, i;
  PetscLogEvent dftEvent;

  PetscFunctionBegin;
  PetscCall(PetscLogEventFindNestedTimer(nstEvent, &entry));
  if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != nstEvent) {
    /* Nested event doesn't exist yet: create it */

    if (nNestedEvents == nNestedEventsAllocated) {
      /* Enlarge and re-allocate nestedEvents if needed */
      PetscNestedEvent *tmp = nestedEvents;
      PetscCall(PetscMalloc1(2 * nNestedEvents, &nestedEvents));
      nNestedEventsAllocated *= 2;
      PetscCall(PetscArraycpy(nestedEvents, tmp, nNestedEvents));
      PetscCall(PetscFree(tmp));
    }

    /* Clear space in nestedEvents for new nested event */
    nNestedEvents++;
    for (i = nNestedEvents - 1; i > entry; i--) nestedEvents[i] = nestedEvents[i - 1];

    /* Create event in nestedEvents */
    nestedEvents[entry].nstEvent = nstEvent;
    nestedEvents[entry].nParents = 1;
    PetscCall(PetscMalloc4(1, &nestedEvents[entry].dftParentsSorted, 1, &nestedEvents[entry].dftEventsSorted, 1, &nestedEvents[entry].dftParents, 1, &nestedEvents[entry].dftEvents));

    /* Fill in new event */
    pentry   = 0;
    dftEvent = (PetscLogEvent)nstEvent;

    nestedEvents[entry].nstEvent                 = nstEvent;
    nestedEvents[entry].dftParents[pentry]       = dftParentActive;
    nestedEvents[entry].dftEvents[pentry]        = dftEvent;
    nestedEvents[entry].dftParentsSorted[pentry] = dftParentActive;
    nestedEvents[entry].dftEventsSorted[pentry]  = dftEvent;

  } else {
    /* Nested event exists: find current dftParentActive among parents */
    PetscLogEvent *dftParentsSorted = nestedEvents[entry].dftParentsSorted;
    PetscLogEvent *dftEvents        = nestedEvents[entry].dftEvents;
    int            nParents         = nestedEvents[entry].nParents;

    PetscCall(PetscLogEventFindDefaultTimer(dftParentActive, dftParentsSorted, nParents, &pentry));

    if (pentry >= nParents || dftParentActive != dftParentsSorted[pentry]) {
      /* dftParentActive not in the list: add it to the list */
      int            i, current_stage;
      PetscLogEvent *dftParents      = nestedEvents[entry].dftParents;
      PetscLogEvent *dftEventsSorted = nestedEvents[entry].dftEventsSorted;
      char           name[100];

      /* Register a new default timer */
      if (!is_event) PetscCall(PetscLogStagePop_Internal());
      PetscCall(PetscLogStageGetCurrent(&current_stage));
      PetscCall(PetscSNPrintf(name, PETSC_STATIC_ARRAY_LENGTH(name), "__Nested %d: %d -> %d", current_stage, (int)dftParentActive, (int)nstEvent));
      if (is_event) {
        PetscCall(PetscLogEventRegister(name, LogNestedEvent, &dftEvent));
        PetscCall(DefaultEventSetNestedEvent(dftEvent, nstEvent));
      } else {
        PetscCall(PetscLogStageRegister(name, &dftEvent));
        // stop the timer for the default stage and start it for the nested stage
        PetscCall(PetscInfo(NULL, "Swapping stage %d for new stage %d\n", (int)-(nstEvent + 2), dftEvent));
        PetscCall(PetscLogStagePush_Internal(dftEvent));
        PetscCall(DefaultStageSetNestedStage(dftEvent, -(nstEvent + 2)));
        dftEvent = -(dftEvent + 2);
      }
      PetscCall(PetscLogEventFindDefaultTimer(dftEvent, dftEventsSorted, nParents, &tentry));

      /* Reallocate parents and dftEvents to make space for new parent */
      PetscCall(PetscMalloc4(1 + nParents, &nestedEvents[entry].dftParentsSorted, 1 + nParents, &nestedEvents[entry].dftEventsSorted, 1 + nParents, &nestedEvents[entry].dftParents, 1 + nParents, &nestedEvents[entry].dftEvents));
      PetscCall(PetscArraycpy(nestedEvents[entry].dftParentsSorted, dftParentsSorted, nParents));
      PetscCall(PetscArraycpy(nestedEvents[entry].dftEventsSorted, dftEventsSorted, nParents));
      PetscCall(PetscArraycpy(nestedEvents[entry].dftParents, dftParents, nParents));
      PetscCall(PetscArraycpy(nestedEvents[entry].dftEvents, dftEvents, nParents));
      PetscCall(PetscFree4(dftParentsSorted, dftEventsSorted, dftParents, dftEvents));

      dftParents       = nestedEvents[entry].dftParents;
      dftEvents        = nestedEvents[entry].dftEvents;
      dftParentsSorted = nestedEvents[entry].dftParentsSorted;
      dftEventsSorted  = nestedEvents[entry].dftEventsSorted;

      nestedEvents[entry].nParents++;
      nParents++;

      for (i = nParents - 1; i > pentry; i--) {
        dftParentsSorted[i] = dftParentsSorted[i - 1];
        dftEvents[i]        = dftEvents[i - 1];
      }
      for (i = nParents - 1; i > tentry; i--) {
        dftParents[i]      = dftParents[i - 1];
        dftEventsSorted[i] = dftEventsSorted[i - 1];
      }

      /* Fill in the new default timer */
      dftParentsSorted[pentry] = dftParentActive;
      dftEvents[pentry]        = dftEvent;
      dftParents[tentry]       = dftParentActive;
      dftEventsSorted[tentry]  = dftEvent;
    } else {
      /* dftParentActive was found: find the corresponding default 'dftEvent'-timer */
      dftEvent = nestedEvents[entry].dftEvents[pentry];
    }
  }

  /* Start the default 'dftEvent'-timer and update the dftParentActive */
  if (is_event) PetscCall(PetscLogEventBeginDefault(dftEvent, t, o1, o2, o3, o4));

  dftParentActive = dftEvent;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/******************************************************************************************/
/* Start a nested event */
static PetscErrorCode PetscLogEventBeginNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBeginNested_Internal(nstEvent, t, o1, o2, o3, o4, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/******************************************************************************************/
/* Start a nested stage */
static PetscErrorCode PetscLogStageBeginHandler_Nested(PetscLogState log_state)
{
  PetscInt stage_id = log_state->stage_stack->stack[log_state->stage_stack->top];

  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL, "Pushing stage %d\n", (int)stage_id));
  PetscCall(PetscLogEventBeginNested_Internal(-(stage_id + 2), 0, NULL, NULL, NULL, NULL, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* End a nested event */
static PetscErrorCode PetscLogEventEndNested_Internal(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4, PetscBool is_event)
{
  int            entry, pentry, nParents;
  PetscLogEvent *dftEventsSorted;

  PetscFunctionBegin;
  /* Find the nested event */
  PetscCall(PetscLogEventFindNestedTimer(nstEvent, &entry));
  PetscCheck(entry < nNestedEvents, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event %d larger than number of events %d", entry, nNestedEvents);
  PetscCheck(nstEvent < 0 || nestedEvents[entry].nstEvent == nstEvent, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event %d had unbalanced begin/end pairs does not match %d", entry, nstEvent);
  dftEventsSorted = nestedEvents[entry].dftEventsSorted;
  nParents        = nestedEvents[entry].nParents;

  /* Find the current default timer among the 'dftEvents' of this event */
  PetscCall(PetscLogEventFindDefaultTimer(dftParentActive, dftEventsSorted, nParents, &pentry));

  PetscCheck(pentry < nParents, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Entry %d is larger than number of parents %d", pentry, nParents);
  PetscCheck(dftEventsSorted[pentry] == dftParentActive, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Active parent is %d, but we seem to be closing %d", dftParentActive, dftEventsSorted[pentry]);

  /* Stop the default timer and update the dftParentActive */
  if (is_event) PetscCall(PetscLogEventEndDefault(dftParentActive, t, o1, o2, o3, o4));
  dftParentActive = nestedEvents[entry].dftParents[pentry];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* End a nested event */
static PetscErrorCode PetscLogEventEndNested(NestedEventId nstEvent, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventEndNested_Internal(nstEvent, t, o1, o2, o3, o4, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStageEndHandler_Nested(PetscLogState log_state)
{
  PetscInt stage_id = log_state->stage_stack->stack[log_state->stage_stack->top];
  PetscInt root_stage;

  PetscFunctionBegin;
  PetscCall(DefaultStageToNestedStage(stage_id, &root_stage));
  PetscCall(PetscInfo(NULL, "Popping stage %d\n", (int)root_stage));
  PetscCall(PetscLogEventEndNested_Internal(-(root_stage + 2), 0, NULL, NULL, NULL, NULL, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscLogSetThreshold - Set the threshold time for logging the events; this is a percentage out of 100, so 1. means any event
          that takes 1 or more percent of the time.

  Logically Collective over `PETSC_COMM_WORLD`

  Input Parameter:
.   newThresh - the threshold to use

  Output Parameter:
.   oldThresh - the previously set threshold value

  Options Database Keys:
. -log_view :filename.xml:ascii_xml - Prints an XML summary of flop and timing information to the file

  Usage:
.vb
      PetscInitialize(...);
      PetscLogNestedBegin();
      PetscLogSetThreshold(0.1,&oldthresh);
       ... code ...
      PetscLogView(viewer);
      PetscFinalize();
.ve

  Level: advanced

.seealso: `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogView()`, `PetscLogTraceBegin()`, `PetscLogDefaultBegin()`,
          `PetscLogNestedBegin()`
@*/
PetscErrorCode PetscLogSetThreshold(PetscLogDouble newThresh, PetscLogDouble *oldThresh)
{
  PetscFunctionBegin;
  if (oldThresh) *oldThresh = thresholdTime;
  if (newThresh == (PetscLogDouble)PETSC_DECIDE) newThresh = 0.01;
  if (newThresh == (PetscLogDouble)PETSC_DEFAULT) newThresh = 0.01;
  thresholdTime = PetscMax(newThresh, 0.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPrintExeSpecs(PetscViewer viewer)
{
  char        arch[128], hostname[128], username[128], pname[PETSC_MAX_PATH_LEN], date[128];
  char        version[256], buildoptions[128] = "";
  PetscMPIInt size;
  size_t      len;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
  PetscCall(PetscGetArchType(arch, sizeof(arch)));
  PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
  PetscCall(PetscGetUserName(username, sizeof(username)));
  PetscCall(PetscGetProgramName(pname, sizeof(pname)));
  PetscCall(PetscGetDate(date, sizeof(date)));
  PetscCall(PetscGetVersion(version, sizeof(version)));

  PetscCall(PetscViewerXMLStartSection(viewer, "runspecification", "Run Specification"));
  PetscCall(PetscViewerXMLPutString(viewer, "executable", "Executable", pname));
  PetscCall(PetscViewerXMLPutString(viewer, "architecture", "Architecture", arch));
  PetscCall(PetscViewerXMLPutString(viewer, "hostname", "Host", hostname));
  PetscCall(PetscViewerXMLPutInt(viewer, "nprocesses", "Number of processes", size));
  PetscCall(PetscViewerXMLPutString(viewer, "user", "Run by user", username));
  PetscCall(PetscViewerXMLPutString(viewer, "date", "Started at", date));
  PetscCall(PetscViewerXMLPutString(viewer, "petscrelease", "Petsc Release", version));

  if (PetscDefined(USE_DEBUG)) PetscCall(PetscStrlcat(buildoptions, "Debug ", sizeof(buildoptions)));
  if (PetscDefined(USE_COMPLEX)) PetscCall(PetscStrlcat(buildoptions, "Complex ", sizeof(buildoptions)));
  if (PetscDefined(USE_REAL_SINGLE)) {
    PetscCall(PetscStrlcat(buildoptions, "Single ", sizeof(buildoptions)));
  } else if (PetscDefined(USE_REAL___FLOAT128)) {
    PetscCall(PetscStrlcat(buildoptions, "Quadruple ", sizeof(buildoptions)));
  } else if (PetscDefined(USE_REAL___FP16)) {
    PetscCall(PetscStrlcat(buildoptions, "Half ", sizeof(buildoptions)));
  }
  if (PetscDefined(USE_64BIT_INDICES)) PetscCall(PetscStrlcat(buildoptions, "Int64 ", sizeof(buildoptions)));
  #if defined(__cplusplus)
  PetscCall(PetscStrlcat(buildoptions, "C++ ", sizeof(buildoptions)));
  #endif
  PetscCall(PetscStrlen(buildoptions, &len));
  if (len) PetscCall(PetscViewerXMLPutString(viewer, "petscbuildoptions", "Petsc build options", buildoptions));
  PetscCall(PetscViewerXMLEndSection(viewer, "runspecification"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintXMLGlobalPerformanceElement(PetscViewer viewer, const char *name, const char *desc, PetscLogDouble local_val, const PetscBool print_average, const PetscBool print_total)
{
  PetscLogDouble min, tot, ratio, avg;
  MPI_Comm       comm;
  PetscMPIInt    rank, size;
  PetscLogDouble valrank[2], max[2];

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  valrank[0] = local_val;
  valrank[1] = (PetscLogDouble)rank;
  PetscCall(MPIU_Allreduce(&local_val, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCall(MPIU_Allreduce(valrank, &max, 1, MPIU_2PETSCLOGDOUBLE, MPI_MAXLOC, comm));
  PetscCall(MPIU_Allreduce(&local_val, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max[0] / min;
  else ratio = 0.0;

  PetscCall(PetscViewerXMLStartSection(viewer, name, desc));
  PetscCall(PetscViewerXMLPutDouble(viewer, "max", NULL, max[0], "%e"));
  PetscCall(PetscViewerXMLPutInt(viewer, "maxrank", "rank at which max was found", (PetscMPIInt)max[1]));
  PetscCall(PetscViewerXMLPutDouble(viewer, "ratio", NULL, ratio, "%f"));
  if (print_average) PetscCall(PetscViewerXMLPutDouble(viewer, "average", NULL, avg, "%e"));
  if (print_total) PetscCall(PetscViewerXMLPutDouble(viewer, "total", NULL, tot, "%e"));
  PetscCall(PetscViewerXMLEndSection(viewer, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintGlobalPerformance(PetscViewer viewer, PetscLogDouble locTotalTime)
{
  PetscLogDouble  flops, mem, red, mess;
  const PetscBool print_total_yes = PETSC_TRUE, print_total_no = PETSC_FALSE, print_average_no = PETSC_FALSE, print_average_yes = PETSC_TRUE;

  PetscFunctionBegin;
  /* Must preserve reduction count before we go on */
  red = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;

  /* Calculate summary information */
  PetscCall(PetscViewerXMLStartSection(viewer, "globalperformance", "Global performance"));

  /*   Time */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "time", "Time (sec)", locTotalTime, print_average_yes, print_total_no));

  /*   Objects */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "objects", "Objects", (PetscLogDouble)petsc_numObjects, print_average_yes, print_total_no));

  /*   Flop */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "mflop", "MFlop", petsc_TotalFlops / 1.0E6, print_average_yes, print_total_yes));

  /*   Flop/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = petsc_TotalFlops / locTotalTime;
  else flops = 0.0;
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "mflops", "MFlop/sec", flops / 1.0E6, print_average_yes, print_total_yes));

  /*   Memory */
  PetscCall(PetscMallocGetMaximumUsage(&mem));
  if (mem > 0.0) PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "memory", "Memory (MiB)", mem / 1024.0 / 1024.0, print_average_yes, print_total_yes));
  /*   Messages */
  mess = 0.5 * (petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "messagetransfers", "MPI Message Transfers", mess, print_average_yes, print_total_yes));

  /*   Message Volume */
  mess = 0.5 * (petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "messagevolume", "MPI Message Volume (MiB)", mess / 1024.0 / 1024.0, print_average_yes, print_total_yes));

  /*   Reductions */
  PetscCall(PetscPrintXMLGlobalPerformanceElement(viewer, "reductions", "MPI Reductions", red, print_average_no, print_total_no));
  PetscCall(PetscViewerXMLEndSection(viewer, "globalperformance"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  const char *name;
  PetscInt id;
  PetscInt parent;
  PetscInt num_descendants;
} PetscNestedEventNode;

static PetscErrorCode PetscLogNestedEventNodesOrderDepthFirst(PetscInt num_nodes, PetscInt parent, PetscNestedEventNode tree[], PetscInt *num_descendants)
{
  PetscInt node, start_loc;
  PetscFunctionBegin;

  node = 0;
  start_loc = 0;
  while (node < num_nodes) {
    if (tree[node].parent == parent) {
      PetscInt num_this_descendants = 0;
      PetscNestedEventNode tmp = tree[start_loc];
      tree[start_loc] = tree[node];
      tree[node] = tmp;
      PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes - start_loc - 1, tree[start_loc].id, &tree[start_loc + 1], &num_this_descendants));
      tree[start_loc].num_descendants = num_this_descendants;
      *num_descendants += 1 + num_this_descendants;
      start_loc += 1 + num_this_descendants;
      node = start_loc;
    } else {
      node++;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedEventNodesFillPerf(PetscInt num_nodes, PetscInt parent, PetscNestedEventNode tree[], PetscEventPerfInfo perf[], PetscStageInfo *current_stage, PetscStageLog stage_log, PetscLogGlobalNames global_stages, PetscLogGlobalNames global_events)
{
  PetscInt node;

  PetscFunctionBegin;
  node = 0;
  while (node < num_nodes) {
    PetscAssert(tree[node].parent == parent, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed tree traversal");
    if (tree[node].id < global_stages->count) {
      // this is a stage
      PetscInt stage_id = global_stages->global_to_local[tree[node].id];
      if (stage_id >= 0) { // stage was called locally, skip subtree
        PetscStageInfo *new_current_stage = &stage_log->array[stage_id];
        PetscCall(PetscLogNestedEventNodesFillPerf(tree[node].num_descendants, tree[node].id, &tree[node + 1], &perf[node + 1], new_current_stage, stage_log, global_stages, global_events));
        if (current_stage) {
          // add back in subtracted values so that stages have timings that are inclusive, like events
          current_stage->perfInfo.time  += new_current_stage->perfInfo.time;
          current_stage->perfInfo.flops += new_current_stage->perfInfo.flops;
          current_stage->perfInfo.numMessages += new_current_stage->perfInfo.numMessages;
          current_stage->perfInfo.messageLength += new_current_stage->perfInfo.messageLength;
          current_stage->perfInfo.numReductions += new_current_stage->perfInfo.numReductions;
        }
        perf[node] = new_current_stage->perfInfo;
      }
    } else {
      // this is an event
      PetscInt event_id = global_events->global_to_local[tree[node].id - global_stages->count];

      if (current_stage && event_id >= 0 && event_id < current_stage->eventLog->num_entries) {
        perf[node] = current_stage->eventLog->array[event_id];
        PetscCall(PetscLogNestedEventNodesFillPerf(tree[node].num_descendants, tree[node].id, &tree[node + 1], &perf[node + 1], current_stage, stage_log, global_stages, global_events));
      }
    }
    node += 1 + tree[node].num_descendants;
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedCreatePerfTree(MPI_Comm comm, PetscStageLog stage_log, PetscLogGlobalNames global_stages, PetscLogGlobalNames global_events, PetscNestedEventNode **tree_p, PetscEventPerfInfo **perf_p)
{
  PetscMPIInt size;
  PetscInt num_nodes = global_stages->count + global_events->count;
  PetscEventPerfInfo *perf;
  PetscNestedEventNode *tree;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(num_nodes, &tree));
  for (PetscInt node = 0; node < num_nodes; node++) {
    tree[node].id = node;
    tree[node].parent = -1;
  }
  for (PetscInt stage = 0; stage < global_stages->count; stage++) {
    PetscInt node = stage;
    PetscInt stage_id = global_stages->global_to_local[stage];

    tree[node].name = global_stages->names[stage];

    if (stage_id >= 0) {
      PetscNestedEvent *nested_event;
      PetscLogEvent    *dftParents;
      PetscLogEvent     parent_id;
      PetscLogEvent    *dftEventsSorted;
      int               entry;
      int               nParents, tentry;
      NestedEventId     root_id;

      PetscCall(DefaultStageToNestedStage(stage_id, &root_id));
      PetscCall(PetscLogEventFindNestedTimer(-(root_id+2), &entry));
      if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != -(root_id + 2)) continue;
      nested_event = &nestedEvents[entry];
      dftParents      = nested_event->dftParents;
      dftEventsSorted = nested_event->dftEventsSorted;
      nParents         = nested_event->nParents;
      PetscCall(PetscLogEventFindDefaultTimer(-(stage_id+2), dftEventsSorted, nParents, &tentry));
      PetscAssert(dftEventsSorted[tentry] == -(stage_id+2), PETSC_COMM_SELF, PETSC_ERR_PLIB, "nested event is unrecognized by root event");
      parent_id = dftParents[tentry];
      if (parent_id >= 0) {
        // parent is an event
        tree[node].parent = global_events->local_to_global[parent_id] + global_stages->count;
      } else if (parent_id <= MAINSTAGE_EVENT) {
        // parent is a stage
        tree[node].parent = global_stages->local_to_global[-(parent_id+2)];
      }
    }
  }
  for (PetscInt event = 0; event < global_events->count; event++) {
    PetscInt node = event + global_stages->count;
    PetscInt event_id = global_events->global_to_local[event];

    tree[node].name = global_events->names[event];
    if (event_id >= 0) {
      PetscNestedEvent *nested_event;
      PetscLogEvent    *dftParents;
      PetscLogEvent     parent_id;
      PetscLogEvent    *dftEventsSorted;
      int               entry;
      int               nParents, tentry;
      NestedEventId     root_id;

      PetscCall(DefaultEventToNestedEvent(event_id, &root_id));
      PetscCall(PetscLogEventFindNestedTimer(root_id, &entry));
      if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != root_id) continue;
      nested_event = &nestedEvents[entry];
      dftParents      = nested_event->dftParents;
      dftEventsSorted = nested_event->dftEventsSorted;
      nParents         = nested_event->nParents;
      PetscCall(PetscLogEventFindDefaultTimer(event_id, dftEventsSorted, nParents, &tentry));
      PetscAssert(dftEventsSorted[tentry] == event_id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "nested event is unrecognized by root event");
      parent_id = dftParents[tentry];
      if (parent_id >= 0) {
        // parent is an event
        tree[node].parent = global_events->local_to_global[parent_id] + global_stages->count;
      } else if (parent_id <= MAINSTAGE_EVENT) {
        // parent is a stage
        tree[node].parent = global_stages->local_to_global[-(parent_id+2)];
      }
    }
  }

  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) { // get missing parents from other processes
    PetscInt *parents;

    PetscCall(PetscMalloc1(num_nodes, &parents));
    for (PetscInt node = 0; node < num_nodes; node++) parents[node] = tree[node].parent;
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, parents, num_nodes, MPIU_INT, MPI_MAX, comm));
    for (PetscInt node = 0; node < num_nodes; node++) tree[node].parent = parents[node];
    PetscCall(PetscFree(parents));
  }

  PetscInt num_descendants = 0;

  PetscCall(PetscLogNestedEventNodesOrderDepthFirst(num_nodes, -1, tree, &num_descendants));
  PetscAssert(num_descendants == num_nodes, comm, PETSC_ERR_PLIB, "Failed tree ordering invariant");

  PetscCall(PetscCalloc1(num_nodes, &perf));
  PetscCall(PetscLogNestedEventNodesFillPerf(num_nodes, -1, tree, perf, NULL, stage_log, global_stages, global_events));

  *tree_p = tree;
  *perf_p = perf;
  PetscFunctionReturn(0);
}

typedef struct {
  MPI_Comm            comm;
  PetscStageLog       stage_log;
  PetscLogGlobalNames global_stages;
  PetscLogGlobalNames global_events;
  PetscNestedEventNode *tree;
  PetscEventPerfInfo   *perf;
} PetscNestedEventTreeNew;

/* Print the global performance: max, max/min, average and total of
 *      time, objects, flops, flops/sec, memory, MPI messages, MPI message lengths, MPI reductions.
 */
static PetscErrorCode PetscPrintXMLNestedLinePerfResults(PetscViewer viewer, const char *name, PetscLogDouble value, PetscLogDouble minthreshold, PetscLogDouble maxthreshold, PetscLogDouble minmaxtreshold)
{
  MPI_Comm       comm; /* MPI communicator in reduction */
  PetscMPIInt    rank; /* rank of this process */
  PetscLogDouble val_in[2], max[2], min[2];
  PetscLogDouble minvalue, maxvalue, tot;
  PetscMPIInt    size;
  PetscMPIInt    minLoc, maxLoc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  val_in[0] = value;
  val_in[1] = (PetscLogDouble)rank;
  PetscCall(MPIU_Allreduce(val_in, max, 1, MPIU_2PETSCLOGDOUBLE, MPI_MAXLOC, comm));
  PetscCall(MPIU_Allreduce(val_in, min, 1, MPIU_2PETSCLOGDOUBLE, MPI_MINLOC, comm));
  maxvalue = max[0];
  maxLoc   = (PetscMPIInt)max[1];
  minvalue = min[0];
  minLoc   = (PetscMPIInt)min[1];
  PetscCall(MPIU_Allreduce(&value, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));

  if (maxvalue < maxthreshold && minvalue >= minthreshold) {
    /* One call per parent or NO value: don't print */
  } else {
    PetscCall(PetscViewerXMLStartSection(viewer, name, NULL));
    if (maxvalue > minvalue * minmaxtreshold) {
      PetscCall(PetscViewerXMLPutDouble(viewer, "avgvalue", NULL, tot / size, "%g"));
      PetscCall(PetscViewerXMLPutDouble(viewer, "minvalue", NULL, minvalue, "%g"));
      PetscCall(PetscViewerXMLPutDouble(viewer, "maxvalue", NULL, maxvalue, "%g"));
      PetscCall(PetscViewerXMLPutInt(viewer, "minloc", NULL, minLoc));
      PetscCall(PetscViewerXMLPutInt(viewer, "maxloc", NULL, maxLoc));
    } else {
      PetscCall(PetscViewerXMLPutDouble(viewer, "value", NULL, tot / size, "%g"));
    }
    PetscCall(PetscViewerXMLEndSection(viewer, name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  int            id;
  PetscLogDouble val;
} PetscSortItem;

static int compareSortItems(const void *item1_, const void *item2_)
{
  PetscSortItem *item1 = (PetscSortItem *)item1_;
  PetscSortItem *item2 = (PetscSortItem *)item2_;
  if (item1->val > item2->val) return -1;
  if (item1->val < item2->val) return +1;
  return 0;
}

static PetscErrorCode PetscLogNestedTreePrintLineNew(PetscViewer viewer, const PetscEventPerfInfo *perfInfo, PetscLogDouble countsPerCall, int parentCount, const char *name, PetscLogDouble totalTime)
{
  PetscLogDouble time = perfInfo->time;
  PetscLogDouble timeMx;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(MPIU_Allreduce(&time, &timeMx, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(PetscViewerXMLPutString(viewer, "name", NULL, name));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "time", time / totalTime * 100.0, 0, 0, 1.02));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "ncalls", parentCount > 0 ? countsPerCall : 1.0, 0.99, 1.01, 1.02));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "mflops", time >= timeMx * 0.001 ? 1e-6 * perfInfo->flops / time : 0, 0, 0.01, 1.05));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "mbps", time >= timeMx * 0.001 ? perfInfo->messageLength / (1024 * 1024 * time) : 0, 0, 0.01, 1.05));
  PetscCall(PetscPrintXMLNestedLinePerfResults(viewer, "nreductsps", time >= timeMx * 0.001 ? perfInfo->numReductions / time : 0, 0, 0.01, 1.05));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNestedNameGetBase(const char name[], const char *base[])
{
  size_t n;
  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &n));
  while (n > 0 && name[n-1] != ';') n--;
  *base = &name[n];
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef enum {
  PETSC_LOG_NESTED_XML,
  PETSC_LOG_NESTED_FLAMEGRAPH
} PetscLogNestedType;

// prints and leaves perf holding self times
static PetscErrorCode PetscLogNestedTreePrintNew(PetscViewer viewer, double total_time, const PetscNestedEventNode *parent_node, PetscEventPerfInfo *parent_info, const PetscNestedEventNode tree[], PetscEventPerfInfo perf[], PetscLogNestedType type)
{
  PetscInt num_children = 0;
  PetscInt num_nodes = parent_node->num_descendants;
  PetscInt *perm;
  PetscReal *times;
  PetscEventPerfInfo other;

  PetscFunctionBegin;
  for (PetscInt node = 0; node < num_nodes; node += 1 + tree[node].num_descendants) {
    PetscAssert(tree[node].parent == parent_node->id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed tree invariant");
    num_children++;
  }
  PetscCall(PetscMemzero(&other, sizeof(other)));
  PetscCall(PetscMalloc2(num_children + 2, &times, num_children + 2, &perm));
  for (PetscInt i = 0, node = 0; node < num_nodes; i++, node += 1 + tree[node].num_descendants) {
    PetscLogDouble child_time = perf[node].time;

    perm[i] = node;
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &child_time, 1, MPI_DOUBLE, MPI_MAX, PetscObjectComm((PetscObject)viewer)));
    times[i] = -child_time;

    parent_info->time          -= perf[node].time;
    parent_info->flops         -= perf[node].flops;
    parent_info->numMessages   -= perf[node].numMessages;
    parent_info->messageLength -= perf[node].messageLength;
    parent_info->numReductions -= perf[node].numReductions;
    if (child_time / total_time < THRESHOLD) {
      PetscEventPerfInfo *add_to = (type == PETSC_LOG_NESTED_XML) ? &other : parent_info;

      add_to->time          += perf[node].time;
      add_to->flops         += perf[node].flops;
      add_to->numMessages   += perf[node].numMessages;
      add_to->messageLength += perf[node].messageLength;
      add_to->numReductions += perf[node].numReductions;
      add_to->count         += perf[node].count;
    }
  }
  perm[num_children] = -1;
  times[num_children] = -parent_info->time;
  perm[num_children + 1] = -2;
  times[num_children + 1] = -other.time;
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &times[num_children], 2, MPI_DOUBLE, MPI_MIN, PetscObjectComm((PetscObject)viewer)));
  if (type == PETSC_LOG_NESTED_FLAMEGRAPH) {
    /* The output is given as an integer in microseconds because otherwise the file cannot be read
     * by apps such as speedscope (https://speedscope.app/). */
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s %" PetscInt64_FMT "\n", parent_node->name, (PetscInt64)(-times[num_children] * 1e6)));
  }
  // sort descending by time
  PetscCall(PetscSortRealWithArrayInt(num_children + 2, times, perm));

  if (type == PETSC_LOG_NESTED_XML) PetscCall(PetscViewerXMLStartSection(viewer, "events", NULL));
  for (PetscInt i = 0; i < num_children + 2; i++) {
    PetscInt node = perm[i];
    PetscLogDouble child_time = -times[i];

    if (child_time / total_time >= THRESHOLD || (node < 0 && child_time > 0.0)) {
      if (type == PETSC_LOG_NESTED_XML) {
        PetscCall(PetscViewerXMLStartSection(viewer, "event", NULL));
        if (node == -1) {
          PetscCall(PetscLogNestedTreePrintLineNew(viewer, parent_info, 0, 0, "self", total_time));
        } else if (node == -2) {
          PetscCall(PetscLogNestedTreePrintLineNew(viewer, &other, ((double) other.count) / ((double) parent_info->count), parent_info->count, "other", total_time));
        } else {
          const char *base_name;
          PetscCall(PetscNestedNameGetBase(tree[node].name, &base_name));
          PetscCall(PetscLogNestedTreePrintLineNew(viewer, &perf[node], ((double) perf[node].count) / ((double) parent_info->count), parent_info->count, base_name, total_time));
          PetscCall(PetscLogNestedTreePrintNew(viewer, total_time, &tree[node], &perf[node], &tree[node+1], &perf[node+1], type));
        }
        PetscCall(PetscViewerXMLEndSection(viewer, "event"));
      } else if (node >= 0) {
        PetscCall(PetscLogNestedTreePrintNew(viewer, total_time, &tree[node], &perf[node], &tree[node+1], &perf[node+1], type));
      }
    }
  }
  if (type == PETSC_LOG_NESTED_XML) PetscCall(PetscViewerXMLEndSection(viewer, "events"));

  PetscCall(PetscFree2(times, perm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogNestedTreePrintTopNew(PetscViewer viewer, PetscNestedEventTreeNew *tree, PetscLogDouble *total_time, PetscLogNestedType type)
{
  PetscNestedEventNode *main_stage;
  PetscNestedEventNode *tree_rem;
  PetscEventPerfInfo   *main_stage_perf;
  PetscEventPerfInfo   *perf_rem;
  PetscLogDouble        time;

  PetscFunctionBegin;
  main_stage = &tree->tree[0];
  tree_rem = &tree->tree[1];
  main_stage_perf = &tree->perf[0];
  perf_rem = &tree->perf[1];
  time = main_stage_perf->time;
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, tree->comm));
  *total_time = time;
  /* Print (or ignore) the children in ascending order of total time */
  if (type == PETSC_LOG_NESTED_XML) {
    PetscCall(PetscViewerXMLStartSection(viewer, "timertree", "Timings tree"));
    PetscCall(PetscViewerXMLPutDouble(viewer, "totaltime", NULL, time, "%f"));
    PetscCall(PetscViewerXMLPutDouble(viewer, "timethreshold", NULL, thresholdTime, "%f"));
  }
  PetscCall(PetscLogNestedTreePrintNew(viewer, main_stage_perf->time, main_stage, main_stage_perf, tree_rem, perf_rem, type));
  if (type == PETSC_LOG_NESTED_XML) PetscCall(PetscViewerXMLEndSection(viewer, "timertree"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  const char    *name;
  PetscLogDouble time;
  PetscLogDouble flops;
  PetscLogDouble numMessages;
  PetscLogDouble messageLength;
  PetscLogDouble numReductions;
} PetscSelfTimer;

static PetscErrorCode PetscCalcSelfTimeNew(PetscViewer viewer, PetscNestedEventTreeNew *tree, PetscSelfTimer **p_self)
{
  PetscInt global_count = tree->global_stages->count + tree->global_events->count;
  PetscInt event_offset = tree->global_stages->count;
  PetscSelfTimer     *perf_by_id;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(global_count, &perf_by_id));

  for (PetscInt i = 0; i < global_count; i++) {
    PetscInt loc = tree->tree[i].id;

    perf_by_id[loc].name = tree->tree[i].name;
    perf_by_id[loc].time = tree->perf[i].time;
    perf_by_id[loc].flops = tree->perf[i].flops;
    perf_by_id[loc].numMessages = tree->perf[i].numMessages;
    perf_by_id[loc].numReductions = tree->perf[i].numReductions;
  }

  for (PetscInt stage = 0; stage < tree->global_stages->count; stage++) {
    PetscInt stage_id = tree->global_stages->global_to_local[stage];
    if (stage_id >= 0) {
      PetscInt root_stage;
      PetscCall(DefaultStageToNestedStage(stage_id, &root_stage));
      if (root_stage != stage_id) {
        PetscInt global_root = tree->global_stages->local_to_global[root_stage];
        perf_by_id[global_root].time += perf_by_id[stage].time;
        perf_by_id[global_root].flops += perf_by_id[stage].flops;
        perf_by_id[global_root].numMessages += perf_by_id[stage].numMessages;
        perf_by_id[global_root].messageLength += perf_by_id[stage].messageLength;
        perf_by_id[global_root].numReductions += perf_by_id[stage].numReductions;
        PetscCall(PetscMemzero(&perf_by_id[stage], sizeof(*perf_by_id)));
      }
    }
  }

  for (PetscInt event = 0; event < tree->global_events->count; event++) {
    PetscInt event_id = tree->global_events->global_to_local[event];
    if (event_id >= 0) {
      PetscInt root_stage;
      PetscCall(DefaultEventToNestedEvent(event_id, &root_stage));
      if (root_stage != event_id) {
        PetscInt global_root = tree->global_events->local_to_global[root_stage] + event_offset;
        PetscInt global_leaf = event + event_offset;

        perf_by_id[global_root].time += perf_by_id[global_leaf].time;
        perf_by_id[global_root].flops += perf_by_id[global_leaf].flops;
        perf_by_id[global_root].numMessages += perf_by_id[global_leaf].numMessages;
        perf_by_id[global_root].messageLength += perf_by_id[global_leaf].messageLength;
        perf_by_id[global_root].numReductions += perf_by_id[global_leaf].numReductions;
        PetscCall(PetscMemzero(&perf_by_id[global_leaf], sizeof(*perf_by_id)));
      }
    }
  }

  *p_self = perf_by_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPrintSelfTime(PetscViewer viewer, const PetscSelfTimer *selftimes, int num_times, PetscLogDouble totalTime)
{
  int                i;
  NestedEventId      nst;
  PetscSortItem     *sortSelfTimes;
  PetscLogDouble    *times, *maxTimes;
  const int          dum_count = 1, dum_parentcount = 1;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));

  PetscCall(PetscMalloc1(num_times, &times));
  PetscCall(PetscMalloc1(num_times, &maxTimes));
  for (nst = 0; nst < num_times; nst++) times[nst] = selftimes[nst].time;
  PetscCall(MPIU_Allreduce(times, maxTimes, num_times, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCall(PetscFree(times));

  PetscCall(PetscMalloc1(num_times + 1, &sortSelfTimes));

  /* Sort the self-timers on basis of the largest time needed */
  for (nst = 0; nst < num_times; nst++) {
    sortSelfTimes[nst].id  = nst;
    sortSelfTimes[nst].val = maxTimes[nst];
  }
  PetscCall(PetscFree(maxTimes));
  qsort(sortSelfTimes, num_times, sizeof(PetscSortItem), compareSortItems);

  PetscCall(PetscViewerXMLStartSection(viewer, "selftimertable", "Self-timings"));
  PetscCall(PetscViewerXMLPutDouble(viewer, "totaltime", NULL, totalTime, "%f"));

  for (i = 0; i < num_times; i++) {
    if ((sortSelfTimes[i].val / totalTime) >= THRESHOLD) {
      NestedEventId      nstEvent = sortSelfTimes[i].id;
      const char        *name;
      PetscEventPerfInfo selfPerfInfo;

      selfPerfInfo.time          = selftimes[nstEvent].time;
      selfPerfInfo.flops         = selftimes[nstEvent].flops;
      selfPerfInfo.numMessages   = selftimes[nstEvent].numMessages;
      selfPerfInfo.messageLength = selftimes[nstEvent].messageLength;
      selfPerfInfo.numReductions = selftimes[nstEvent].numReductions;

      PetscCall(PetscNestedNameGetBase(selftimes[nstEvent].name, &name));
      PetscCall(PetscViewerXMLStartSection(viewer, "event", NULL));
      PetscCall(PetscLogNestedTreePrintLineNew(viewer, &selfPerfInfo, dum_count, dum_parentcount, name, totalTime));
      PetscCall(PetscViewerXMLEndSection(viewer, "event"));
    }
  }
  PetscCall(PetscViewerXMLEndSection(viewer, "selftimertable"));
  PetscCall(PetscFree(sortSelfTimes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogComputeEventNestedName(PetscStageLog, NestedEventId, char **, char **);

static PetscErrorCode PetscStageLogComputeStageNestedName(PetscStageLog stage_log, NestedEventId stage_id, char **nested_event_names, char **nested_stage_names)
{
  PetscNestedEvent *nested_event;
  PetscLogEvent    *dftParents;
  PetscLogEvent    parent_id;
  PetscLogEvent *dftEventsSorted;
  int            entry;
  int            nParents, tentry;
  char           buf[BUFSIZ];
  NestedEventId root_id;

  PetscFunctionBegin;
  if (nested_stage_names[stage_id]) PetscFunctionReturn(PETSC_SUCCESS);
  if (-(stage_id+2) == MAINSTAGE_EVENT) {
    PetscCall(PetscStrallocpy(stage_log->array[stage_id].name, &nested_stage_names[stage_id]));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DefaultStageToNestedStage(stage_id, &root_id));
  PetscCall(PetscLogEventFindNestedTimer(-(root_id+2), &entry));
  if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != -(root_id + 2)) PetscFunctionReturn(PETSC_SUCCESS);
  nested_event = &nestedEvents[entry];
  dftParents      = nested_event->dftParents;
  dftEventsSorted = nested_event->dftEventsSorted;
  nParents         = nested_event->nParents;
  PetscCall(PetscLogEventFindDefaultTimer(-(stage_id+2), dftEventsSorted, nParents, &tentry));
  PetscAssert(dftEventsSorted[tentry] == -(stage_id+2), PETSC_COMM_SELF, PETSC_ERR_PLIB, "nested event is unrecognized by root event");
  parent_id = dftParents[tentry];
  if (parent_id >= 0) {
    PetscCall(PetscStageLogComputeEventNestedName(stage_log, parent_id, nested_event_names, nested_stage_names));
    PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s;%s", nested_event_names[parent_id], stage_log->array[root_id].name));
  } else {
    if (parent_id <= MAINSTAGE_EVENT) {
      parent_id = -(parent_id+2);
      PetscCall(PetscStageLogComputeStageNestedName(stage_log, parent_id, nested_event_names, nested_stage_names));
      PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s;%s", nested_stage_names[parent_id], stage_log->array[root_id].name));
    } else {
      PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s", stage_log->array[root_id].name));
    }
  }
  PetscCall(PetscStrallocpy(buf, &nested_stage_names[stage_id]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogComputeEventNestedName(PetscStageLog stage_log, NestedEventId event_id, char **nested_event_names, char **nested_stage_names)
{
  PetscNestedEvent *nested_event;
  PetscEventRegLog event_log;
  PetscLogEvent    *dftParents;
  PetscLogEvent    parent_id;
  PetscLogEvent *dftEventsSorted;
  int            entry;
  int            nParents, tentry;
  char           buf[BUFSIZ];
  NestedEventId root_id;

  PetscFunctionBegin;
  if (nested_event_names[event_id]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DefaultEventToNestedEvent(event_id, &root_id));
  PetscCall(PetscLogEventFindNestedTimer(root_id, &entry));
  if (entry >= nNestedEvents || nestedEvents[entry].nstEvent != root_id) PetscFunctionReturn(PETSC_SUCCESS);
  nested_event = &nestedEvents[entry];
  dftParents      = nested_event->dftParents;
  dftEventsSorted = nested_event->dftEventsSorted;
  nParents         = nested_event->nParents;
  PetscCall(PetscLogEventFindDefaultTimer(event_id, dftEventsSorted, nParents, &tentry));
  PetscAssert(dftEventsSorted[tentry] == event_id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "nested event is unrecognized by root event");
  parent_id = dftParents[tentry];
  PetscCall(PetscLogGetEventLog(&event_log));
  if (parent_id >= 0) {
    PetscCall(PetscStageLogComputeEventNestedName(stage_log, parent_id, nested_event_names, nested_stage_names));
    PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s;%s", nested_event_names[parent_id], event_log->array[root_id].name));
  } else {
    if (parent_id <= MAINSTAGE_EVENT) {
      parent_id = -(parent_id+2);
      PetscCall(PetscStageLogComputeStageNestedName(stage_log, parent_id, nested_event_names, nested_stage_names));
      PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s;%s", nested_stage_names[parent_id], event_log->array[root_id].name));
    } else {
      PetscCall(PetscSNPrintf(buf, BUFSIZ, "%s", event_log->array[root_id].name));
    }
  }
  PetscCall(PetscStrallocpy(buf, &nested_event_names[event_id]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageLogNestNames(PetscStageLog stage_log)
{
  PetscInt num_stages = stage_log->num_entries;
  PetscEventRegLog event_log;
  PetscCall(PetscLogGetEventLog(&event_log));
  PetscInt num_events = event_log->num_entries;
  char **nested_stage_names;
  char **nested_event_names;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(num_stages, &nested_stage_names));
  PetscCall(PetscCalloc1(num_events, &nested_event_names));

  for (NestedEventId i = 0; i < num_events; i++) PetscCall(PetscStageLogComputeEventNestedName(stage_log, i, nested_event_names, nested_stage_names));

  for (NestedEventId i = 0; i < num_stages; i++) PetscCall(PetscStageLogComputeStageNestedName(stage_log, i, nested_event_names, nested_stage_names));

  for (NestedEventId i = 0; i < num_events; i++) {
    if (nested_event_names[i]) {
      PetscCall(PetscFree(event_log->array[i].name));
      event_log->array[i].name = nested_event_names[i];
    }
  }

  for (NestedEventId i = 0; i < num_stages; i++) {
    if (nested_stage_names[i]) {
      PetscCall(PetscFree(stage_log->array[i].name));
      stage_log->array[i].name = nested_stage_names[i];
    }
  }

  PetscCall(PetscFree(nested_stage_names));
  PetscCall(PetscFree(nested_event_names));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogView_Nested(PetscViewer viewer)
{
  PetscLogDouble        locTotalTime, globTotalTime;
  PetscStageLog         stage_log_orig;
  PetscLogGlobalNames global_stages, global_events;
  PetscStageLog stage_log_nested;
  PetscNestedEventNode *tree;
  PetscNestedEventTreeNew tree_traversal;
  PetscEventPerfInfo *perf;
  PetscSelfTimer     *self_timers;
  MPI_Comm              comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCall(PetscViewerInitASCII_XML(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "<!-- PETSc Performance Summary: -->\n"));
  PetscCall(PetscViewerXMLStartSection(viewer, "petscroot", NULL));

  // Print global information about this run
  PetscCall(PetscPrintExeSpecs(viewer));
  PetscCall(PetscLogGetDefaultHandler(&stage_log_orig));
  locTotalTime = stage_log_orig->array[0].perfInfo.time; // Main stage time
  PetscCall(PetscPrintGlobalPerformance(viewer, locTotalTime));

  // Get nested names that can be used to unique global identities of stages and events
  PetscCall(PetscStageLogDuplicate(stage_log_orig, &stage_log_nested));
  PetscCall(PetscStageLogNestNames(stage_log_nested));
  PetscCall(PetscStageLogCreateGlobalStageNames(comm, stage_log_nested, &global_stages));
  PetscCall(PetscStageLogCreateGlobalEventNames(comm, stage_log_nested, &global_events));

  // Sort the performance data into a tree (depth-first storage linearization
  PetscCall(PetscLogNestedCreatePerfTree(comm, stage_log_nested, global_stages, global_events, &tree, &perf));
  tree_traversal.comm = comm;
  tree_traversal.global_events = global_events;
  tree_traversal.global_stages = global_stages;
  tree_traversal.perf = perf;
  tree_traversal.tree = tree;
  tree_traversal.stage_log = stage_log_nested;
  PetscCall(PetscLogNestedTreePrintTopNew(viewer, &tree_traversal, &globTotalTime, PETSC_LOG_NESTED_XML));

  // flat self-time (collapsing nested events to their root events
  PetscCall(PetscCalcSelfTimeNew(viewer, &tree_traversal, &self_timers));
  PetscCall(PetscPrintSelfTime(viewer, self_timers, global_stages->count + global_events->count, globTotalTime));
  PetscCall(PetscFree(self_timers));

  PetscCall(PetscFree(perf));
  PetscCall(PetscFree(tree));
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));
  PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
  PetscCall(PetscStageLogDestroy(stage_log_nested));

  PetscCall(PetscViewerXMLEndSection(viewer, "petscroot"));
  PetscCall(PetscViewerFinalASCII_XML(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Print nested logging information to a file suitable for reading into a Flame Graph.
 *
 * The format consists of a semicolon-separated list of events and the event duration in microseconds (which must be an integer).
 * An example output would look like:
 *   Main Stage;MatAssemblyBegin 1
 *   Main Stage;MatAssemblyEnd 10
 *   Main Stage;MatView 302
 *   Main Stage;KSPSetUp 98
 *   Main Stage;KSPSetUp;VecSet 5
 *   Main Stage;KSPSolve 150
 *
 * This option may be requested from the command line by passing in the flag `-log_view :<somefile>.txt:ascii_flamegraph`.
 */
PetscErrorCode PetscLogView_Flamegraph(PetscViewer viewer)
{
  PetscStageLog         stage_log_orig;
  MPI_Comm              comm;
  PetscLogDouble        globTotalTime;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)viewer);
  PetscCall(PetscLogGetDefaultHandler(&stage_log_orig));
  {
    PetscLogGlobalNames global_stages, global_events;
    PetscStageLog stage_log_nested;
    PetscNestedEventNode *tree;
    PetscNestedEventTreeNew tree_traversal;
    PetscEventPerfInfo *perf;

    PetscCall(PetscStageLogDuplicate(stage_log_orig, &stage_log_nested));
    PetscCall(PetscStageLogNestNames(stage_log_nested));
    PetscCall(PetscStageLogCreateGlobalStageNames(comm, stage_log_nested, &global_stages));
    PetscCall(PetscStageLogCreateGlobalEventNames(comm, stage_log_nested, &global_events));
    PetscCall(PetscLogNestedCreatePerfTree(comm, stage_log_nested, global_stages, global_events, &tree, &perf));
    tree_traversal.comm = comm;
    tree_traversal.global_events = global_events;
    tree_traversal.global_stages = global_stages;
    tree_traversal.perf = perf;
    tree_traversal.tree = tree;
    tree_traversal.stage_log = stage_log_nested;
    PetscCall(PetscLogNestedTreePrintTopNew(viewer, &tree_traversal, &globTotalTime, PETSC_LOG_NESTED_FLAMEGRAPH));
    PetscCall(PetscFree(perf));
    PetscCall(PetscFree(tree));
    PetscCall(PetscLogGlobalNamesDestroy(&global_events));
    PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
    PetscCall(PetscStageLogDestroy(stage_log_nested));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
