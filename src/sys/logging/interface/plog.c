
/*
      PETSc code to log object creation and destruction and PETSc events.

      This provides the public API used by the rest of PETSc and by users.

      These routines use a private API that is not used elsewhere in PETSc and is not
      accessible to users. The private API is defined in logimpl.h and the utils directory.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/
#include <petsctime.h>
#include <petscviewer.h>
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  #include <../src/sys/perfstubs/timer.h>
#endif
#include <../src/sys/logging/interface/plog.h>
#include <../src/sys/logging/impls/default/logdefault.h>

#if defined(PETSC_USE_LOG)
#include <petscmachineinfo.h>
#include <petscconfiginfo.h>

// This file, and only this file, is for functions that interact with the global logging state

#if defined(PETSC_HAVE_THREADSAFETY)
PetscInt           petsc_log_gid = -1; /* Global threadId counter */
PETSC_TLS PetscInt petsc_log_tid = -1; /* Local threadId */
#endif

/* used in the MPI_XXX() count macros in petsclog.h */

/* Global counters */
PetscLogDouble petsc_BaseTime        = 0.0;
PetscLogDouble petsc_TotalFlops      = 0.0; /* The number of flops */
PetscLogDouble petsc_send_ct         = 0.0; /* The number of sends */
PetscLogDouble petsc_recv_ct         = 0.0; /* The number of receives */
PetscLogDouble petsc_send_len        = 0.0; /* The total length of all sent messages */
PetscLogDouble petsc_recv_len        = 0.0; /* The total length of all received messages */
PetscLogDouble petsc_isend_ct        = 0.0; /* The number of immediate sends */
PetscLogDouble petsc_irecv_ct        = 0.0; /* The number of immediate receives */
PetscLogDouble petsc_isend_len       = 0.0; /* The total length of all immediate send messages */
PetscLogDouble petsc_irecv_len       = 0.0; /* The total length of all immediate receive messages */
PetscLogDouble petsc_wait_ct         = 0.0; /* The number of waits */
PetscLogDouble petsc_wait_any_ct     = 0.0; /* The number of anywaits */
PetscLogDouble petsc_wait_all_ct     = 0.0; /* The number of waitalls */
PetscLogDouble petsc_sum_of_waits_ct = 0.0; /* The total number of waits */
PetscLogDouble petsc_allreduce_ct    = 0.0; /* The number of reductions */
PetscLogDouble petsc_gather_ct       = 0.0; /* The number of gathers and gathervs */
PetscLogDouble petsc_scatter_ct      = 0.0; /* The number of scatters and scattervs */

/* Thread Local storage */
PETSC_TLS PetscLogDouble petsc_TotalFlops_th      = 0.0;
PETSC_TLS PetscLogDouble petsc_send_ct_th         = 0.0;
PETSC_TLS PetscLogDouble petsc_recv_ct_th         = 0.0;
PETSC_TLS PetscLogDouble petsc_send_len_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_recv_len_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_isend_ct_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_irecv_ct_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_isend_len_th       = 0.0;
PETSC_TLS PetscLogDouble petsc_irecv_len_th       = 0.0;
PETSC_TLS PetscLogDouble petsc_wait_ct_th         = 0.0;
PETSC_TLS PetscLogDouble petsc_wait_any_ct_th     = 0.0;
PETSC_TLS PetscLogDouble petsc_wait_all_ct_th     = 0.0;
PETSC_TLS PetscLogDouble petsc_sum_of_waits_ct_th = 0.0;
PETSC_TLS PetscLogDouble petsc_allreduce_ct_th    = 0.0;
PETSC_TLS PetscLogDouble petsc_gather_ct_th       = 0.0;
PETSC_TLS PetscLogDouble petsc_scatter_ct_th      = 0.0;

  #if defined(PETSC_HAVE_DEVICE)
PetscLogDouble petsc_ctog_ct        = 0.0; /* The total number of CPU to GPU copies */
PetscLogDouble petsc_gtoc_ct        = 0.0; /* The total number of GPU to CPU copies */
PetscLogDouble petsc_ctog_sz        = 0.0; /* The total size of CPU to GPU copies */
PetscLogDouble petsc_gtoc_sz        = 0.0; /* The total size of GPU to CPU copies */
PetscLogDouble petsc_ctog_ct_scalar = 0.0; /* The total number of CPU to GPU copies */
PetscLogDouble petsc_gtoc_ct_scalar = 0.0; /* The total number of GPU to CPU copies */
PetscLogDouble petsc_ctog_sz_scalar = 0.0; /* The total size of CPU to GPU copies */
PetscLogDouble petsc_gtoc_sz_scalar = 0.0; /* The total size of GPU to CPU copies */
PetscLogDouble petsc_gflops         = 0.0; /* The flops done on a GPU */
PetscLogDouble petsc_gtime          = 0.0; /* The time spent on a GPU */

PETSC_TLS PetscLogDouble petsc_ctog_ct_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_gtoc_ct_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_ctog_sz_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_gtoc_sz_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_ctog_ct_scalar_th = 0.0;
PETSC_TLS PetscLogDouble petsc_gtoc_ct_scalar_th = 0.0;
PETSC_TLS PetscLogDouble petsc_ctog_sz_scalar_th = 0.0;
PETSC_TLS PetscLogDouble petsc_gtoc_sz_scalar_th = 0.0;
PETSC_TLS PetscLogDouble petsc_gflops_th         = 0.0;
PETSC_TLS PetscLogDouble petsc_gtime_th          = 0.0;
  #endif

  #if defined(PETSC_HAVE_THREADSAFETY)
PetscErrorCode PetscAddLogDouble(PetscLogDouble *tot, PetscLogDouble *tot_th, PetscLogDouble tmp)
{
  *tot_th += tmp;
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  *tot += tmp;
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  return PETSC_SUCCESS;
}

PetscErrorCode PetscAddLogDoubleCnt(PetscLogDouble *cnt, PetscLogDouble *tot, PetscLogDouble *cnt_th, PetscLogDouble *tot_th, PetscLogDouble tmp)
{
  *cnt_th = *cnt_th + 1;
  *tot_th += tmp;
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  *tot += (PetscLogDouble)(tmp);
  *cnt += *cnt + 1;
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  return PETSC_SUCCESS;
}

PetscInt PetscLogGetTid(void)
{
  if (petsc_log_tid < 0) {
    PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
    petsc_log_tid = ++petsc_log_gid;
    PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  }
  return petsc_log_tid;
}

  #endif

PetscLogState petsc_log_state = NULL;

static PetscErrorCode PetscLogGetHandler(PetscLogHandlerType type, PetscLogHandler *handler)
{
  PetscFunctionBegin;
  PetscValidPointer(handler, 1);
  *handler = NULL;
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    if (PetscLogHandlers[i] && PetscLogHandlers[i]->impl->type == type) {
      *handler = PetscLogHandlers[i];
    }
  }
  if (*handler == NULL) {
    if (type == PETSC_LOG_HANDLER_DEFAULT) fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    else if (type == PETSC_LOG_HANDLER_NESTED) fprintf(stderr, "PETSC ERROR: Nested logging has not been enabled.\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogGetDefaultHandler(PetscLogHandler *default_handler)
{
  PetscFunctionBegin;
  PetscCall(PetscLogGetHandler(PETSC_LOG_HANDLER_DEFAULT, default_handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogGetNestedHandler(PetscLogHandler *nested_handler)
{
  PetscFunctionBegin;
  PetscCall(PetscLogGetHandler(PETSC_LOG_HANDLER_NESTED, nested_handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGetState(PetscLogState *state_p)
{
  PetscFunctionBegin;
  PetscValidPointer(state_p, 1);
  if (!petsc_log_state) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *state_p = petsc_log_state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tracing event logging variables */
static PetscBool PetscLogInitializeCalled = PETSC_FALSE;

PETSC_INTERN PetscErrorCode PetscLogInitialize(void)
{
  int              stage;
  PetscLogRegistry registry;
  //PetscBool        opt;

  PetscFunctionBegin;
  if (PetscLogInitializeCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscLogInitializeCalled = PETSC_TRUE;

  /* Setup default logging structures */
  PetscCall(PetscLogStateCreate(&petsc_log_state));
  registry = petsc_log_state->registry;
  PetscCall(PetscLogRegistryStageRegister(registry, "Main Stage", &stage));

  #if defined(PETSC_HAVE_THREADSAFETY)
  petsc_log_tid = 0;
  petsc_log_gid = 0;
  #endif

  /* All processors sync here for more consistent logging */
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&petsc_BaseTime));
  PetscCall(PetscLogStagePush(stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogFinalize(void)
{
  PetscLogState state;

  PetscFunctionBegin;
  #if defined(PETSC_HAVE_THREADSAFETY)
  if (eventInfoMap_th) {
    PetscEventPerfInfo **array;
    PetscInt             n, off = 0;

    PetscCall(PetscHMapEventGetSize(eventInfoMap_th, &n));
    PetscCall(PetscMalloc1(n, &array));
    PetscCall(PetscHMapEventGetVals(eventInfoMap_th, &off, array));
    for (PetscInt i = 0; i < n; i++) PetscCall(PetscFree(array[i]));
    PetscCall(PetscFree(array));
    PetscCall(PetscHMapEventDestroy(&eventInfoMap_th));
  }
  #endif

  /* Resetting phase */
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateDestroy(state));

  petsc_TotalFlops          = 0.0;
  petsc_BaseTime            = 0.0;
  petsc_TotalFlops          = 0.0;
  petsc_send_ct             = 0.0;
  petsc_recv_ct             = 0.0;
  petsc_send_len            = 0.0;
  petsc_recv_len            = 0.0;
  petsc_isend_ct            = 0.0;
  petsc_irecv_ct            = 0.0;
  petsc_isend_len           = 0.0;
  petsc_irecv_len           = 0.0;
  petsc_wait_ct             = 0.0;
  petsc_wait_any_ct         = 0.0;
  petsc_wait_all_ct         = 0.0;
  petsc_sum_of_waits_ct     = 0.0;
  petsc_allreduce_ct        = 0.0;
  petsc_gather_ct           = 0.0;
  petsc_scatter_ct          = 0.0;
  petsc_TotalFlops_th       = 0.0;
  petsc_send_ct_th          = 0.0;
  petsc_recv_ct_th          = 0.0;
  petsc_send_len_th         = 0.0;
  petsc_recv_len_th         = 0.0;
  petsc_isend_ct_th         = 0.0;
  petsc_irecv_ct_th         = 0.0;
  petsc_isend_len_th        = 0.0;
  petsc_irecv_len_th        = 0.0;
  petsc_wait_ct_th          = 0.0;
  petsc_wait_any_ct_th      = 0.0;
  petsc_wait_all_ct_th      = 0.0;
  petsc_sum_of_waits_ct_th  = 0.0;
  petsc_allreduce_ct_th     = 0.0;
  petsc_gather_ct_th        = 0.0;
  petsc_scatter_ct_th       = 0.0;

  #if defined(PETSC_HAVE_DEVICE)
  petsc_ctog_ct    = 0.0;
  petsc_gtoc_ct    = 0.0;
  petsc_ctog_sz    = 0.0;
  petsc_gtoc_sz    = 0.0;
  petsc_gflops     = 0.0;
  petsc_gtime      = 0.0;
  petsc_ctog_ct_th = 0.0;
  petsc_gtoc_ct_th = 0.0;
  petsc_ctog_sz_th = 0.0;
  petsc_gtoc_sz_th = 0.0;
  petsc_gflops_th  = 0.0;
  petsc_gtime_th   = 0.0;
  #endif

  PETSC_LARGEST_CLASSID    = PETSC_SMALLEST_CLASSID;
  PETSC_OBJECT_CLASSID     = 0;
  petsc_log_state          = NULL;
  PetscLogInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogIsActive - Check if logging is currently in progress.

  Not Collective

  Output Parameter:
. isActive - `PETSC_TRUE` if logging is in progress, `PETSC_FALSE` otherwise

  Level: beginner

.seealso: [](ch_profiling), `PetscLogDefaultBegin()`, `PetscLogAllBegin()`, `PetscLogSet()`
@*/
PetscErrorCode PetscLogIsActive(PetscBool *isActive)
{
  PetscFunctionBegin;
  *isActive = (petsc_log_state) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogActions - Determines whether actions are logged for the graphical viewer.

  Not Collective

  Input Parameter:
. flag - `PETSC_TRUE` if actions are to be logged

  Options Database Key:
. -log_exclude_actions - Turns off actions logging

  Level: intermediate

  Note:
  Logging of actions continues to consume more memory as the program
  runs. Long running programs should consider turning this feature off.
.seealso: [](ch_profiling), `PetscLogStagePush()`, `PetscLogStagePop()`
@*/
PetscErrorCode PetscLogActions(PetscBool flag)
{
  PetscLogHandler default_handler;

  PetscFunctionBegin;
  PetscCall(PetscLogGetDefaultHandler(&default_handler));
  PetscCall(PetscLogHandlerDefaultSetLogActions(default_handler, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogObjects - Determines whether objects are logged for the graphical viewer.

  Not Collective

  Input Parameter:
. flag - `PETSC_TRUE` if objects are to be logged

  Options Database Key:
. -log_exclude_objects - Turns off objects logging

  Level: intermediate

  Note:
  Logging of objects continues to consume more memory as the program
  runs. Long running programs should consider turning this feature off.

.seealso: [](ch_profiling), `PetscLogStagePush()`, `PetscLogStagePop()`
@*/
PetscErrorCode PetscLogObjects(PetscBool flag)
{
  PetscLogHandler default_handler;

  PetscFunctionBegin;
  PetscCall(PetscLogGetDefaultHandler(&default_handler));
  PetscCall(PetscLogHandlerDefaultSetLogObjects(default_handler, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Stage Functions --------------------------------------------------*/
/*@C
  PetscLogStageRegister - Attaches a character string name to a logging stage.

  Not Collective

  Input Parameter:
. sname - The name to associate with that stage

  Output Parameter:
. stage - The stage number

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStagePush()`, `PetscLogStagePop()`
@*/
PetscErrorCode PetscLogStageRegister(const char sname[], PetscLogStage *stage)
{
  PetscLogState    state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateStageRegister(state, sname, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogStagePush_Internal(PetscLogStage stage)
{
  PetscLogState   state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateStagePush(state, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogStagePush - This function pushes a stage on the logging stack. Events started and stopped until `PetscLogStagePop()` will be associated with the stage

  Not Collective

  Input Parameter:
. stage - The stage on which to log

  Usage:
  If the option -log_view is used to run the program containing the
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]
      PetscLogStagePush(1);
      [stage 1 of code]
      PetscLogStagePop();
      PetscBarrier(...);
      [more stage 0 of code]
      PetscFinalize();
.ve

  Level: intermediate

  Note:
  Use `PetscLogStageRegister()` to register a stage.

.seealso: [](ch_profiling), `PetscLogStagePop()`, `PetscLogStageRegister()`, `PetscBarrier()`
@*/
PetscErrorCode PetscLogStagePush(PetscLogStage stage)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i];
    if (h && h->impl->stage_push) PetscCall((*(h->impl->stage_push))(state, stage, h->ctx));
  }
  PetscCall(PetscLogStateStagePush(state, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogStagePop - This function pops a stage from the logging stack that was pushed with `PetscLogStagePush()`

  Not Collective

  Usage:
  If the option -log_view is used to run the program containing the
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]
      PetscLogStagePush(1);
      [stage 1 of code]
      PetscLogStagePop();
      PetscBarrier(...);
      [more stage 0 of code]
      PetscFinalize();
.ve

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStagePush()`, `PetscLogStageRegister()`, `PetscBarrier()`
@*/
PetscErrorCode PetscLogStagePop(void)
{
  PetscLogState state;
  PetscLogStage current_stage;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  current_stage = petsc_log_state->current_stage;
  PetscCall(PetscLogStateStagePop(state));
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i];
    if (h && h->impl->stage_pop) PetscCall((*(h->impl->stage_pop))(petsc_log_state, current_stage, h->ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageSetActive - Sets if a stage is used for `PetscLogEventBegin()` and `PetscLogEventEnd()`.

  Not Collective

  Input Parameters:
+ stage    - The stage
- isActive - The activity flag, `PETSC_TRUE` for logging, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: intermediate

  Note:
  If this is set to `PETSC_FALSE` the logging acts as if the stage did not exist

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
@*/
PetscErrorCode PetscLogStageSetActive(PetscLogStage stage, PetscBool isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateStageSetActive(state, stage, isActive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageGetActive - Checks if a stage is used for `PetscLogEventBegin()` and `PetscLogEventEnd()`.

  Not Collective

  Input Parameter:
. stage    - The stage

  Output Parameter:
. isActive - The activity flag, `PETSC_TRUE` for logging, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
@*/
PetscErrorCode PetscLogStageGetActive(PetscLogStage stage, PetscBool *isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateStageGetActive(state, stage, isActive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageSetVisible - Determines stage visibility in `PetscLogView()`

  Not Collective

  Input Parameters:
+ stage     - The stage
- isVisible - The visibility flag, `PETSC_TRUE` to print, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: intermediate

  Developer Note:
  What does visible mean, needs to be documented.

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogView()`
@*/
PetscErrorCode PetscLogStageSetVisible(PetscLogStage stage, PetscBool isVisible)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscStageRegLogSetVisible(state->registry->stages, stage, isVisible));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageGetVisible - Returns stage visibility in `PetscLogView()`

  Not Collective

  Input Parameter:
. stage     - The stage

  Output Parameter:
. isVisible - The visibility flag, `PETSC_TRUE` to print, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogView()`
@*/
PetscErrorCode PetscLogStageGetVisible(PetscLogStage stage, PetscBool *isVisible)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscStageRegLogGetVisible(state->registry->stages, stage, isVisible));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogStageGetId - Returns the stage id when given the stage name.

  Not Collective

  Input Parameter:
. name  - The stage name

  Output Parameter:
. stage - The stage, , or -1 if no stage with that name exists

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
@*/
PetscErrorCode PetscLogStageGetId(const char name[], PetscLogStage *stage)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscStageRegLogGetId(state->registry->stages, name, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Event Functions --------------------------------------------------*/

/*@C
  PetscLogEventRegister - Registers an event name for logging operations

  Not Collective

  Input Parameters:
+ name   - The name associated with the event
- classid - The classid associated to the class for this event, obtain either with
           `PetscClassIdRegister()` or use a predefined one such as `KSP_CLASSID`, `SNES_CLASSID`, the predefined ones
           are only available in C code

  Output Parameter:
. event - The event id for use with `PetscLogEventBegin()` and `PetscLogEventEnd()`.

  Example of Usage:
.vb
      PetscLogEvent USER_EVENT;
      PetscClassId classid;
      PetscLogDouble user_event_flops;
      PetscClassIdRegister("class name",&classid);
      PetscLogEventRegister("User event name",classid,&USER_EVENT);
      PetscLogEventBegin(USER_EVENT,0,0,0,0);
         [code segment to monitor]
         PetscLogFlops(user_event_flops);
      PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

  Level: intermediate

  Notes:
  PETSc automatically logs library events if the code has been
  configured with --with-log (which is the default) and
  -log_view or -log_all is specified.  `PetscLogEventRegister()` is
  intended for logging user events to supplement this PETSc
  information.

  PETSc can gather data for use with the utilities Jumpshot
  (part of the MPICH distribution).  If PETSc has been compiled
  with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
  MPICH), the user can employ another command line option, -log_mpe,
  to create a logfile, "mpe.log", which can be visualized
  Jumpshot.

  The classid is associated with each event so that classes of events
  can be disabled simultaneously, such as all matrix events. The user
  can either use an existing classid, such as `MAT_CLASSID`, or create
  their own as shown in the example.

  If an existing event with the same name exists, its event handle is
  returned instead of creating a new event.

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogFlops()`,
          `PetscLogEventActivate()`, `PetscLogEventDeactivate()`, `PetscClassIdRegister()`
@*/
PetscErrorCode PetscLogEventRegister(const char name[], PetscClassId classid, PetscLogEvent *event)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateEventRegister(state, name, classid, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventSetCollective - Indicates that a particular event is collective.

  Not Collective

  Input Parameters:
+ event - The event id
- collective - Boolean flag indicating whether a particular event is collective

  Level: developer

  Notes:
  New events returned from `PetscLogEventRegister()` are collective by default.

  Collective events are handled specially if the -log_sync is used. In that case the logging saves information about
  two parts of the event; the time for all the MPI ranks to synchronize and then the time for the actual computation/communication
  to be performed. This option is useful to debug imbalance within the computations or communications

.seealso: [](ch_profiling), `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventRegister()`
@*/
PetscErrorCode PetscLogEventSetCollective(PetscLogEvent event, PetscBool collective)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscEventRegLogSetCollective(state->registry->events, event, collective));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventIncludeClass - Activates event logging for a PETSc object class in every stage.

  Not Collective

  Input Parameter:
. classid - The object class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventActivateClass()`, `PetscLogEventDeactivateClass()`, `PetscLogEventActivate()`, `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventIncludeClass(PetscClassId classid)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateEventIncludeClass(state, classid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventExcludeClass - Deactivates event logging for a PETSc object class in every stage.

  Not Collective

  Input Parameter:
. classid - The object class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.

  Level: developer

  Note:
  If a class is excluded then events associated with that class are not logged.

.seealso: [](ch_profiling), `PetscLogEventDeactivateClass()`, `PetscLogEventActivateClass()`, `PetscLogEventDeactivate()`, `PetscLogEventActivate()`
@*/
PetscErrorCode PetscLogEventExcludeClass(PetscClassId classid)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateEventExcludeClass(state, classid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventActivate - Indicates that a particular event should be logged.

  Not Collective

  Input Parameter:
. event - The event id

  Usage:
.vb
      PetscLogEventDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in include/petsclog.h)
  or an event number obtained with `PetscLogEventRegister()`.

.seealso: [](ch_profiling), `PlogEventDeactivate()`, `PlogEventDeactivatePush()`, `PetscLogEventDeactivatePop()`
@*/
PetscErrorCode PetscLogEventActivate(PetscLogEvent event)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateEventActivate(state, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventDeactivate - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameter:
. event - The event id

  Usage:
.vb
      PetscLogEventDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscLogEventRegister()`).

.seealso: [](ch_profiling), `PetscLogEventActivate()`, `PetscLogEventDeactivatePush()`, `PetscLogEventDeactivatePop()`
@*/
PetscErrorCode PetscLogEventDeactivate(PetscLogEvent event)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateEventDeactivate(state, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventDeactivatePush - Indicates that a particular event should not be logged until `PetscLogEventDeactivatePop()` is called

  Not Collective

  Input Parameter:
. event - The event id

  Usage:
.vb
      PetscLogEventDeactivatePush(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventDeactivatePop(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscLogEventRegister()`).

.seealso: [](ch_profiling), `PetscLogEventActivate()`, `PetscLogEventDeactivatePop()`, `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventDeactivatePush(PetscLogEvent event)
{
  PetscFunctionBegin;
  // TODO
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventDeactivatePop - Indicates that a particular event should again be logged after the logging was turned off with `PetscLogEventDeactivatePush()`

  Not Collective

  Input Parameter:
. event - The event id

  Usage:
.vb
      PetscLogEventDeactivatePush(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventDeactivatePop(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscLogEventRegister()`).

.seealso: [](ch_profiling), `PetscLogEventActivate()`, `PetscLogEventDeactivatePush()`
@*/
PetscErrorCode PetscLogEventDeactivatePop(PetscLogEvent event)
{
  PetscFunctionBegin;
  // TODO
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventSetActiveAll - Turns on logging of all events

  Not Collective

  Input Parameters:
+ event    - The event id
- isActive - The activity flag determining whether the event is logged

  Level: advanced

.seealso: [](ch_profiling), `PlogEventActivate()`, `PlogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventSetActiveAll(PetscLogEvent event, PetscBool isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  for (PetscInt stage = 0; stage < state->registry->stages->num_entries; stage++) {
    PetscCall((isActive ? PetscBTSet : PetscBTClear)(state->active, stage + (event + 1) * state->bt_num_stages));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventActivateClass - Activates event logging for a PETSc object class for the current stage

  Not Collective

  Input Parameter:
. classid - The event class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventIncludeClass()`, `PetscLogEventExcludeClass()`, `PetscLogEventDeactivateClass()`, `PetscLogEventActivate()`, `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventActivateClass(PetscClassId classid)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateEventActivateClass(state, classid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventDeactivateClass - Deactivates event logging for a PETSc object class for the current stage

  Not Collective

  Input Parameter:
. classid - The event class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventIncludeClass()`, `PetscLogEventExcludeClass()`, `PetscLogEventActivateClass()`, `PetscLogEventActivate()`, `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventDeactivateClass(PetscClassId classid)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateEventDeactivateClass(state, classid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PetscLogEventSync - Synchronizes the beginning of a user event.

   Synopsis:
   #include <petsclog.h>
   PetscErrorCode PetscLogEventSync(int e,MPI_Comm comm)

   Collective

   Input Parameters:
+  e - integer associated with the event obtained from PetscLogEventRegister()
-  comm - an MPI communicator

   Usage:
.vb
     PetscLogEvent USER_EVENT;
     PetscLogEventRegister("User event",0,&USER_EVENT);
     PetscLogEventSync(USER_EVENT,PETSC_COMM_WORLD);
     PetscLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
     PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Level: developer

   Note:
   This routine should be called only if there is not a
   `PetscObject` available to pass to `PetscLogEventBegin()`.

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`
M*/

/*MC
   PetscLogEventBegin - Logs the beginning of a user event.

   Synopsis:
   #include <petsclog.h>
   PetscErrorCode PetscLogEventBegin(int e,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)

   Not Collective

   Input Parameters:
+  e - integer associated with the event obtained from PetscLogEventRegister()
-  o1,o2,o3,o4 - objects associated with the event, or 0

   Fortran Synopsis:
   void PetscLogEventBegin(int e,PetscErrorCode ierr)

   Usage:
.vb
     PetscLogEvent USER_EVENT;
     PetscLogDouble user_event_flops;
     PetscLogEventRegister("User event",0,&USER_EVENT);
     PetscLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PetscLogFlops(user_event_flops);
     PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Level: intermediate

   Developer Note:
     `PetscLogEventBegin()` and `PetscLogEventBegin()` return error codes instead of explicitly handling the
     errors that occur in the macro directly because other packages that use this macros have used them in their
     own functions or methods that do not return error codes and it would be disruptive to change the current
     behavior.

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventEnd()`, `PetscLogFlops()`
M*/

/*MC
   PetscLogEventEnd - Log the end of a user event.

   Synopsis:
   #include <petsclog.h>
   PetscErrorCode PetscLogEventEnd(int e,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)

   Not Collective

   Input Parameters:
+  e - integer associated with the event obtained with PetscLogEventRegister()
-  o1,o2,o3,o4 - objects associated with the event, or 0

   Fortran Synopsis:
   void PetscLogEventEnd(int e,PetscErrorCode ierr)

   Usage:
.vb
     PetscLogEvent USER_EVENT;
     PetscLogDouble user_event_flops;
     PetscLogEventRegister("User event",0,&USER_EVENT,);
     PetscLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PetscLogFlops(user_event_flops);
     PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Level: intermediate

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogFlops()`
M*/

/*@C
  PetscLogEventGetId - Returns the event id when given the event name.

  Not Collective

  Input Parameter:
. name  - The event name

  Output Parameter:
. event - The event, or -1 if no event with that name exists

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogStageGetId()`
@*/
PetscErrorCode PetscLogEventGetId(const char name[], PetscLogEvent *event)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscEventRegLogGetEvent(state->registry->events, name, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Output Functions -------------------------------------------------*/

/*@C
  PetscLogDump - Dumps logs of objects to a file. This file is intended to
  be read by bin/petscview. This program no longer exists.

  Collective on `PETSC_COMM_WORLD`

  Input Parameter:
. name - an optional file name

  Usage:
.vb
     PetscInitialize(...);
     PetscLogDefaultBegin(); or PetscLogAllBegin();
     ... code ...
     PetscLogDump(filename);
     PetscFinalize();
.ve

  Level: advanced

  Note:
  The default file name is Log.<rank> where <rank> is the MPI process rank. If no name is specified,
  this file will be used.

.seealso: [](ch_profiling), `PetscLogDefaultBegin()`, `PetscLogAllBegin()`, `PetscLogView()`
@*/
PetscErrorCode PetscLogDump(const char sname[])
{
  PetscLogHandler     handler;

  PetscFunctionBegin;
  PetscCall(PetscLogGetDefaultHandler(&handler));
  PetscCall(PetscLogDump_Default(handler, sname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogView - Prints a summary of the logging.

  Collective over MPI_Comm

  Input Parameter:
.  viewer - an ASCII viewer

  Options Database Keys:
+  -log_view [:filename] - Prints summary of log information
.  -log_view :filename.py:ascii_info_detail - Saves logging information from each process as a Python file
.  -log_view :filename.xml:ascii_xml - Saves a summary of the logging information in a nested format (see below for how to view it)
.  -log_view :filename.txt:ascii_flamegraph - Saves logging information in a format suitable for visualising as a Flame Graph (see below for how to view it)
.  -log_view_memory - Also display memory usage in each event
.  -log_view_gpu_time - Also display time in each event for GPU kernels (Note this may slow the computation)
.  -log_all - Saves a file Log.rank for each MPI rank with details of each step of the computation
-  -log_trace [filename] - Displays a trace of what each process is doing

  Level: beginner

  Notes:
  It is possible to control the logging programmatically but we recommend using the options database approach whenever possible
  By default the summary is printed to stdout.

  Before calling this routine you must have called either PetscLogDefaultBegin() or PetscLogNestedBegin()

  If PETSc is configured with --with-logging=0 then this functionality is not available

  To view the nested XML format filename.xml first copy  ${PETSC_DIR}/share/petsc/xml/performance_xml2html.xsl to the current
  directory then open filename.xml with your browser. Specific notes for certain browsers
$    Firefox and Internet explorer - simply open the file
$    Google Chrome - you must start up Chrome with the option --allow-file-access-from-files
$    Safari - see https://ccm.net/faq/36342-safari-how-to-enable-local-file-access
  or one can use the package http://xmlsoft.org/XSLT/xsltproc2.html to translate the xml file to html and then open it with
  your browser.
  Alternatively, use the script ${PETSC_DIR}/lib/petsc/bin/petsc-performance-view to automatically open a new browser
  window and render the XML log file contents.

  The nested XML format was kindly donated by Koos Huijssen and Christiaan M. Klaij  MARITIME  RESEARCH  INSTITUTE  NETHERLANDS

  The Flame Graph output can be visualised using either the original Flame Graph script (https://github.com/brendangregg/FlameGraph)
  or using speedscope (https://www.speedscope.app).
  Old XML profiles may be converted into this format using the script ${PETSC_DIR}/lib/petsc/bin/xml2flamegraph.py.

.seealso: [](ch_profiling), `PetscLogDefaultBegin()`, `PetscLogDump()`
@*/
PetscErrorCode PetscLogView(PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;
  int               stage;
  PetscLogState     state;
  PetscIntStack     temp_stack;
  PetscLogHandler   handler;
  PetscBool         is_empty;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  /* Pop off any stages the user forgot to remove */
  PetscCall(PetscIntStackCreate(&temp_stack));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  while (stage >= 0) {
    PetscCall(PetscLogStagePop());
    PetscCall(PetscIntStackPush(temp_stack, stage));
    PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(isascii, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Currently can only view logging to ASCII");
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_XML || format == PETSC_VIEWER_ASCII_FLAMEGRAPH) {
    PetscCall(PetscLogGetNestedHandler(&handler));
    PetscCall(PetscLogView_Nested(handler, viewer));
  } else {
    PetscCall(PetscLogGetDefaultHandler(&handler));
    PetscCall(PetscLogView_Default(handler, viewer));
    //if (format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_ASCII_INFO) {
    //  PetscCall(PetscLogView_Default(handler, viewer));
    //} else if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    //  PetscCall(PetscLogView_Detailed(handler, viewer));
    //} else if (format == PETSC_VIEWER_ASCII_CSV) {
    //  PetscCall(PetscLogView_CSV(viewer));
    //} else if (format == PETSC_VIEWER_ASCII_XML) {
    //  PetscCall(PetscLogView_Nested(viewer));
    //} else if (format == PETSC_VIEWER_ASCII_FLAMEGRAPH) {
    //  PetscCall(PetscLogView_Flamegraph(viewer));
    //}
  }
  PetscCall(PetscIntStackEmpty(temp_stack, &is_empty));
  while (!is_empty) {
    PetscCall(PetscIntStackPop(temp_stack, &stage));
    PetscCall(PetscLogStagePush(stage));
  }
  PetscCall(PetscIntStackDestroy(temp_stack));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogViewFromOptions - Processes command line options to determine if/how a `PetscLog` is to be viewed.

  Collective on `PETSC_COMM_WORLD`

  Level: developer

.seealso: [](ch_profiling), `PetscLogView()`
@*/
PetscErrorCode PetscLogViewFromOptions(void)
{
  PetscInt          n_max = PETSC_LOG_VIEW_FROM_OPTIONS_MAX;
  PetscViewer       viewers[PETSC_LOG_VIEW_FROM_OPTIONS_MAX];
  PetscViewerFormat formats[PETSC_LOG_VIEW_FROM_OPTIONS_MAX];
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetViewers(PETSC_COMM_WORLD, NULL, NULL, "-log_view", &n_max, viewers, formats, &flg));
  for (PetscInt i = 0; i < n_max; i++) {
    PetscCall(PetscViewerPushFormat(viewers[i], formats[i]));
    PetscCall(PetscLogView(viewers[i]));
    PetscCall(PetscViewerPopFormat(viewers[i]));
    PetscCall(PetscViewerDestroy(&(viewers[i])));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*----------------------------------------------- Counter Functions -------------------------------------------------*/
/*@C
   PetscGetFlops - Returns the number of flops used on this processor
   since the program began.

   Not Collective

   Output Parameter:
   flops - number of floating point operations

   Level: intermediate

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   `PetscLogFlops()` to increment this counter to include flops for the
   application code.

   A separate counter `PetscLogGPUFlops()` logs the flops that occur on any GPU associated with this MPI rank

.seealso: [](ch_profiling), `PetscLogGPUFlops()`, `PetscTime()`, `PetscLogFlops()`
@*/
PetscErrorCode PetscGetFlops(PetscLogDouble *flops)
{
  PetscFunctionBegin;
  *flops = petsc_TotalFlops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogObjectState(PetscObject obj, const char format[], ...)
{
  PetscLogHandler default_handler;
  va_list Argp;

  PetscFunctionBegin;
  PetscCall(PetscLogGetDefaultHandler(&default_handler));
  va_start(Argp, format);
  PetscCall(PetscLogDefaultHandlerLogObjectState(default_handler, obj, format, Argp));
  va_end(Argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PetscLogFlops - Adds floating point operations to the global counter.

   Synopsis:
   #include <petsclog.h>
   PetscErrorCode PetscLogFlops(PetscLogDouble f)

   Not Collective

   Input Parameter:
.  f - flop counter

   Usage:
.vb
     PetscLogEvent USER_EVENT;
     PetscLogEventRegister("User event",0,&USER_EVENT);
     PetscLogEventBegin(USER_EVENT,0,0,0,0);
        [code segment to monitor]
        PetscLogFlops(user_flops)
     PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

   Level: intermediate

   Note:
   A global counter logs all PETSc flop counts.  The user can use
   PetscLogFlops() to increment this counter to include flops for the
   application code.

.seealso: [](ch_profiling), `PetscLogGPUFlops()`, `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscGetFlops()`
M*/

/*MC
   PetscPreLoadBegin - Begin a segment of code that may be preloaded (run twice)
    to get accurate timings

   Synopsis:
   #include <petsclog.h>
   void PetscPreLoadBegin(PetscBool  flag,char *name);

   Not Collective

   Input Parameters:
+   flag - `PETSC_TRUE` to run twice, `PETSC_FALSE` to run once, may be overridden
           with command line option -preload true or -preload false
-   name - name of first stage (lines of code timed separately with `-log_view`) to
           be preloaded

   Usage:
.vb
     PetscPreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PetscPreLoadStage("second stage");
       lines of code
     PetscPreLoadEnd();
.ve

   Level: intermediate

   Note:
    Only works in C/C++, not Fortran

     Flags available within the macro.
+    PetscPreLoadingUsed - true if we are or have done preloading
.    PetscPreLoadingOn - true if it is CURRENTLY doing preload
.    PetscPreLoadIt - 0 for the first computation (with preloading turned off it is only 0) 1 for the second
-    PetscPreLoadMax - number of times it will do the computation, only one when preloading is turned on
     The first two variables are available throughout the program, the second two only between the `PetscPreLoadBegin()`
     and `PetscPreLoadEnd()`

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
M*/

/*MC
   PetscPreLoadEnd - End a segment of code that may be preloaded (run twice)
    to get accurate timings

   Synopsis:
   #include <petsclog.h>
   void PetscPreLoadEnd(void);

   Not Collective

   Usage:
.vb
     PetscPreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PetscPreLoadStage("second stage");
       lines of code
     PetscPreLoadEnd();
.ve

   Level: intermediate

   Note:
    Only works in C/C++ not fortran

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadStage()`
M*/

/*MC
   PetscPreLoadStage - Start a new segment of code to be timed separately.
    to get accurate timings

   Synopsis:
   #include <petsclog.h>
   void PetscPreLoadStage(char *name);

   Not Collective

   Usage:
.vb
     PetscPreLoadBegin(PETSC_TRUE,"first stage);
       lines of code
       PetscPreLoadStage("second stage");
       lines of code
     PetscPreLoadEnd();
.ve

   Level: intermediate

   Note:
    Only works in C/C++ not fortran

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`
M*/

  #if PetscDefined(HAVE_DEVICE)
    #include <petsc/private/deviceimpl.h>

PetscBool PetscLogGpuTimeFlag = PETSC_FALSE;

/*
   This cannot be called by users between PetscInitialize() and PetscFinalize() at any random location in the code
   because it will result in timing results that cannot be interpreted.
*/
static PetscErrorCode PetscLogGpuTime_Off(void)
{
  PetscLogGpuTimeFlag = PETSC_FALSE;
  return PETSC_SUCCESS;
}

/*@C
     PetscLogGpuTime - turn on the logging of GPU time for GPU kernels

  Options Database Key:
.   -log_view_gpu_time - provide the GPU times in the -log_view output

   Level: advanced

  Notes:
    Turning on the timing of the
    GPU kernels can slow down the entire computation and should only be used when studying the performance
    of operations on GPU such as vector operations and matrix-vector operations.

    This routine should only be called once near the beginning of the program. Once it is started it cannot be turned off.

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogGpuFlops()`, `PetscLogGpuTimeEnd()`, `PetscLogGpuTimeBegin()`
@*/
PetscErrorCode PetscLogGpuTime(void)
{
  if (!PetscLogGpuTimeFlag) PetscCall(PetscRegisterFinalize(PetscLogGpuTime_Off));
  PetscLogGpuTimeFlag = PETSC_TRUE;
  return PETSC_SUCCESS;
}

/*@C
  PetscLogGpuTimeBegin - Start timer for device

  Level: intermediate

  Notes:
    When CUDA or HIP is enabled, the timer is run on the GPU, it is a separate logging of time devoted to GPU computations (excluding kernel launch times).

    When CUDA or HIP is not available, the timer is run on the CPU, it is a separate logging of time devoted to GPU computations (including kernel launch times).

    There is no need to call WaitForCUDA() or WaitForHIP() between `PetscLogGpuTimeBegin()` and `PetscLogGpuTimeEnd()`

    This timer should NOT include times for data transfers between the GPU and CPU, nor setup actions such as allocating space.

    The regular logging captures the time for data transfers and any CPU activities during the event

    It is used to compute the flop rate on the GPU as it is actively engaged in running a kernel.

  Developer Notes:
    The GPU event timer captures the execution time of all the kernels launched in the default stream by the CPU between `PetscLogGpuTimeBegin()` and `PetsLogGpuTimeEnd()`.

    `PetscLogGpuTimeBegin()` and `PetsLogGpuTimeEnd()` insert the begin and end events into the default stream (stream 0). The device will record a time stamp for the
    event when it reaches that event in the stream. The function xxxEventSynchronize() is called in `PetsLogGpuTimeEnd()` to block CPU execution,
    but not continued GPU execution, until the timer event is recorded.

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogGpuFlops()`, `PetscLogGpuTimeEnd()`, `PetscLogGpuTime()`
@*/
PetscErrorCode PetscLogGpuTimeBegin(void)
{
  PetscFunctionBegin;
  if (!PetscLogPLB || !PetscLogGpuTimeFlag) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscDefined(HAVE_DEVICE)) {
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextBeginTimer_Internal(dctx));
  } else {
    PetscCall(PetscTimeSubtract(&petsc_gtime));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogGpuTimeEnd - Stop timer for device

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogGpuFlops()`, `PetscLogGpuTimeBegin()`
@*/
PetscErrorCode PetscLogGpuTimeEnd(void)
{
  PetscFunctionBegin;
  if (!PetscLogPLE || !PetscLogGpuTimeFlag) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscDefined(HAVE_DEVICE)) {
    PetscDeviceContext dctx;
    PetscLogDouble     elapsed;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextEndTimer_Internal(dctx, &elapsed));
    petsc_gtime += (elapsed / 1000.0);
  } else {
    PetscCall(PetscTimeAdd(&petsc_gtime));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #endif /* end of PETSC_HAVE_DEVICE */

#else /* end of -DPETSC_USE_LOG section */

PetscErrorCode PetscLogObjectState(PetscObject obj, const char format[], ...)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_USE_LOG*/

PetscClassId PETSC_LARGEST_CLASSID = PETSC_SMALLEST_CLASSID;
PetscClassId PETSC_OBJECT_CLASSID  = 0;

/*@C
  PetscClassIdRegister - Registers a new class name for objects and logging operations in an application code.

  Not Collective

  Input Parameter:
. name   - The class name

  Output Parameter:
. oclass - The class id or classid

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventRegister()`
@*/
PetscErrorCode PetscClassIdRegister(const char name[], PetscClassId *oclass)
{
#if defined(PETSC_USE_LOG)
  PetscLogState state;
#endif

  PetscFunctionBegin;
  *oclass = ++PETSC_LARGEST_CLASSID;
#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscClassRegLogRegister(state->registry->classes, name, *oclass));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerDestroy(PetscLogHandler *handler_p)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  handler = *handler_p;
  *handler_p = NULL;
  if (handler->impl->destroy) PetscCall((*(handler->impl->destroy))(handler->ctx));
  PetscCall(PetscFree(handler->impl));
  PetscCall(PetscFree(handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogEventGetPerfInfo - Return the performance information about the given event in the given stage

  Input Parameters:
+ stage - The stage number or `PETSC_DETERMINE` for the current stage
- event - The event number

  Output Parameter:
. info - This structure is filled with the performance information

  Level: Intermediate

  Note:
  This is a low level routine used by the logging functions in PETSc
@*/
PetscErrorCode PetscLogEventGetPerfInfo(PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo *info)
{
  PetscLogHandler default_handler;
  PetscEventPerfInfo *event_info;

  PetscFunctionBegin;
  PetscValidPointer(info, 3);
  PetscCall(PetscLogGetDefaultHandler(&default_handler));
  if (stage < 0) PetscCall(PetscLogStageGetCurrent(&stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfInfo(default_handler, stage, event, &event_info));
  *info = *event_info;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogEventSetDof - Set the nth number of degrees of freedom of a numerical problem associated with this event

  Not Collective

  Input Parameters:
+ event - The event id to log
. n     - The dof index, in [0, 8)
- dof   - The number of dofs

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Note:
  This is to enable logging of convergence

.seealso: `PetscLogEventSetError()`, `PetscLogEventRegister()`
@*/
PetscErrorCode PetscLogEventSetDof(PetscLogEvent event, PetscInt n, PetscLogDouble dof)
{
  PetscLogHandler     default_handler;
  PetscLogState       state;
  PetscEventPerfInfo *event_info;
  PetscLogStage     stage;

  PetscFunctionBegin;
  PetscCheck(!(n < 0) && !(n > 7), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %" PetscInt_FMT " is not in [0, 8)", n);
  PetscCall(PetscLogGetDefaultHandler(&default_handler));
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfInfo(default_handler, stage, event, &event_info));
  event_info->dof[n] = dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogEventSetError - Set the nth error associated with a numerical problem associated with this event

  Not Collective

  Input Parameters:
+ event - The event id to log
. n     - The error index, in [0, 8)
- error - The error

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Notes:
  This is to enable logging of convergence, and enable users to interpret the errors as they wish. For example,
  as different norms, or as errors for different fields

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscLogEventSetDof()`, `PetscLogEventRegister()`
@*/
PetscErrorCode PetscLogEventSetError(PetscLogEvent event, PetscInt n, PetscLogDouble error)
{
  PetscLogHandler     default_handler;
  PetscEventPerfInfo *event_info;
  int               stage;

  PetscFunctionBegin;
  PetscCheck(!(n < 0) && !(n > 7), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %" PetscInt_FMT " is not in [0, 8)", n);
  PetscCall(PetscLogGetDefaultHandler(&default_handler));
  PetscCall(PetscLogStageGetCurrent(&stage));
  PetscCall(PetscLogHandlerDefaultGetEventPerfInfo(default_handler, stage, event, &event_info));
  event_info->errors[n] = error;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_LOG) && defined(PETSC_HAVE_MPE)
#include <mpe.h>

PetscBool PetscBeganMPE = PETSC_FALSE;

PETSC_INTERN PetscErrorCode PetscLogEventBeginMPE(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_INTERN PetscErrorCode PetscLogEventEndMPE(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);

/*@C
   PetscLogMPEBegin - Turns on MPE logging of events. This creates large log files
   and slows the program down.

   Collective over `PETSC_COMM_WORLD`

   Options Database Key:
. -log_mpe - Prints extensive log information

   Level: advanced

   Note:
   A related routine is `PetscLogDefaultBegin()` (with the options key -log_view), which is
   intended for production runs since it logs only flop rates and object
   creation (and should not significantly slow the programs).

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogDefaultBegin()`, `PetscLogAllBegin()`, `PetscLogEventActivate()`,
          `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogMPEBegin(void)
{
  PetscFunctionBegin;
  /* Do MPE initialization */
  if (!MPE_Initialized_logging()) { /* This function exists in mpich 1.1.2 and higher */
    PetscCall(PetscInfo(0, "Initializing MPE.\n"));
    PetscCall(MPE_Init_log());

    PetscBeganMPE = PETSC_TRUE;
  } else {
    PetscCall(PetscInfo(0, "MPE already initialized. Not attempting to reinitialize.\n"));
  }
  PetscCall(PetscLogSet(PetscLogEventBeginMPE, PetscLogEventEndMPE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscLogMPEDump - Dumps the MPE logging info to file for later use with Jumpshot.

   Collective over `PETSC_COMM_WORLD`

   Level: advanced

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogMPEBegin()`
@*/
PetscErrorCode PetscLogMPEDump(const char sname[])
{
  char name[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (PetscBeganMPE) {
    PetscCall(PetscInfo(0, "Finalizing MPE.\n"));
    if (sname) {
      PetscCall(PetscStrncpy(name, sname, sizeof(name)));
    } else {
      PetscCall(PetscGetProgramName(name, sizeof(name)));
    }
    PetscCall(MPE_Finish_log(name));
  } else {
    PetscCall(PetscInfo(0, "Not finalizing MPE (not started by PETSc).\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PETSC_RGB_COLORS_MAX 39
static const char *PetscLogMPERGBColors[PETSC_RGB_COLORS_MAX] = {"OliveDrab:      ", "BlueViolet:     ", "CadetBlue:      ", "CornflowerBlue: ", "DarkGoldenrod:  ", "DarkGreen:      ", "DarkKhaki:      ", "DarkOliveGreen: ",
                                                                 "DarkOrange:     ", "DarkOrchid:     ", "DarkSeaGreen:   ", "DarkSlateGray:  ", "DarkTurquoise:  ", "DeepPink:       ", "DarkKhaki:      ", "DimGray:        ",
                                                                 "DodgerBlue:     ", "GreenYellow:    ", "HotPink:        ", "IndianRed:      ", "LavenderBlush:  ", "LawnGreen:      ", "LemonChiffon:   ", "LightCoral:     ",
                                                                 "LightCyan:      ", "LightPink:      ", "LightSalmon:    ", "LightSlateGray: ", "LightYellow:    ", "LimeGreen:      ", "MediumPurple:   ", "MediumSeaGreen: ",
                                                                 "MediumSlateBlue:", "MidnightBlue:   ", "MintCream:      ", "MistyRose:      ", "NavajoWhite:    ", "NavyBlue:       ", "OliveDrab:      "};

/*@C
  PetscLogMPEGetRGBColor - This routine returns a rgb color useable with `PetscLogEventRegister()`

  Not collective. Maybe it should be?

  Output Parameter:
. str - character string representing the color

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventRegister()`
@*/
PetscErrorCode PetscLogMPEGetRGBColor(const char *str[])
{
  static int idx = 0;

  PetscFunctionBegin;
  *str = PetscLogMPERGBColors[idx];
  idx  = (idx + 1) % PETSC_RGB_COLORS_MAX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_USE_LOG && PETSC_HAVE_MPE */
