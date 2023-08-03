#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

/* TSEventCalcSigns() - helper function.
   Fills array sign[] with signs of array f[]. If abs(f[i]) < vtol[i], the zero sign is taken.
   All arrays should have length 'nev'
*/
static inline void TSEventCalcSigns(PetscInt nev, const PetscReal *f, const PetscReal *vtol, PetscInt *sign)
{
  for (PetscInt i = 0; i < nev; i++) {
    if (PetscAbsReal(f[i]) < vtol[i]) sign[i] = 0;
    else sign[i] = PetscSign(f[i]);
  }
}

/*
  TSEventInitialize - Initializes TSEvent for TSSolve
*/
PetscErrorCode TSEventInitialize(TSEvent event, TS ts, PetscReal t, Vec U)
{
  PetscFunctionBegin;
  if (!event) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(event, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscValidHeaderSpecific(U, VEC_CLASSID, 4);
  event->ptime_prev = t;
  event->iterctr    = 0;
  event->processing = PETSC_FALSE;
  PetscCall((*event->eventhandler)(ts, t, U, event->fvalue_prev, event->ctx));
  TSEventCalcSigns(event->nevents, event->fvalue_prev, event->vtol, event->fsign_prev); // by this moment event->vtol should have been defined
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSEventDestroy(TSEvent *event)
{
  PetscFunctionBegin;
  PetscAssertPointer(event, 1);
  if (!*event) PetscFunctionReturn(PETSC_SUCCESS);
  if (--(*event)->refct > 0) {
    *event = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscFree((*event)->fvalue_prev));
  PetscCall(PetscFree((*event)->fvalue));
  PetscCall(PetscFree((*event)->fvalue_right));
  PetscCall(PetscFree((*event)->fsign_prev));
  PetscCall(PetscFree((*event)->fsign));
  PetscCall(PetscFree((*event)->fsign_right));
  PetscCall(PetscFree((*event)->side));
  PetscCall(PetscFree((*event)->side_prev));
  PetscCall(PetscFree((*event)->justrefined_AB));
  PetscCall(PetscFree((*event)->gamma_AB));
  PetscCall(PetscFree((*event)->direction));
  PetscCall(PetscFree((*event)->terminate));
  PetscCall(PetscFree((*event)->events_zero));
  PetscCall(PetscFree((*event)->vtol));

  for (PetscInt i = 0; i < (*event)->recsize; i++) PetscCall(PetscFree((*event)->recorder.eventidx[i]));
  PetscCall(PetscFree((*event)->recorder.eventidx));
  PetscCall(PetscFree((*event)->recorder.nevents));
  PetscCall(PetscFree((*event)->recorder.stepnum));
  PetscCall(PetscFree((*event)->recorder.time));

  PetscCall(PetscViewerDestroy(&(*event)->monitor));
  PetscCall(PetscFree(*event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSetPostEventStep - Set the time step to use immediately following the event

  Logically Collective

  Input Parameters:
+ ts - time integration context
- dt - post event step

  Options Database Keys:
. -ts_event_post_event_step <dt> - time step after the event; zero value - to keep using previous time steps

  Notes:
  `TSSetPostEventStep()` allows one to set a time step that is used immediately following an event.
  If a positive real number is specified, it will be applied as is.
  However, if `TSAdapt` is allowed to interfere, a large 'dt' may get truncated, resulting in a smaller actual post-event step.

  The post-event time step should be selected based on the post-event dynamics.
  If the dynamics are stiff, or a significant jump in the equations or the state vector has taken place at the event,
  a conservative (small) step should be employed. If not, then a larger time step may be appropriate.

  In the latter case, instead of explicitly setting the post-event time step,
  the user may also choose a special option of keeping the time steps used before the event, which is a sort of 'petsc-decide' strategy.
  For this, use special value 0. It will signal the TS to directly step to the time point it planned to visit prior to the event detection.
  E.g. if a step t0 -> t1 was planned originally, and an event 'te' occurred, t0 < te < t1, then after the event the TS will step: te -> t1.
  Moreover, in this situation the originally planned subsequent step t1 -> t2 will also be preserved.

  This function can be called not only in the initial setup, but also inside the postevent callback set with `TSSetEventHandler()`,
  affecting the post-event step for the current event, and the subsequent ones.
  So, the strategy of the post-event time step definition can be adjusted on the fly.
  If several events have been triggered in the given time point, still a single postevent handler is invoked,
  and the user is to figure out what post-event time step is more appropriate in this situation.

  By default (on `TSSetEventHandler()` call), the post-event time step is set equal to the (initial) `TS` time step.

  Level: advanced

  .seealso: [](ch_ts), `TS`, `TSEvent`, `TSSetEventHandler()`
@*/
PetscErrorCode TSSetPostEventStep(TS ts, PetscReal dt)
{
  PetscFunctionBegin;
  ts->event->timestep_postevent = dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSetPostEventIntervalStep - Set the time-step used immediately following an event interval

  Logically Collective

  Input Parameters:
+ ts - time integration context
- dt - post event interval step

  Options Database Keys:
. -ts_event_post_eventinterval_step <dt> time-step after event interval

  Notes:
  This function is deprecated, and its invocation has no effect. Use `TSSetPostEventStep()`.
  This (original) manual page is kept for reference, but is effectively irrelevant.

  `TSSetPostEventIntervalStep()` allows one to set a time-step that is used immediately following an event interval.

  This function should be called from the postevent function set with `TSSetEventHandler()`.

  The post event interval time-step should be selected based on the dynamics following the event.
  If the dynamics are stiff, a conservative (small) step should be used.
  If not, then a larger time-step can be used.

  Level: advanced

  .seealso: [](ch_ts), `TS`, `TSEvent`, `TSSetEventHandler()`
@*/
PetscErrorCode TSSetPostEventIntervalStep(TS ts, PetscReal dt)
{
  PetscFunctionBegin;
  //ts->event->timestep_posteventinterval = dt;
  /* This deprecated function does nothing. Attempting to reproduce its original behaviour,
     i.e. setting the second (not the first) step after event, would break the logic of the TSEventHandler new code.
  */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSetEventTolerances - Set tolerances for event zero crossings

  Logically Collective

  Input Parameters:
+ ts   - time integration context
. tol  - tolerance, `PETSC_DECIDE` or `PETSC_DEFAULT` to leave the current value
- vtol - array of tolerances or `NULL`, used in preference to `tol` if present

  Options Database Key:
. -ts_event_tol <tol> - tolerance for event zero crossing

  Notes:
  One must call `TSSetEventHandler()` before setting the tolerances.

  The size of `vtol` should be equal to the number of events on the given process.

  This function can be also called from the postevent callback set with `TSSetEventHandler()`,
  to adjust the tolerances on the fly.

  Level: beginner

  .seealso: [](ch_ts), `TS`, `TSEvent`, `TSSetEventHandler()`
@*/
PetscErrorCode TSSetEventTolerances(TS ts, PetscReal tol, PetscReal vtol[])
{
  TSEvent event;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (vtol) PetscAssertPointer(vtol, 3);
  PetscCheck(ts->event, PetscObjectComm((PetscObject)ts), PETSC_ERR_USER, "Must set the events first by calling TSSetEventHandler()");

  event = ts->event;
  if (vtol) {
    for (PetscInt i = 0; i < event->nevents; i++) event->vtol[i] = vtol[i];
  } else {
    if (tol != (PetscReal)PETSC_DECIDE && tol != (PetscReal)PETSC_DEFAULT) {
      for (PetscInt i = 0; i < event->nevents; i++) event->vtol[i] = tol;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSSetEventHandler - Sets the functions and parameters used for detecting and handling the events

  Logically Collective on TS

  Input Parameters:
+ ts           - the `TS` context obtained from `TSCreate()`
. nevents      - number of events managed by the given MPI process
. direction    - direction of zero crossing to be detected (one for each local event).
                 `-1` => zero crossing in negative direction,
                 `+1` => zero crossing in positive direction, `0` => both ways
. terminate    - flag to indicate whether time stepping should be terminated after
                 event is detected (one for each local event)
. eventhandler - event monitoring routine, which defines the event-functions
. postevent    - [optional] user post-event function; this function can change the solution, ODE etc at the time of the event
- ctx          - [optional] user-defined context for private data for the
                 event monitor and post-event routine (use `NULL` if no
                 context is desired)

  Calling sequence of `eventhandler`:
$   PetscErrorCode eventhandler(TS ts, PetscReal t, Vec U, PetscReal fvalue[], void* ctx)
+ ts     - the `TS` context
. t      - current time
. U      - current solution
. fvalue - output array with values of local event-functions (the length is `nevents`) for time t and state-vector U
- ctx    - the context passed with `TSSetEventHandler()`

  Calling sequence of `postevent`:
$   PetscErrorCode postevent(TS ts, PetscInt nevents_zero, PetscInt events_zero[], PetscReal t, Vec U, PetscBool forwardsolve, void *ctx)
+ ts           - the `TS` context
. nevents_zero - number of triggered local events (whose event function is marked as crossing zero, and direction is appropriate)
. events_zero  - indices of the triggered local events
. t            - current time
. U            - current solution
. forwardsolve - flag to indicate whether `TS` is doing a forward solve (`PETSC_TRUE`) or adjoint solve (`PETSC_FALSE`)
- ctx          - the context passed with `TSSetEventHandler()`

  Options Database Keys:
+ -ts_event_tol <tol>                       - tolerance for zero crossing check of event-functions
. -ts_event_monitor                         - print choices made by event handler
. -ts_event_recorder_initial_size <recsize> - initial size of event recorder
. -ts_event_post_event_step <dt>            - time step after event
- -ts_event_dt_min <dt>                     - minimum time step considered for TSEvent

  Notes:
  The event-functions should be defined in the `eventhandler` callback using the components of solution `U` and/or time `t`.
  Note that `U` is `PetscScalar`-valued, and the event-functions are `PetscReal`-valued. It is the user's responsibility to
  properly handle this difference, e.g. by applying `PetscRealPart()` or other appropriate conversion means.

  The full set of events is distributed (by the user design) across MPI processes, with each process defining its own local sub-set of events.
  However, event resolution, and the `postevent` callback invocation are performed synchronously on all processes, including
  those processes which have not currently triggered any events.

  Level: intermediate

.seealso: [](ch_ts), `TSEvent`, `TSCreate()`, `TSSetTimeStep()`, `TSSetConvergedReason()`
@*/
PetscErrorCode TSSetEventHandler(TS ts, PetscInt nevents, PetscInt direction[], PetscBool terminate[], PetscErrorCode (*eventhandler)(TS ts, PetscReal t, Vec U, PetscReal fvalue[], void *ctx), PetscErrorCode (*postevent)(TS ts, PetscInt nevents_zero, PetscInt events_zero[], PetscReal t, Vec U, PetscBool forwardsolve, void *ctx), void *ctx)
{
  TSAdapt   adapt;
  PetscReal hmin;
  TSEvent   event;
  PetscBool flg;
#if defined PETSC_USE_REAL_SINGLE
  PetscReal tol = 1e-4;
#else
  PetscReal tol = 1e-6;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (nevents) {
    PetscAssertPointer(direction, 3);
    PetscAssertPointer(terminate, 4);
  }
  PetscCall(PetscNew(&event));
  PetscCall(PetscMalloc1(nevents, &event->fvalue_prev));
  PetscCall(PetscMalloc1(nevents, &event->fvalue));
  PetscCall(PetscMalloc1(nevents, &event->fvalue_right));
  PetscCall(PetscMalloc1(nevents, &event->fsign_prev));
  PetscCall(PetscMalloc1(nevents, &event->fsign));
  PetscCall(PetscMalloc1(nevents, &event->fsign_right));
  PetscCall(PetscMalloc1(nevents, &event->side));
  PetscCall(PetscMalloc1(nevents, &event->side_prev));
  PetscCall(PetscMalloc1(nevents, &event->justrefined_AB));
  PetscCall(PetscMalloc1(nevents, &event->gamma_AB));
  PetscCall(PetscMalloc1(nevents, &event->direction));
  PetscCall(PetscMalloc1(nevents, &event->terminate));
  PetscCall(PetscMalloc1(nevents, &event->events_zero));
  PetscCall(PetscMalloc1(nevents, &event->vtol));
  for (PetscInt i = 0; i < nevents; i++) {
    event->direction[i]      = direction[i];
    event->terminate[i]      = terminate[i];
    event->justrefined_AB[i] = PETSC_FALSE;
    event->gamma_AB[i]       = 1;
    event->side[i]           = 2;
    event->side_prev[i]      = 0;
  }
  event->iterctr            = 0;
  event->processing         = PETSC_FALSE;
  event->nevents            = nevents;
  event->eventhandler       = eventhandler;
  event->postevent          = postevent;
  event->ctx                = ctx;
  event->timestep_postevent = ts->time_step;
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptGetStepLimits(adapt, &hmin, NULL));
  event->timestep_min = hmin;

  event->recsize = 8; /* Initial size of the recorder */
  PetscOptionsBegin(((PetscObject)ts)->comm, ((PetscObject)ts)->prefix, "TS Event options", "TS");
  {
    PetscCall(PetscOptionsReal("-ts_event_tol", "Tolerance for zero crossing check of event-functions", "TSSetEventTolerances", tol, &tol, NULL));
    PetscCall(PetscOptionsName("-ts_event_monitor", "Print choices made by event handler", "", &flg));
    PetscCall(PetscOptionsInt("-ts_event_recorder_initial_size", "Initial size of event recorder", "", event->recsize, &event->recsize, NULL));
    PetscCall(PetscOptionsReal("-ts_event_post_event_step", "Time step after event", "", event->timestep_postevent, &event->timestep_postevent, NULL));
    PetscCall(PetscOptionsDeprecated("-ts_event_post_eventinterval_step", NULL, "3.20", "Use -ts_event_post_event_step"));
    PetscCall(PetscOptionsReal("-ts_event_dt_min", "Minimum time step considered for TSEvent", "", event->timestep_min, &event->timestep_min, NULL));
  }
  PetscOptionsEnd();

  PetscCall(PetscMalloc1(event->recsize, &event->recorder.time));
  PetscCall(PetscMalloc1(event->recsize, &event->recorder.stepnum));
  PetscCall(PetscMalloc1(event->recsize, &event->recorder.nevents));
  PetscCall(PetscMalloc1(event->recsize, &event->recorder.eventidx));
  for (PetscInt i = 0; i < event->recsize; i++) PetscCall(PetscMalloc1(event->nevents, &event->recorder.eventidx[i]));
  /* Initialize the event recorder */
  event->recorder.ctr = 0;

  for (PetscInt i = 0; i < event->nevents; i++) event->vtol[i] = tol;
  if (flg) PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, "stdout", &event->monitor));

  PetscCall(TSEventDestroy(&ts->event));
  ts->event        = event;
  ts->event->refct = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TSEventRecorderResize - Resizes (2X) the event recorder arrays whenever the recording limit (event->recsize)
                          is reached.
*/
static PetscErrorCode TSEventRecorderResize(TSEvent event)
{
  PetscReal *time;
  PetscInt  *stepnum, *nevents;
  PetscInt **eventidx;
  PetscInt   fact = 2;

  PetscFunctionBegin;
  /* Create larger arrays */
  PetscCall(PetscMalloc1(fact * event->recsize, &time));
  PetscCall(PetscMalloc1(fact * event->recsize, &stepnum));
  PetscCall(PetscMalloc1(fact * event->recsize, &nevents));
  PetscCall(PetscMalloc1(fact * event->recsize, &eventidx));
  for (PetscInt i = 0; i < fact * event->recsize; i++) PetscCall(PetscMalloc1(event->nevents, &eventidx[i]));

  /* Copy over data */
  PetscCall(PetscArraycpy(time, event->recorder.time, event->recsize));
  PetscCall(PetscArraycpy(stepnum, event->recorder.stepnum, event->recsize));
  PetscCall(PetscArraycpy(nevents, event->recorder.nevents, event->recsize));
  for (PetscInt i = 0; i < event->recsize; i++) PetscCall(PetscArraycpy(eventidx[i], event->recorder.eventidx[i], event->recorder.nevents[i]));

  /* Destroy old arrays */
  for (PetscInt i = 0; i < event->recsize; i++) PetscCall(PetscFree(event->recorder.eventidx[i]));
  PetscCall(PetscFree(event->recorder.eventidx));
  PetscCall(PetscFree(event->recorder.nevents));
  PetscCall(PetscFree(event->recorder.stepnum));
  PetscCall(PetscFree(event->recorder.time));

  /* Set pointers */
  event->recorder.time     = time;
  event->recorder.stepnum  = stepnum;
  event->recorder.nevents  = nevents;
  event->recorder.eventidx = eventidx;

  /* Update the size */
  event->recsize *= fact;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Helper routine to handle user postevents and recording
*/
static PetscErrorCode TSPostEvent(TS ts, PetscReal t, Vec U)
{
  TSEvent   event        = ts->event;
  PetscBool restart      = PETSC_FALSE;
  PetscBool terminate    = PETSC_FALSE;
  PetscBool statechanged = PETSC_FALSE;
  PetscInt  ctr, stepnum;
  PetscBool inflag[3], outflag[3];
  PetscBool forwardsolve = PETSC_TRUE; // Flag indicating that TS is doing a forward solve

  PetscFunctionBegin;
  if (event->postevent) {
    PetscObjectState state_prev, state_post;
    PetscCall(PetscObjectStateGet((PetscObject)U, &state_prev));
    PetscCall((*event->postevent)(ts, event->nevents_zero, event->events_zero, t, U, forwardsolve, event->ctx)); // TODO update 'restart' here!
    PetscCall(PetscObjectStateGet((PetscObject)U, &state_post));
    if (state_prev != state_post) {
      restart      = PETSC_TRUE;
      statechanged = PETSC_TRUE;
    }
  }

  // Handle termination events and step restart
  for (PetscInt i = 0; i < event->nevents_zero; i++)
    if (event->terminate[event->events_zero[i]]) terminate = PETSC_TRUE;
  inflag[0] = restart;
  inflag[1] = terminate;
  inflag[2] = statechanged;
  PetscCall(MPIU_Allreduce(inflag, outflag, 3, MPIU_BOOL, MPI_LOR, ((PetscObject)ts)->comm));
  restart      = outflag[0];
  terminate    = outflag[1];
  statechanged = outflag[2];
  if (restart) PetscCall(TSRestartStep(ts));
  if (terminate) PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_EVENT));

  /* Recalculate the functions and signs if states have been changed by the user postevent callback.
     Note! If the state HAS NOT changed, the previous 'event->fsign' is kept, which:
     - might have been defined using the previous (now-possibly-overridden) event->vtol,
     - might have been set to zero on reaching a small time step rather than using the vtol criterion.
     This will enforce keeping event->fsign = 0 where the zero-crossings were actually triggered,
     resulting in a more consistent behaviour of fsign's.
  */
  if (statechanged) {
    if (event->monitor) PetscCall(PetscPrintf(((PetscObject)ts)->comm, "TSEvent: at time %g the vector state has been changed by PostEvent, recalculating fvalues and signs\n", (double)t));
    PetscCall(VecLockReadPush(U));
    PetscCall((*event->eventhandler)(ts, t, U, event->fvalue, event->ctx));
    PetscCall(VecLockReadPop(U));
    TSEventCalcSigns(event->nevents, event->fvalue, event->vtol, event->fsign); // note, event->vtol might have been changed by the postevent()
  }

  // Record the event in the event recorder
  PetscCall(TSGetStepNumber(ts, &stepnum));
  ctr = event->recorder.ctr;
  if (ctr == event->recsize) PetscCall(TSEventRecorderResize(event));
  event->recorder.time[ctr]    = t;
  event->recorder.stepnum[ctr] = stepnum;
  event->recorder.nevents[ctr] = event->nevents_zero;
  for (PetscInt i = 0; i < event->nevents_zero; i++) event->recorder.eventidx[ctr][i] = event->events_zero[i];
  event->recorder.ctr++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* (modified) Anderson-Bjorck variant of regula falsi method, refines [tleft, t] or [t, tright] based on 'side' (-1 or +1).
   The scaling parameter is defined based on the 'justrefined' flag, the history of repeats of 'side', and the threshold.
   To escape certain failure modes, the algorithm may drift towards the bisection rule.
   The value pointed to by 'side_prev' gets updated.
   This function returns the new time step.

   The underlying pure Anderson-Bjorck algorithm was taken as described in
   J.M. Fernandez-Diaz, C.O. Menendez-Perez "A common framework for modified Regula Falsi methods and new methods of this kind", 2023.
   The modifications subsequently introduced have little effect on the behaviour for simple cases requiring only a few iterations
   (some minor convergence slowdown may take place though), but the effect becomes more pronounced for tough cases requiring many iterations.
   For the latter situation the speed-up may be order(s) of magnitude compared to the classical Anderson-Bjorck.
   The modifications (the threshold trick, and the drift towards bisection) were tested and tweaked
   based on a number of test functions from the mentioned paper.
*/
static inline PetscReal RefineAndersonBjorck(PetscReal tleft, PetscReal t, PetscReal tright, PetscReal fleft, PetscReal f, PetscReal fright, PetscInt side, PetscInt *side_prev, PetscBool justrefined, PetscReal *gamma)
{
  PetscReal      new_dt, scal = 1.0, scalB = 1.0, threshold = 0.0, power;
  PetscInt       reps     = 0;
  const PetscInt REPS_CAP = 8; // an upper bound to be imposed on 'reps' (set to 8, somewhat arbitrary number, found after some tweaking)

  // Preparations
  if (justrefined) {
    if (*side_prev * side > 0) *side_prev += side;     // the side keeps repeating -> increment the side counter (-ve or +ve)
    else *side_prev = side;                            // reset the counter
    reps      = PetscMin(*side_prev * side, REPS_CAP); // the length of the recent side-repeat series, including the current 'side'
    threshold = PetscPowReal(0.5, reps) * 0.1;         // ad-hoc strategy for threshold calculation (involved some tweaking)
  } else *side_prev = side;                            // initial reset of the counter

  // Time step calculation
  if (side == -1) {
    if (justrefined && fright != 0.0 && fleft != 0.0) {
      scal  = (fright - f) / fright;
      scalB = -f / fleft;
    }
  } else { // must be side == +1
    if (justrefined && fleft != 0.0 && fright != 0.0) {
      scal  = (fleft - f) / fleft;
      scalB = -f / fright;
    }
  }

  if (scal < threshold) scal = 0.5;
  if (reps > 1) *gamma *= scal; // side did not switch since the last time, accumulate gamma
  else *gamma = 1.0;            // side switched -> reset gamma
  power = PetscMax(0.0, (reps - 2.0) / (REPS_CAP - 2.0));
  scal  = PetscPowReal(scalB / *gamma, power) * (*gamma); // mix the Anderson-Bjorck scaling and Bisection scaling

  if (side == -1) new_dt = (scal * fleft * t - f * tleft) / (scal * fleft - f) - tleft;
  else new_dt = (f * tright - scal * fright * t) / (f - scal * fright) - t;
  /* In tough cases (e.g. a polynomial of high order), there is a failure mode for the standard Anderson-Bjorck,
     when the new proposed point jumps from one end-point of the bracket to the other, however the bracket is contracting very slowly.
     A larger threshold for 'scal' prevents entering this mode.
     On the other hand, if the iteration gets stuck near one end-point of the bracket, and the 'side' does not switch for a while,
     the 'scal' drifts towards the bisection approach (via scalB), ensuring stable convergence.
  */
  return new_dt;
}

/* Checks if the current point (t) is the zero-crossing location, based on the event-function signs and direction[].
   The situation (fsign_prev, fsign) = (0, 0) is treated as staying in the near-zero-zone of the previous zero-crossing.
   This function may update event->side[].
*/
static PetscErrorCode TSEventTestZero(TS ts)
{
  TSEvent event = ts->event;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < event->nevents; i++) {
    if (event->fsign[i] == 0) { // found the potential event location
      if (event->fsign_prev[i] < 0 && event->direction[i] >= 0) event->side[i] = 0;
      if (event->fsign_prev[i] > 0 && event->direction[i] <= 0) event->side[i] = 0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Checks if [fleft, f] or [f, fright] are 'brackets', i.e. intervals with the sign change, satisfying the 'direction'.
   The right interval is only checked if iterctr > 0 (i.e. Anderson-Bjorck refinement has started).
   The intervals like [0, x] and [x, 0] are not counted as brackets.
   The function returns the 'side' value: -1 (left, or both are brackets), +1 (only right one), +2 (neither).
*/
static inline PetscInt TSEventTestBracket(PetscInt fsign_left, PetscInt fsign, PetscInt fsign_right, PetscInt direction, PetscInt iterctr)
{
  PetscInt side = 2;
  if (fsign_left * fsign < 0 && fsign * direction >= 0) side = -1;
  if (side != -1 && iterctr > 0 && fsign * fsign_right < 0 && fsign_right * direction >= 0) side = 1;
  return side;
}

/* A helper function for capping the time steps, accounting for time span points.
   It uses 'event->timestep_cache' as a time step to calculate the tolerance for tspan points detection. This
   is done since the event resolution may result in significant time step refinement, and we don't use these small steps for tolerances.
   To enhance the consistency of tspan points detection, tolerance 'tspan->worktol' is reused later in the TSSolve iteration.
   If a user-defined step is cut by this function, the input uncut step is saved to adapt->dt_span_cached.
   Flag 'user_dt' indicates if the step was defined by user.
*/
static inline PetscReal TSEvent_dt_cap(TS ts, PetscReal t, PetscReal dt, PetscBool user_dt)
{
  PetscReal res = dt;
  if (ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
    PetscReal maxdt    = ts->max_time - t; // this may be overriden by tspan
    PetscBool cut_made = PETSC_FALSE;
    PetscReal eps      = 10 * PETSC_MACHINE_EPSILON;
    if (ts->tspan) {
      PetscInt   ctr = ts->tspan->spanctr;
      PetscInt   Ns  = ts->tspan->num_span_times;
      PetscReal *st  = ts->tspan->span_times;

      if (ts->tspan->worktol == 0) ts->tspan->worktol = ts->tspan->reltol * ts->event->timestep_cache + ts->tspan->abstol; // in case TSAdaptChoose() has not defined it
      if (ctr < Ns && PetscIsCloseAtTol(t, st[ctr], ts->tspan->worktol, 0)) {                                              // just hit a time span point
        if (ctr + 1 < Ns) maxdt = st[ctr + 1] - t;                                                                         // ok to use the next time span point
        else maxdt = ts->max_time - t;                                                                                     // can't use the next time span point: they have finished
      } else if (ctr < Ns) maxdt = st[ctr] - t;                                                                            // haven't hit a time span point, use the nearest one
    }
    maxdt = PetscMin(maxdt, ts->max_time - t);
    PetscCheck((maxdt > eps) || (PetscAbsReal(maxdt) <= eps && PetscIsCloseAtTol(t, ts->max_time, eps, 0)), PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Unexpected state: bad maxdt in TSEvent_dt_cap()");

    if (PetscIsCloseAtTol(dt, maxdt, eps, 0)) res = maxdt; // no cut
    else {
      if (dt > maxdt) {
        res      = maxdt; // yes cut
        cut_made = PETSC_TRUE;
      } else res = dt; // no cut
    }
    if (ts->adapt && user_dt) { // only update dt_span_cached for the user-defined step
      if (cut_made) ts->adapt->dt_span_cached = dt;
      else ts->adapt->dt_span_cached = 0;
    }
  }
  return res;
}

/* A helper function for updating the left-end values
*/
static inline void TSEvent_update_left(TSEvent event, PetscReal t)
{
  event->ptime_prev = t;
  for (PetscInt i = 0; i < event->nevents; i++) {
    event->fvalue_prev[i] = event->fvalue[i];
    event->fsign_prev[i]  = event->fsign[i];
  }
}

/* A helper function for updating the right-end values
*/
static inline void TSEvent_update_right(TSEvent event, PetscReal t)
{
  event->ptime_right = t;
  for (PetscInt i = 0; i < event->nevents; i++) {
    event->fvalue_right[i] = event->fvalue[i];
    event->fsign_right[i]  = event->fsign[i];
  }
}

/* TSEventHandler() - the main function to perform a single iteration of event resolution.
   Developer notes:
   1) The 'event->iterctr > 0' is used as an indicator that Anderson-Bjorck refinement has started.
   2) If event->iterctr == 0, then justrefined_AB[i] is always false.
   3) The right-end quantities: ptime_right, fvalue_right[i] and fsign_right[i] are only guaranteed to be valid for event->iterctr > 0.
   4) If event->iterctr > 0, then event->processing is PETSC_TRUE; the opposite may not hold.

   The intervals containing the potential zero-crossings are called 'brackets'.
   The algorithm first finds a bracket, and then sequentially subdivides it, generating a sequence
   of brackets whose length tends to zero. The bracket subdivision involves the (modified) Anderson-Bjorck method.
*/
PetscErrorCode TSEventHandler(TS ts)
{
  TSEvent     event;
  PetscReal   t, dt_min;
  Vec         U;
  PetscMPIInt rank;
  PetscInt    minsidein = 2, minsideout = 2; // minsideout is sync on all ranks
  PetscBool   finished = PETSC_FALSE;        // should stay sync on all ranks

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscCallMPI(MPI_Comm_rank(((PetscObject)ts)->comm, &rank)); // 'rank' is used in the event->monitor reports

  if (!ts->event) PetscFunctionReturn(PETSC_SUCCESS);
  event               = ts->event;
  event->nevents_zero = 0;
  for (PetscInt i = 0; i < event->nevents; i++) event->side[i] = 2; // side's are reset on each new iteration
  if (event->iterctr == 0)
    for (PetscInt i = 0; i < event->nevents; i++) event->justrefined_AB[i] = PETSC_FALSE;

  PetscCall(TSGetTime(ts, &t));
  if (!event->processing) { // update the caches
    PetscReal dt;
    PetscCall(TSGetTimeStep(ts, &dt));
    event->ptime_cache    = t;
    event->timestep_cache = dt; // the next TS move is planned to be: t -> t+dt
  }

  PetscCall(TSGetSolution(ts, &U));
  PetscCall(VecLockReadPush(U));
  PetscCall((*event->eventhandler)(ts, t, U, event->fvalue, event->ctx)); // fill fvalue's at point 't'
  PetscCall(VecLockReadPop(U));
  TSEventCalcSigns(event->nevents, event->fvalue, event->vtol, event->fsign); // fill fvalue signs
  PetscCall(TSEventTestZero(ts));                                             // check if the current point 't' is the event location; event->side[] may get updated

  for (PetscInt i = 0; i < event->nevents; i++) { // check for brackets on the left/right of 't'
    if (event->side[i] != 0) event->side[i] = TSEventTestBracket(event->fsign_prev[i], event->fsign[i], event->fsign_right[i], event->direction[i], event->iterctr);
    minsidein = PetscMin(minsidein, event->side[i]);
  }
  PetscCall(MPIU_Allreduce(&minsidein, &minsideout, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)ts)));
  /* minsideout (sync on all ranks) indicates the minimum of the following states:
     -1 : [ptime_prev, t] is a bracket
     +1 : [t, ptime_right] is a bracket
      0 : t is a zero-crossing
      2 : none of the above
  */
  if (minsideout == -1 || minsideout == +1) { // this if-branch will refine the left/right bracket
    PetscReal dti_min = PETSC_MAX_REAL;
    for (PetscInt i = 0; i < event->nevents; i++) {
      if (event->side[i] == minsideout) { // only refine the appropriate brackets
        PetscReal dti = RefineAndersonBjorck(event->ptime_prev, t, event->ptime_right, event->fvalue_prev[i], event->fvalue[i], event->fvalue_right[i], event->side[i], &event->side_prev[i], event->justrefined_AB[i], &event->gamma_AB[i]);
        dti_min       = PetscMin(dti_min, dti);
      }
    }
    PetscCall(MPIU_Allreduce(&dti_min, &dt_min, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)ts)));

    if (PetscAbsReal(dt_min) < event->timestep_min) { // check if the time step is small
      finished = PETSC_TRUE;
      for (PetscInt i = 0; i < event->nevents; i++)
        if (event->side[i] == minsideout) {
          event->events_zero[event->nevents_zero++] = i;
          event->fsign[i]                           = 0; // note, the sign = 0 is enforced here, irrespective of the vtol criterion
          if (event->monitor)
            PetscCall(PetscViewerASCIIPrintf(event->monitor, "[%" PetscInt_FMT "] TSEvent: iter %" PetscInt_FMT " - Event %" PetscInt_FMT " accepting time %g as event location, due to reaching too small time step %g while refining the bracket\n", (PetscInt)rank,
                                             event->iterctr, i, (double)t, (double)dt_min));
        }
    }

    if (minsideout == -1) { // minsideout == -1, update the right-end values, retain the left-end values
      TSEvent_update_right(event, t);
      if (!finished) { // handle the rollback; note: 'finished' flag is sync on all ranks
        PetscCall(TSRollBack(ts));
        PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_ITERATING)); // e.g. to override TS_CONVERGED_TIME on reaching ts->max_time
      }
    } else TSEvent_update_left(event, t); // minsideout == +1, update the left-end values, retain the right-end values

    for (PetscInt i = 0; i < event->nevents; i++) { // update the "Anderson-Bjorck" flags
      if (event->side[i] == minsideout) {
        event->justrefined_AB[i] = PETSC_TRUE; // only for these i's Anderson-Bjorck was invoked
        if (event->monitor && !finished)
          PetscCall(PetscViewerASCIIPrintf(event->monitor, "[%" PetscInt_FMT "] TSEvent: iter %" PetscInt_FMT " - Event %" PetscInt_FMT " refining the bracket with sign change [%g - %g], next stepping to %g\n", (PetscInt)rank, event->iterctr, i,
                                           (double)event->ptime_prev, (double)event->ptime_right, (double)(event->ptime_prev + dt_min)));
      } else event->justrefined_AB[i] = PETSC_FALSE; // for these i's Anderson-Bjorck was not invoked
    }
    event->iterctr++;
    event->processing = PETSC_TRUE;
  } else if (minsideout == 0) { // found the appropriate zero-crossing (and no brackets to the left), finishing!
    finished = PETSC_TRUE;
    for (PetscInt i = 0; i < event->nevents; i++)
      if (event->side[i] == minsideout) {
        event->events_zero[event->nevents_zero++] = i;
        if (event->monitor)
          PetscCall(PetscViewerASCIIPrintf(event->monitor, "[%" PetscInt_FMT "] TSEvent: iter %" PetscInt_FMT " - Event %" PetscInt_FMT " zero crossing located at time %g (tol=%g)\n", (PetscInt)rank, event->iterctr, i, (double)t, (double)event->vtol[i]));
      }
    event->iterctr++;
    event->processing = PETSC_TRUE;
  } else { // minsideout == 2: no brackets, no zero-crossings
    PetscCheck(event->iterctr == 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Unexpected state (event->iterctr != 0) in TSEventHandler()");
    if (event->processing) PetscCall(TSSetTimeStep(ts, TSEvent_dt_cap(ts, t, event->timestep_cache, PETSC_FALSE)));
    event->processing = PETSC_FALSE;
  }

  if (finished) { // finished handling the current event
    PetscCall(TSPostEvent(ts, t, U));

    PetscReal dt;
    PetscBool user_dt = PETSC_FALSE;
    if (event->timestep_postevent > 0) {
      dt                = event->timestep_postevent; // user-provided post-event dt
      event->processing = PETSC_FALSE;
      user_dt           = PETSC_TRUE;
    } else {
      dt                = event->ptime_cache - t; // 'petsc-decide' the post-event dt
      event->processing = PETSC_TRUE;
      if (PetscAbsReal(dt) < PETSC_SMALL) {
        dt                = event->timestep_cache; // we hit the event, continue with the cached time step
        event->processing = PETSC_FALSE;
      }
    }
    PetscCall(TSSetTimeStep(ts, TSEvent_dt_cap(ts, t, dt, user_dt)));
    event->iterctr = 0;
  } // if-finished

  if (event->iterctr == 0) TSEvent_update_left(event, t); // not found an event, or finished the event
  else {
    PetscCall(TSGetTime(ts, &t));                                             // update 't' to account for potential rollback
    PetscCall(TSSetTimeStep(ts, TSEvent_dt_cap(ts, t, dt_min, PETSC_FALSE))); // continue resolving the event
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSAdjointEventHandler(TS ts)
{
  TSEvent   event;
  PetscReal t;
  Vec       U;
  PetscInt  ctr;
  PetscBool forwardsolve = PETSC_FALSE; // Flag indicating that TS is doing an adjoint solve

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (!ts->event) PetscFunctionReturn(PETSC_SUCCESS);
  event = ts->event;

  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetSolution(ts, &U));

  ctr = event->recorder.ctr - 1;
  if (ctr >= 0 && PetscAbsReal(t - event->recorder.time[ctr]) < PETSC_SMALL) {
    // Call the user postevent function
    if (event->postevent) {
      PetscCall((*event->postevent)(ts, event->recorder.nevents[ctr], event->recorder.eventidx[ctr], t, U, forwardsolve, event->ctx));
      event->recorder.ctr--;
    }
  }

  PetscCall(PetscBarrier((PetscObject)ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSGetNumEvents - Get the number of events defined on the given MPI process

  Logically Collective

  Input Parameter:
. ts - the `TS` context

  Output Parameter:
. nevents - the number of local events on each MPI process

  Level: intermediate

.seealso: [](ch_ts), `TSEvent`, `TSSetEventHandler()`
@*/
PetscErrorCode TSGetNumEvents(TS ts, PetscInt *nevents)
{
  PetscFunctionBegin;
  *nevents = ts->event->nevents;
  PetscFunctionReturn(PETSC_SUCCESS);
}
