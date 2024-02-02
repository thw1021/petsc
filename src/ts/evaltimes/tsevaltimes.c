#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

// TSEvaluationTimesSetFromOptions - to be called from TSSetFromOptions(); may construct ts->evaltimes if necessary
PetscErrorCode TSEvaluationTimesSetFromOptions(TS ts, PetscOptionItems PetscOptionsObject)
{
  PetscBool flg1, flg2, flg3;
  PetscReal evtimes[100];
  PetscInt  nt1 = PETSC_STATIC_ARRAY_LENGTH(evtimes);
  PetscInt  nt2 = nt1;
  char      range[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscOptionsHeadBegin(PetscOptionsObject, "TSEvaluationTimes options");
  PetscCall(PetscOptionsRealArray("-ts_time_span", "Array of evaluation time points /overrides init_time, max_time/", "TSEvaluationTimesAddArray", evtimes, &nt1, &flg1));
  PetscCall(PetscOptionsRealArray("-ts_eval_times", "Array of evaluation time points", "TSEvaluationTimesAddArray", evtimes, &nt2, &flg2));
  PetscCall(PetscOptionsString("-ts_eval_times_uniform", "Triplet x,y,n <=> n evenly spaced evaluation time points in [x,y]", "TSEvaluationTimesAddUniform", "", range, sizeof(range), &flg3));
  PetscOptionsHeadEnd();

  PetscCheck(!((flg1 && flg2) || (flg1 && flg3) || (flg2 && flg3)), PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_INCOMP, "Can only use one of the options at a time: -ts_time_span, -ts_eval_times, -ts_eval_times_uniform");
  if (flg1) PetscCall(TSEvaluationTimesSetDefaultSchedule(ts, nt1, 0.0, 0.0, evtimes, PETSC_TRUE));  // overrides t0, tmax
  if (flg2) PetscCall(TSEvaluationTimesSetDefaultSchedule(ts, nt2, 0.0, 0.0, evtimes, PETSC_FALSE)); // does not override t0, tmax
  if (flg3) {
    PetscReal   x[2];
    PetscInt    n;
    const char *value;
    const char *errmsg = "In option '-ts_eval_times_uniform': string expected 'real,real,int' / string provided '%.100s'";
    PetscToken  token;

    PetscCall(PetscTokenCreate(range, ',', &token));
    for (PetscInt i = 0; i < 3; i++) {
      PetscCall(PetscTokenFind(token, &value));
      PetscCheck(value, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, errmsg, range);
      if (i < 2) PetscCall(PetscOptionsStringToReal(value, &x[i]));
      else PetscCall(PetscOptionsStringToInt(value, &n));
    }
    PetscCall(PetscTokenFind(token, &value));
    PetscCheck(!value, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, errmsg, range);
    PetscCall(PetscTokenDestroy(&token));
    PetscCall(TSEvaluationTimesSetDefaultSchedule(ts, n, x[0], x[1], NULL, PETSC_FALSE)); // does not override t0, tmax
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimesCreate - constructs ts->evaltimes, should be followed by adding a schedule.
static PetscErrorCode TSEvaluationTimesCreate(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscCheck(!ts->evaltimes, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "evaltimes == NULL is expected");
  PetscCall(PetscNew(&ts->evaltimes));
  ts->evaltimes->assembled = PETSC_FALSE; // the evaluation times is now empty, adding a schedule is expected
  ts->evaltimes->refct     = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimesSchedule_LockVecs - locks ets->vecs[i]
static PetscErrorCode TSEvaluationTimesSchedule_LockVecs(TSEvaluationTimesSchedule ets)
{
  PetscFunctionBegin;
  if (!ets || !PetscDefined(USE_DEBUG)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(ets, 1);
  if (ets->vecs)
    for (PetscInt i = 0; i < ets->len; i++)
      if (ets->vecs[i]) PetscCall(VecLockReadPush(ets->vecs[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimesSchedule_UnlockVecs - unlocks ets->vecs[i]
static PetscErrorCode TSEvaluationTimesSchedule_UnlockVecs(TSEvaluationTimesSchedule ets)
{
  PetscFunctionBegin;
  if (!ets || !PetscDefined(USE_DEBUG)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(ets, 1);
  if (ets->vecs)
    for (PetscInt i = 0; i < ets->len; i++)
      if (ets->vecs[i]) PetscCall(VecLockReadPop(ets->vecs[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimesSchedule_DestroyVecs - routine to VecDestroy the ets->vecs[i]
static PetscErrorCode TSEvaluationTimesSchedule_DestroyVecs(TSEvaluationTimesSchedule ets)
{
  PetscFunctionBegin;
  if (ets && ets->vecs)
    for (PetscInt i = 0; i < ets->len; i++) PetscCall(VecDestroy(&ets->vecs[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimesSchedule_Destroy - destroys TSEvaluationTimesSchedule
static PetscErrorCode TSEvaluationTimesSchedule_Destroy(PetscCtxRt obj)
{
  TSEvaluationTimesSchedule ets = (TSEvaluationTimesSchedule)*(void **)obj;

  PetscFunctionBegin;
  PetscAssertPointer(obj, 1);
  if (!ets) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(!ets->c_locked, PETSC_COMM_SELF, PETSC_ERR_COR, "TSEvaluationTimesSchedule '%.100s' is locked for modification; missing TSEvaluationTimesRestoreSolutions()?", ets->name);

  PetscCall(PetscFree(ets->times));
  PetscCall(PetscFree(ets->inds_global));
  PetscCall(TSEvaluationTimesSchedule_DestroyVecs(ets));
  PetscCall(PetscFree(ets->vecs));
  PetscCall(PetscFree(ets));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimes_Clear - worker function for clearing TSEvaluationTimes; it destroys the input structure, and sets it to NULL
static PetscErrorCode TSEvaluationTimes_Clear(TSEvaluationTimes *evaltimes)
{
  PetscFunctionBegin;
  PetscAssertPointer(evaltimes, 1);
  if (!*evaltimes) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree((*evaltimes)->times_global));
  PetscCall(PetscObjectListDestroy(&(*evaltimes)->schedlist));
  PetscCall(PetscFree(*evaltimes));
  *evaltimes = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSEvaluationTimesReset - Destroys the vectors saved in TSEvaluationTimes, resets the inner counters

  Logically Collective

  Input Parameter:
. ts - time integration context

  Level: intermediate

  Notes:
  This routine may be useful e.g. if one has run `TSSolve()` recording the vectors to the
  `TSEvaluationTimes`, and now wants to re-run the `TSSolve()`. Although the second `TSSolve()` will always
  overwrite the old saved vectors when necessary, calling `TSEvaluationTimesReset()` first
  will ensure nothing is left from the previous run.
  The evaluation time points are not affected by this function.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesDestroy()`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesGetSolutions()`
@*/
PetscErrorCode TSEvaluationTimesReset(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (ts->evaltimes) {
    PetscContainer            cont;
    TSEvaluationTimesSchedule ets;

    PetscCheck(ts->evaltimes->assembled, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "Need to set up TSEvaluationTimes first; use TSEvaluationTimesSetUp()");
    for (PetscObjectList it = ts->evaltimes->schedlist; it; it = it->next) {
      cont = (PetscContainer)it->obj;
      PetscCall(PetscContainerGetPointer(cont, (void **)&ets));
      PetscCheck(!ets->c_locked, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "TSEvaluationTimesSchedule '%.100s' is locked for modification; missing TSEvaluationTimesRestoreSolutions()?", ets->name);
      ets->ctr   = 0;
      ets->start = ets->end = -1;
      PetscCall(TSEvaluationTimesSchedule_DestroyVecs(ets)); // individual vectors are deleted, the array is not
    }
    ts->evaltimes->ctr_global = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSEvaluationTimesDestroy - Deletes TSEvaluationTimes from TS

  Logically Collective

  Input Parameter:
. ts - time integration context whose `TSEvaluationTimes` is to be deleted

  Level: intermediate

  Notes:
  This function is equivalent to 'manually' deleting all TSEvaluationTimesSchedule's.

  On destroying the TS, `TSEvaluationTimesDestroy()` is called automatically.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesSchedule`, `TSEvaluationTimesReset()`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesGetSolutions()`
@*/
PetscErrorCode TSEvaluationTimesDestroy(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);

  if (!ts->evaltimes) PetscFunctionReturn(PETSC_SUCCESS);
  if (--ts->evaltimes->refct > 0) {
    ts->evaltimes = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TSEvaluationTimes_Clear(&ts->evaltimes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimes_Add - Common part for TSEvaluationTimesAddArray(), TSEvaluationTimesAddUniform()
// If a TSEvaluationTimesSchedule is added/replaced, 'pets' returns the schedule object to simplify further access.
// Adding a schedule to a NULL 'evaltimes' invokes its construction, deleting the last schedule from 'evaltimes' invokes its destruction.
static PetscErrorCode TSEvaluationTimes_Add(TS ts, const char *name, PetscInt nprivate, PetscErrorCode (*handler)(TS ts, PetscInt iunion, PetscInt iprivate, PetscReal t, Vec full, Vec *sub, void *ctx), void *ctx, TSEvaluationTimesSchedule *pets)
{
  PetscContainer  cont    = NULL;
  const PetscBool add_ets = (nprivate > 0 && handler ? PETSC_TRUE : PETSC_FALSE);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  if (ctx) PetscAssertPointer(ctx, 5);
  PetscAssertPointer(pets, 6);

  *pets = NULL;
  if (!ts->evaltimes && add_ets) PetscCall(TSEvaluationTimesCreate(ts)); // note, !ts->evaltimes && !add_ets => !cont => error
  if (ts->evaltimes) PetscCall(PetscObjectListFind(ts->evaltimes->schedlist, name, (PetscObject *)&cont));
  if (cont) {                                                             // 'name' has been found
    PetscCall(PetscObjectListAdd(&ts->evaltimes->schedlist, name, NULL)); // first, delete the existing item
    if (!ts->evaltimes->schedlist && !add_ets) {
      PetscCall(TSEvaluationTimesDestroy(ts)); // if the list has become empty, and no plans to add -> destroy evaltimes
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else PetscCheck(add_ets, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "TSEvaluationTimesSchedule with name '%.100s' is absent in the list, cannot delete it", name);

  if (add_ets) { // add new TSEvaluationTimesSchedule to the list (or replace the former one)
    TSEvaluationTimesSchedule ets;

    PetscCall(PetscNew(&ets));                                   // alloc. zeroed memory
    PetscCall(PetscStrncpy(ets->name, name, sizeof(ets->name))); // fill ets ...
    ets->len = nprivate;
    PetscCall(PetscCalloc1(nprivate, &ets->times));
    PetscCall(PetscCalloc1(nprivate, &ets->inds_global));
    PetscCall(PetscCalloc1(nprivate, &ets->vecs));
    ets->c_times  = NULL;
    ets->c_vecs   = NULL;
    ets->c_locked = PETSC_FALSE;
    ets->ctr      = 0;
    ets->start    = -1;
    ets->end      = -1;
    ets->ctx      = ctx;
    ets->handler  = handler;

    cont = NULL;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)ts), &cont));
    PetscCall(PetscContainerSetPointer(cont, (void *)ets));
    PetscCall(PetscContainerSetCtxDestroy(cont, TSEvaluationTimesSchedule_Destroy));
    PetscCall(PetscObjectListAdd(&ts->evaltimes->schedlist, name, (PetscObject)cont)); // 'cont' gets referenced via the list
    PetscCall(PetscObjectDereference((PetscObject)cont));                              // need to remove the locally created reference to 'cont'
    *pets = ets;
  }
  ts->evaltimes->assembled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSEvaluationTimesAddArray - Add a TSEvaluationTimesSchedule, i.e. a set of points where time stepper stops and records the solution to array

  Logically Collective

  Input Parameters:
+ ts         - time integration context
. name       - reference name for the `TSEvaluationTimesSchedule`
. nprivate   - number of evaluation time points
. schedtimes - sorted array of evaluation time points (values are copied over, length = `nprivate`)
. handler    - callback to be invoked at the evaluation time points
- ctx        - [optional] user-defined context for private data for `handler()` routine, use `NULL` if not needed

  Calling sequence of `handler`:
+ ts       - the `TS` context
. iunion   - index of current point in the union of all evaluation time points
. iprivate - index of current point within the given `TSEvaluationTimesSchedule`
. t        - current time
. full     - current solution vector
. sub      - the output vector, should be created by user, or left as `NULL` (default)
- ctx      - context provided as the last argument to `TSEvaluationTimesAddArray()`

  Options Database Keys:
+ -ts_time_span t1,...tn  - solutions will be saved at the time points listed (schedule 'default' is added); will use t1 and tn as initial and max TS times
- -ts_eval_times t1,...tn - solutions will be saved at the time points listed (schedule 'default' is added)

  Level: intermediate

  Notes:
  The evaluation time points `schedtimes` should be sorted in increasing order.

  Calling this function multiple times results in adding several different TSEvaluationTimesSchedule's, each one having
  its unique reference `name`.

  Calling `TSEvaluationTimesAddArray()` twice with the same `name` overwrites the `TSEvaluationTimesSchedule` previously set.
  Besides, if `NULL` is provided as the callback, the `TSEvaluationTimesSchedule` with the given `name` is removed from the
  `TSEvaluationTimes` collection. Providing nprivate <= 0 or schedtimes == NULL has the same effect.

  Options `-ts_time_span`, `-ts_eval_times` are equivalent to adding/overriding a schedule named "default", adopting callback
  `TSEvaluationTimesDefaultHandler()`, and the evaluation time points listed. The first option (`-ts_time_span`) also overrides
  the initial and max times of `TS` by t1 and tn respectively. The options are activated via `TSSetFromOptions()`.

  After calling `TSEvaluationTimesAddArray()` one or multiple times, the internal data structures need to be set up by
  `TSEvaluationTimesSetUp()` before running the time stepper.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesSchedule`, `TSEvaluationTimesAddUniform()`, `TSEvaluationTimesDefaultHandler()`, `TSEvaluationTimesSetUp()`, `TSEvaluationTimesGetSolutions()`
@*/
PetscErrorCode TSEvaluationTimesAddArray(TS ts, const char *name, PetscInt nprivate, const PetscReal *schedtimes, PetscErrorCode (*handler)(TS ts, PetscInt iunion, PetscInt iprivate, PetscReal t, Vec full, Vec *sub, void *ctx), void *ctx)
{
  TSEvaluationTimesSchedule ets;

  PetscFunctionBegin;
  if (schedtimes) PetscAssertPointer(schedtimes, 4);
  else nprivate = 0;

  PetscCall(TSEvaluationTimes_Add(ts, name, nprivate, handler, ctx, &ets)); // construction/destruction may take place here
  if (ets) PetscCall(PetscArraycpy(ets->times, schedtimes, nprivate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSEvaluationTimesAddUniform - Add a TSEvaluationTimesSchedule defined by a [min, max] range with uniform steps

  Logically Collective

  Input Parameters:
+ ts       - time integration context
. name     - reference name for the `TSEvaluationTimesSchedule`
. nprivate - number of evaluation time points
. min      - lower bound of the evaluation time points
. max      - upper bound of the evaluation time points
. handler  - callback to be invoked at the evaluation time points
- ctx      - [optional] user-defined context for private data for `handler()` routine, use `NULL` if not needed

  Calling sequence of `handler`:
+ ts       - the `TS` context
. iunion   - index of current point in the union of all evaluation time points
. iprivate - index of current point within the given `TSEvaluationTimesSchedule`
. t        - current time
. full     - current solution vector
. sub      - the output vector, should be created by user, or left as `NULL` (default)
- ctx      - context provided as the last argument to `TSEvaluationTimesAddUniform()`

  Options Database Key:
. -ts_eval_times_uniform x,y,n - solutions will be saved at n evenly spaced time points in [x,y] (schedule 'default' is added)

  Level: intermediate

  Notes:
  This function is equivalent to `TSEvaluationTimesAddArray()`. The only difference is how the array of evaluation time points is defined.
  For `TSEvaluationTimesAddArray()` the array is provided explicitly. For `TSEvaluationTimesAddUniform()` the array of length `nprivate`
  is constructed from the `[min, max]` range, using a uniform step between points.
.vb
  step = (max - min)/(nprivate - 1)
.ve
  If nprivate == 1 and min == max, a single-point array is constructed.

  Option `-ts_eval_times_uniform` adds/overrides a schedule named "default", adopting callback `TSEvaluationTimesDefaultHandler()`,
  and evenly spaced evaluation time points. The option is activated via `TSSetFromOptions()`.

  After calling `TSEvaluationTimesAddArray()` or `TSEvaluationTimesAddUniform()` one or multiple times,
  the internal data structures need to be set up by `TSEvaluationTimesSetUp()` before running the time stepper.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesSchedule`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesDefaultHandler()`, `TSEvaluationTimesSetUp()`, `TSEvaluationTimesGetSolutions()`
@*/
PetscErrorCode TSEvaluationTimesAddUniform(TS ts, const char *name, PetscInt nprivate, PetscReal min, PetscReal max, PetscErrorCode (*handler)(TS ts, PetscInt iunion, PetscInt iprivate, PetscReal t, Vec full, Vec *sub, void *ctx), void *ctx)
{
  TSEvaluationTimesSchedule ets;

  PetscFunctionBegin;
  PetscCheck(nprivate != 1 || min == max, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "For a single-point range, min == max must be used, currently min = %g, max = %g", (double)min, (double)max);
  PetscCheck(nprivate <= 1 || min < max, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "For a multi-point (n = %" PetscInt_FMT ") range we need min < max, currently min = %g, max = %g", nprivate, (double)min, (double)max);

  PetscCall(TSEvaluationTimes_Add(ts, name, nprivate, handler, ctx, &ets)); // construction/destruction may take place here
  if (ets) {
    PetscReal       x    = min;
    const PetscReal step = (nprivate == 1 ? 0 : (max - min) / (nprivate - 1));

    PetscAssert(nprivate > 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "nprivate out of range");
    for (PetscInt i = 0; i < nprivate - 1; i++, x += step) ets->times[i] = x;
    ets->times[nprivate - 1] = max;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSEvaluationTimesDefaultHandler - A callback that creates the copies of the full solution vectors

  Logically Collective

  Input Parameters:
+ ts       - the `TS` context
. iunion   - index of current point in the union of all evaluation time points
. iprivate - index of current point within the given `TSEvaluationTimesSchedule`
. t        - current time
. full     - current solution vector
. sub      - the output vector, should be created by user, or left as `NULL` (default)
- ctx      - user-provided context

  Level: intermediate

  Notes:
  This function is not intended to be called by user. Rather, use it as a callback in
  `TSEvaluationTimesAddArray()` or `TSEvaluationTimesAddUniform()`.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesSchedule`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesAddUniform()`, `TSEvaluationTimesSetUp()`
@*/
PetscErrorCode TSEvaluationTimesDefaultHandler(TS ts, PetscInt iunion, PetscInt iprivate, PetscReal t, Vec full, Vec *sub, void *ctx)
{
  PetscFunctionBegin;
  (void)ts;
  (void)iunion;
  (void)iprivate;
  (void)t;
  (void)ctx;
  PetscCall(VecDuplicate(full, sub));
  PetscCall(VecCopy(full, *sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscClangLinter pragma disable: -fdoc-section-header-maybe-header
/*@
  TSEvaluationTimesSetUp - Set up the data structures after adding TSEvaluationTimesSchedule's

  Logically Collective

  Input Parameters:
+ ts               - time integration context
- override_t0_tmax - flag signalling to override the initial/end time in `TS` by the min/max of the evaluation time points

  Level: intermediate

  Notes:
  After calling `TSEvaluationTimesAddArray()` or `TSEvaluationTimesAddUniform()` one or multiple times,
  the internal data structures should be set up by calling this function.
  This makes the `TSEvaluationTimes` object ready for work in `TSSolve()`.

  The evaluation time points in each individual `TSEvaluationTimesSchedule` should be sorted in increasing order.
  To engage `TSEvaluationTimes` in `TSSolve()`, the user must also set `TS_EXACTFINALTIME_MATCHSTEP` for the `TS`.
  It is recommended that any of the following calls and definitions take place before the `TSEvaluationTimesSetUp()` call
.vb
  - Defining the initial solution time (t0),
  - Defining the maximum time (tmax),
  - Defining dt_min_rel, dt_min_abs for TSAdapt,
  - Calling TSSetFromOptions().
.ve
  This rule, if violated, does not result in an error. But if followed, it allows a more correct merging of the
  closely spaced evaluation time points near t0 and tmax, and between. Hence, immediately before `TSSolve()`, the following
  sequence of function calls makes sense\:
.vb
  ...
  TSSetFromOptions();
  TSEvaluationTimesSetUp();
  TSSolve();
  ...
.ve

  If one calls `TSEvaluationTimesSetUp()` with `override_t0_tmax` set to `PETSC_TRUE`, the initial time and maximum time for `TS`
  will be taken from the min/max of the evaluation time points (at least two evaluation time points should be defined).

  Note, calling this function destroys the vectors previously saved in `TSEvaluationTimes`.
  The internal counters are also reset.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesAddUniform()`
@*/
PetscErrorCode TSEvaluationTimesSetUp(TS ts, PetscBool override_t0_tmax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (ts->evaltimes) {
    PetscContainer            cont;
    TSEvaluationTimesSchedule ets;
    PetscBool                 sorted, finished = PETSC_FALSE;
    PetscInt                  i = 0, count = 0;     // indexing of ets's
    PetscInt                  cglob = 0;            // global (union) indices
    PetscInt                  ct0 = -1, ctmax = -1; // 'cglob' corresponding to t0, tmax
    PetscReal                 t0, tmax, last;
    PetscReal                 globmin = PETSC_MAX_REAL, globmax = PETSC_MIN_REAL;
    PetscInt                  **arrinds, *cur, *len; // size = count
    PetscReal                 **arrtimes;            // size = count
    PetscReal                 rtol = 10 * PETSC_MACHINE_EPSILON;    // rel. tolerance for merging the points
    PetscReal                 atol = PetscSqrtReal(PETSC_REAL_MIN); // abs. tolerance for merging the points

    // I. Make preliminary checks, count the ets's, fill the fast-access arrays
    for (PetscObjectList it = ts->evaltimes->schedlist; it; it = it->next) {
      cont = (PetscContainer)it->obj;
      PetscCall(PetscContainerGetPointer(cont, (void **)&ets));
      PetscCall(PetscSortedReal(ets->len, ets->times, &sorted));
      PetscCheck(sorted, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "Time points not sorted in TSEvaluationTimesSchedule '%.100s'", ets->name);
      PetscCheck(!ets->c_locked, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "TSEvaluationTimesSchedule '%.100s' is locked for modification; missing TSEvaluationTimesRestoreSolutions()?", ets->name);
      count++;
    }
    PetscCall(PetscCalloc1(count, &arrinds));  // helper array for fast access, indexing: arrinds[j][cur[j]], 0 <= j < count
    PetscCall(PetscCalloc1(count, &arrtimes)); // helper array for fast access, indexing: arrtimes[j][cur[j]]
    PetscCall(PetscCalloc1(count, &cur));      // current index in each 'ets', 0 <= cur[j] < len[j]
    PetscCall(PetscCalloc1(count, &len));      // len in each 'ets'
    for (PetscObjectList it = ts->evaltimes->schedlist; it; it = it->next) {
      cont = (PetscContainer)it->obj;
      PetscCall(PetscContainerGetPointer(cont, (void **)&ets));
      arrinds[i]  = ets->inds_global;
      arrtimes[i] = ets->times;
      len[i]      = ets->len;
      if (ets->len > 0 && ets->times[0] < globmin) globmin = ets->times[0];
      if (ets->len > 0 && ets->times[ets->len - 1] > globmax) globmax = ets->times[ets->len - 1];
      i++;
    }
    PetscAssert(i == count, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Bad counter value");
    last = globmin - PetscMax(1.0, PetscAbsReal(globmin) * 0.1); // a safe starting value for 'last'

    if (override_t0_tmax) {
      PetscCheck(count > 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Cannot override t0 and tmax using empty TSEvaluationTimes");
      PetscCheck(globmin < globmax, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Cannot override t0 and tmax, need two distinct evaluation time points");
      PetscCall(TSSetTime(ts, globmin));
      PetscCall(TSSetMaxTime(ts, globmax));
    }
    PetscCall(TSGetTime(ts, &t0));      // get the initial time
    PetscCall(TSGetMaxTime(ts, &tmax)); // get the final time
    if (ts->adapt) PetscCall(TSAdaptGetMinStep(ts->adapt, &rtol, &atol));

    // II. Fill the ets->inds_global[] (i.e. arrinds), estimate 'len_global' (i.e. the final cglob)
    while (!finished) { // each iteration: add one point from one of ets's
      PetscInt  minind = -1;
      PetscReal minval = PETSC_MAX_REAL;
      for (PetscInt j = 0; j < count; j++) {
        if (cur[j] < len[j] && arrtimes[j][cur[j]] < minval) {
          minind = j;
          minval = arrtimes[j][cur[j]];
        }
      }
      // now 'minind' is the index of 'ets' whose point 'minval' we take
      if (minind == -1) finished = PETSC_TRUE;
      else { // transform 'minval' to avoid very small intervals
        PetscBool update_ct0 = PETSC_FALSE, update_ctmax = PETSC_FALSE;
        if (PetscIsCloseAtTol(minval, t0, rtol, atol)) { // slightly shift if too close to 't0'
          update_ct0 = PETSC_TRUE;
          minval     = t0;
        } else if (PetscIsCloseAtTol(minval, tmax, rtol, atol)) { // otherwise slightly shift if too close to 'tmax'
          update_ctmax = PETSC_TRUE;
          minval       = tmax;
        }
        if (PetscIsCloseAtTol(minval, last, rtol, atol)) { // merge with the last added point if too close to it
          cglob--;
          minval = last;
        }
        PetscAssert(cglob >= 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Bad index value");
        if (update_ct0) ct0 = cglob;
        if (update_ctmax) ctmax = cglob;
        arrinds[minind][cur[minind]] = cglob;
        cur[minind]++;
        last = minval;
        cglob++;
      }
    }

    // III. Fill 'times_global', clean up
    ts->evaltimes->len_global = cglob;
    PetscCall(PetscFree(ts->evaltimes->times_global));
    PetscCall(PetscCalloc1(ts->evaltimes->len_global, &ts->evaltimes->times_global));
    for (PetscInt j = 0; j < ts->evaltimes->len_global; j++) ts->evaltimes->times_global[j] = PETSC_MAX_REAL;
    for (PetscInt j = 0; j < count; j++) {    // over all ets's
      for (PetscInt k = 0; k < len[j]; k++) { // inside each ets
        const PetscInt G = arrinds[j][k];
        PetscAssert(0 <= G && G < ts->evaltimes->len_global, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Bad global index");
        if (arrtimes[j][k] < ts->evaltimes->times_global[G]) ts->evaltimes->times_global[G] = arrtimes[j][k]; // find min over {j,k} corresponding to G
      }
    }
    if (ct0 != -1) ts->evaltimes->times_global[ct0] = t0;       // if necessary, t0 and tmax are explicitly added to 'times_global',
    if (ctmax != -1) ts->evaltimes->times_global[ctmax] = tmax; //    also note, t0 and tmax may not be present within arrtimes[j][k]

    if (PetscDefined(USE_DEBUG))
      for (PetscInt j = 0; j < ts->evaltimes->len_global; j++) { PetscAssert(ts->evaltimes->times_global[j] != PETSC_MAX_REAL, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "'times_global' not filled properly"); }

    ts->evaltimes->assembled = PETSC_TRUE;
    PetscCall(TSEvaluationTimesReset(ts));
    PetscCall(PetscFree(arrinds));
    PetscCall(PetscFree(arrtimes));
    PetscCall(PetscFree(cur));
    PetscCall(PetscFree(len));
  } else PetscCheck(!override_t0_tmax, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Cannot override t0 and tmax using empty TSEvaluationTimes");

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TSEvaluationTimesSetDefaultSchedule - helper function to add/override a schedule named "default", setting callback TSEvaluationTimesDefaultHandler(),
  either for array 'schedtimes' (if it is != NULL), or for a discretised range [min, max] (otherwise).
*/
PetscErrorCode TSEvaluationTimesSetDefaultSchedule(TS ts, PetscInt nprivate, PetscReal min, PetscReal max, const PetscReal *schedtimes, PetscBool override_t0_tmax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscCheck(nprivate >= 1, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "Number of evaluation time points >= 1 is required, provided: %" PetscInt_FMT, nprivate);
  if (schedtimes) PetscAssertPointer(schedtimes, 5);

  if (schedtimes) PetscCall(TSEvaluationTimesAddArray(ts, "default", nprivate, schedtimes, TSEvaluationTimesDefaultHandler, NULL)); // if evaltimes is NULL, it will be constructed
  else PetscCall(TSEvaluationTimesAddUniform(ts, "default", nprivate, min, max, TSEvaluationTimesDefaultHandler, NULL));            // if evaltimes is NULL, it will be constructed
  PetscCall(TSEvaluationTimesSetUp(ts, override_t0_tmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSetTimeSpan - Deprecated, use `TSEvaluationTimesAddArray()`

  Logically Collective

  Input Parameters:
+ ts          - time integration context
. n           - number of the time points (>=2)
- time_points - sorted array of the time points

  Options Database Key:
. -ts_time_span t1,...tn - sets the evaluation times, overrides init_time, max_time

  Level: deprecated

  Notes:
  This function is deprecated. It sets the evaluation time points, where the solution vectors will be saved to an array.

  Under the hood, this function adds/overrides a schedule named "default", with points `time_points`,
  and callback `TSEvaluationTimesDefaultHandler()`. The initial and final `TS` times init_time, max_time
  are set respectively to the first and the last entries of `time_points`.
  The runtime option `-ts_time_span` has the same effect.

  Array `time_points` should be sorted in increasing order, and `TS_EXACTFINALTIME_MATCHSTEP` must be set for the `TS`.

  Instead of `TSSetTimeSpan()` it is recommended to use `TSEvaluationTimesAddArray()` and the associated functions.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesAddUniform()`, `TSEvaluationTimesSetUp()`, `TSGetEvaluationTimes()`, `TSGetEvaluationSolutions()`
@*/
PetscErrorCode TSSetTimeSpan(TS ts, PetscInt n, PetscReal *time_points)
{
  PetscFunctionBegin;
  PetscCall(TSEvaluationTimesSetDefaultSchedule(ts, n, 0.0, 0.0, time_points, PETSC_TRUE)); // override t0, tmax
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSetEvaluationTimes - Deprecated, use `TSEvaluationTimesAddArray()`

  Logically Collective

  Input Parameters:
+ ts          - time integration context
. n           - number of the time points
- time_points - sorted array of the time points

  Options Database Key:
. -ts_eval_times t1,...,tn - sets the evaluation times

  Level: deprecated

  Notes:
  This function is deprecated. It sets the evaluation time points, where the solution vectors will be saved to an array.

  Under the hood, this function adds/overrides a schedule named "default", with points `time_points`,
  and callback `TSEvaluationTimesDefaultHandler()`.
  The runtime option `-ts_eval_times` has the same effect.

  Array `time_points` should be sorted in increasing order, and `TS_EXACTFINALTIME_MATCHSTEP` must be set for the `TS`.

  Instead of `TSSetEvaluationTimes()` it is recommended to use `TSEvaluationTimesAddArray()` and the associated functions.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesAddUniform()`, `TSEvaluationTimesSetUp()`, `TSSetTimeSpan()`, `TSGetEvaluationTimes()`, `TSGetEvaluationSolutions()`
@*/
PetscErrorCode TSSetEvaluationTimes(TS ts, PetscInt n, PetscReal *time_points)
{
  PetscFunctionBegin;
  PetscCall(TSEvaluationTimesSetDefaultSchedule(ts, n, 0.0, 0.0, time_points, PETSC_FALSE)); // don't override t0, tmax
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimes_GetScheduleNoLock - function to get data, without locking (which MAY BE UNSAFE!)
static PetscErrorCode TSEvaluationTimes_GetScheduleNoLock(TS ts, const char *name, PetscInt *nprivate, PetscInt *idxstart, PetscInt *idxend, PetscReal **schedtimes, Vec **sols, TSEvaluationTimesSchedule *pets)
{
  PetscContainer cont = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  if (nprivate) PetscAssertPointer(nprivate, 3);
  if (idxstart) PetscAssertPointer(idxstart, 4);
  if (idxend) PetscAssertPointer(idxend, 5);
  if (schedtimes) PetscAssertPointer(schedtimes, 6);
  if (sols) PetscAssertPointer(sols, 7);
  if (pets) PetscAssertPointer(pets, 8);

  if (ts->evaltimes) PetscCall(PetscObjectListFind(ts->evaltimes->schedlist, name, (PetscObject *)&cont));
  if (cont) { // found the evaltimes->schedule: return its data
    TSEvaluationTimesSchedule ets = NULL;
    PetscCall(PetscContainerGetPointer(cont, (void **)&ets));
    PetscCheck(ets, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Failure with 'ets'");

    if (nprivate) *nprivate = ets->len;
    if (idxstart) *idxstart = ets->start;
    if (idxend) *idxend = ets->end;
    if (schedtimes) *schedtimes = ets->times;
    if (sols) *sols = ets->vecs;
    if (pets) *pets = ets;
  } else { // not found the evaltimes->schedule: return "zeros"
    if (nprivate) *nprivate = 0;
    if (idxstart) *idxstart = -1;
    if (idxend) *idxend = -1;
    if (schedtimes) *schedtimes = NULL;
    if (sols) *sols = NULL;
    if (pets) *pets = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSEvaluationTimesGetSolutions - Get the vectors (and other data) saved by TSEvaluationTimesSchedule

  Logically Collective

  Input Parameters:
+ ts   - time integration context
- name - reference name for the `TSEvaluationTimesSchedule`, as defined in `TSEvaluationTimesAddArray()`

  Output Parameters:
+ nprivate   - total number of time points for this TSEvaluationTimesSchedule (0, if the schedule is not found)
. idxstart   - the start index in [0, nprivate) for the points visited
. idxend     - the end index in [0, nprivate) for the points visited
. schedtimes - the time points originally set for this TSEvaluationTimesSchedule, array length = nprivate
- sols       - the vectors saved at evaluation time points during the TSSolve(s), array length = nprivate

  Level: intermediate

  Notes:
  The data returned by this function is stored internally in `TS`. After use, it MUST be restored by a matching call of
  `TSEvaluationTimesRestoreSolutions()` with the same arguments. Until the restore call, any modification or deletion
  of the given `TSEvaluationTimesSchedule` will be prohibited. However, simultaneously accessing the data
  of different TSEvaluationTimesSchedule's is fine.

  This function can also be used to check if schedule `name` is present in `TSEvaluationTimes` collection.
  If the schedule exists, `nprivate`, `schedtimes`, `sols` get non-zero values (the data requested).
  If the schedule is absent, `nprivate`, `schedtimes`, `sols` get zeros; `idxstart` and `idxend` get -1.

  Any of the output argument pointers may be `NULL` if no output is required for them.

  The [idxstart, idxend) is the sub-range in [0, nprivate), these are the indices of the evaluation time points
  actually visited during the recent `TSSolve()` run(s). Note, `idxend` is the past-the-end index.
  If the time stepper starts too late in time, or terminates too early, it may not visit all the evaluation time points provided.
  If no evaluation time points were visited at all, idxstart = -1, idxend = -1.

  Even if the `TSEvaluationTimesSchedule` was added by `TSEvaluationTimesAddUniform()`, the output `schedtimes` returns
  a full-fledged array corresponding to the range provided originally.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesSchedule`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesAddUniform()`, `TSEvaluationTimesRestoreSolutions()`
@*/
PetscErrorCode TSEvaluationTimesGetSolutions(TS ts, const char *name, PetscInt *nprivate, PetscInt *idxstart, PetscInt *idxend, PetscReal **schedtimes, Vec **sols)
{
  TSEvaluationTimesSchedule ets = NULL;

  PetscFunctionBegin;
  PetscCall(TSEvaluationTimes_GetScheduleNoLock(ts, name, nprivate, idxstart, idxend, schedtimes, sols, &ets));
  if (ets) { // if ets is found, save cache and lock it
    PetscCheck(!ets->c_locked, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "TSEvaluationTimesRestoreSolutions() should be called before a new call to TSEvaluationTimesGetSolutions() for schedule '%.100s'", ets->name);

    ets->c_locked = PETSC_TRUE;
    if (schedtimes) ets->c_times = ets->times; // cache the pointer for check at restore
    else ets->c_times = NULL;
    if (sols) {
      PetscCall(TSEvaluationTimesSchedule_LockVecs(ets));
      ets->c_vecs = ets->vecs;
    } else ets->c_vecs = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSEvaluationTimesRestoreSolutions - Restores the vectors (and other data) after `TSEvaluationTimesGetSolutions()`

  Logically Collective

  Input Parameters:
+ ts   - time integration context
- name - reference name for the `TSEvaluationTimesSchedule`, as defined in `TSEvaluationTimesAddArray()`

  Output Parameters:
+ nprivate   - total number of time points for this TSEvaluationTimesSchedule
. idxstart   - the start index in [0, nprivate) for the points visited
. idxend     - the end index in [0, nprivate) for the points visited
. schedtimes - the time points originally set for this TSEvaluationTimesSchedule, array length = nprivate
- sols       - the vectors saved at evaluation time points during the TSSolve(s), array length = nprivate

  Level: intermediate

  Notes:
  `TSEvaluationTimesRestoreSolutions()` restores the data queried by `TSEvaluationTimesGetSolutions()`,
  so that the underlying `TSEvaluationTimesSchedule` object becomes unlocked for further modifications.
  The arguments passed to this function must match those passed to `TSEvaluationTimesGetSolutions()`.

  If schedule `name` is not present in `TSEvaluationTimes` collection, calling this function is not necessary.
  In this case calling and not calling are equally correct.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesSchedule`, `TSEvaluationTimesAddArray()`, `TSEvaluationTimesAddUniform()`, `TSEvaluationTimesGetSolutions()`
@*/
PetscErrorCode TSEvaluationTimesRestoreSolutions(TS ts, const char *name, PetscInt *nprivate, PetscInt *idxstart, PetscInt *idxend, PetscReal **schedtimes, Vec **sols)
{
  PetscContainer cont = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(name, 2);
  if (nprivate) PetscAssertPointer(nprivate, 3);
  if (idxstart) PetscAssertPointer(idxstart, 4);
  if (idxend) PetscAssertPointer(idxend, 5);
  if (schedtimes) PetscAssertPointer(schedtimes, 6);
  if (sols) PetscAssertPointer(sols, 7);

  if (ts->evaltimes) PetscCall(PetscObjectListFind(ts->evaltimes->schedlist, name, (PetscObject *)&cont));
  if (cont) { // if the evaltimes->schedule exists, unlock it
    TSEvaluationTimesSchedule ets = NULL;

    PetscCall(PetscContainerGetPointer(cont, (void **)&ets));
    PetscCheck(ets, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Failure with 'ets'");
    PetscCheck(ets->c_locked, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "TSEvaluationTimesGetSolutions() should be called before trying to TSEvaluationTimesRestoreSolutions() for schedule '%.100s'", ets->name);

    PetscCheck(!schedtimes || *schedtimes == ets->c_times, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Non-matching argument 'schedtimes' between get/restore calls for schedule '%.100s'", ets->name);
    PetscCheck(schedtimes || NULL == ets->c_times, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Non-matching argument 'schedtimes' between get/restore calls for schedule '%.100s'", ets->name);
    PetscCheck(!sols || *sols == ets->c_vecs, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Non-matching argument 'sols' between get/restore calls for schedule '%.100s'", ets->name);
    PetscCheck(sols || NULL == ets->c_vecs, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Non-matching argument 'sols' between get/restore calls for schedule '%.100s'", ets->name);

    if (nprivate) *nprivate = 0;
    if (idxstart) *idxstart = -1;
    if (idxend) *idxend = -1;
    if (schedtimes) *schedtimes = ets->c_times = NULL;
    if (sols) {
      PetscCall(TSEvaluationTimesSchedule_UnlockVecs(ets));
      *sols = ets->c_vecs = NULL;
    }
    ets->c_locked = PETSC_FALSE;
  } else { // evaltimes->schedule does not exist -> make some extra checks to enforce proper use of get/restore
    PetscCheck(!nprivate || *nprivate == 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Schedule '%.100s' is not found -> the restore call expects *nprivate == 0", name);
    PetscCheck(!idxstart || *idxstart == -1, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Schedule '%.100s' is not found -> the restore call expects *idxstart == -1", name);
    PetscCheck(!idxend || *idxend == -1, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Schedule '%.100s' is not found -> the restore call expects *idxend == -1", name);
    PetscCheck(!schedtimes || *schedtimes == NULL, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Schedule '%.100s' is not found -> the restore call expects *schedtimes == NULL", name);
    PetscCheck(!sols || *sols == NULL, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Schedule '%.100s' is not found -> the restore call expects *sols == NULL", name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSGetTimeSpan - Deprecated, use `TSEvaluationTimesGetSolutions()`

  Not Collective

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameters:
+ n           - number of the time points
- time_points - array of the time points

  Level: deprecated

  Note:
  This function is deprecated (and may be unsafe, as it accesses schedule "default" without locking), use `TSEvaluationTimesGetSolutions()`.

  It returns the array of time span points (evaluation times), and its length, set via `TSSetTimeSpan()` or `TSSetEvaluationTimes()`,
  or via options `-ts_time_span`, `-ts_eval_times`, `-ts_eval_times_uniform`.
  If nothing was set, zero length is returned. Both `n` and `time_points` can be `NULL`.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesGetSolutions()`, `TSEvaluationTimesRestoreSolutions()`, `TSSetTimeSpan()`, `TSSetEvaluationTimes()`
@*/
PetscErrorCode TSGetTimeSpan(TS ts, PetscInt *n, const PetscReal **time_points)
{
  PetscFunctionBegin;
  PetscCall(TSEvaluationTimes_GetScheduleNoLock(ts, "default", n, NULL, NULL, (PetscReal **)time_points, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSGetEvaluationTimes - Deprecated, use `TSEvaluationTimesGetSolutions()`

  Not Collective

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameters:
+ n           - number of the time points
- time_points - array of the time points

  Level: deprecated

  Note:
  This function is deprecated (and may be unsafe, as it accesses schedule "default" without locking), use `TSEvaluationTimesGetSolutions()`.

  This function returns the array of evaluation times, and its length, set via `TSSetEvaluationTimes()` or `TSSetTimeSpan()`,
  or via options `-ts_time_span`, `-ts_eval_times`, `-ts_eval_times_uniform`.
  If nothing was set, zero length is returned. Both `n` and `time_points` can be `NULL`.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesGetSolutions()`, `TSEvaluationTimesRestoreSolutions()`, `TSSetTimeSpan()`, `TSSetEvaluationTimes()`
 @*/
PetscErrorCode TSGetEvaluationTimes(TS ts, PetscInt *n, const PetscReal **time_points)
{
  PetscFunctionBegin;
  PetscCall(TSEvaluationTimes_GetScheduleNoLock(ts, "default", n, NULL, NULL, (PetscReal **)time_points, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSGetTimeSpanSolutions - Deprecated, use `TSEvaluationTimesGetSolutions()`

  Not Collective

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameters:
+ nsol - number of solutions
- sols - array of solution vectors

  Level: deprecated

  Note:
  This function is deprecated (and may be unsafe, as it accesses schedule "default" without locking), use `TSEvaluationTimesGetSolutions()`.

  It returns the number of solutions and the array of solution vectors saved at the time span points (evaluation times).
  The time span points are those defined by 'TSSetTimeSpan()' or `TSSetEvaluationTimes()`,
  or by options `-ts_time_span`, `-ts_eval_times`, `-ts_eval_times_uniform`.
  If no points are defined, zero array length is returned. Both `nsol` and `sols` can be `NULL`.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesGetSolutions()`, `TSEvaluationTimesRestoreSolutions()`, `TSSetTimeSpan()`, `TSGetEvaluationSolutions()`
@*/
PetscErrorCode TSGetTimeSpanSolutions(TS ts, PetscInt *nsol, Vec **sols)
{
  PetscInt start = -1, end;

  PetscFunctionBegin;
  if (nsol) PetscAssertPointer(nsol, 2);
  if (sols) PetscAssertPointer(sols, 3);
  PetscCall(TSEvaluationTimes_GetScheduleNoLock(ts, "default", NULL, &start, &end, NULL, sols, NULL));
  if (start != -1) {
    if (nsol) *nsol = end - start;
    if (sols) *sols = *sols + start; // the resulting *sols stores the recorded solutions, corresponding to the original indices [start, end)
  } else {
    if (nsol) *nsol = 0;
    if (sols) *sols = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSGetEvaluationSolutions - Deprecated, use `TSEvaluationTimesGetSolutions()`

  Not Collective

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameters:
+ nsol      - number of solutions (length of sol_times and sols)
. sol_times - array of solution times corresponding to the solution vectors
- sols      - array of solution vectors

  Level: deprecated

  Note:
  This function is deprecated (and may be unsafe, as it accesses schedule "default" without locking), use `TSEvaluationTimesGetSolutions()`.

  It returns the number of solutions, the array of solution times, and the array of solution vectors saved at the evaluation times.
  The simulation may finish before the end of the evaluation times, or start after their beginning, in which case the solutions
  will only be saved at a subset (`sol_times`) of the original evaluation times.
  The evaluation times are those defined by `TSSetEvaluationTimes()` or 'TSSetTimeSpan()',
  or by options `-ts_time_span`, `-ts_eval_times`, `-ts_eval_times_uniform`.
  If no points are defined, zero array length is returned. Any of `nsol`, `sol_times`, or `sols` can be `NULL`.

.seealso: [](ch_ts), `TS`, `TSEvaluationTimes`, `TSEvaluationTimesGetSolutions()`, `TSEvaluationTimesRestoreSolutions()`, `TSSetEvaluationTimes()`, `TSGetEvaluationTimes()`
@*/
PetscErrorCode TSGetEvaluationSolutions(TS ts, PetscInt *nsol, const PetscReal **sol_times, Vec **sols)
{
  PetscInt start = -1, end;

  PetscFunctionBegin;
  if (nsol) PetscAssertPointer(nsol, 2);
  if (sol_times) PetscAssertPointer(sol_times, 3);
  if (sols) PetscAssertPointer(sols, 4);
  PetscCall(TSEvaluationTimes_GetScheduleNoLock(ts, "default", NULL, &start, &end, (PetscReal **)sol_times, sols, NULL));
  if (start != -1) {
    if (nsol) *nsol = end - start;
    if (sol_times) *sol_times = *sol_times + start;
    if (sols) *sols = *sols + start; // the resulting *sols stores the recorded solutions, corresponding to the original indices [start, end)
  } else {
    if (nsol) *nsol = 0;
    if (sol_times) *sol_times = NULL;
    if (sols) *sols = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TSEvaluationTimes_Hit - detects if an evaluation time point has been hit ('hit state' condition); the function doesn't change the state.
  The detection of the evaluation time points is designed to fully avoid any use of tolerances (except during the initial point-merging on setup).
  The design relies on monitoring the current 'state', and making the time stepper change the state in a predictable way.
  Now use the following notation for further explanation of the concepts.
  t   - current time,
  c   - current ctr_global,
  e_c - current evaluation time point,
  N   - number of points (len_global).
  The three guys {t, c, e_c} describe the current state, which can be either 'clean' or 'hit'.
  >> "clean state" ("looking for the next point") means that condition 1, or 2, or 3 holds.
    1. t < e_c, and c is the minimum possible index to satisfy this condition.
    2. t >= e_i for all i, and c == N.
    3. evaltimes == NULL.
  >> "hit state" ("just hit a point") means that e_c <= t. In this case the design also ensures |t - e_c| ~ machine epsilon,
    but this near-equality is not checked explicitly in the code.
  The allowed state transitions are as follows.
  * hit -> clean, done by increasing counter c, and saving the vectors, via TSEvaluationTimesSaveVecs().      | {t - fixed, c - changes}
    Due to the possible floating point arithmetic errors, this may require several c increments, not just one.|
  * clean -> hit, done by making a TS step to the next evaluation time point. The relevant step size is       | {t - changes, c - fixed}
    found by TSEvaluationTimesNext(), and a "no-undershoot guarantee" is provided by Delta_with_overshoot().  |
  * clean -> clean, done by making a TS step not reaching an evaluation time point.                           | {t - changes, c - fixed}
  In the beginning of TSSolve, TSEvaluationTimesSetUpCounters() sets a state which is either 'clean', or 'hit'-with-strict-equality.
  During transition between different TSSolve's, the final 'clean state' from solve-1 becomes a 'clean state' for solve-2.
  Point ts->max_time is processed like other e_c's, but it doesn't save any vectors, and doesn't set the 'hit state'.
*/
static PetscBool TSEvaluationTimes_Hit(TS ts, PetscReal t)
{
  if (!ts->evaltimes) return PETSC_FALSE;
  else {
    PetscReal *e = ts->evaltimes->times_global;
    PetscInt   N = ts->evaltimes->len_global;
    PetscInt   c = ts->evaltimes->ctr_global;

    if (c < N) return (e[c] <= t) ? PETSC_TRUE : PETSC_FALSE; // when e[c] <= t, the overall design tacitly assumes e[c] ~= t
    else return PETSC_FALSE;
  }
}

/*
  TSEvaluationTimesSchedule_SaveVec - takes 'x' through 'handler', saves the result to 'vecs', updates the counter.
  'ind_glob' is the global index of the current time point.
  The current time point is skipped for this 'ets' if 'ind_glob' is not in ets->inds_global.
*/
static PetscErrorCode TSEvaluationTimesSchedule_SaveVec(TS ts, TSEvaluationTimesSchedule ets, Vec x, PetscInt ind_glob)
{
  PetscReal t;
  Vec       xout = NULL; // note, if 'xout' is to be meaningfully defined, it must be created by user in ets->handler(), otherwise it stays NULL

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(ets, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscCheck(!ets->c_locked, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "TSEvaluationTimesSchedule '%.100s' is locked for modification; missing TSEvaluationTimesRestoreSolutions()?", ets->name);
  PetscAssert(ets->ctr >= 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Counter 'ctr' out of range");
  PetscCheck(ets->ctr >= ets->len || ind_glob <= ets->inds_global[ets->ctr], PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Unexpected counter / index");

  while (ets->ctr < ets->len && ind_glob == ets->inds_global[ets->ctr]) { // process the repeating 'inds_global', if any
    PetscCall(TSGetTime(ts, &t));
    PetscCallBack("Processing the vector before saving in TSEvaluationTimes", (*ets->handler)(ts, ind_glob, ets->ctr, t, x, &xout, ets->ctx));
    if (ets->vecs[ets->ctr]) PetscCall(VecDestroy(&ets->vecs[ets->ctr]));
    ets->vecs[ets->ctr] = xout; // note, destruction of this Vec will be done automatically (not by user)

    if (ets->start == -1) ets->start = ets->ctr;
    else ets->start = PetscMin(ets->start, ets->ctr);
    ets->end = PetscMax(ets->end, ets->ctr + 1);
    ets->ctr++;
  }
  // if (ets->ctr >= ets->len) : finished with the given TSEvaluationTimesSchedule, return.
  // if (ind_glob < ets->inds_global[ets->ctr]) : the TSEvaluationTimesSchedule now expects a more posterior point, return.
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TSEvaluationTimes_SaveVec - transform/save 'x' in all schedules where applicable, update the (global) counter 'c'.
  The 'hit state' should be on, i.e. e_c <= t, e_c ~= t.
*/
static PetscErrorCode TSEvaluationTimes_SaveVec(TS ts, Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  if (ts->evaltimes) {
    PetscAssert(0 <= ts->evaltimes->ctr_global && ts->evaltimes->ctr_global < ts->evaltimes->len_global, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Counter out of range");
    PetscCheck(ts->evaltimes->assembled, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "Need to set up TSEvaluationTimes first; use TSEvaluationTimesSetUp()");

    for (PetscObjectList it = ts->evaltimes->schedlist; it; it = it->next) {
      PetscContainer            cont;
      TSEvaluationTimesSchedule ets;

      cont = (PetscContainer)it->obj;
      PetscCall(PetscContainerGetPointer(cont, (void **)&ets));
      PetscCall(TSEvaluationTimesSchedule_SaveVec(ts, ets, x, ts->evaltimes->ctr_global));
    }
    ts->evaltimes->ctr_global++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TSEvaluationTimesSaveVecs - saves the vectors while the 'hit state' is on.
  The function exits in a 'clean state' (see TSEvaluationTimes_Hit for terminology).
  Vector x and time t are fixed.
*/
PetscErrorCode TSEvaluationTimesSaveVecs(TS ts, Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);

  if (ts->evaltimes) {
    PetscCheck(ts->evaltimes->assembled, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "Need to set up TSEvaluationTimes first; use TSEvaluationTimesSetUp()");
    while (TSEvaluationTimes_Hit(ts, ts->ptime)) PetscCall(TSEvaluationTimes_SaveVec(ts, x)); // inside: evaltimes->ctr_global++, so the 'hit state' will eventually change to 'clean state'
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSEvaluationTimesSchedule_SetUpCounter - set ets->ctr according to ctr_global
static PetscErrorCode TSEvaluationTimesSchedule_SetUpCounter(TSEvaluationTimesSchedule ets, PetscInt ctr_global)
{
  PetscInt loc;

  PetscFunctionBegin;
  PetscAssertPointer(ets, 1);
  PetscCall(PetscFindInt(ctr_global, ets->len, ets->inds_global, &loc));
  if (loc < 0) loc = -(loc + 1);
  if (loc < ets->len)
    while (loc > 0 && ets->inds_global[loc - 1] == ets->inds_global[loc]) loc--; // rewind a bit in case of repeating values
  ets->ctr = loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TSEvaluationTimesSetUpCounters - set the correct initial value for 'ctr_global', and the private counters.
  The function sets a 'clean state', or 'hit state' with strict equality.
  It should be called in the beginning of TSSolve().
*/
PetscErrorCode TSEvaluationTimesSetUpCounters(TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);

  if (ts->evaltimes) {
    PetscReal        t0;
    PetscInt         c          = ts->evaltimes->ctr_global;
    PetscInt         N          = ts->evaltimes->len_global;
    const PetscReal *e          = ts->evaltimes->times_global;
    PetscBool        fullsearch = PETSC_TRUE;

    PetscAssert(N > 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "'len_global' should be positive");
    PetscAssert(0 <= c && c <= N, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "'ctr_global' out of range");
    PetscCheck(ts->evaltimes->assembled, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "Need to set up TSEvaluationTimes first; use TSEvaluationTimesSetUp()");
    PetscCall(TSGetTime(ts, &t0));

    // 1. Set the global counter
    // 1.1 First check if seamless transition with the current 'c' is possible, {t0 - fixed, c - fixed}. This is done not only
    // to avoid calling the search routine, but also to preserve logic in passing the counters in a series of TSSolve's.
    if (c < N) {
      if (c == 0) fullsearch = (t0 <= e[0] ? PETSC_FALSE : PETSC_TRUE);                 // {t0 < e[0]}=clean, or {t0 == e[0]}=hit
      if (c > 0) fullsearch = (e[c - 1] <= t0 && t0 < e[c] ? PETSC_FALSE : PETSC_TRUE); // {e[c-1] <= t0 < e[c]}=clean
    } else fullsearch = (t0 >= e[N - 1] ? PETSC_FALSE : PETSC_TRUE);                    // {t0 >= e[N-1], and c == N}=clean

    // 1.2 Otherwise do a full search, {t0 - fixed, c - changes}.
    if (fullsearch) {
      PetscInt r = 0;

      PetscCall(PetscFindReal(t0, N, e, 0.0, &r));
      PetscAssert(r < 0, PetscObjectComm((PetscObject)ts), PETSC_ERR_PLIB, "Unexpected output from PetscFindReal()");
      c = -(r + 1); // the result is: {t0 == e[c]}=hit, or {e[c-1] < t0 < e[c]}=clean, or {t0 > e[i] for all i, and c == N}=clean
    }
    ts->evaltimes->ctr_global = c;

    // 2. Set the private counters
    for (PetscObjectList it = ts->evaltimes->schedlist; it; it = it->next) {
      TSEvaluationTimesSchedule ets;
      PetscContainer            cont;

      cont = (PetscContainer)it->obj;
      PetscCall(PetscContainerGetPointer(cont, (void **)&ets));
      PetscCall(TSEvaluationTimesSchedule_SetUpCounter(ets, ts->evaltimes->ctr_global));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Delta_with_overshoot - returns dt = t1 - t0, possibly with a "minimal" (machine epsilon) floating point correction
  towards +inf, ensuring t0 + dt >= t1, i.e. excluding undershoots in adding dt to t0.
  For instance, 0.1 + (0.45 - 0.1) < 0.45, but 0.1 + Delta_with_overshoot(0.1, 0.45) > 0.45 by machine epsilon.
  In this example, there is no floating point number 'dt' to exactly hit 0.45.

  For presicion == __fp16, this function returns the plain difference t1 - t0, since the current
  PetscNextafter implementation for __fp16 outputs the unchanged input.

  Note. The while-loop is safeguarded from infinite spinning with a counter.
  Normally the counter should not cause the loop to exit, however if it does, the "no-undershoot" guarantee may be void.
*/
static PetscReal Delta_with_overshoot(const PetscReal t0, const PetscReal t1)
{
  PetscReal       dt              = t1 - t0;
  PetscReal       t1_mutable      = t1;
  const PetscBool t1_dominates_dt = (PetscAbsReal(t1) >= PetscAbsReal(dt) ? PETSC_TRUE : PETSC_FALSE);
  PetscInt        count           = 0; // safeguard counter
  while (t0 + dt < t1 && count < 5) {
    if (t1_dominates_dt) {
      t1_mutable = PetscNextafter(t1_mutable, PETSC_MAX_REAL); // slightly increase t1_mutable to recalculate/increase dt
      dt = t1_mutable - t0;
    } else dt = PetscNextafter(dt, PETSC_MAX_REAL); // slightly increase dt
    count++;
  }
  return dt;
}

/*
  TSEvaluationTimesNext - finds the next point in the (global) evaluation times.
  t       - current point reached by TS.
  next_t  - [output] next point in the evaluation times (next_t > t),
                     or max_time, or PETSC_MAX_REAL.
  next_dt - [output] step size to reach 'next_t' with "no undershoot" guarantee, i.e.
                     ensuring t + next_dt >= next_t, where possible overshoot ~ machine epsilon.
*/
PetscErrorCode TSEvaluationTimesNext(TS ts, PetscReal t, PetscReal *next_t, PetscReal *next_dt)
{
  PetscReal o_next_t = PETSC_MAX_REAL; // next point
  PetscReal o_next_dt;                 // next step size

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  if (next_t) PetscAssertPointer(next_t, 3);
  if (next_dt) PetscAssertPointer(next_dt, 4);

  if (ts->evaltimes) {
    PetscReal *e = ts->evaltimes->times_global;
    PetscInt   N = ts->evaltimes->len_global;
    PetscInt   c = ts->evaltimes->ctr_global;
    PetscCheck(ts->evaltimes->assembled, PetscObjectComm((PetscObject)ts), PETSC_ERR_COR, "Need to set up TSEvaluationTimes first; use TSEvaluationTimesSetUp()");

    if (TSEvaluationTimes_Hit(ts, t)) o_next_t = (c + 1 < N ? e[c + 1] : PETSC_MAX_REAL); // hit state
    else o_next_t = (c < N ? e[c] : PETSC_MAX_REAL); // clean state
  }
  if (t < ts->max_time) o_next_t = PetscMin(o_next_t, ts->max_time);
  o_next_dt = (o_next_t < PETSC_MAX_REAL ? Delta_with_overshoot(t, o_next_t) : PETSC_MAX_REAL);

  if (next_t) *next_t = o_next_t;
  if (next_dt) *next_dt = o_next_dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}
