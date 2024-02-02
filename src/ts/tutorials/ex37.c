#include <petscts.h>

static char help[] = "Simple 3*3 linear problem demonstrating TSEvaluationTimes and TSEvent\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"
                     "z_dot =  1\n\n"

                     "The following evaluation times schedules are available:\n"
                     "1) Schedule 'user' (present by default) with points [0.01, 0.21, 1.01, ..., 6.21, 6.99, 7.21,... 10.01, 10.21]\n"
                     "                                   plus the points: {3, 4, 4+D, 5-D, 5, 6-D, 6, 6+D} with user-defined 'D'\n"
                     "2) Schedule 'default' can be added by PETSc options -ts_time_span, -ts_eval_times or -ts_eval_times_uniform\n"
                     "3) Schedule 'range' can be added by the application option -range, see below.\n"
                     "4) Schedules 'dummy1'...'dummy5' are temporarily added/removed for test purposes.\n"
                     "All these schedules use different callbacks, see the code for details.\n\n"

                     "The following event indicator functions are involved:\n"
                     "- two polynomial event functions on rank-0 and last-rank, with zeros: 1.05, 9.05, both terminating\n"
                     "- one event function on rank = '1%size', equal to sin(pi*t), zeros = 1,...,10,...\n"
                     "  on event at t==5, the solution is changed: z = -z\n\n"

                     "TSSolve() is called consecutively 4 times: [t0 .. 1.05] [1.05 .. 9.05] [9.05 .. ts_max_time] [ts_max_time .. tmax]\n"
                     "After the 1st TSSolve, the solution's x, y are multipled by 2.0\n"
                     "After the 2nd TSSolve, the TSEvaluationTimes is reset (or cleared, with option -clear)\n"
                     "After the 3rd TSSolve, schedule 'user' is deleted, and current time is shifted by -tshift\n\n"

                     "Application options:\n"
                     "-dir     d : zero-crossing direction for events: 0, 1, -1 (default = 0)\n"
                     "-flg       : additional output in Postevent (default = nothing)\n"
                     "-errtol  e : error tolerance, for printing 'pass/fail' for located events etc. (default = 1e-4)\n"
                     "-errtol2 t : error tolerance for checking the solution norm2 accuracy (default = 0.03)\n"
                     "-nouser    : disables adding the schedule 'user'\n"
                     "-clear     : calls TSEvaluationTimesDestroy() instead of TSEvaluationTimesReset() after the 2nd TSSolve()\n"
                     "-D       z : a (small) real number to define the additional time points for schedule 'user' (default = 0.02)\n"
                     "-range   T : with T = min,max,N defining N evaluation time points in [min, max]\n"
                     "-tshift  s : current time += s after 3rd TSSolve() (default = 0.0)\n"
                     "-tmax    M : max time for the 4th TSSolve() (default = 12.0)\n"
                     "-msg     m : an optional final message to print for documenting the tests\n";

#define MAX_NFUNC 5   // max event functions per rank
#define MAX_NEV   500 // max zero crossings for each rank

typedef struct {
  PetscMPIInt rank, size;
  PetscReal   pi;
  PetscReal   fvals[MAX_NFUNC]; // helper array for reporting the residuals
  PetscReal   evres[MAX_NEV];   // times of found zero-crossings
  PetscReal   ref[MAX_NEV];     // reference times of zero-crossings, for checking
  PetscInt    cnt;              // counter
  PetscInt    cntref;           // actual length of 'ref' on the given rank
  PetscBool   flg;              // flag for additional print in PostEvent
  PetscReal   errtol;           // error tolerance, for printing 'pass/fail' for located events (default = 1e-4)
  PetscReal   errtol2;          // error tolerance for checking the solution norm2 accuracy (default = 0.03)
  Mat         A;                // system matrix
} AppCtx;

typedef struct {
  const char *name; // used only for test purposes
} Ctx2;

PetscErrorCode RHSFunc(TS ts, PetscReal t, Vec U, Vec F, void *ctx);
PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx);
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx);
PetscErrorCode Fill_mat(PetscReal coeff, PetscMPIInt rnk, Mat A);                    // Fills the system matrix (3*3) on rank-0
PetscErrorCode Handler_user(TS, PetscInt, PetscInt, PetscReal, Vec, Vec *, void *);  // Callback for evaluation times schedule 'user'
PetscErrorCode Handler_range(TS, PetscInt, PetscInt, PetscReal, Vec, Vec *, void *); // Callback for evaluation times schedule 'range'
PetscErrorCode Print_schedule_vecs(TS, const char *, PetscReal, PetscReal, PetscReal, PetscInt, PetscReal, PetscReal);
PetscErrorCode Print_user_start_end(TS, const char *);
PetscErrorCode Print_len_all_scheds(TS, const char *);

int main(int argc, char **argv)
{
  TS                     ts;
  Vec                    sol;
  PetscInt               n, dir0 = 0, ind0 = 0;
  PetscInt               m; // local size of A
  PetscReal              D = 0.02;
  char                   range[256]; // for schedule 'range'
  char                   msg[1024];  // final message, for documenting the tests
  PetscInt               dir[MAX_NFUNC];
  PetscBool              term[MAX_NFUNC];
  PetscBool              nouser = PETSC_FALSE, clear = PETSC_FALSE, adaptnone, pass_t, pass_dt;
  const char            *msg_t, *msg_dt;
  PetscScalar           *x;
  PetscReal              evtimes[30], tlast, maxtime, t0, dt0, dtlast, t4 = 0.0, tshift = 0.0, tcur, tmax = 12.0, post1, post2, dtlast_expected;
  PetscReal              t1 = 1.05; // initial time for the second TSSolve
  AppCtx                 ctx;
  Ctx2                   ctx2;
  TSConvergedReason      reason;
  TSAdapt                adapt;
  TSExactFinalTimeOption eft;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
  range[0]    = 0;
  msg[0]      = 0;
  ctx.pi      = PetscAcosReal(-1.0);
  ctx.cnt     = 0;
  ctx.cntref  = 0;
  ctx.flg     = PETSC_FALSE;
  ctx.errtol  = 1e-4;
  ctx.errtol2 = 0.03;
  ctx2.name   = "Range";

  // The linear problem has a 3*3 matrix. The matrix is constant
  m = (!ctx.rank ? 3 : 0);
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, m, m, PETSC_DETERMINE, PETSC_DETERMINE, 2, NULL, 0, NULL, &ctx.A));
  PetscCallBack("Fill_mat", Fill_mat(0.2, ctx.rank, ctx.A));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_LINEAR));

  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunc, &ctx));
  PetscCall(TSSetRHSJacobian(ts, ctx.A, ctx.A, TSComputeRHSJacobianConstant, NULL));

  PetscCall(TSSetTimeStep(ts, 0.099));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetMaxSteps(ts, 10000));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &dir0, NULL));                     // desired zero-crossing direction for events
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));                       // flag for additional output
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol", &ctx.errtol, NULL));           // error tolerance for located events etc.
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol2", &ctx.errtol2, NULL));         // error tolerance for checking the solution norm2 accuracy
  PetscCall(PetscOptionsHasName(NULL, NULL, "-nouser", &nouser));                     // disables adding the schedule 'user'
  PetscCall(PetscOptionsHasName(NULL, NULL, "-clear", &clear));                       // calls TSEvaluationTimesDestroy() instead of TSEvaluationTimesReset() after 2nd TSSolve()
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-D", &D, NULL));                         // small number for evaluation times schedule 'user'
  PetscCall(PetscOptionsGetString(NULL, NULL, "-range", range, sizeof(range), NULL)); // defines the evaluation times schedule 'range'
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tshift", &tshift, NULL));               // shift for the current time after 3rd TSSolve()
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tmax", &tmax, NULL));                   // max time for the 4th TSSolve()
  PetscCall(PetscOptionsGetString(NULL, NULL, "-msg", msg, sizeof(msg), NULL));       // final message, for documenting the tests

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ts_event_post_event_step", &post1, NULL));        // for checking
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ts_event_post_event_second_step", &post2, NULL)); // for checking

  // Set the event handling
  n = 0;               // event counter
  if (ctx.rank == 0) { // first event (with termination) -- on rank-0
    dir[n]    = dir0;
    term[n++] = PETSC_TRUE;
    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.05; // reference value, for the test purposes
  }
  if (ctx.rank == ctx.size - 1) { // second event (with termination) -- on last rank
    dir[n]    = dir0;
    term[n++] = PETSC_TRUE;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 9.05; // reference value, for the test purposes
  }
  if (ctx.rank == 1 % ctx.size) { // third event -- on rank == 1%ctx.size
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;

    for (PetscInt i = -19; i < MAX_NEV - 22; i++) {
      if ((i + 20) % 2 == 1 && dir0 <= 0) ctx.ref[ctx.cntref++] = i; // reference values, for the test purposes
      if ((i + 20) % 2 == 0 && dir0 >= 0) ctx.ref[ctx.cntref++] = i;
    }
  }
  if (ctx.cntref > 0) PetscCall(PetscSortReal(ctx.cntref, ctx.ref));
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));
  PetscCall(TSSetFromOptions(ts)); // remember, the option -ts_time_span overrides init_time and max_time

  // Set the different additional evaluation times schedules
  for (PetscInt i = 0; i < 11; i++) {
    evtimes[2 * i]     = 0.01 + i + (i == 7 ? -0.02 : 0);
    evtimes[2 * i + 1] = 0.21 + i;
  }
  evtimes[22] = 3;
  evtimes[23] = 4;
  evtimes[24] = 4 + D;
  evtimes[25] = 5 - D;
  evtimes[26] = 5;
  evtimes[27] = 6 - D;
  evtimes[28] = 6;
  evtimes[29] = 6 + D;
  PetscCall(PetscSortReal(PETSC_STATIC_ARRAY_LENGTH(evtimes), evtimes));
  if (!nouser) PetscCall(TSEvaluationTimesAddArray(ts, "user", PETSC_STATIC_ARRAY_LENGTH(evtimes), evtimes, Handler_user, &ctx));

  // Add 5 dummy schedules
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy1", 1, evtimes + 0, TSEvaluationTimesDefaultHandler, NULL));
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy2", 2, evtimes + 1, TSEvaluationTimesDefaultHandler, NULL));
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy3", 3, evtimes + 2, TSEvaluationTimesDefaultHandler, NULL));
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy4", 1, evtimes, TSEvaluationTimesDefaultHandler, NULL));
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy4", 4, evtimes + 3, TSEvaluationTimesDefaultHandler, NULL)); // override what was set above, for test purposes
  PetscCall(TSEvaluationTimesAddUniform(ts, "dummy5", 5, 2.0, 3.0, TSEvaluationTimesDefaultHandler, NULL));

  if (range[0]) {
    PetscReal  x[2];
    PetscInt   n;
    const char *value;
    PetscToken token;

    PetscCall(PetscTokenCreate(range, ',', &token));
    for (PetscInt i = 0; i < 3; i++) {
      PetscCall(PetscTokenFind(token, &value));
      PetscCheck(value, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Bad string '%.100s' in option -range", range);
      if (i < 2) PetscCall(PetscOptionsStringToReal(value, &x[i]));
      else PetscCall(PetscOptionsStringToInt(value, &n));
    }
    PetscCall(PetscTokenDestroy(&token));
    PetscCall(TSEvaluationTimesAddUniform(ts, "range", n, x[0], x[1], Handler_range, &ctx2));
  }
  PetscCall(TSEvaluationTimesSetUp(ts, PETSC_FALSE)); // PETSC_FALSE = do not override t0, tmax
  PetscCall(Print_len_all_scheds(ts, "after first TSEvaluationTimesSetUp()"));

  // Modify the evaluation times schedules: remove all 'dummyX' in different ways, for test purposes
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy2", 1, NULL, TSEvaluationTimesDefaultHandler, NULL));
  PetscCall(TSEvaluationTimesAddUniform(ts, "dummy4", 0, 2.0, 3.0, TSEvaluationTimesDefaultHandler, NULL));
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy3", 0, evtimes, TSEvaluationTimesDefaultHandler, NULL));
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy1", 1, evtimes, NULL, NULL));
  PetscCall(TSEvaluationTimesAddArray(ts, "dummy5", -1, evtimes, TSEvaluationTimesDefaultHandler, NULL));
  if (!nouser) {
    PetscCall(TSEvaluationTimesAddArray(ts, "user", PETSC_STATIC_ARRAY_LENGTH(evtimes), evtimes, NULL, &ctx));         // delete 'user'
    PetscCall(TSEvaluationTimesAddArray(ts, "user", PETSC_STATIC_ARRAY_LENGTH(evtimes), evtimes, Handler_user, &ctx)); // add 'user' again
  }
  PetscCall(Print_len_all_scheds(ts, "before second TSEvaluationTimesSetUp()")); // this call is ok even before the setup call
  PetscCall(Print_user_start_end(ts, "before second TSEvaluationTimesSetUp()"));
  PetscCall(TSEvaluationTimesSetUp(ts, PETSC_FALSE)); // call setup again to adopt the changes in schedules

  PetscCall(TSGetTime(ts, &t0)); // initial time
  PetscCall(MatCreateVecs(ctx.A, &sol, NULL));
  PetscCall(VecGetArray(sol, &x));
  if (!ctx.rank) { // initial conditions
    x[0] = 0;      // sin(0)
    x[1] = 1;      // cos(0)
    x[2] = t0;     // i.e. z(0) = 0
  }
  PetscCall(VecRestoreArray(sol, &x));
  PetscCall(TSGetTimeStep(ts, &dt0));

  // Solution: 1 - 2 - 3 - 4
  PetscCall(Print_len_all_scheds(ts, "before 1st TSSolve"));
  PetscCall(Print_user_start_end(ts, "before 1st TSSolve"));

  // 1
  PetscCall(TSSolve(ts, sol)); // normal exit at t=1.05 by event
  PetscCall(Print_user_start_end(ts, "after 1st TSSolve"));
  PetscCall(Print_schedule_vecs(ts, "user", ctx.errtol2, t0, tmax, dir0, t4, 0.0));

  // 2
  PetscCall(TSGetTime(ts, &t1));
  PetscCall(VecGetArray(sol, &x));
  if (!ctx.rank) { // change the solution after 1st TSSolve
    x[0] *= 2;     // if 't1' is an evaluation time point, it only records the solution before change
    x[1] *= 2;
  }
  PetscCall(VecRestoreArray(sol, &x));
  PetscCall(TSSolve(ts, sol));                              // normal exit at t=9.05 by event
  PetscCall(Print_len_all_scheds(ts, "after 2nd TSSolve")); // print the schedules stuff before resetting
  PetscCall(Print_schedule_vecs(ts, "user", ctx.errtol2, t0, t1, dir0, t4, 0.0));
  PetscCall(Print_schedule_vecs(ts, "default", ctx.errtol, t0, t1, dir0, t4, 0.0));
  PetscCall(Print_schedule_vecs(ts, "range", ctx.errtol, t0, t1, dir0, t4, 0.0));
  if (!clear) {
    PetscCall(TSEvaluationTimesReset(ts));
    PetscCall(Print_len_all_scheds(ts, "after TSEvaluationTimesReset"));
  } else {
    PetscCall(TSEvaluationTimesDestroy(ts));
    PetscCall(Print_len_all_scheds(ts, "after TSEvaluationTimesDestroy"));
    if (!nouser) PetscCall(TSEvaluationTimesAddArray(ts, "user", PETSC_STATIC_ARRAY_LENGTH(evtimes), evtimes, Handler_user, &ctx)); // add 'user' back
    PetscCall(TSEvaluationTimesSetUp(ts, PETSC_FALSE));                                                                             // after adding the schedules, need to setup
  }
  PetscCall(Print_schedule_vecs(ts, "user", ctx.errtol2, t0, t1, dir0, t4, 0.0));
  PetscCall(Print_schedule_vecs(ts, "default", ctx.errtol, t0, t1, dir0, t4, 0.0));
  PetscCall(Print_schedule_vecs(ts, "range", ctx.errtol, t0, t1, dir0, t4, 0.0));

  // 3
  PetscCall(TSSolve(ts, sol)); // normal exit at ts_max_time
  PetscCall(Print_user_start_end(ts, "after 3rd TSSolve"));
  if (!nouser) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Removing schedule 'user' (this also causes evaluation times to reset)...\n"));
    PetscCall(TSEvaluationTimesAddArray(ts, "user", 0, NULL, TSEvaluationTimesDefaultHandler, NULL)); // remove schedule 'user' (need to call setup then)
    PetscCall(TSEvaluationTimesSetUp(ts, PETSC_FALSE));                                               // PETSC_FALSE = do not override t0, tmax
  } else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Schedule 'user' is absent, no need to remove\n"));
  PetscCall(TSGetTime(ts, &tcur));
  t4 = tcur;
  tcur += tshift; // shift the current time, for test purposes
  if (tshift) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Shift time: %g -> %g\n", (double)t4, (double)tcur));
  PetscCall(TSSetTime(ts, tcur));

  // 4
  PetscCall(TSSetMaxTime(ts, tmax));
  PetscCall(TSSolve(ts, sol)); // normal exit at tmax
  PetscCall(Print_user_start_end(ts, "after 4th TSSolve"));

  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "== %s ==\n", TSConvergedReasons[reason]));

  // Events: the 4 columns printed are: [RANK] [time of event] [error w.r.t. reference] ["pass"/"fail"]
  while (ind0 < ctx.cntref && ctx.ref[ind0] <= t0) ind0++; // offset for the starting reference index
  for (PetscInt j = 0; j < ctx.cnt; j++) {
    PetscReal err = 10.0;
    if (j + ind0 < ctx.cntref) err = PetscAbsReal(ctx.evres[j] - ctx.ref[j + ind0]);
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d\t%g\t%g\t%s\n", ctx.rank, (double)ctx.evres[j], (double)err, err < ctx.errtol ? "pass" : "fail"));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  // Evaluation times: print [t] ["solution"] ["pass"/"fail"], on rank-0
  PetscCall(Print_schedule_vecs(ts, "user", ctx.errtol2, t0, t1, dir0, t4, tshift));
  PetscCall(Print_schedule_vecs(ts, "default", ctx.errtol, t0, t1, dir0, t4, tshift));
  PetscCall(Print_schedule_vecs(ts, "range", ctx.errtol, t0, t1, dir0, t4, tshift));

  // Print the final time and time step
  PetscCall(TSGetTime(ts, &tlast));
  PetscCall(TSGetMaxTime(ts, &maxtime));
  PetscCall(TSGetTimeStep(ts, &dtlast));
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTNONE, &adaptnone));
  PetscCall(TSGetExactFinalTime(ts, &eft));
  pass_t = PetscAbsReal(tlast - maxtime) < ctx.errtol ? PETSC_TRUE : PETSC_FALSE;

  dtlast_expected = dt0;
  if (post2 > 0) dtlast_expected = post2;
  if (post1 > 0) dtlast_expected = post1;
  pass_dt = PetscAbsReal(dtlast_expected - dtlast) < ctx.errtol ? PETSC_TRUE : PETSC_FALSE;

  if (eft == TS_EXACTFINALTIME_MATCHSTEP) {
    msg_t  = pass_t ? "pass" : "fail";
    msg_dt = pass_dt ? "pass" : "fail";
  } else msg_t = msg_dt = "****";
  if (!adaptnone) msg_dt = "(adapt != none)";
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(1) Initial time = %g, final time reached = %g, max time = %g, %s\n", (double)t0, (double)tlast, (double)maxtime, msg_t));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(2) Initial dt = %g, final dt = %g, %s\n(1) or (2): %s\n", (double)dt0, (double)dtlast, msg_dt, pass_t || pass_dt ? "pass" : "fail"));
  if (msg[0]) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "========================\n%s\n========================\n", msg));

  PetscCall(MatDestroy(&ctx.A));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&sol));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode RHSFunc(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscScalar *x;
  AppCtx      *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(TSComputeRHSFunctionLinear(ts, t, U, F, ctx));
  PetscCall(VecGetArray(F, &x));
  if (!Ctx->rank) x[2] += 1;
  PetscCall(VecRestoreArray(F, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// User callback for defining the event-functions
PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx)
{
  PetscInt n   = 0;
  AppCtx  *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  // for the test purposes, event-functions are defined based on 't' only
  // first event -- on rank-0
  if (Ctx->rank == 0) {
    if (t < 2.05) gval[n++] = 0.5 * (1 - PetscPowReal(t - 2.05, 12));
    else gval[n++] = 0.5;
  }

  // second event -- on last rank
  if (Ctx->rank == Ctx->size - 1) {
    if (t > 8.05) gval[n++] = 0.25 * (1 - PetscPowReal(t - 8.05, 12));
    else gval[n++] = 0.25;
  }

  // third event -- on rank = 1%ctx.size
  if (Ctx->rank == 1 % Ctx->size) gval[n++] = PetscSinReal(Ctx->pi * t);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// User callback for the post-event stuff
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx)
{
  AppCtx *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  if (Ctx->flg) {
    PetscCallBack("EventFunction", EventFunction(ts, t, U, Ctx->fvals, ctx));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] At t = %g : %" PetscInt_FMT " events triggered, fvalues =", Ctx->rank, (double)t, nev_zero));
    for (PetscInt j = 0; j < nev_zero; j++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\t%g", (double)Ctx->fvals[evs_zero[j]]));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  }

  if (Ctx->cnt + nev_zero < MAX_NEV)
    for (PetscInt i = 0; i < nev_zero; i++) Ctx->evres[Ctx->cnt++] = t; // save the repeating zeros (if any) separately for easier/unified testing

  if (PetscAbsReal(t - 5.0) < 0.01) { // t == 5: solution change
    PetscScalar *x;
    PetscCall(VecGetArray(U, &x));
    if (!Ctx->rank) x[2] = -x[2];
    PetscCall(VecRestoreArray(U, &x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Fills the system matrix (3*3) on rank-0
PetscErrorCode Fill_mat(PetscReal coeff, PetscMPIInt rnk, Mat A)
{
  PetscInt    inds[2], m = (!rnk ? 2 : 0);
  PetscScalar vals[4];

  PetscFunctionBeginUser;
  inds[0] = 0;
  inds[1] = 1;
  vals[0] = 0;
  vals[1] = coeff;
  vals[2] = -coeff;
  vals[3] = 0;
  PetscCall(MatSetValues(A, m, inds, m, inds, vals, INSERT_VALUES));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Callback for evaluation times schedule 'user'. Output subvector *sub = {full[0],full[1]}
PetscErrorCode Handler_user(TS ts, PetscInt iunion, PetscInt iprivate, PetscReal t, Vec full, Vec *sub, void *ctx)
{
  PetscInt     m = 0;
  PetscScalar *x, *y;
  AppCtx      *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  if (PetscIsCloseAtTol(t, 4.0, 10 * PETSC_MACHINE_EPSILON, 0.0)) PetscFunctionReturn(PETSC_SUCCESS); // skip saving a vector at t == 4.0

  if (!Ctx->rank) m = 2;                                                                // the local size of 'sub'
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)full), m, PETSC_DETERMINE, sub)); // create the output vector

  PetscCall(VecGetArray(full, &x));
  PetscCall(VecGetArray(*sub, &y));
  if (m) {
    y[0] = x[0];
    y[1] = x[1];
  }
  PetscCall(VecRestoreArray(full, &x));
  PetscCall(VecRestoreArray(*sub, &y));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ETS 'user ' point %g, union index %4" PetscInt_FMT ", private index %4" PetscInt_FMT "\n", (double)t, iunion, iprivate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Callback for evaluation times schedule 'range'. This function only does printing, the output *sub will be NULL
PetscErrorCode Handler_range(TS ts, PetscInt iunion, PetscInt iprivate, PetscReal t, Vec full, Vec *sub, void *ctx)
{
  Ctx2 *ctx2 = (Ctx2 *)ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ETS '%.100s' point %g, union index %4" PetscInt_FMT ", private index %4" PetscInt_FMT "\n", ctx2->name, (double)t, iunion, iprivate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Helper function for reporting the results saved by different schedules
PetscErrorCode Print_schedule_vecs(TS ts, const char *name, PetscReal tol, PetscReal t0, PetscReal t1, PetscInt dir0, PetscReal t4, PetscReal shift)
{
  PetscInt   len, i0, i1;
  PetscInt   type = 2; // 0 = 'user', 1 = 'default', 2 = any other (i.e. 'range')
  PetscBool  match;
  PetscReal *times;
  Vec       *sols;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcmp(name, "user", &match));
  if (match) type = 0;
  PetscCall(PetscStrcmp(name, "default", &match));
  if (match) type = 1;
  // "range" type = 2

  PetscCall(TSEvaluationTimesGetSolutions(ts, name, &len, &i0, &i1, &times, &sols));
  if (len) { // TSEvaluationTimesSchedule 'name' exists
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "---\nEvaluation times schedule '%.100s': %" PetscInt_FMT " point(s), visited subrange: [%" PetscInt_FMT ", %" PetscInt_FMT ")\n", name, len, i0, i1));
    for (PetscInt i = 0; i < len; i++) {
      PetscScalar val = 0, ref = 0; // calculated solution, reference solution

      if (type == 0) { // for 'user': check norm2 of subvector
        PetscReal norm2 = 0;

        ref = (times[i] < t1 || PetscIsCloseAtTol(times[i], t1, 10 * PETSC_MACHINE_EPSILON, PetscSqrtReal(PETSC_REAL_MIN)) ? 1.0 : 2.0); // account for solution change after the first TSSolve
        if (sols[i]) PetscCall(VecNorm(sols[i], NORM_2, &norm2));
        val = norm2;
      } else if (type == 1) { // for 'default': check z-coordinate
        ref = times[i];
        if (times[i] >= t4 + shift) ref -= shift;            // account for the time shift after the third TSSolve
        if (times[i] >= 5 && t0 < 5 && dir0 <= 0) ref -= 10; // account for solution change at t = 5
        if (sols[i]) {
          PetscInt           m;
          const PetscScalar *x;

          PetscCall(VecGetLocalSize(sols[i], &m));
          PetscCall(VecGetArrayRead(sols[i], &x));
          if (m >= 3) val = x[2]; // z-coordinate
          PetscCall(VecRestoreArrayRead(sols[i], &x));
        }
      } // else, for 'range': vectors were not saved, do nothing

      if (sols[i]) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%g\t%g\t%s\n", (double)times[i], (double)PetscRealPart(val), PetscAbsScalar(val - ref) < tol ? "pass" : "fail"));
      else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%g\t%g\t%s\n", (double)times[i], (double)PetscRealPart(val), "***"));
    }
  } else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "---\nEvaluation times schedule '%.100s' not found\n", name));

  PetscCall(TSEvaluationTimesRestoreSolutions(ts, name, &len, &i0, &i1, &times, &sols));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Prints start and end indices for 'user'
PetscErrorCode Print_user_start_end(TS ts, const char *msg)
{
  PetscInt    len, i0, i1;
  const char *name = "user";

  PetscFunctionBeginUser;
  PetscCall(TSEvaluationTimesGetSolutions(ts, name, &len, &i0, &i1, NULL, NULL));
  if (len) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "### Check '%.100s' %.100s: start %" PetscInt_FMT ", end %" PetscInt_FMT "\n", name, msg, i0, i1));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "### Check '%.100s' %.100s: not found\n", name, msg));
  PetscCall(TSEvaluationTimesRestoreSolutions(ts, name, &len, &i0, &i1, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Prints length of the 3+5 schedules
PetscErrorCode Print_len_all_scheds(TS ts, const char *msg)
{
  PetscInt    Lu, Ld, Lr, Ldum[5];
  const char *nu      = "user";
  const char *nd      = "default";
  const char *nr      = "range";
  const char *ndum[5] = {"dummy1", "dummy2", "dummy3", "dummy4", "dummy5"};

  PetscFunctionBeginUser;
  PetscCall(TSEvaluationTimesGetSolutions(ts, nu, &Lu, NULL, NULL, NULL, NULL)); // here, test simultaneously accessing multiple schedules
  PetscCall(TSEvaluationTimesGetSolutions(ts, nd, &Ld, NULL, NULL, NULL, NULL));
  PetscCall(TSEvaluationTimesGetSolutions(ts, nr, &Lr, NULL, NULL, NULL, NULL));
  for (PetscInt i = 0; i < 5; i++) PetscCall(TSEvaluationTimesGetSolutions(ts, ndum[i], &Ldum[i], NULL, NULL, NULL, NULL));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">> Schedule '%7.100s' %.100s: %" PetscInt_FMT " point(s)%s\n", nu, msg, Lu, Lu ? "" : " <empty>"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">> Schedule '%7.100s' %.100s: %" PetscInt_FMT " point(s)%s\n", nd, msg, Ld, Ld ? "" : " <empty>"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">> Schedule '%7.100s' %.100s: %" PetscInt_FMT " point(s)%s\n", nr, msg, Lr, Lr ? "" : " <empty>"));
  for (PetscInt i = 0; i < 5; i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">> Schedule '%7.100s' %.100s: %" PetscInt_FMT " point(s)%s\n", ndum[i], msg, Ldum[i], Ldum[i] ? "" : " <empty>"));

  PetscCall(TSEvaluationTimesRestoreSolutions(ts, nu, &Lu, NULL, NULL, NULL, NULL));
  PetscCall(TSEvaluationTimesRestoreSolutions(ts, nd, &Ld, NULL, NULL, NULL, NULL));
  PetscCall(TSEvaluationTimesRestoreSolutions(ts, nr, &Lr, NULL, NULL, NULL, NULL));
  for (PetscInt i = 0; i < 5; i++) PetscCall(TSEvaluationTimesRestoreSolutions(ts, ndum[i], &Ldum[i], NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*---------------------------------------------------------------------------------------------*/
/*TEST
  test:
    suffix: 1
    args: -ts_time_step 0.123 -ts_type alpha -D -2.95
    nsize: 2

  test:
    suffix: 2
    args: -ts_time_step 0.123 -ts_type alpha -D 3.05 -clear
    nsize: 2

  test:
    suffix: 3
    args: -ts_time_step 0.123 -ts_type alpha -D -2.95 -ts_init_time 0.01 -range -2,2,20
    nsize: 2

  test:
    suffix: 4
    args: -ts_time_step 0.123 -ts_type alpha -D -2.95 -ts_init_time 0.01 -range -2,2,20 -clear
    nsize: 2

  test:
    suffix: 5
    args: -ts_time_step 0.123 -ts_type alpha -D -2.95 -ts_init_time 0.01
    nsize: 4

  test:
    suffix: 6
    requires: !single
    args: -ts_time_step 0.123 -ts_type alpha -D -2.95 -range -2e-10,2e-10,11 -ts_adapt_dt_min_abs 1e-10
    args: -msg "TS starts at t=0. The 11 points in schedule =range= are merged into three: -2e-10, 0.0, 1.2e-10 (the actual tolerance for merging here is 1e-10)"

  test:
    suffix: 7
    requires: !single
    args: -ts_time_step 0.123 -ts_type alpha -D -2.95 -range -2e-10,2e-10,11 -ts_init_time -0.07 -ts_adapt_dt_min_abs 1e-10
    args: -msg "Like suffix-6, but t0=-0.07 is far from =range=, so the 11 =range= points are merged into four: -2e-10, -0.8e-10, 0.4e-10, 1.6e-10"

  test:
    suffix: 8
    requires: !single
    args: -ts_time_step 0.123 -ts_type alpha -D -2.95 -ts_init_time 1.0 -range 1.04999999995,1.05000000035,11 -clear
    args: -ts_event_tol 1e-10 -ts_adapt_dt_min_abs 1e-10
    args: -msg "The 11 points in =range= are merged into four"
    nsize: 2

  testset:
    args: -range 1.05,3.05,11 -ts_max_time 10.055 -ts_time_step 0.123 -ts_type rk -ts_adapt_type none
    args: -msg "Note, event handling is not designed for shifting the time between TSSolves, and may not function properly (Jul 2026); see ts.c code: TSSolve() -> TSEventInitialize()"

    test:
      suffix: 9
      args: -malloc_debug -ts_eval_times 0,1.1,2.2,10,10.5,10.8,11.1,11.5,11.8,12

    test:
      suffix: 10
      args: -ts_eval_times 0,1.1,2.2,10,10.5,10.8,11.1,11.5,11.8,12 -tshift 0.44

    test:
      suffix: 11
      args: -ts_eval_times 0,1.1,2.2,10,10.5,10.8,11.1,11.5,11.8,12 -tshift 0.445

    test:
      suffix: 12
      args: -ts_eval_times 0,1.1,2.2,10,10.04,10.5,10.8,11.1,11.5,11.8,12 -tshift 0.8 -nouser

    test:
      suffix: 13
      args: -ts_eval_times 0,1.1,2.2,10,10.04,10.5,10.8,11.1,11.5,11.8,12 -tshift 0.8 -nouser -dir 1

    test:
      suffix: 14
      args: -ts_eval_times 0,1.1,2.2,10,10.04,10.5,10.8,11.1,11.5,11.8,12 -tshift -0.045 -nouser

  test:
    suffix: 15
    args: -ts_init_time 0 -ts_max_time 10.055 -ts_time_step 0.123 -nouser -ts_eval_times_uniform 10,12,21

  test:
    suffix: 16
    args: -ts_max_time 10.055 -ts_time_step 0.123 -nouser -msg "Empty evaluation times"

  test:
    suffix: 17
    requires: !single
    args: -nouser -ts_monitor -errtol 1e-20 -malloc_debug -ts_adapt_dt_min_abs 1e-20
    args: -ts_eval_times 0,1e-21,1e-11,2e-11,3e-11,7e-11,10e-11 -ts_init_time -2e-11 -ts_max_time 10e-11 -tmax 10e-11
    args: -msg "Evaluation time points 0,1e-21 are merged into one; the others are not. Also checking here the internal counters setup"

  test:
    suffix: 18
    requires: !single
    args: -nouser -ts_monitor -errtol 0.015 -ts_event_tol 2 -ts_type rk -ts_adapt_type none -ts_time_step 0.9 -ts_adapt_dt_min_rel 1e-10
    args: -ts_eval_times -1000000010,-1000000009.99,-1000000009,-1000000008,-1000000007,-1000000003,-1000000000
    args: -ts_init_time -1000000012 -ts_max_time -999999999.5 -tmax -999999999.5
    args: -msg "The first two evaluation time points are merged. Event tolerance is large - to disable the sine event"

  test:
    suffix: 19
    args: -ts_eval_times 1.05,9.05 -ts_init_time 0 -ts_max_time 10 -ts_event_post_event_second_step 0.123 -errtol2 0.05
TEST*/
