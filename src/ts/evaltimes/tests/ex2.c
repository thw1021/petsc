#include <petscts.h>

static char help[] = "Simple 3*3 linear problem with TSEvaluationTimes and events\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"
                     "z_dot =  1\n"

                     "The following evaluation times schedules are available:\n"
                     "1) Schedule 'user' (present by default) with points [0.01, 0.21, 1.01, ..., 6.21, 6.99, 7.21,... 9.21]\n"
                     "                                   plus the points: {3, 4, 4+D, 5-D, 5, 6-D, 6, 6+D} with user-defined 'D'\n"
                     "2) Schedule 'default' can be added by PETSc options -ts_time_span, -ts_eval_times or -ts_eval_times_uniform\n"
                     "3) Schedule 'range' can be added by the application option -range, see below.\n"
                     "These schedules use different callbacks, see the code for details.\n"

                     "The following event functions are involved:\n"
                     "- two polynomial event functions on rank-0 and last-rank (with zeros: 1.05, 9.05[terminating])\n"
                     "- one event function on rank = '1%size', equal to sin(pi*t), zeros = 1,...,10\n"
                     "  on event at t==5, solution is changed: z = -z\n"

                     "Application options:\n"
                     "-dir     d : zero-crossing direction for events: 0, 1, -1 (default = 0)\n"
                     "-flg       : additional output in Postevent (default = nothing)\n"
                     "-errtol  e : error tolerance, for printing 'pass/fail' for located events etc. (default = 1e-4)\n"
                     "-errtol2 t : error tolerance for checking the solution norm2 accuracy (default = 0.03)\n"
                     "-nouser    : disables adding the schedule 'user'\n"
                     "-D       z : a (small) real number to define the additional evaluation time points for 'user' (default = 0.02)\n"
                     "-range   T : with T = min,max,N defining N evaluation time points in [min, max]\n"
                     "-msg     m : an optional final message to print for documenting the tests\n";

#define MAX_NFUNC 5   // max event functions per rank
#define MAX_NEV   500 // max zero crossings for each rank

#if PetscDefined(USE_REAL___FLOAT128)
  #define Z01 0.01Q
  #define Z21 0.21Q
#else
  #define Z01 0.01
  #define Z21 0.21
#endif

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
PetscErrorCode Print_schedule_vecs(TS, const char *, PetscReal, PetscReal, PetscInt);
PetscErrorCode Print_user_start_end(TS, const char *);
PetscErrorCode Print_len_all_scheds(TS, const char *);

int main(int argc, char **argv)
{
  TS                ts;
  Vec               sol;
  PetscInt          n, dir0 = 0, ind0 = 0;
  PetscInt          m; // local size of A
  PetscReal         D = Z01 * 2;
  char              range[256]; // for schedule 'range'
  char              msg[1024];  // final message, for documenting the tests
  PetscInt          dir[MAX_NFUNC];
  PetscBool         term[MAX_NFUNC];
  PetscBool         nouser = PETSC_FALSE, adaptnone;
  PetscScalar      *x;
  PetscReal         evtimes[28], tlast, tlast_expected, maxtime, t0, dt0, dtlast;
  AppCtx            ctx;
  Ctx2              ctx2;
  TSConvergedReason reason;
  TSAdapt           adapt;

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
  // PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &dir0, NULL));                     // desired zero-crossing direction for events
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));                       // flag for additional output
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol", &ctx.errtol, NULL));           // error tolerance for located events etc.
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol2", &ctx.errtol2, NULL));         // error tolerance for checking the solution norm2 accuracy
  PetscCall(PetscOptionsHasName(NULL, NULL, "-nouser", &nouser));                     // disables adding the schedule 'user'
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-D", &D, NULL));                         // small number for evaluation times schedule 'user'
  PetscCall(PetscOptionsGetString(NULL, NULL, "-range", range, sizeof(range), NULL)); // defines the evaluation times schedule 'range'
  PetscCall(PetscOptionsGetString(NULL, NULL, "-msg", msg, sizeof(msg), NULL));       // final message, for documenting the tests

  // Set the event handling
  n = 0;               // event counter
  if (ctx.rank == 0) { // first event -- on rank-0
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;
    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.05;
  }
  if (ctx.rank == ctx.size - 1) { // second event (with termination) -- on last rank
    dir[n]    = dir0;
    term[n++] = PETSC_TRUE;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 9.05;
  }
  if (ctx.rank == 1 % ctx.size) { // third event -- on rank == 1%ctx.size
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;

    for (PetscInt i = -19; i < MAX_NEV - 22; i++) {
      if ((i + 20) % 2 == 1 && dir0 <= 0) ctx.ref[ctx.cntref++] = i;
      if ((i + 20) % 2 == 0 && dir0 >= 0) ctx.ref[ctx.cntref++] = i;
    }
  }
  if (ctx.cntref > 0) PetscCall(PetscSortReal(ctx.cntref, ctx.ref));
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));

  // Set the different evaluation times schedules
  for (PetscInt i = 0; i < 10; i++) {
    evtimes[2 * i]     = Z01 + i + (i == 7 ? -Z01 * 2 : 0);
    evtimes[2 * i + 1] = Z21 + i;
  }
  evtimes[20] = 3;
  evtimes[21] = 4;
  evtimes[22] = 4 + D;
  evtimes[23] = 5 - D;
  evtimes[24] = 5;
  evtimes[25] = 6 - D;
  evtimes[26] = 6;
  evtimes[27] = 6 + D;
  PetscCall(PetscSortReal(PETSC_STATIC_ARRAY_LENGTH(evtimes), evtimes));
  if (!nouser) PetscCall(TSEvaluationTimesAddArray(ts, "user", PETSC_STATIC_ARRAY_LENGTH(evtimes), evtimes, Handler_user, &ctx));

  if (range[0]) {
    PetscReal   x[2];
    PetscInt    n;
    const char *value;
    PetscToken  token;

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
  PetscCall(Print_user_start_end(ts, "before SetUp"));
  PetscCall(TSEvaluationTimesSetUp(ts, PETSC_FALSE)); // PETSC_FALSE = do not override t0, tmax

  /*
    The options -ts_init_time, -ts_max_time allow t0 and tmax to be overridden.
    If the evaluation times schedule (named 'default') is set via option -ts_time_span,
    t0 and tmax are overridden again, now accounting for all schedules added, including 'user' and 'range'.
  */
  PetscCall(TSSetFromOptions(ts));

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

  // Solution
  PetscCall(Print_user_start_end(ts, "before Solve"));
  PetscCall(Print_len_all_scheds(ts, "before Solve"));
  PetscCall(TSSolve(ts, sol));
  PetscCall(Print_user_start_end(ts, "after Solve"));
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
  PetscCall(Print_schedule_vecs(ts, "user", ctx.errtol2, t0, dir0));
  PetscCall(Print_schedule_vecs(ts, "default", ctx.errtol, t0, dir0));
  PetscCall(Print_schedule_vecs(ts, "range", ctx.errtol, t0, dir0));

  // Print the final time and time step
  PetscCall(TSGetTime(ts, &tlast));
  PetscCall(TSGetMaxTime(ts, &maxtime));
  PetscCall(TSGetTimeStep(ts, &dtlast));
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTNONE, &adaptnone));
  tlast_expected = (dir0 == 1 || t0 >= 9.05 ? maxtime : PetscMin(maxtime, 9.05));
  if (PetscDefined(USE_REAL___FLOAT128) && maxtime == PETSC_MAX_REAL)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial time = %g, final time reached = %g, max time = 1.19e+4932, %s\n", (double)t0, (double)tlast, PetscAbsReal(tlast - tlast_expected) < ctx.errtol ? "pass" : "fail"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial time = %g, final time reached = %g, max time = %g, %s\n", (double)t0, (double)tlast, (double)maxtime, PetscAbsReal(tlast - tlast_expected) < ctx.errtol ? "pass" : "fail"));
  if (adaptnone) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial time step = %g, final time step = %g, adapt == none, %s\n", (double)dt0, (double)dtlast, PetscAbsReal(dt0 - dtlast) < ctx.errtol ? "match" : "NO-match"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial time step = %g, final time step = %g, adapt != none, %s\n", (double)dt0, (double)dtlast, "***"));
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
  // for the test purposes, event-functions are defined based on t
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
    for (PetscInt i = 0; i < nev_zero; i++) Ctx->evres[Ctx->cnt++] = t; // save the repeating zeros separately for easier/unified testing

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
  if (PetscIsCloseAtTol(t, 4.0, 10 * PETSC_MACHINE_EPSILON, 0.0)) PetscFunctionReturn(PETSC_SUCCESS); // skip saving this particular vector

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
PetscErrorCode Print_schedule_vecs(TS ts, const char *name, PetscReal tol, PetscReal t0, PetscInt dir0)
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

        ref = 1.0;
        if (sols[i]) PetscCall(VecNorm(sols[i], NORM_2, &norm2));
        val = norm2;
      } else if (type == 1) { // for 'default': check z-coordinate
        ref = times[i];
        if ((times[i] > 5 || PetscIsCloseAtTol(times[i], 5, 10 * PETSC_MACHINE_EPSILON, 0.0)) && t0 < 5 && dir0 <= 0) ref -= 10; // account for solution change at t = 5
        if (sols[i]) {
          PetscInt           m;
          const PetscScalar *x;

          PetscCall(VecGetLocalSize(sols[i], &m));
          PetscCall(VecGetArrayRead(sols[i], &x));
          if (m >= 3) val = x[2]; // z-coordinate
          PetscCall(VecRestoreArrayRead(sols[i], &x));
        }
      } // else, for 'range': vectors are not saved, do nothing

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

// Prints length of the three schedules
PetscErrorCode Print_len_all_scheds(TS ts, const char *msg)
{
  PetscInt    Lu, Ld, Lr;
  const char *nu = "user";
  const char *nd = "default";
  const char *nr = "range";

  PetscFunctionBeginUser;
  PetscCall(TSEvaluationTimesGetSolutions(ts, nu, &Lu, NULL, NULL, NULL, NULL)); // here, test simultaneously accessing multiple schedules
  PetscCall(TSEvaluationTimesGetSolutions(ts, nd, &Ld, NULL, NULL, NULL, NULL));
  PetscCall(TSEvaluationTimesGetSolutions(ts, nr, &Lr, NULL, NULL, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">> Schedule '%7.100s' %.100s: %" PetscInt_FMT " point(s)%s\n", nu, msg, Lu, Lu ? "" : " <empty>"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">> Schedule '%7.100s' %.100s: %" PetscInt_FMT " point(s)%s\n", nd, msg, Ld, Ld ? "" : " <empty>"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, ">> Schedule '%7.100s' %.100s: %" PetscInt_FMT " point(s)%s\n", nr, msg, Lr, Lr ? "" : " <empty>"));
  PetscCall(TSEvaluationTimesRestoreSolutions(ts, nu, &Lu, NULL, NULL, NULL, NULL));
  PetscCall(TSEvaluationTimesRestoreSolutions(ts, nd, &Ld, NULL, NULL, NULL, NULL));
  PetscCall(TSEvaluationTimesRestoreSolutions(ts, nr, &Lr, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*---------------------------------------------------------------------------------------------*/
/*TEST
  test:
    suffix: 1
    args: -range -1,8,5 -ts_init_time 4 -ts_max_time 10 -D 0

  test:
    suffix: 2
    args: -ts_time_span 1.5,1.5,4,4,4,5,5 -D 0
    args: -msg "Option ts_time_span triggers overriding t0 and tmax, accounting for all schedules, i.e. 'user', 'default'"

  test:
    suffix: 3
    args: -D 0

  test:
    suffix: 4
    args: -ts_max_time 5.2 -ts_exact_final_time stepover -nouser

  test:
    suffix: 5
    args: -ts_max_time 5.99 -ts_exact_final_time interpolate -nouser -ts_adapt_type basic

  test:
    suffix: 6
    requires: double
    args: -ts_eval_times 1.5,1.5,4,4,4,5,5 -D 0 -range 2.5000000000001,2.5000000000002,5 -ts_adapt_dt_min_rel 1e-13 -ts_init_time 0.01 -ts_max_time 9.21
    args: -msg "Check merging of repeating points and closely spaced points in TSEvaluationTimes; all points in \"range\" are merged into one point"

  test:
    suffix: 7
    args: -D 0.0001 -range 2.5,2.501,5 -ts_adapt_dt_min_abs 0.11001 -ts_event_post_event_step 0.001 -ts_adapt_type basic
    args: -ts_event_dt_min_abs 1e-15
    args: -msg "Check that -ts_adapt_dt_min_abs does not prevent smaller steps: the postevent step (check via -ts_monitor), and small steps from the evaluation times"

  test:
    suffix: 8
    requires: double
    args: -D 1e-12 -range 0,1e-12,5 -ts_init_time 4 -ts_max_time 6 -ts_adapt_dt_min_abs 2e-12 -ts_eval_times 0
    args: -msg "Merging points at beginning, near t0, near tmax. Dummy -ts_eval_times is added to trigger one more TSEvaluationTimesSetUp() on setting from options"

  test:
    suffix: 9
    requires: double
    args: -D 1e-12 -range 0,1e-12,5 -ts_init_time 3.9 -ts_max_time 6.5 -ts_adapt_dt_min_abs 3e-12 -ts_eval_times 0

  test:
    suffix: 10
    requires: double
    args: -D 1e-12 -range 10,10.000000000001,9 -ts_init_time 4 -ts_max_time 6 -ts_adapt_dt_min_rel 1e-12 -ts_eval_times 10
    args: -msg "Merging points near t0 (4), near tmax (6), at the end (10)"

  test:
    suffix: 11
    requires: double
    args: -D 1e-12 -range 10,10.000000000001,9 -ts_eval_times 0,1e-12,2e-12 -ts_adapt_dt_min_abs 3e-12
    args: -msg "More merging..."

  test:
    suffix: 12
    requires: double
    args: -D 1e-12 -range 10,10.000000000001,9 -ts_eval_times 0,1e-12,2e-12 -dir 1 -ts_adapt_dt_min_abs 3e-12 -ts_max_time 10
    args: -msg "Event(t==5) is skipped, so no solution change there; event(t==9.05) is skipped, so no termination there"

  test:
    suffix: 13
    requires: double
    args: -D 1e-9 -range 10,10.000000002,8 -ts_eval_times 0,0.6e-9,1.2e-9 -dir 1 -ts_adapt_dt_min_abs 1.1e-9 -ts_max_time 10.1
    args: -msg "The tightly spaced evaluation time points are chosen such that some are merged, some are not"

  test:
    suffix: 14
    requires: !single
    args: -nouser -ts_time_span 0,1e-11,5e-11,9e-11 -ts_adapt_dt_min_rel 0.1
    args: -msg "Very small time span, close to zero; the relative tolerance used keeps the points NOT merged"

  test:
    suffix: 15
    requires: !single
    args: -nouser -ts_time_span 1,1.0001,1.0005,1.0009 -ts_adapt_dt_min_rel 1e-3 -errtol 1e-3
    args: -msg "The time span points are merged into one point, but the time span sets distinct t0 and tmax. Only one time step is made"

  test:
    suffix: 16
    requires: !single
    args: -nouser -range 0,2e-10,11 -ts_time_span -2e-11,2.2e-10 -ts_event_tol 1e-14 -ts_monitor -malloc_debug
    args: -msg "Points are not merged"

  test:
    suffix: 17
    args: -range -1,8,5 -ts_init_time 9.1 -ts_max_time 15
    args: -msg "Late start => missing some (nearly all) evaluation time points"
    nsize: 4

  test:
    suffix: 18
    args: -D 0.001 -ts_adapt_type basic -ts_type {{theta rk alpha arkimex rosw bdf}}

  test:
    suffix: 19
    requires: !single
    args: -ts_time_step 0.123 -ts_type alpha -D -2.95 -ts_eval_times 0,0,0,0,2e-10,2e-10,2e-10
    nsize: 2

  test:
    suffix: 20
    requires: !single
    args: -ts_time_step 0.123 -ts_type alpha -D -4.01 -ts_time_span 0,0,0,0,2e-10,2e-10,2e-10
    args: -msg "t0, tmax will be set from TSEvaluationTimes (all schedules)"
    nsize: 2

  test:
    suffix: 21
    args: -ts_time_step 0.123 -ts_type alpha -D 0 -ts_init_time 5.95
    nsize: 3

  test:
    suffix: 22
    args: -ts_init_time 0.1 -ts_time_step 0.12345 -nouser -range 0.05,10.05,21 -ts_max_time 10.01

  test:
    suffix: 23
    args: -ts_init_time 0.1 -ts_time_step 0.12345 -nouser -dir 1 -range 0.04,10.04,21 -ts_max_time 1.06

  test:
    suffix: 24
    args: -ts_init_time 0.1 -ts_time_step 0.12345 -nouser -dir 0 -range 0.04,10.04,21 -ts_max_time 1.06 -malloc_debug

  test:
    suffix: 25
    args: -ts_init_time 0.1 -ts_time_step 0.12345 -nouser -dir 0 -range 0.04,10.04,21 -ts_max_time 1.05
    nsize: 2

  test:
    suffix: 26
    args: -ts_init_time 0.1 -ts_time_step 0.12345 -nouser -dir 0 -range 0.06,10.06,21 -ts_max_time 1.07

  test:
    suffix: 27
    args: -ts_monitor -ts_init_time 0.1 -ts_time_step 0.12345 -nouser -ts_adapt_monitor -ts_max_time 0.3 -malloc_debug

  test:
    suffix: 28
    args: -ts_monitor -ts_init_time 0.1 -ts_time_step 0.12345 -nouser -ts_adapt_monitor -ts_max_time 0.15

  test:
    suffix: 29
    args: -ts_init_time 0 -ts_time_step 0.12345 -ts_max_time 9.05

  test:
    suffix: 30
    args: -ts_init_time 0 -ts_time_step 0.12345 -ts_max_time 9.05 -dir 1

  test:
    suffix: 31
    args: -ts_monitor -ts_event_monitor -ts_init_time 2 -ts_time_step 0.12345 -ts_adapt_monitor -ts_max_time 1
    args: -msg "Check t0 > tmax"

  test:
    suffix: 32
    args: -ts_monitor -ts_event_monitor -ts_init_time 2.01 -ts_time_step 0.12345 -ts_adapt_monitor -ts_max_time 1
    args: -msg "Check t0 > tmax"

  test:
    suffix: 33
    args: -ts_monitor -ts_event_monitor -ts_init_time 2.01 -ts_time_step 0.12345 -ts_adapt_monitor -ts_max_time 2.01
    args: -msg "Check t0 == tmax"

  test:
    suffix: 34
    requires: !single !__float128
    args: -ts_monitor -ts_time_step 10 -ts_type alpha
    args: -msg "Illustrate how evaluation times can be used for defining an arbitrary schedule of time steps; the initial dt is large enough"

  test:
    suffix: 35
    args: -ts_adapt_type none -ts_time_step 0.1127 -D 0.0015 -ts_max_time 5.5
    args: -msg "Step size is increased to 0.1138 before evaltimes t = 3.0 and 4.0 (check with -ts_monitor)"
TEST*/
