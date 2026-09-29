#include <petscts.h>

static char help[] = "A small (3-dim) non-linear problem with TSEvaluationTimes and events:\n"
                     "u_dot =  v\n"
                     "v_dot = -B^2*u + 2*(1 - w/15)*v\n"
                     "w_dot =  B,\n"
                     "where B == 2*w - w^2/15\n\n"

                     "The exact solution is:\n"
                     "u = sin(w)\n"
                     "v = cos(w)*B\n"
                     "w = 30/(1 + exp(-2*t+10))\n"
                     "The suggested time interval for solution is [0, 10].\n"
                     "Near the middle, i.e. around [4, 6], the frequency of oscillations gets higher,\n"
                     "causing the time step adaptor to decrease steps by one order of magnitude.\n\n"

                     "The following evaluation times schedules are available:\n"
                     "1) Schedule 'user' (present by default) with points [0.01, 0.21, 1.01, ..., 6.21, 6.99, 7.21,... 9.21]\n"
                     "                                   plus the points: {3, 4, 4+D, 5-D, 5, 6-D, 6, 6+D} with user-defined 'D'\n"
                     "2) Schedule 'default' can be added by PETSc options -ts_time_span, -ts_eval_times or -ts_eval_times_uniform\n"
                     "3) Schedule 'range' can be added by the application option -range, see below.\n"
                     "These schedules use different callbacks, see the code for details.\n\n"

                     "The following event functions are involved:\n"
                     "- two polynomial event functions on rank='1%size' and last-rank (with zeros: 1.05, 9.05[terminating])\n"
                     "- one event function on rank-0, equal to u(t), with nine zeros in [3, 7]\n\n"

                     "Application options:\n"
                     "-dir     d : zero-crossing direction for events: 0, 1, -1 (default = 0)\n"
                     "-flg       : additional output in Postevent (default = nothing)\n"
                     "-errtol  e : error tolerance, for printing 'pass/fail' for located events etc. (default = 0.01)\n"
                     "-errtol2 t : error tolerance for checking norm2 in schedule 'user' (default = 0.003)\n"
                     "-nouser    : disables adding the schedule 'user'\n"
                     "-D       z : a (small) real number to define the additional evaluation time points for 'user' (default = 0.02)\n"
                     "-range   T : with T = min,max,N defining N evaluation time points in [min, max]\n"
                     "-msg     m : an optional final message to print for documenting the tests\n";

#define MAX_NFUNC 3  // max event indicator functions per rank
#define MAX_NEV   20 // max zero crossings for each rank

typedef struct {
  PetscMPIInt rank, size;
  PetscReal   fvals[MAX_NFUNC]; // helper array for reporting the residuals
  PetscReal   evres[MAX_NEV];   // times of found zero-crossings
  PetscReal   ref[MAX_NEV];     // reference times of zero-crossings, for checking
  PetscInt    cnt;              // counter
  PetscInt    cntref;           // actual length of 'ref' on the given rank
  PetscBool   flg;              // flag for additional print in PostEvent
  PetscReal   errtol;           // error tolerance, for printing 'pass/fail' for located events etc. (default = 0.01)
  PetscReal   errtol2;          // error tolerance for checking norm2 in schedule 'user' (default = 0.003)
  Mat         A;                // system matrix
} AppCtx;

typedef struct {
  const char *name; // used only for test purposes
} Ctx2;

PetscErrorCode RHSFunc(TS ts, PetscReal t, Vec U, Vec F, void *ctx);
PetscErrorCode RHSJac(TS ts, PetscReal t, Vec U, Mat A, Mat P, void *ctx);
PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx);
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx);
PetscErrorCode Handler_user(TS, PetscInt, PetscInt, PetscReal, Vec, Vec *, void *);  // Callback for evaluation times schedule 'user'
PetscErrorCode Handler_range(TS, PetscInt, PetscInt, PetscReal, Vec, Vec *, void *); // Callback for evaluation times schedule 'range'
PetscErrorCode Print_schedule_vecs(TS, const char *, PetscReal);
PetscErrorCode Print_user_start_end(TS, const char *);
PetscErrorCode Print_len_all_scheds(TS, const char *);

int main(int argc, char **argv)
{
  TS                ts;
  Vec               sol;
  PetscInt          n, dir0 = 0, ind0 = 0;
  PetscInt          m; // local size of A
  PetscReal         D = 0.02;
  char              range[256]; // for schedule 'range'
  char              msg[1024];  // final message, for documenting the tests
  PetscInt          dir[MAX_NFUNC];
  PetscBool         term[MAX_NFUNC];
  PetscBool         nouser = PETSC_FALSE;
  PetscScalar      *x;
  PetscReal         evtimes[28], tlast, tlast_expected, maxtime, t0, dt0, dtlast;
  AppCtx            ctx;
  Ctx2              ctx2;
  TSConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
  range[0]    = 0;
  msg[0]      = 0;
  ctx.cnt     = 0;
  ctx.cntref  = 0;
  ctx.flg     = PETSC_FALSE;
  ctx.errtol  = 0.01;
  ctx.errtol2 = 0.003;
  ctx2.name   = "Range";

  // The Jacobian is a 3*3 matrix
  m = (!ctx.rank ? 3 : 0);
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, m, m, PETSC_DETERMINE, PETSC_DETERMINE, 3, NULL, 0, NULL, &ctx.A));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));

  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunc, &ctx));
  PetscCall(TSSetRHSJacobian(ts, ctx.A, ctx.A, RHSJac, &ctx));

  PetscCall(TSSetTimeStep(ts, 0.099));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetMaxSteps(ts, 10000));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &dir0, NULL));                     // desired zero-crossing direction for events
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));                       // flag for additional output
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol", &ctx.errtol, NULL));           // error tolerance for located events etc.
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol2", &ctx.errtol2, NULL));         // error tolerance for checking norm2 in schedule 'user'
  PetscCall(PetscOptionsHasName(NULL, NULL, "-nouser", &nouser));                     // disables adding the schedule 'user'
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-D", &D, NULL));                         // small number for evaluation times schedule 'user'
  PetscCall(PetscOptionsGetString(NULL, NULL, "-range", range, sizeof(range), NULL)); // defines the evaluation times schedule 'range'
  PetscCall(PetscOptionsGetString(NULL, NULL, "-msg", msg, sizeof(msg), NULL));       // final message, for documenting the tests

  // Set the event handling
  n = 0;                          // event counter
  if (ctx.rank == 1 % ctx.size) { // first event -- on rank == 1%ctx.size
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;
    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 1.05;
  }
  if (ctx.rank == ctx.size - 1) { // second event (with termination) -- on last rank
    dir[n]    = dir0;
    term[n++] = PETSC_TRUE;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 9.05;
  }
  if (ctx.rank == 0) { // third event -- on rank-0
    dir[n]    = dir0;
    term[n++] = PETSC_FALSE;

    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 3.92708;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 4.60963;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 5.04723;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 5.50505;
    if (dir0 <= 0) ctx.ref[ctx.cntref++] = 6.39817;

    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 4.33585;
    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 4.83631;
    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 5.26251;
    if (dir0 >= 0) ctx.ref[ctx.cntref++] = 5.82082;
  }
  if (ctx.cntref > 0) PetscCall(PetscSortReal(ctx.cntref, ctx.ref));
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));

  // Set the different evaluation times schedules
  for (PetscInt i = 0; i < 10; i++) {
    evtimes[2 * i]     = 0.01 + i + (i == 7 ? -0.02 : 0);
    evtimes[2 * i + 1] = 0.21 + i;
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
  PetscCall(TSEvaluationTimesSetUp(ts, PETSC_FALSE)); // PETSC_FALSE = do not override t0, tmax

  /*
    The options -ts_init_time, -ts_max_time allow t0 and tmax to be overridden.
    If the evaluation times schedule (named 'default') is set via PETSc option -ts_time_span,
    t0 and tmax are overridden again, now accounting for all schedules added, including 'user' and 'range'.
  */
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSGetTime(ts, &t0)); // initial time
  PetscCall(MatCreateVecs(ctx.A, &sol, NULL));
  PetscCall(VecGetArray(sol, &x));
  if (!ctx.rank) { // initial conditions
    x[2] = 30 / (1 + PetscExpReal(-2 * t0 + 10));
    x[1] = PetscCosReal(x[2]) * (2 * x[2] - x[2] * x[2] / 15);
    x[0] = PetscSinReal(x[2]);
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
  PetscCall(Print_schedule_vecs(ts, "user", ctx.errtol2));
  PetscCall(Print_schedule_vecs(ts, "default", ctx.errtol));
  PetscCall(Print_schedule_vecs(ts, "range", 0.0));

  // Print the final time and time step
  PetscCall(TSGetTime(ts, &tlast));
  PetscCall(TSGetMaxTime(ts, &maxtime));
  PetscCall(TSGetTimeStep(ts, &dtlast));

  tlast_expected = (dir0 == 1 || t0 >= 9.05 ? maxtime : PetscMin(maxtime, 9.05));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial time = %g, final time reached = %g, max time = %g, %s\n", (double)t0, (double)tlast, (double)maxtime, PetscAbsReal(tlast - tlast_expected) < ctx.errtol * 1e-3 ? "pass" : "fail"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initial time step = %g, final time step = %g\n", (double)dt0, (double)dtlast));
  if (msg[0]) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "========================\n%s\n========================\n", msg));

  PetscCall(MatDestroy(&ctx.A));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&sol));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode RHSFunc(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;
  AppCtx            *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));
  if (!Ctx->rank) {
    f[2] = 2 * u[2] - u[2] * u[2] / 15; // == B
    f[1] = -f[2] * f[2] * u[0] + 2 * (1 - u[2] / 15) * u[1];
    f[0] = u[1];
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSJac(TS ts, PetscReal t, Vec U, Mat A, Mat P, void *ctx)
{
  const PetscScalar *u;
  PetscScalar        p[9], B, dBdw;
  PetscInt           n      = 0;
  PetscInt           idx[3] = {0, 1, 2};
  AppCtx            *Ctx    = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  if (!Ctx->rank) {
    B    = 2 * u[2] - u[2] * u[2] / 15;
    dBdw = 2 - 2 * u[2] / 15;
    n    = 3;
    p[0] = 0; // 1st row, for u_dot = v
    p[1] = 1;
    p[2] = 0;
    p[3] = -B * B; // 2nd row, for v_dot = -B^2*u + 2*(1 - w/15)*v
    p[4] = 2 * (1 - u[2] / 15);
    p[5] = -2 * (B * dBdw * u[0] + u[1] / 15);
    p[6] = 0; // 3rd row, for w_dot = B
    p[7] = 0;
    p[8] = dBdw;
  }
  PetscCall(VecRestoreArrayRead(U, &u));

  PetscCall(MatSetValues(P, n, idx, n, idx, p, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (A != P) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// User callback for defining the event indicator functions
PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx)
{
  PetscInt           n = 0;
  const PetscScalar *u;
  AppCtx            *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  // first event -- on rank = 1%ctx.size
  if (Ctx->rank == 1 % Ctx->size) {
    if (t < 2.05) gval[n++] = 0.5 * (1 - PetscPowReal(t - 2.05, 12));
    else gval[n++] = 0.5;
  }

  // second event -- on last rank
  if (Ctx->rank == Ctx->size - 1) {
    if (t > 8.05) gval[n++] = 0.25 * (1 - PetscPowReal(t - 8.05, 12));
    else gval[n++] = 0.25;
  }

  // third event -- on rank-0
  PetscCall(VecGetArrayRead(U, &u));
  if (Ctx->rank == 0) gval[n++] = PetscRealPart(u[0]);
  PetscCall(VecRestoreArrayRead(U, &u));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Callback for evaluation times schedule 'user'. For 'full' = {u, v, w}, the output subvector *sub = {u, v/B(w)}
PetscErrorCode Handler_user(TS ts, PetscInt iunion, PetscInt iprivate, PetscReal t, Vec full, Vec *sub, void *ctx)
{
  PetscInt     m = 0;
  PetscScalar *x, *y;
  AppCtx      *Ctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  if (!Ctx->rank) m = 2;                                                                // the local size of 'sub'
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)full), m, PETSC_DETERMINE, sub)); // create the output vector

  PetscCall(VecGetArray(full, &x));
  PetscCall(VecGetArray(*sub, &y));
  if (m) {
    y[0] = x[0];
    y[1] = x[1] / (2 * x[2] - x[2] * x[2] / 15);
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
PetscErrorCode Print_schedule_vecs(TS ts, const char *name, PetscReal tol)
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

      if (type == 0) { // for 'user': check norm2 of the subvector
        PetscReal norm2 = 0;

        ref = 1.0;
        if (sols[i]) PetscCall(VecNorm(sols[i], NORM_2, &norm2));
        val = norm2;
      } else if (type == 1) { // for 'default': check u (1-st coordinate)
        ref = PetscSinReal(30 / (1 + PetscExpReal(-2 * times[i] + 10)));
        if (sols[i]) {
          PetscInt           m;
          const PetscScalar *x;

          PetscCall(VecGetLocalSize(sols[i], &m));
          PetscCall(VecGetArrayRead(sols[i], &x));
          if (m >= 3) val = x[0];
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
  testset:
    args: -ts_adapt_type basic -ts_rtol 5e-5 -ts_atol 5e-5

    test:
      suffix: 1
      args: -ts_type bdf -ts_bdf_order 4 -errtol2 0.01
      nsize: 4

    test:
      suffix: 2
      args: -ts_type rk -dir -1 -errtol 0.008
      nsize: 3

    test:
      suffix: 3
      args: -ts_type rosw -dir 1
      nsize: 2

    test:
      suffix: 4
      requires: double
      args: -ts_type bdf -ts_bdf_order 6 -nouser -range 5,9,5
      args: -ts_time_span 5,5.5,5.5000000001,5.5000000005,5.5000000007,6,7,8,9,10 -ts_adapt_dt_min_abs 3e-10 -errtol 1e-3
      args: -msg "Note, some evaluation time points are merged"
      nsize: 2
TEST*/
