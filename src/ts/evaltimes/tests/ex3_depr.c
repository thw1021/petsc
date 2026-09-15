#include <petscts.h>

#pragma GCC diagnostic warning "-Wdeprecated-declarations"

static char help[] = "Simple 3*3 linear problem with time span (TSEvaluationTimes) and events\n"
                     "x_dot =  0.2*y\n"
                     "y_dot = -0.2*x\n"
                     "z_dot =  1\n\n"

                     "Using time span (via old interface) with points [0.01, 0.21, 1.01, ..., 6.21, 6.99, 7.21,... 9.21]\n"
                     "                               plus the points: {3, 4, 4+D, 5-D, 5, 6-D, 6, 6+D} with user-defined 'D'\n\n"

                     "The following event functions are involved:\n"
                     "- two polynomial event functions on rank-0 and last-rank, with zeros: 1.05, 9.05, both terminating\n"
                     "- one event function on rank = '1%size', equal to sin(pi*t), zeros = 1,...,10\n"
                     "  on event at t==5, solution is changed: z = -z\n\n"

                     "TSSolve() is called consecutively 4 times: [t0 .. 1.05] [1.05 .. 9.05] [9.05 .. ts_max_time] [ts_max_time .. tmax]\n\n"

                     "Options:\n"
                     "-dir     d : zero-crossing direction for events: 0, 1, -1 (default = 0)\n"
                     "-flg       : additional output in Postevent (default = nothing)\n"
                     "-errtol  e : error tolerance, for printing 'pass/fail' for located events etc. (default = 1e-4)\n"
                     "-notspan   : disables adding the time span\n"
                     "-D       z : a (small) real number to define the additional time span points (default = 0.02)\n"
                     "-tmax    M : max time for the 4th TSSolve() (default = 12.0)\n";

#define MAX_NFUNC 5    // max event functions per rank
#define MAX_NEV   1000 // max zero crossings for each rank

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
  Mat         A;                // system matrix
} AppCtx;

PetscErrorCode RHSFunc(TS ts, PetscReal t, Vec U, Vec F, void *ctx);
PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscReal gval[], void *ctx);
PetscErrorCode Postevent(TS ts, PetscInt nev_zero, PetscInt evs_zero[], PetscReal t, Vec U, PetscBool fwd, void *ctx);
PetscErrorCode Fill_mat(PetscReal coeff, PetscMPIInt rnk, Mat A); // Fills the system matrix (3*3) on rank-0

int main(int argc, char **argv)
{
  TS                ts;
  Vec               sol;
  PetscInt          n, dir0 = 0, ind0 = 0;
  PetscInt          m; // local size of A
  PetscReal         D = 0.02;
  PetscInt          dir[MAX_NFUNC];
  PetscBool         term[MAX_NFUNC], notspan = PETSC_FALSE;
  PetscScalar      *x;
  PetscReal         tspan[28], tlast, maxtime, t0, tmax = 12.0;
  AppCtx            ctx;
  TSConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
  ctx.pi     = PetscAcosReal(-1.0);
  ctx.cnt    = 0;
  ctx.cntref = 0;
  ctx.flg    = PETSC_FALSE;
  ctx.errtol = 1e-4;

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

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dir", &dir0, NULL));           // desired zero-crossing direction for events
  PetscCall(PetscOptionsHasName(NULL, NULL, "-flg", &ctx.flg));             // flag for additional output
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-errtol", &ctx.errtol, NULL)); // error tolerance for located events
  PetscCall(PetscOptionsHasName(NULL, NULL, "-notspan", &notspan));         // disables adding the time span
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-D", &D, NULL));               // small number for time span
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tmax", &tmax, NULL));         // max time for the 4th TSSolve()

  // Set the event handling
  n = 0;               // event counter
  if (ctx.rank == 0) { // first event (with termination) -- on rank-0
    dir[n]    = dir0;
    term[n++] = PETSC_TRUE;
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

    for (PetscInt i = 1; i < MAX_NEV - 2; i++) {
      if (i % 2 == 1 && dir0 <= 0) ctx.ref[ctx.cntref++] = i;
      if (i % 2 == 0 && dir0 >= 0) ctx.ref[ctx.cntref++] = i;
    }
  }
  if (ctx.cntref > 0) PetscCall(PetscSortReal(ctx.cntref, ctx.ref));
  PetscCall(TSSetEventHandler(ts, n, dir, term, EventFunction, Postevent, &ctx));

  // Set the time span
  for (PetscInt i = 0; i < 10; i++) {
    tspan[2 * i]     = 0.01 + i + (i == 7 ? -0.02 : 0);
    tspan[2 * i + 1] = 0.21 + i;
  }
  tspan[20] = 3;
  tspan[21] = 4;
  tspan[22] = 4 + D;
  tspan[23] = 5 - D;
  tspan[24] = 5;
  tspan[25] = 6 - D;
  tspan[26] = 6;
  tspan[27] = 6 + D;
  PetscCall(PetscSortReal(28, tspan));
  if (!notspan) PetscCall(TSSetTimeSpan(ts, 28, tspan)); // using the deprecated time span interface (overrides t0, tmax)
  PetscCall(TSSetFromOptions(ts));                       // here, the time span may be overridden from options (in which case t0, tmax are overridden again)

  PetscCall(TSGetTime(ts, &t0)); // initial time
  PetscCall(MatCreateVecs(ctx.A, &sol, NULL));
  PetscCall(VecGetArray(sol, &x));
  if (!ctx.rank) { // initial conditions
    x[0] = 0;      // sin(0)
    x[1] = 1;      // cos(0)
    x[2] = t0;     // i.e. z(0) = 0
  }
  PetscCall(VecRestoreArray(sol, &x));

  // Solution
  PetscCall(TSSolve(ts, sol)); // normal exit at t=1.05 by event
  PetscCall(TSSolve(ts, sol)); // normal exit at t=9.05 by event
  PetscCall(TSSolve(ts, sol)); // normal exit at ts_max_time / timespan_last
  PetscCall(TSSetMaxTime(ts, tmax));
  PetscCall(TSSolve(ts, sol)); // normal exit at tmax
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

  // Time span: print [t] [Vec_z] ["pass"/"fail"], on rank-0
  {
    PetscInt         len1, len2;
    const PetscReal *times;
    Vec             *sols;

    // Use the deprecated time span interface:
    PetscCall(TSGetTimeSpan(ts, &len1, &times));
    PetscCall(TSGetTimeSpanSolutions(ts, &len2, &sols));
    PetscCheck(len1 >= len2, PETSC_COMM_WORLD, PETSC_ERR_COR, "Inconsistent sizes! %" PetscInt_FMT " %" PetscInt_FMT, len1, len2);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------\nTime span: %" PetscInt_FMT " point(s)\n", len2));
    for (PetscInt i = 0; i < len2; i++) {
      PetscScalar val = 0;        // found solution
      PetscScalar ref = times[i]; // reference solution (may be irrelevant if the actual recording started after times[0])

      if (times[i] >= 5 && t0 < 5 && dir0 <= 0) ref -= 10; // account for solution change at t = 5
      if (sols[i]) {
        PetscCall(VecGetArray(sols[i], &x));
        if (!ctx.rank) val = x[2]; // Vec_z
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%g\t%g\t%s\n", (double)times[i], (double)PetscRealPart(val), PetscAbsScalar(val - ref) < ctx.errtol ? "pass" : "fail"));
        PetscCall(VecRestoreArray(sols[i], &x));
      } else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%g\t%g\t%s\n", (double)times[i], (double)PetscRealPart(val), "***"));
    }
  }

  // Print the final time
  PetscCall(TSGetTime(ts, &tlast));
  PetscCall(TSGetMaxTime(ts, &maxtime));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final time = %g, max time = %g, %s\n", (double)tlast, (double)maxtime, PetscAbsReal(tlast - maxtime) < ctx.errtol ? "pass" : "fail"));

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
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] At t = %20.16g : %" PetscInt_FMT " events triggered, fvalues =", Ctx->rank, (double)t, nev_zero));
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
/*---------------------------------------------------------------------------------------------*/
/*TEST
  test:
    suffix: 1
    args: -ts_time_span 0.1,0.3,0.5,1,1.05,3,5,7,10 -ts_time_step 0.123
    nsize: 2

  test:
    suffix: 2
    args: -ts_type rk -ts_adapt_type basic -D 0.001
TEST*/
