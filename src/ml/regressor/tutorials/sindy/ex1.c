static char help[] = "Discovers Lorenz dynamics from simulated data using PETSc TS and PetscRegressor LASSO.\n\n"
                     "This is a SINDy (Sparse Identification of Nonlinear Dynamics) example. The phases are:\n"
                     "  1. Simulate the Lorenz system with TS to gather state snapshots U.\n"
                     "  2. Build a polynomial library Theta(U) of candidate terms and estimate U' by finite differences.\n"
                     "  3. For each state component, solve the sparse regression Theta * xi_j = U'_{:,j} with PetscRegressor (LASSO).\n"
                     "  4. Optionally, integrate the discovered system with another TS and print the predicted final state.\n\n";

#include <petscts.h>
#include <petscregressor.h>

#define SINDY_NSTATE   3  /* number of state variables (x, y, z) */
#define SINDY_NFEATURE 10 /* polynomial library size: order 2 in 3 variables */

typedef struct {
  PetscReal sigma, rho, beta; /* truth Lorenz parameters used by LorenzRHS */
  Mat       U;                /* snapshot matrix (nsteps+1) x SINDY_NSTATE, seq-dense */
  /* predict-phase fields driven by SindyRHS */
  Mat Xi;        /* coefficient matrix SINDY_NFEATURE x SINDY_NSTATE, seq-dense */
  Vec theta_row; /* length-SINDY_NFEATURE scratch vector */
} AppCtx;

/* Evaluate the polynomial library row {1, x, y, z, x^2, xy, xz, y^2, yz, z^2} at a single state. */
static void BuildLibraryRow(const PetscScalar x[SINDY_NSTATE], PetscScalar row[SINDY_NFEATURE])
{
  PetscScalar xv = x[0], yv = x[1], zv = x[2];

  row[0] = 1.0;
  row[1] = xv;
  row[2] = yv;
  row[3] = zv;
  row[4] = xv * xv;
  row[5] = xv * yv;
  row[6] = xv * zv;
  row[7] = yv * yv;
  row[8] = yv * zv;
  row[9] = zv * zv;
}

static PetscErrorCode LorenzRHS(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  const PetscScalar *x;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(F, &f));
  f[0] = user->sigma * (x[1] - x[0]);
  f[1] = x[0] * (user->rho - x[2]) - x[1];
  f[2] = x[0] * x[1] - user->beta * x[2];
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Stash the solution at each accepted step into row `step` of ctx->U. */
static PetscErrorCode SnapshotMonitor(TS ts, PetscInt step, PetscReal t, Vec X, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  const PetscScalar *x;
  PetscInt           cols[SINDY_NSTATE] = {0, 1, 2};

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(MatSetValues(user->U, 1, &step, SINDY_NSTATE, cols, x, INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fill Theta from U using BuildLibraryRow. Both matrices are column-major seq-dense. */
static PetscErrorCode BuildLibrary(Mat U, Mat Theta)
{
  const PetscScalar *u;
  PetscScalar       *th;
  PetscInt           m;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(U, &m, NULL));
  PetscCall(MatDenseGetArrayRead(U, &u));
  PetscCall(MatDenseGetArrayWrite(Theta, &th));
  for (PetscInt i = 0; i < m; i++) {
    PetscScalar state[SINDY_NSTATE];
    PetscScalar feat[SINDY_NFEATURE];

    state[0] = u[0 * m + i];
    state[1] = u[1 * m + i];
    state[2] = u[2 * m + i];
    BuildLibraryRow(state, feat);
    for (PetscInt k = 0; k < SINDY_NFEATURE; k++) th[k * m + i] = feat[k];
  }
  PetscCall(MatDenseRestoreArrayWrite(Theta, &th));
  PetscCall(MatDenseRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 2nd-order central interior, 1st-order forward/backward at endpoints (matches numpy.gradient). */
static PetscErrorCode FiniteDifferenceDerivatives(Mat U, Mat Uprime, PetscReal dt)
{
  const PetscScalar *u;
  PetscScalar       *up;
  PetscInt           m;
  PetscReal          inv_dt  = 1.0 / dt;
  PetscReal          inv_2dt = 0.5 / dt;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(U, &m, NULL));
  PetscCheck(m >= 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Need at least 2 snapshots for finite differences");
  PetscCall(MatDenseGetArrayRead(U, &u));
  PetscCall(MatDenseGetArrayWrite(Uprime, &up));
  for (PetscInt j = 0; j < SINDY_NSTATE; j++) {
    const PetscScalar *col_u  = u + j * m;
    PetscScalar       *col_up = up + j * m;

    col_up[0]     = (col_u[1] - col_u[0]) * inv_dt;
    col_up[m - 1] = (col_u[m - 1] - col_u[m - 2]) * inv_dt;
    for (PetscInt i = 1; i < m - 1; i++) col_up[i] = (col_u[i + 1] - col_u[i - 1]) * inv_2dt;
  }
  PetscCall(MatDenseRestoreArrayWrite(Uprime, &up));
  PetscCall(MatDenseRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* RHS for the discovered system: F = Xi^T * theta(X). */
static PetscErrorCode SindyRHS(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  const PetscScalar *x;
  PetscScalar       *th;
  PetscScalar        state[SINDY_NSTATE];
  PetscScalar        feat[SINDY_NFEATURE];

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  for (PetscInt k = 0; k < SINDY_NSTATE; k++) state[k] = x[k];
  PetscCall(VecRestoreArrayRead(X, &x));

  BuildLibraryRow(state, feat);
  PetscCall(VecGetArray(user->theta_row, &th));
  for (PetscInt k = 0; k < SINDY_NFEATURE; k++) th[k] = feat[k];
  PetscCall(VecRestoreArray(user->theta_row, &th));

  PetscCall(MatMultTranspose(user->Xi, user->theta_row, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print each discovered equation, suppressing terms whose coefficient magnitude is below display_thresh.
   The threshold sits above the LASSO noise floor and below the smallest true Lorenz coefficient,
   making the symbolic output robust to per-run wobble in LASSO solutions. */
static PetscErrorCode ViewDiscoveredEquations(Mat Xi, PetscReal display_thresh, PetscViewer viewer)
{
  static const char *const state_names[SINDY_NSTATE]     = {"dx/dt", "dy/dt", "dz/dt"};
  static const char *const feature_names[SINDY_NFEATURE] = {"1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"};
  const PetscScalar       *xi;
  PetscInt                 m;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(Xi, &m, NULL));
  PetscCheck(m == SINDY_NFEATURE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Xi has unexpected row count %" PetscInt_FMT, m);
  PetscCall(MatDenseGetArrayRead(Xi, &xi));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Discovered system (terms with |coef| > %.3g):\n", (double)display_thresh));
  for (PetscInt j = 0; j < SINDY_NSTATE; j++) {
    const PetscScalar *col   = xi + j * SINDY_NFEATURE;
    PetscBool          first = PETSC_TRUE;

    PetscCall(PetscViewerASCIIPrintf(viewer, "  %s =", state_names[j]));
    for (PetscInt k = 0; k < SINDY_NFEATURE; k++) {
      PetscReal v = PetscRealPart(col[k]);

      if (PetscAbsReal(v) < display_thresh) continue;
      if (first) {
        if (k == 0) PetscCall(PetscViewerASCIIPrintf(viewer, " %.3f", (double)v));
        else PetscCall(PetscViewerASCIIPrintf(viewer, " %.3f %s", (double)v, feature_names[k]));
        first = PETSC_FALSE;
      } else {
        const char *sep  = (v >= 0.0) ? " + " : " - ";
        PetscReal   absv = PetscAbsReal(v);

        if (k == 0) PetscCall(PetscViewerASCIIPrintf(viewer, "%s%.3f", sep, (double)absv));
        else PetscCall(PetscViewerASCIIPrintf(viewer, "%s%.3f %s", sep, (double)absv, feature_names[k]));
      }
    }
    if (first) PetscCall(PetscViewerASCIIPrintf(viewer, " 0"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  PetscCall(MatDenseRestoreArrayRead(Xi, &xi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx         ctx;
  TS             ts;
  Vec            X;
  Mat            Theta, Uprime, Xi;
  PetscRegressor regressor;
  PetscReal      sindy_dt       = 0.01;
  PetscReal      sindy_tmax     = 10.0;
  PetscReal      predict_tmax   = 5.0;
  PetscReal      lambda         = 20.0;
  PetscReal      display_thresh = 0.05;
  PetscReal      ic_x = -8.0, ic_y = 8.0, ic_z = 27.0;
  PetscBool      skip_predict  = PETSC_FALSE;
  PetscBool      view_xi_raw   = PETSC_FALSE;
  PetscBool      fit_intercept = PETSC_FALSE;
  PetscInt       nsteps, actual_steps;
  PetscScalar   *x_ptr;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  ctx.sigma     = 10.0;
  ctx.rho       = 28.0;
  ctx.beta      = 8.0 / 3.0;
  ctx.U         = NULL;
  ctx.Xi        = NULL;
  ctx.theta_row = NULL;

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Lorenz/SINDy tutorial options:", "PetscRegressor");
  PetscCall(PetscOptionsReal("-lorenz_sigma", "Lorenz sigma (truth)", "ex1.c", ctx.sigma, &ctx.sigma, NULL));
  PetscCall(PetscOptionsReal("-lorenz_rho", "Lorenz rho (truth)", "ex1.c", ctx.rho, &ctx.rho, NULL));
  PetscCall(PetscOptionsReal("-lorenz_beta", "Lorenz beta (truth)", "ex1.c", ctx.beta, &ctx.beta, NULL));
  PetscCall(PetscOptionsReal("-sindy_ic_x", "Initial condition x", "ex1.c", ic_x, &ic_x, NULL));
  PetscCall(PetscOptionsReal("-sindy_ic_y", "Initial condition y", "ex1.c", ic_y, &ic_y, NULL));
  PetscCall(PetscOptionsReal("-sindy_ic_z", "Initial condition z", "ex1.c", ic_z, &ic_z, NULL));
  PetscCall(PetscOptionsReal("-sindy_dt", "Data-generation time step", "ex1.c", sindy_dt, &sindy_dt, NULL));
  PetscCall(PetscOptionsReal("-sindy_tmax", "Data-generation horizon", "ex1.c", sindy_tmax, &sindy_tmax, NULL));
  PetscCall(PetscOptionsReal("-sindy_predict_tmax", "Predict-phase horizon", "ex1.c", predict_tmax, &predict_tmax, NULL));
  PetscCall(PetscOptionsBool("-sindy_skip_predict", "Skip the predict phase", "ex1.c", skip_predict, &skip_predict, NULL));
  PetscCall(PetscOptionsReal("-sindy_lambda", "LASSO regularizer weight", "ex1.c", lambda, &lambda, NULL));
  PetscCall(PetscOptionsReal("-sindy_display_threshold", "Pretty-print cutoff for discovered coefficients", "ex1.c", display_thresh, &display_thresh, NULL));
  PetscCall(PetscOptionsBool("-sindy_view_xi_raw", "Debug: full MatView of Xi", "ex1.c", view_xi_raw, &view_xi_raw, NULL));
  PetscCall(PetscOptionsBool("-sindy_fit_intercept", "Fit a separate intercept (Theta already contains a constant column; default off)", "ex1.c", fit_intercept, &fit_intercept, NULL));
  PetscOptionsEnd();

  nsteps = (PetscInt)PetscFloorReal(sindy_tmax / sindy_dt + 0.5);
  PetscCheck(nsteps >= 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Need at least 2 data-generation steps (got %" PetscInt_FMT ")", nsteps);

  /* ---- Phase 1: data generation ---- */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, nsteps + 1, SINDY_NSTATE, NULL, &ctx.U));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, SINDY_NSTATE, &X));
  PetscCall(VecGetArray(X, &x_ptr));
  x_ptr[0] = ic_x;
  x_ptr[1] = ic_y;
  x_ptr[2] = ic_z;
  PetscCall(VecRestoreArray(X, &x_ptr));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSRKSetType(ts, TSRK4));
  PetscCall(TSSetRHSFunction(ts, NULL, LorenzRHS, &ctx));
  PetscCall(TSSetTimeStep(ts, sindy_dt));
  PetscCall(TSSetMaxTime(ts, sindy_tmax));
  PetscCall(TSSetMaxSteps(ts, nsteps));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSMonitorSet(ts, SnapshotMonitor, &ctx, NULL));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts, X));
  PetscCall(TSGetStepNumber(ts, &actual_steps));
  PetscCheck(actual_steps == nsteps, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected %" PetscInt_FMT " timesteps but TS performed %" PetscInt_FMT, nsteps, actual_steps);
  PetscCall(MatAssemblyBegin(ctx.U, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx.U, MAT_FINAL_ASSEMBLY));
  PetscCall(TSDestroy(&ts));

  /* ---- Phase 2: build Theta and U', fit LASSO column by column ---- */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, nsteps + 1, SINDY_NFEATURE, NULL, &Theta));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, nsteps + 1, SINDY_NSTATE, NULL, &Uprime));
  PetscCall(BuildLibrary(ctx.U, Theta));
  PetscCall(FiniteDifferenceDerivatives(ctx.U, Uprime, sindy_dt));
  PetscCall(MatAssemblyBegin(Theta, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Theta, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Uprime, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Uprime, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, SINDY_NFEATURE, SINDY_NSTATE, NULL, &Xi));
  PetscCall(MatZeroEntries(Xi));

  PetscCall(PetscRegressorCreate(PETSC_COMM_WORLD, &regressor));
  PetscCall(PetscRegressorSetType(regressor, PETSCREGRESSORLINEAR));
  PetscCall(PetscRegressorLinearSetType(regressor, REGRESSOR_LINEAR_LASSO));
  PetscCall(PetscRegressorLinearSetFitIntercept(regressor, fit_intercept));
  PetscCall(PetscRegressorSetRegularizerWeight(regressor, lambda));
  PetscCall(PetscRegressorSetFromOptions(regressor));

  for (PetscInt j = 0; j < SINDY_NSTATE; j++) {
    Vec y_j, xi_j, Xi_col;

    PetscCall(MatDenseGetColumnVecRead(Uprime, j, &y_j));
    PetscCall(PetscRegressorFit(regressor, Theta, y_j));
    PetscCall(PetscRegressorLinearGetCoefficients(regressor, &xi_j));
    PetscCall(MatDenseGetColumnVecWrite(Xi, j, &Xi_col));
    PetscCall(VecCopy(xi_j, Xi_col));
    PetscCall(MatDenseRestoreColumnVecWrite(Xi, j, &Xi_col));
    PetscCall(MatDenseRestoreColumnVecRead(Uprime, j, &y_j));
    /* Reset clears Tao/Vec/Mat state so the next Fit() rebuilds with the new target.
       Without it, the linear impl reuses stale setup state from the prior column. */
    PetscCall(PetscRegressorReset(regressor));
  }
  PetscCall(MatAssemblyBegin(Xi, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Xi, MAT_FINAL_ASSEMBLY));

  PetscCall(ViewDiscoveredEquations(Xi, display_thresh, PETSC_VIEWER_STDOUT_WORLD));
  if (view_xi_raw) PetscCall(MatView(Xi, PETSC_VIEWER_STDOUT_WORLD));

  /* ---- Phase 3: predict ---- */
  if (!skip_predict) {
    TS        ts_pred;
    Vec       X_pred;
    PetscReal tf;
    PetscInt  steps_pred;

    ctx.Xi = Xi;
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, SINDY_NFEATURE, &ctx.theta_row));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, SINDY_NSTATE, &X_pred));
    PetscCall(VecGetArray(X_pred, &x_ptr));
    x_ptr[0] = ic_x;
    x_ptr[1] = ic_y;
    x_ptr[2] = ic_z;
    PetscCall(VecRestoreArray(X_pred, &x_ptr));

    PetscCall(TSCreate(PETSC_COMM_WORLD, &ts_pred));
    PetscCall(TSSetType(ts_pred, TSRK));
    PetscCall(TSRKSetType(ts_pred, TSRK4));
    PetscCall(TSSetRHSFunction(ts_pred, NULL, SindyRHS, &ctx));
    PetscCall(TSSetTimeStep(ts_pred, sindy_dt));
    PetscCall(TSSetMaxTime(ts_pred, predict_tmax));
    PetscCall(TSSetMaxSteps(ts_pred, (PetscInt)PetscFloorReal(predict_tmax / sindy_dt + 0.5)));
    PetscCall(TSSetExactFinalTime(ts_pred, TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetSolution(ts_pred, X_pred));
    /* No TSSetFromOptions: the predict TS deliberately ignores top-level -ts_* options so the
       integrator settings come from this tutorial alone. Use -sindy_skip_predict to disable. */

    PetscCall(TSSolve(ts_pred, X_pred));
    PetscCall(TSGetSolveTime(ts_pred, &tf));
    PetscCall(TSGetStepNumber(ts_pred, &steps_pred));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Predict phase: %" PetscInt_FMT " steps to t = %.3f\n", steps_pred, (double)tf));
    PetscCall(VecDestroy(&X_pred));
    PetscCall(VecDestroy(&ctx.theta_row));
    PetscCall(TSDestroy(&ts_pred));
  }

  PetscCall(PetscRegressorDestroy(&regressor));
  PetscCall(MatDestroy(&Xi));
  PetscCall(MatDestroy(&Uprime));
  PetscCall(MatDestroy(&Theta));
  PetscCall(MatDestroy(&ctx.U));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
     suffix: lasso
     nsize: 1
     args: -ts_type rk -ts_rk_type 4 -sindy_dt 0.01 -sindy_tmax 10.0 -sindy_lambda 20.0 -sindy_display_threshold 0.05 -sindy_predict_tmax 5.0

TEST*/
