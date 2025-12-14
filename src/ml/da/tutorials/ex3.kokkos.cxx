/* Data assimilation framework header (provides PetscDA) */
#include "petscda.h"
/* PETSc DMDA header (provides DM, DMDA functionality) */
#include <petscdmda.h>
#include <petscts.h>
#include <petscvec.h>
#include <Kokkos_Core.hpp>

static char help[] = "Shallow water dam-break test case with LETKF data assimilation.\n"
                     "Implements 1D shallow water equations with 2 DOF per grid point (h, hu).\n\n"
                     "Example usage:\n"
                     "  ./ex3.kokkos -steps 1000 -burn 100 -obs_freq 5 -obs_error 0.1 -da_view -ensemble_size 30\n\n";

/* Default parameter values */
#define DEFAULT_N             (2 * Q_NUM_OBSERVATIONS_MAX) /* 80 grid points */
#define DEFAULT_STEPS         1000
#define DEFAULT_BURN          100
#define DEFAULT_OBS_FREQ      5
#define DEFAULT_RANDOM_SEED   12345
#define DEFAULT_G             9.81
#define DEFAULT_DT            0.01
#define DEFAULT_OBS_ERROR_STD 0.1
#define DEFAULT_ENSEMBLE_SIZE 30
#define SPINUP_STEPS          0 /* No spinup for dam-break */

/* Minimum valid parameter values */
#define MIN_N              1
#define MIN_ENSEMBLE_SIZE  2
#define MIN_OBS_FREQ       1
#define PROGRESS_INTERVALS 10

typedef struct {
  DM        da; /* 1D periodic DM storing the shallow water state */
  PetscInt  n;  /* State dimension (number of grid points) */
  PetscReal g;  /* Gravitational constant */
  PetscReal dx; /* Grid spacing */
  PetscReal dt; /* Integration time step size */
  TS        ts; /* Reusable time stepper for efficiency */
} ShallowWaterCtx;

/*
  ShallowWaterRHS - Compute the right-hand side of the shallow water equations
  
  Uses a simple upwind scheme for flux calculation.
*/
static PetscErrorCode ShallowWaterRHS(TS ts, PetscReal t, Vec X, Vec F_vec, void *ctx)
{
  ShallowWaterCtx   *sw = (ShallowWaterCtx *)ctx;
  Vec                X_local;
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt           xs, xm, i;
  const PetscInt     ndof = 2; /* h and hu */

  PetscFunctionBeginUser;
  (void)ts;
  (void)t;

  PetscCall(DMDAGetCorners(sw->da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMGetLocalVector(sw->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(sw->da, X, INSERT_VALUES, X_local));
  PetscCall(VecGetArrayRead(X_local, &x));
  PetscCall(VecGetArray(F_vec, &f));

  /* Compute fluxes using a simple upwind scheme */
  for (i = xs; i < xs + xm; i++) {
    /* Extract state variables - DMDA handles periodic boundaries through DMGlobalToLocal */
    PetscReal h  = x[i * ndof];
    PetscReal hu = x[i * ndof + 1];
    PetscReal u  = (h > 1e-10) ? (hu / h) : 0.0;

    /* Access neighboring points - DMDA with periodic boundaries already sets correct halo values */
    PetscReal h_im1  = x[(i - 1) * ndof];
    PetscReal hu_im1 = x[(i - 1) * ndof + 1];
    PetscReal u_im1  = (h_im1 > 1e-10) ? (hu_im1 / h_im1) : 0.0;

    PetscReal h_ip1  = x[(i + 1) * ndof];
    PetscReal hu_ip1 = x[(i + 1) * ndof + 1];
    PetscReal u_ip1  = (h_ip1 > 1e-10) ? (hu_ip1 / h_ip1) : 0.0;

    /* Compute fluxes */
    PetscReal flux_h_left  = (u_im1 > 0) ? hu_im1 : hu;
    PetscReal flux_h_right = (u > 0) ? hu : hu_ip1;

    PetscReal flux_hu_left  = (u_im1 > 0) ? (hu_im1 * u_im1 + 0.5 * sw->g * h_im1 * h_im1) : (hu * u + 0.5 * sw->g * h * h);
    PetscReal flux_hu_right = (u > 0) ? (hu * u + 0.5 * sw->g * h * h) : (hu_ip1 * u_ip1 + 0.5 * sw->g * h_ip1 * h_ip1);

    /* Update RHS */
    f[i * ndof]     = -(flux_h_right - flux_h_left) / sw->dx;
    f[i * ndof + 1] = -(flux_hu_right - flux_hu_left) / sw->dx;
  }

  PetscCall(VecRestoreArrayRead(X_local, &x));
  PetscCall(VecRestoreArray(F_vec, &f));
  PetscCall(DMRestoreLocalVector(sw->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterContextCreate - Create and initialize a shallow water context with reusable TS object
*/
static PetscErrorCode ShallowWaterContextCreate(DM da, PetscInt n, PetscReal g, PetscReal dt, ShallowWaterCtx **ctx)
{
  ShallowWaterCtx *sw;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&sw));
  sw->da = da;
  sw->n  = n;
  sw->g  = g;
  sw->dx = 1.0 / n; /* Domain is [0, 1] */
  sw->dt = dt;

  PetscCall(TSCreate(PETSC_COMM_SELF, &sw->ts));
  PetscCall(TSSetProblemType(sw->ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(sw->ts, NULL, ShallowWaterRHS, sw));
  PetscCall(TSSetType(sw->ts, TSRK));
  PetscCall(TSRKSetType(sw->ts, TSRK4));
  PetscCall(TSSetTimeStep(sw->ts, dt));
  PetscCall(TSSetMaxSteps(sw->ts, 1));
  PetscCall(TSSetMaxTime(sw->ts, dt));
  PetscCall(TSSetExactFinalTime(sw->ts, TS_EXACTFINALTIME_MATCHSTEP));

  *ctx = sw;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterContextDestroy - Destroy a shallow water context and its TS object
*/
static PetscErrorCode ShallowWaterContextDestroy(ShallowWaterCtx **ctx)
{
  PetscFunctionBeginUser;
  if (!ctx || !*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSDestroy(&(*ctx)->ts));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterStep - Advance state vector one time step using shallow water dynamics
*/
static PetscErrorCode ShallowWaterStep(Vec x_in, Vec x_out, void *ctx)
{
  ShallowWaterCtx *sw = (ShallowWaterCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(TSSetTime(sw->ts, 0.0));
  if (x_in != x_out) PetscCall(VecCopy(x_in, x_out));
  PetscCall(TSSolve(sw->ts, x_out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ShallowWaterSolution - Dam-break initial condition
  
  Sets initial condition with discontinuity in water height at x=0.5
*/
static PetscErrorCode ShallowWaterSolution(PetscReal x, PetscReal *h, PetscReal *hu)
{
  PetscFunctionBeginUser;
  /* Dam-break initial condition */
  if (x < 0.5) {
    *h = 2.0; /* Water height (h) - higher on left side */
  } else {
    *h = 1.0; /* Water height (h) - lower on right side */
  }
  *hu = 0.0; /* Momentum (hu) - initially zero */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateObservationMatrix - Create observation matrix H for shallow water
  
  Observes water height (h) at every other grid point.
  This creates a sparse matrix mapping from full state (n*ndof) to observations.
  For n=80 grid points, we observe at points 0, 2, 4, ..., 78 giving nobs=40 observations.
*/
static PetscErrorCode CreateObservationMatrix(PetscInt n, PetscInt ndof, PetscInt nobs, Mat *H)
{
  PetscInt i;

  PetscFunctionBeginUser;
  PetscCheck(nobs == Q_NUM_OBSERVATIONS_MAX, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Number of observations (%" PetscInt_FMT ") must equal Q_NUM_OBSERVATIONS_MAX (%d)", nobs, Q_NUM_OBSERVATIONS_MAX);
  PetscCheck(n == 2 * Q_NUM_OBSERVATIONS_MAX, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Number of grid points (%" PetscInt_FMT ") must equal 2*Q_NUM_OBSERVATIONS_MAX (%d)", n, 2 * Q_NUM_OBSERVATIONS_MAX);

  /* Create observation matrix H (nobs × n*ndof) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nobs, n * ndof, 1, NULL, 0, NULL, H));

  /* Observe water height (h) at every other grid point */
  for (i = 0; i < nobs; i++) {
    PetscInt grid_point = 2 * i; /* Observe at points 0, 2, 4, ..., 78 */
    PetscCall(MatSetValue(*H, i, grid_point * ndof, 1.0, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateLocalizationMatrix - Create and initialize full localization matrix Q for shallow water
  
  Q is a (state_size × obs_size) matrix that specifies which observations affect each state variable.
  For full localization, each state variable uses all observations.
*/
static PetscErrorCode CreateLocalizationMatrix(PetscInt state_size, PetscInt obs_size, Mat *Q)
{
  PetscInt i, j;

  PetscFunctionBeginUser;
  /* Create Q matrix (state_size × obs_size)
     Each row will have obs_size non-zeros (all observations affect each state variable) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, state_size, obs_size, obs_size, NULL, 0, NULL, Q));

  /* Initialize with full localization: each state variable uses all observations */
  for (i = 0; i < state_size; i++) {
    for (j = 0; j < obs_size; j++) PetscCall(MatSetValue(*Q, i, j, 1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ValidateParameters - Validate input parameters and apply constraints
*/
static PetscErrorCode ValidateParameters(PetscInt *n, PetscInt *nobs, PetscInt *steps, PetscInt *burn, PetscInt *obs_freq, PetscInt *ensemble_size, PetscReal *dt, PetscReal *g, PetscReal *obs_error_std)
{
  PetscFunctionBeginUser;
  PetscCheck(*n > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "State dimension n must be positive, got %" PetscInt_FMT, *n);
  PetscCheck(*n == 2 * Q_NUM_OBSERVATIONS_MAX, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "For LETKF, n must be 2*Q_NUM_OBSERVATIONS_MAX (%d), got %" PetscInt_FMT, 2 * Q_NUM_OBSERVATIONS_MAX, *n);
  PetscCheck(*nobs == Q_NUM_OBSERVATIONS_MAX, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Number of observations must be Q_NUM_OBSERVATIONS_MAX (%d), got %" PetscInt_FMT, Q_NUM_OBSERVATIONS_MAX, *nobs);
  PetscCheck(*steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative, got %" PetscInt_FMT, *steps);
  PetscCheck(*ensemble_size >= MIN_ENSEMBLE_SIZE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least %d for meaningful statistics, got %" PetscInt_FMT, MIN_ENSEMBLE_SIZE, *ensemble_size);

  if (*obs_freq < MIN_OBS_FREQ) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency adjusted from %" PetscInt_FMT " to %d\n", *obs_freq, MIN_OBS_FREQ));
    *obs_freq = MIN_OBS_FREQ;
  }
  if (*obs_freq > *steps && *steps > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency (%" PetscInt_FMT ") > total steps (%" PetscInt_FMT "), no observations will be assimilated.\n", *obs_freq, *steps));
  if (*burn > *steps) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Burn-in steps (%" PetscInt_FMT ") exceeds total steps (%" PetscInt_FMT "), setting burn = steps\n", *burn, *steps));
    *burn = *steps;
  }

  PetscCheck(*dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step dt must be positive, got %g", (double)*dt);
  PetscCheck(*obs_error_std > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Observation error std must be positive, got %g", (double)*obs_error_std);
  PetscCheck(PetscIsNormalReal(*g), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Gravitational constant g must be a normal real number");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeRMSE - Compute root mean square error between two vectors
*/
static PetscErrorCode ComputeRMSE(Vec v1, Vec v2, Vec work, PetscInt n, PetscReal *rmse)
{
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(VecWAXPY(work, -1.0, v2, v1));
  PetscCall(VecNorm(work, NORM_2, &norm));
  *rmse = norm / PetscSqrtReal((PetscReal)n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  /* Configuration parameters */
  const PetscInt ndof          = 2; /* Degrees of freedom per grid point: h and hu */
  PetscInt       n             = DEFAULT_N;
  PetscInt       steps         = DEFAULT_STEPS;
  PetscInt       burn          = DEFAULT_BURN;
  PetscInt       obs_freq      = DEFAULT_OBS_FREQ;
  PetscInt       random_seed   = DEFAULT_RANDOM_SEED;
  PetscInt       ensemble_size = DEFAULT_ENSEMBLE_SIZE;
  PetscReal      g             = DEFAULT_G;
  PetscReal      dt            = DEFAULT_DT;
  PetscReal      obs_error_std = DEFAULT_OBS_ERROR_STD;

  /* PETSc objects */
  ShallowWaterCtx *sw_ctx = NULL;
  DM               da_state;
  PetscDA          daas;
  Vec              x0, x_mean, x_forecast;
  Vec              truth_state, rmse_work;
  Vec              observation, obs_noise, obs_error_var;
  PetscRandom      rng;
  Mat              Q = NULL; /* Localization matrix */
  Mat              H = NULL; /* Observation operator matrix */

  /* Statistics tracking */
  PetscReal rmse_forecast = 0.0, rmse_analysis = 0.0;
  PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscInt  n_stat_steps = 0;
  PetscInt  obs_count    = 0;
  PetscInt  step, progress_interval;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Kokkos initialization deferred to Phase 5 optimization */

  /* Parse command-line options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Shallow Water LETKF Example", NULL);
  PetscCall(PetscOptionsInt("-n", "Number of grid points", "", n, &n, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsInt("-burn", "Burn-in steps excluded from statistics", "", burn, &burn, NULL));
  PetscCall(PetscOptionsInt("-obs_freq", "Observation frequency", "", obs_freq, &obs_freq, NULL));
  PetscCall(PetscOptionsReal("-g", "Gravitational constant", "", g, &g, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", "", dt, &dt, NULL));
  PetscCall(PetscOptionsReal("-obs_error", "Observation error standard deviation", "", obs_error_std, &obs_error_std, NULL));
  PetscCall(PetscOptionsInt("-ensemble_size", "Number of ensemble members", "", ensemble_size, &ensemble_size, NULL));
  PetscCall(PetscOptionsInt("-random_seed", "Random seed for ensemble perturbations", "", random_seed, &random_seed, NULL));
  PetscOptionsEnd();

  /* LETKF constraint: nobs = Q_NUM_OBSERVATIONS_MAX, observe every other point */
  PetscInt nobs = Q_NUM_OBSERVATIONS_MAX;

  /* Validate and constrain parameters */
  PetscCall(ValidateParameters(&n, &nobs, &steps, &burn, &obs_freq, &ensemble_size, &dt, &g, &obs_error_std));

  /* Calculate progress reporting interval */
  progress_interval = (steps >= PROGRESS_INTERVALS) ? (steps / PROGRESS_INTERVALS) : 1;

  /* Create 1D periodic DM for state space with ndof=2 */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, n, ndof, 2, NULL, &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));

  /* Create shallow water context with reusable TS object */
  PetscCall(ShallowWaterContextCreate(da_state, n, g, dt, &sw_ctx));

  /* Initialize random number generator */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rng));
  PetscCall(PetscRandomSetSeed(rng, (unsigned long)random_seed));
  PetscCall(PetscRandomSetFromOptions(rng));
  PetscCall(PetscRandomSeed(rng));

  /* Initialize state vectors */
  PetscCall(DMCreateGlobalVector(da_state, &x0));

  /* Set dam-break initial condition */
  {
    PetscScalar *x_array;
    PetscInt     xs, xm, i;
    PetscCall(DMDAGetCorners(da_state, &xs, NULL, NULL, &xm, NULL, NULL));
    PetscCall(VecGetArray(x0, &x_array));
    for (i = xs; i < xs + xm; i++) {
      PetscReal x = ((PetscReal)i + 0.5) / n;
      PetscReal h, hu;
      PetscCall(ShallowWaterSolution(x, &h, &hu));
      x_array[i * ndof]     = h;
      x_array[i * ndof + 1] = hu;
    }
    PetscCall(VecRestoreArray(x0, &x_array));
  }

  /* Initialize truth trajectory */
  PetscCall(VecDuplicate(x0, &truth_state));
  PetscCall(VecCopy(x0, truth_state));
  PetscCall(VecDuplicate(x0, &rmse_work));

  /* No spinup for dam-break problem - start from discontinuous IC */
  if (SPINUP_STEPS > 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinning up truth for %d steps...\n", SPINUP_STEPS));
    for (int k = 0; k < SPINUP_STEPS; k++) PetscCall(ShallowWaterStep(truth_state, truth_state, sw_ctx));
  }

  /* Initialize observation vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &observation));
  PetscCall(VecSetSizes(observation, PETSC_DECIDE, nobs));
  PetscCall(VecSetFromOptions(observation));
  PetscCall(VecDuplicate(observation, &obs_noise));
  PetscCall(VecDuplicate(observation, &obs_error_var));
  PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));

  /* Initialize ensemble statistics vectors */
  PetscCall(VecDuplicate(x0, &x_mean));
  PetscCall(VecDuplicate(x0, &x_forecast));

  /* Create and configure PetscDA for ensemble data assimilation */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &daas));
  PetscCall(PetscDASetType(daas, PETSCDALETKF));                   /* Set LETKF type */
  PetscCall(PetscDASetSizes(daas, n * ndof, nobs, ensemble_size)); /* State size includes ndof */
  PetscCall(PetscDASetNDOF(daas, ndof));                           /* Set number of degrees of freedom per grid point */
  PetscCall(PetscDASetFromOptions(daas));
  PetscCall(PetscDASetUp(daas));
  PetscCall(PetscDAViewFromOptions(daas, NULL, "-da_view"));
  PetscCall(PetscDASetObsErrorVariance(daas, obs_error_var));

  /* Create and set localization matrix Q */
  PetscCall(CreateLocalizationMatrix(n * ndof, nobs, &Q));
  PetscCall(PetscDALETKFSetLocalization(daas, Q));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization matrix Q created: %dx%d, full localization (all weights = 1.0)\n", n * ndof, nobs));

  /* Create observation matrix H (nobs=40, observing h at every other grid point) */
  PetscCall(CreateObservationMatrix(n, ndof, nobs, &H));

  /* Initialize ensemble members */
  PetscCall(InitializeEnsemble(daas, x0, ensemble_size, obs_error_std, rng));

  /* Print configuration summary */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Shallow Water LETKF Example\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "============================\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "  State dimension       : %" PetscInt_FMT " (%" PetscInt_FMT " grid points × %d DOF)\n"
                        "  Observation dimension : %" PetscInt_FMT "\n"
                        "  Ensemble size         : %" PetscInt_FMT "\n"
                        "  Gravitational const   : %.4f\n"
                        "  Time step (dt)        : %.4f\n"
                        "  Total steps           : %" PetscInt_FMT "\n"
                        "  Burn-in steps         : %" PetscInt_FMT "\n"
                        "  Observation frequency : %" PetscInt_FMT "\n"
                        "  Observation noise std : %.3f\n"
                        "  Random seed           : %" PetscInt_FMT "\n"
                        "  Localization          : Full (40 obs per vertex)\n\n",
                        n * ndof, n, ndof, nobs, ensemble_size, (double)g, (double)dt, steps, burn, obs_freq, (double)obs_error_std, random_seed));

  /* Main assimilation cycle: forecast and analysis steps */
  for (step = 0; step <= steps; step++) {
    PetscReal time = step * dt;

    /* Forecast step: compute ensemble mean and forecast RMSE */
    PetscCall(PetscDAComputeEnsembleMean(daas, x_mean));
    PetscCall(VecCopy(x_mean, x_forecast));
    PetscCall(ComputeRMSE(x_forecast, truth_state, rmse_work, n * ndof, &rmse_forecast));
    rmse_analysis = rmse_forecast;

    /* Analysis step: assimilate observations when available */
    if (step % obs_freq == 0 && step > 0) {
      /* Generate synthetic noisy observations from truth using observation matrix H */
      Vec truth_obs;
      PetscCall(VecCreate(PETSC_COMM_WORLD, &truth_obs));
      PetscCall(VecSetSizes(truth_obs, PETSC_DECIDE, nobs));
      PetscCall(VecSetFromOptions(truth_obs));

      /* Apply H to get observations: y = H*x_true */
      PetscCall(MatMult(H, truth_state, truth_obs));

      /* Add observation noise */
      PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
      PetscCall(VecWAXPY(observation, 1.0, obs_noise, truth_obs));

      /* Perform LETKF analysis with observation matrix H */
      PetscCall(PetscDAAnalysis(daas, observation, H));

      /* Clean up */
      PetscCall(VecDestroy(&truth_obs));

      /* Compute analysis RMSE */
      PetscCall(PetscDAComputeEnsembleMean(daas, x_mean));
      PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, n * ndof, &rmse_analysis));
      obs_count++;
    }

    /* Accumulate statistics after burn-in period */
    if (step >= burn) {
      sum_rmse_forecast += rmse_forecast;
      sum_rmse_analysis += rmse_analysis;
      n_stat_steps++;
    }

    /* Progress reporting */
    if ((step % progress_interval == 0) || (step == steps) || (step == 0))
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f%s\n", step, (double)time, (double)rmse_forecast, (double)rmse_analysis, (step < burn) ? " [burn-in]" : ""));

    /* Propagate ensemble and truth trajectory */
    if (step < steps) {
      PetscCall(PetscDAApplyModel(daas, ShallowWaterStep, sw_ctx));
      PetscCall(ShallowWaterStep(truth_state, truth_state, sw_ctx));
    }
  }

  /* Report final statistics */
  if (n_stat_steps > 0) {
    PetscReal avg_rmse_forecast = sum_rmse_forecast / n_stat_steps;
    PetscReal avg_rmse_analysis = sum_rmse_analysis / n_stat_steps;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStatistics (%" PetscInt_FMT " post-burn-in steps):\n", n_stat_steps));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==================================================\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (forecast) : %.5f\n", (double)avg_rmse_forecast));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (analysis) : %.5f\n", (double)avg_rmse_analysis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Observations used    : %" PetscInt_FMT "\n\n", obs_count));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nWarning: No post-burn-in statistics collected (burn >= steps)\n\n"));
  }

  /* Cleanup */
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Q));
  PetscCall(VecDestroy(&x_forecast));
  PetscCall(VecDestroy(&x_mean));
  PetscCall(VecDestroy(&obs_error_var));
  PetscCall(VecDestroy(&obs_noise));
  PetscCall(VecDestroy(&observation));
  PetscCall(VecDestroy(&rmse_work));
  PetscCall(VecDestroy(&truth_state));
  PetscCall(VecDestroy(&x0));
  PetscCall(PetscDADestroy(&daas));
  PetscCall(DMDestroy(&da_state));
  PetscCall(ShallowWaterContextDestroy(&sw_ctx));
  PetscCall(PetscRandomDestroy(&rng));

  /* Kokkos finalization deferred to Phase 5 optimization */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: !complex kokkos_kernels
    diff_args: -j
    args: -steps 100 -burn 10 -obs_freq 5 -obs_error 0.1 -da_view -ensemble_size 30 -da_sqrt_type cholesky

  test:
    suffix: eigen
    diff_args: -j
    requires: !complex kokkos_kernels
    args: -steps 100 -burn 10 -obs_freq 5 -obs_error 0.1 -ensemble_size 30 -da_sqrt_type eigen

TEST*/
