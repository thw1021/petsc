/* Data assimilation framework header (provides PetscDA) */
#include "petscda.h"
/* PETSc DMDA header (provides DM, DMDA functionality) */
#include <petscdmda.h>
#include <petscts.h>
#include <petscvec.h>
#include <Kokkos_Core.hpp>

static char help[] = "Deterministic LETKF example for the Lorenz-96 model. See "
                     "Algorithm 6.4 of \n"
                     "Asch, Bocquet, and Nodet (2016) \"Data Assimilation\" "
                     "(SIAM, doi:10.1137/1.9781611974546).\n\n"
                     "Example usage:\n"
                     "  ./ex2.kokkos -steps 105000 -burn 5000 -obs_freq 1 -obs_error 1 -da_view -ensemble_size 30\n"
                     "  Expected result: Similar to ETKF with full localization\n\n";

/* Default parameter values */
#define DEFAULT_N             40
#define DEFAULT_STEPS         200
#define DEFAULT_BURN          100
#define DEFAULT_OBS_FREQ      5
#define DEFAULT_RANDOM_SEED   12345
#define DEFAULT_F             8.0
#define DEFAULT_DT            0.05
#define DEFAULT_OBS_ERROR_STD 1.0
#define DEFAULT_ENSEMBLE_SIZE 30
#define SPINUP_STEPS          2000

/* Minimum valid parameter values */
#define MIN_N              1
#define MIN_ENSEMBLE_SIZE  2
#define MIN_OBS_FREQ       1
#define PROGRESS_INTERVALS 10

/* LETKF constraint: Fixed number of observations per vertex */
#define Q_NUM_OBSERVATIONS_MAX 40

typedef struct {
  DM        da;   /* 1D periodic DM storing the Lorenz-96 state */
  PetscInt  n;    /* State dimension (number of grid points) */
  PetscReal F;    /* Constant forcing term in the Lorenz-96 equations */
  PetscReal dt;   /* Integration time step size */
  TS        ts;   /* Reusable time stepper for efficiency */
  PetscReal time; /* Current simulation time */
  PetscInt  step; /* Current simulation step */
} Lorenz96Ctx;

/*
  Lorenz96RHS - Compute the right-hand side of the Lorenz-96 equations
*/
static PetscErrorCode Lorenz96RHS(TS ts, PetscReal t, Vec X, Vec F_vec, void *ctx)
{
  Lorenz96Ctx       *l95 = (Lorenz96Ctx *)ctx;
  Vec                X_local;
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt           xs, xm, i;

  PetscFunctionBeginUser;
  (void)ts;
  (void)t;

  PetscCall(DMDAGetCorners(l95->da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMGetLocalVector(l95->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecGetArray(l95->da, F_vec, &f));

  for (i = xs; i < xs + xm; i++) f[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + l95->F;

  PetscCall(DMDAVecRestoreArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecRestoreArray(l95->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(l95->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96ContextCreate - Create and initialize a Lorenz96 context with reusable TS object
*/
static PetscErrorCode Lorenz96ContextCreate(DM da, PetscInt n, PetscReal F, PetscReal dt, Lorenz96Ctx **ctx)
{
  Lorenz96Ctx *l95;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&l95));
  l95->da   = da;
  l95->n    = n;
  l95->F    = F;
  l95->dt   = dt;
  l95->time = 0.0;
  l95->step = 0;

  PetscCall(TSCreate(PetscObjectComm((PetscObject)da), &l95->ts));
  PetscCall(TSSetProblemType(l95->ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(l95->ts, NULL, Lorenz96RHS, l95));
  PetscCall(TSSetType(l95->ts, TSRK));
  PetscCall(TSRKSetType(l95->ts, TSRK4));
  PetscCall(TSSetTimeStep(l95->ts, dt));
  PetscCall(TSSetMaxSteps(l95->ts, 1));
  PetscCall(TSSetMaxTime(l95->ts, dt));
  PetscCall(TSSetExactFinalTime(l95->ts, TS_EXACTFINALTIME_MATCHSTEP));

  *ctx = l95;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96ContextDestroy - Destroy a Lorenz96 context and its TS object
*/
static PetscErrorCode Lorenz96ContextDestroy(Lorenz96Ctx **ctx)
{
  PetscFunctionBeginUser;
  if (!ctx || !*ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSDestroy(&(*ctx)->ts));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96Step - Advance state vector one time step using Lorenz-96 dynamics
*/
static PetscErrorCode Lorenz96Step(Vec x_in, Vec x_out, void *ctx)
{
  Lorenz96Ctx *l95 = (Lorenz96Ctx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(TSSetTime(l95->ts, l95->time));
  PetscCall(TSSetMaxSteps(l95->ts, l95->step + 1));
  PetscCall(TSSetStepNumber(l95->ts, l95->step));
  PetscCall(TSSetTimeStep(l95->ts, l95->dt));
  PetscCall(TSSetMaxTime(l95->ts, l95->time + l95->dt));

  if (x_in != x_out) PetscCall(VecCopy(x_in, x_out));
  PetscCall(TSSolve(l95->ts, x_out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  CreateIdentityObservationMatrix - Create identity observation matrix H for Lorenz-96

  For the fully observed case, H is an nxn identity matrix representing y = H*x where
  each observation corresponds directly to a state variable.

  Input Parameter:
. n - State dimension (number of grid points)

  Output Parameter:
. H - Identity observation matrix (n x n), sparse AIJ format
*/
static PetscErrorCode CreateIdentityObservationMatrix(PetscInt n, Mat *H)
{
  PetscInt i;

  PetscFunctionBeginUser;
  /* Create identity observation matrix H (n x n) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 0, NULL, H));

  /* Set diagonal entries to 1.0 for identity mapping */
  for (i = 0; i < n; i++) PetscCall(MatSetValue(*H, i, i, 1.0, INSERT_VALUES));

  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ValidateParameters - Validate input parameters and apply constraints
*/
static PetscErrorCode ValidateParameters(PetscInt *n, PetscInt *steps, PetscInt *burn, PetscInt *obs_freq, PetscInt *ensemble_size, PetscReal *dt, PetscReal *F, PetscReal *obs_error_std)
{
  PetscFunctionBeginUser;
  PetscCheck(*n > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "State dimension n must be positive, got %" PetscInt_FMT, *n);
  PetscCheck(*steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative, got %" PetscInt_FMT, *steps);
  PetscCheck(*ensemble_size >= MIN_ENSEMBLE_SIZE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least %" PetscInt_FMT " for meaningful statistics, got %" PetscInt_FMT, (PetscInt)MIN_ENSEMBLE_SIZE, *ensemble_size);

  /* LETKF constraint: n must equal Q_NUM_OBSERVATIONS_MAX for fully observed case */
  PetscCheck(*n == Q_NUM_OBSERVATIONS_MAX, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "For fully observed case, n (%" PetscInt_FMT ") must equal Q_NUM_OBSERVATIONS_MAX (%" PetscInt_FMT ")", *n, (PetscInt)Q_NUM_OBSERVATIONS_MAX);

  if (*obs_freq < MIN_OBS_FREQ) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency adjusted from %" PetscInt_FMT " to %" PetscInt_FMT "\n", *obs_freq, (PetscInt)MIN_OBS_FREQ));
    *obs_freq = MIN_OBS_FREQ;
  }
  if (*obs_freq > *steps && *steps > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency (%" PetscInt_FMT ") > total steps (%" PetscInt_FMT "), no observations will be assimilated.\n", *obs_freq, *steps));
  if (*burn > *steps) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Burn-in steps (%" PetscInt_FMT ") exceeds total steps (%" PetscInt_FMT "), setting burn = steps\n", *burn, *steps));
    *burn = *steps;
  }

  PetscCheck(*dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step dt must be positive, got %g", (double)*dt);
  PetscCheck(*obs_error_std > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Observation error std must be positive, got %g", (double)*obs_error_std);
  PetscCheck(PetscIsNormalReal(*F), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Forcing parameter F must be a normal real number");
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

/*
  CreateLocalizationMatrix - Create and initialize full localization matrix Q
  
  For the fully observed case (n = Q_NUM_OBSERVATIONS_MAX), Q is a dense nxn
  matrix with all entries = 1.0, meaning each vertex uses all observations.
*/
static PetscErrorCode CreateLocalizationMatrix(PetscInt n, Mat *Q)
{
  PetscInt i, j;

  PetscFunctionBeginUser;
  /* Verify constraint */
  PetscCheck(n == Q_NUM_OBSERVATIONS_MAX, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "For fully observed case, n (%" PetscInt_FMT ") must equal Q_NUM_OBSERVATIONS_MAX (%" PetscInt_FMT ")", n, (PetscInt)Q_NUM_OBSERVATIONS_MAX);

  /* Create Q matrix (n x n for identity observation operator)
     Each row will have exactly Q_NUM_OBSERVATIONS_MAX non-zeros */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, Q_NUM_OBSERVATIONS_MAX, NULL, 0, NULL, Q));

  /* Initialize with full localization (all weights = 1.0)
     Each vertex i uses all n observations */
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) PetscCall(MatSetValue(*Q, i, j, 1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*Q, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Q, MAT_FINAL_ASSEMBLY));

  /* Validate: Check each row has exactly Q_NUM_OBSERVATIONS_MAX non-zeros */
  for (i = 0; i < n; i++) {
    PetscInt           ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(*Q, i, &ncols, &cols, &vals));
    PetscCheck(ncols == Q_NUM_OBSERVATIONS_MAX, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Row %" PetscInt_FMT " has %" PetscInt_FMT " non-zeros, expected %" PetscInt_FMT, i, ncols, (PetscInt)Q_NUM_OBSERVATIONS_MAX);
    PetscCall(MatRestoreRow(*Q, i, &ncols, &cols, &vals));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  /* Configuration parameters */
  PetscInt  n             = DEFAULT_N;
  PetscInt  steps         = DEFAULT_STEPS;
  PetscInt  burn          = DEFAULT_BURN;
  PetscInt  obs_freq      = DEFAULT_OBS_FREQ;
  PetscInt  random_seed   = DEFAULT_RANDOM_SEED;
  PetscInt  ensemble_size = DEFAULT_ENSEMBLE_SIZE;
  PetscReal F             = DEFAULT_F;
  PetscReal dt            = DEFAULT_DT;
  PetscReal obs_error_std = DEFAULT_OBS_ERROR_STD;

  /* PETSc objects */
  Lorenz96Ctx *l95_ctx = NULL;
  DM           da_state;
  PetscDA      daas;
  Vec          x0, x_mean, x_forecast;
  Vec          truth_state, rmse_work;
  Vec          observation, obs_noise, obs_error_var;
  PetscRandom  rng;
  Mat          Q = NULL; /* Localization matrix */
  Mat          H = NULL; /* Observation operator matrix */

  /* Statistics tracking */
  PetscReal rmse_forecast = 0.0, rmse_analysis = 0.0;
  PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscInt  n_stat_steps = 0;
  PetscInt  obs_count    = 0;
  PetscInt  step, progress_interval;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Kokkos initialization deferred to Phase 5 optimization */

  /* Parse command-line options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Lorenz-96 Example", NULL);
  PetscCall(PetscOptionsInt("-n", "State dimension", "", n, &n, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", "", steps, &steps, NULL));
  PetscCall(PetscOptionsInt("-burn", "Burn-in steps excluded from statistics", "", burn, &burn, NULL));
  PetscCall(PetscOptionsInt("-obs_freq", "Observation frequency", "", obs_freq, &obs_freq, NULL));
  PetscCall(PetscOptionsReal("-F", "Forcing parameter", "", F, &F, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", "", dt, &dt, NULL));
  PetscCall(PetscOptionsReal("-obs_error", "Observation error standard deviation", "", obs_error_std, &obs_error_std, NULL));
  PetscCall(PetscOptionsInt("-ensemble_size", "Number of ensemble members", "", ensemble_size, &ensemble_size, NULL));
  PetscCall(PetscOptionsInt("-random_seed", "Random seed for ensemble perturbations", "", random_seed, &random_seed, NULL));
  PetscOptionsEnd();

  /* Validate and constrain parameters */
  PetscCall(ValidateParameters(&n, &steps, &burn, &obs_freq, &ensemble_size, &dt, &F, &obs_error_std));

  /* Calculate progress reporting interval */
  progress_interval = (steps >= PROGRESS_INTERVALS) ? (steps / PROGRESS_INTERVALS) : 1;

  /* Create 1D periodic DM for state space */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, n, 1, 2, NULL, &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));

  /* Create Lorenz96 context with reusable TS object */
  PetscCall(Lorenz96ContextCreate(da_state, n, F, dt, &l95_ctx));

  /* Initialize random number generator */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rng));
  PetscCall(PetscRandomSetSeed(rng, (unsigned long)random_seed));
  PetscCall(PetscRandomSetFromOptions(rng));
  PetscCall(PetscRandomSeed(rng));

  /* Initialize state vectors */
  PetscCall(DMCreateGlobalVector(da_state, &x0));
  PetscCall(PetscRandomSetInterval(rng, -.1 * F, .1 * F));
  PetscCall(VecSetRandom(x0, rng));
  PetscCall(PetscRandomSetInterval(rng, 0, 1));

  /* Initialize truth trajectory */
  PetscCall(VecDuplicate(x0, &truth_state));
  PetscCall(VecCopy(x0, truth_state));
  PetscCall(VecDuplicate(x0, &rmse_work));

  /* Spin up truth to get onto attractor */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Spinning up truth for %" PetscInt_FMT " steps...\n", (PetscInt)SPINUP_STEPS));
  for (int k = 0; k < SPINUP_STEPS; k++) PetscCall(Lorenz96Step(truth_state, truth_state, l95_ctx));

  /* Initialize observation vectors */
  PetscCall(VecDuplicate(x0, &observation));
  PetscCall(VecDuplicate(x0, &obs_noise));
  PetscCall(VecDuplicate(x0, &obs_error_var));
  PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));

  /* Initialize ensemble statistics vectors */
  PetscCall(VecDuplicate(x0, &x_mean));
  PetscCall(VecDuplicate(x0, &x_forecast));

  /* Create and configure PetscDA for ensemble data assimilation */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &daas));
  PetscCall(PetscDASetType(daas, PETSCDALETKF)); /* Set LETKF type */
  /* Note: ndof defaults to 1 (scalar field) - perfect for Lorenz-96 */
  PetscCall(PetscDASetSizes(daas, n, n, ensemble_size));
  PetscCall(PetscDASetFromOptions(daas));
  PetscCall(PetscDASetUp(daas));
  PetscCall(PetscDAViewFromOptions(daas, NULL, "-da_view"));
  PetscCall(PetscDASetObsErrorVariance(daas, obs_error_var));

  /* Create and set localization matrix Q */
  PetscCall(CreateLocalizationMatrix(n, &Q));
  PetscCall(PetscDALETKFSetLocalization(daas, Q));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Localization matrix Q created: %" PetscInt_FMT " x %" PetscInt_FMT ", full localization (all weights = 1.0)\n", n, n));

  /* Create identity observation matrix H */
  PetscCall(CreateIdentityObservationMatrix(n, &H));

  /* Initialize ensemble members from spun-up truth state */
  PetscCall(InitializeEnsemble(daas, truth_state, ensemble_size, obs_error_std, rng));

  /* Print configuration summary */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Lorenz-96 LETKF Example\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "======================\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "  State dimension       : %" PetscInt_FMT "\n"
                        "  Ensemble size         : %" PetscInt_FMT "\n"
                        "  Forcing parameter (F) : %.4f\n"
                        "  Time step (dt)        : %.4f\n"
                        "  Total steps           : %" PetscInt_FMT "\n"
                        "  Burn-in steps         : %" PetscInt_FMT "\n"
                        "  Observation frequency : %" PetscInt_FMT "\n"
                        "  Observation noise std : %.3f\n"
                        "  Random seed           : %" PetscInt_FMT "\n"
                        "  Localization          : Full (Q_NUM_OBS_MAX = %" PetscInt_FMT ")\n\n",
                        n, ensemble_size, (double)F, (double)dt, steps, burn, obs_freq, (double)obs_error_std, random_seed, (PetscInt)Q_NUM_OBSERVATIONS_MAX));

  /* Main assimilation cycle: forecast and analysis steps */
  for (step = 0; step <= steps; step++) {
    PetscReal time = step * dt;

    /* Forecast step: compute ensemble mean and forecast RMSE */
    PetscCall(PetscDAComputeEnsembleMean(daas, x_mean));
    PetscCall(VecCopy(x_mean, x_forecast));
    PetscCall(ComputeRMSE(x_forecast, truth_state, rmse_work, n, &rmse_forecast));
    rmse_analysis = rmse_forecast;

    /* Analysis step: assimilate observations when available */
    if (step % obs_freq == 0 && step > 0) {
      /* Generate synthetic noisy observations from truth */
      PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
      PetscCall(VecWAXPY(observation, 1.0, obs_noise, truth_state));

      /* Perform LETKF analysis with observation matrix H */
      PetscCall(PetscDAAnalysis(daas, observation, H));

      /* Compute analysis RMSE */
      PetscCall(PetscDAComputeEnsembleMean(daas, x_mean));
      PetscCall(ComputeRMSE(x_mean, truth_state, rmse_work, n, &rmse_analysis));
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
      l95_ctx->time = step * dt;
      l95_ctx->step = step;
      PetscCall(PetscDAApplyModel(daas, Lorenz96Step, l95_ctx));
      PetscCall(Lorenz96Step(truth_state, truth_state, l95_ctx));
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
  PetscCall(Lorenz96ContextDestroy(&l95_ctx));
  PetscCall(PetscRandomDestroy(&rng));

  /* Kokkos finalization deferred to Phase 5 optimization */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: !complex kokkos_kernels
    diff_args: -j
    args: -steps 120 -burn 10 -obs_freq 1 -obs_error 1 -da_view -ensemble_size 30

  test:
    suffix: chol
    diff_args: -j
    requires: !complex kokkos_kernels
    args: -steps 120 -burn 10 -obs_freq 1 -obs_error .5 -da_view -ensemble_size 30 -da_sqrt_type cholesky

  test:
    suffix: etkf
    diff_args: -j
    requires: !complex kokkos_kernels
    args: -steps 120 -burn 10 -obs_freq 1 -obs_error 1 -da_view -ensemble_size 30 -petscda_type etkf

TEST*/
