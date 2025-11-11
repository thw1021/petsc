#include "petscda.h"
#include <petscdmda.h>
#include <petscts.h>
#include <petscvec.h>

static char help[] = "Deterministic ETKF example for the Lorenz-96 model. See "
                     "Algorithm 6.4 of \n"
                     "Asch, Bocquet, and Nodet (2016) \"Data Assimilation\" "
                     "(SIAM, doi:10.1137/1.9781611974546).\n\n"
                     "Example usage:\n"
                     "  ./ex1 -steps 105000 -burn 5000 -obs_freq 1 -obs_error 1 -petscda_view -ensemble_size 30\n"
                     "  Expected result: Mean RMSE (analysis): ~0.474040\n\n";

/* \begin{algorithm}
\caption{Ensemble Transform Kalman Filter (ETKF) - Deterministic}
\begin{algorithmic}[1]
\State \textbf{Initialize:} Ensemble $\mathbf{E}_0 = [\mathbf{x}_0^{(1)}, \ldots, \mathbf{x}_0^{(m)}]$ \Comment{$\mathbf{E}_0 \in \mathbb{R}^{n \times m}$}
\For{$k = 1, 2, \ldots, K$}
    \State \textbf{// Forecast Step}
    \For{$i = 1$ to $m$}
        \State $\mathbf{x}_k^{f,(i)} = \mathcal{M}_{k-1}(\mathbf{x}_{k-1}^{a,(i)})$ \Comment{$\mathbf{x}_k^{f,(i)} \in \mathbb{R}^{n}$}
    \EndFor
    \State Assemble forecast matrix: $\mathbf{E}_k^f = [\mathbf{x}_k^{f,(1)}, \ldots, \mathbf{x}_k^{f,(m)}]$ \Comment{$\mathbf{E}_k^f \in \mathbb{R}^{n \times m}$}
    \State
    \State \textbf{// Analysis Step (if observation available)}
    \If{observation $\mathbf{y}_k$ available}
        \State Compute ensemble mean: $\overline{\mathbf{x}}_k^f = \frac{1}{m}\sum_{i=1}^m \mathbf{x}_k^{f,(i)}$ \Comment{$\overline{\mathbf{x}}_k^f \in \mathbb{R}^{n}$}
        \State Compute anomalies: $\mathbf{X}_k = \frac{1}{\sqrt{m-1}}(\mathbf{E}_k^f - \overline{\mathbf{x}}_k^f \mathbf{1}^T)$ \Comment{$\mathbf{X}_k \in \mathbb{R}^{n \times m}$}
        \State Apply observation operator: $\mathbf{Z}_k = \mathcal{H}(\mathbf{E}_k^f)$ \Comment{$\mathbf{Z}_k \in \mathbb{R}^{b \times m}$}
        \State Compute obs ensemble mean: $\overline{\mathbf{y}}_k = \frac{1}{m}\sum_{i=1}^m \mathcal{H}(\mathbf{x}_k^{f,(i)})$ \Comment{$\overline{\mathbf{y}}_k \in \mathbb{R}^{b}$}
        \State Compute obs anomalies: $\mathbf{S}_k = \frac{1}{\sqrt{m-1}}\mathbf{R}^{-1/2}(\mathbf{Z}_k - \overline{\mathbf{y}}_k\mathbf{1}^T)$ \Comment{$\mathbf{S}_k \in \mathbb{R}^{b \times m}$}
        \State Compute raw innovation: $\boldsymbol{\delta}_k = \mathbf{y}_k - \overline{\mathbf{y}}_k$ \Comment{$\boldsymbol{\delta}_k \in \mathbb{R}^{b}$}
        \State Whiten innovation: $\tilde{\boldsymbol{\delta}}_k = \mathbf{R}^{-1/2}\boldsymbol{\delta}_k$ \Comment{$\tilde{\boldsymbol{\delta}}_k \in \mathbb{R}^{b}$}
        \State Compute transform matrix: $\mathbf{T}_k = (\mathbf{I}_m + \mathbf{S}_k^T\mathbf{S}_k)^{-1}$ \Comment{$\mathbf{T}_k \in \mathbb{R}^{m \times m}$}
        \State Compute weight vector: $\mathbf{w}_k = \mathbf{T}_k \mathbf{S}_k^T \tilde{\boldsymbol{\delta}}_k$ \Comment{$\mathbf{w}_k \in \mathbb{R}^{m}$}
        \State Form deterministic map: $\mathbf{G}_k = \mathbf{w}_k\mathbf{1}^T + \sqrt{m-1}\,\mathbf{T}_k^{1/2}\mathbf{U}$ \Comment{$\mathbf{G}_k \in \mathbb{R}^{m \times m}$}
        \State Update ensemble: $\mathbf{E}_k^a = \overline{\mathbf{x}}_k^f\mathbf{1}^T + \mathbf{X}_k \mathbf{G}_k$ \Comment{$\mathbf{E}_k^a \in \mathbb{R}^{n \times m}$}
        \State where $\mathbf{U}$ is orthogonal with $\mathbf{U}\mathbf{1} = \mathbf{1}$ (the PETSc implementation stores it as \texttt{da->U} and defaults to $\mathbf{I}_m$)
    \Else
        \State $\mathbf{E}_k^a = \mathbf{E}_k^f$ \Comment{$\mathbf{E}_k^a \in \mathbb{R}^{n \times m}$}
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}
 */

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

/* Minimum valid parameter values */
#define MIN_N              1
#define MIN_ENSEMBLE_SIZE  2
#define MIN_OBS_FREQ       1
#define PROGRESS_INTERVALS 10

typedef struct {
  DM        da; /* 1D periodic DM storing the Lorenz-96 state */
  PetscInt  n;  /* State dimension (number of grid points) */
  PetscReal F;  /* Constant forcing term in the Lorenz-96 equations */
  PetscReal dt; /* Integration time step size */
  TS        ts; /* Reusable time stepper for efficiency */
} Lorenz96Ctx;

/*
  Lorenz96RHS - Compute the right-hand side of the Lorenz-96 equations

  Input Parameters:
+ ts    - The time-stepping context (unused but required by interface)
. t     - Current time (unused but required by interface)
. X     - State vector
- ctx   - User context (Lorenz96Ctx)

  Output Parameter:
. F_vec - RHS vector (tendency)
*/
static PetscErrorCode Lorenz96RHS(TS ts, PetscReal t, Vec X, Vec F_vec, void *ctx)
{
  Lorenz96Ctx       *l95 = (Lorenz96Ctx *)ctx;
  Vec                X_local;
  const PetscScalar *x;
  PetscScalar       *f;
  PetscInt           xs, xm, i;

  PetscFunctionBeginUser;
  (void)ts; /* Mark as intentionally unused to avoid compiler warnings */
  (void)t;

  /* Work with a local (ghosted) vector so the Lorenz-96 stencil has the
   * required neighbors for periodic boundary conditions. */
  PetscCall(DMDAGetCorners(l95->da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMGetLocalVector(l95->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecGetArray(l95->da, F_vec, &f));

  /* Standard Lorenz-96 tendency: (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F. */
  for (i = xs; i < xs + xm; i++) f[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + l95->F;

  PetscCall(DMDAVecRestoreArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecRestoreArray(l95->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(l95->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96ContextCreate - Create and initialize a Lorenz96 context with reusable TS object

  Input Parameters:
+ da - DM for state space
. n  - State dimension
. F  - Forcing parameter
- dt - Time step size

  Output Parameter:
. ctx - Initialized Lorenz96 context
*/
static PetscErrorCode Lorenz96ContextCreate(DM da, PetscInt n, PetscReal F, PetscReal dt, Lorenz96Ctx **ctx)
{
  Lorenz96Ctx *l95;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&l95));
  l95->da = da;
  l95->n  = n;
  l95->F  = F;
  l95->dt = dt;

  /* Create and configure a reusable time stepper to avoid repeated allocation/deallocation */
  PetscCall(TSCreate(PETSC_COMM_SELF, &l95->ts));
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

  Input Parameters:
+ x_in - Initial state vector
- ctx  - Lorenz96 context (contains reusable TS)

  Output Parameter:
. x_out - State vector after one time step

  Notes:
  Uses a single explicit RK4 step with the pre-configured TS object for efficiency.
*/
static PetscErrorCode Lorenz96Step(Vec x_in, Vec x_out, void *ctx)
{
  Lorenz96Ctx *l95 = (Lorenz96Ctx *)ctx;

  PetscFunctionBeginUser;
  /* Reset the TS time for each integration (required for proper RK4 stepping) */
  PetscCall(TSSetTime(l95->ts, 0.0));
  PetscCall(VecCopy(x_in, x_out));
  PetscCall(TSSolve(l95->ts, x_out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Lorenz96ObsIdentity - Identity observation operator

  Input Parameters:
+ x   - State vector
- ctx - User context (unused)

  Output Parameter:
. y - Observation vector (copy of state)
*/
static PetscErrorCode Lorenz96ObsIdentity(Vec x, Vec y, void *ctx)
{
  PetscFunctionBeginUser;
  (void)ctx; /* Mark as intentionally unused */
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ValidateParameters - Validate input parameters and apply constraints

  Input/Output Parameters:
+ n             - State dimension
. steps         - Number of time steps
. burn          - Burn-in steps
. obs_freq      - Observation frequency
. ensemble_size - Ensemble size
. dt            - Time step
. F             - Forcing parameter
- obs_error_std - Observation error standard deviation
*/
static PetscErrorCode ValidateParameters(PetscInt *n, PetscInt *steps, PetscInt *burn, PetscInt *obs_freq, PetscInt *ensemble_size, PetscReal *dt, PetscReal *F, PetscReal *obs_error_std)
{
  PetscFunctionBeginUser;
  /* Validate and constrain integer parameters */
  PetscCheck(*n > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "State dimension n must be positive, got %" PetscInt_FMT, *n);
  PetscCheck(*steps >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of steps must be non-negative, got %" PetscInt_FMT, *steps);
  PetscCheck(*ensemble_size >= MIN_ENSEMBLE_SIZE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least %d for meaningful statistics, got %" PetscInt_FMT, MIN_ENSEMBLE_SIZE, *ensemble_size);

  /* Apply constraints */
  if (*obs_freq < MIN_OBS_FREQ) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Observation frequency adjusted from %" PetscInt_FMT " to %d\n", *obs_freq, MIN_OBS_FREQ));
    *obs_freq = MIN_OBS_FREQ;
  }
  if (*burn > *steps) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Burn-in steps (%" PetscInt_FMT ") exceeds total steps (%" PetscInt_FMT "), setting burn = steps\n", *burn, *steps));
    *burn = *steps;
  }

  /* Validate real-valued parameters */
  PetscCheck(*dt > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Time step dt must be positive, got %g", (double)*dt);
  PetscCheck(*obs_error_std > 0.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Observation error std must be positive, got %g", (double)*obs_error_std);
  PetscCheck(PetscIsNormalReal(*F), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Forcing parameter F must be a normal real number");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  InitializeEnsemble - Initialize ensemble members with Gaussian perturbations

  Input Parameters:
+ da_ctx        - PetscDA context
. x0            - Background state
. ensemble_size - Number of ensemble members
. obs_error_std - Standard deviation for perturbations
- rng           - Random number generator

  Notes:
  Each ensemble member is initialized as x0 + Gaussian(0, obs_error_std)
*/
static PetscErrorCode InitializeEnsemble(PetscDA da_ctx, Vec x0, PetscInt ensemble_size, PetscReal obs_error_std, PetscRandom rng)
{
  Vec      member, spread;
  PetscInt i;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x0, &member));
  PetscCall(VecDuplicate(x0, &spread));

  /* Populate the ensemble by perturbing the background state with Gaussian draws */
  for (i = 0; i < ensemble_size; i++) {
    PetscCall(VecCopy(x0, member));
    PetscCall(VecSetRandomGaussian(spread, rng, 0.0, obs_error_std));
    PetscCall(VecAXPY(member, 1.0, spread));
    PetscCall(PetscDASetEnsembleMember(da_ctx, i, member));
  }

  PetscCall(VecDestroy(&member));
  PetscCall(VecDestroy(&spread));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeRMSE - Compute root mean square error between two vectors

  Input Parameters:
+ v1 - First vector
. v2 - Second vector
- n  - Vector dimension (for normalization)

  Output Parameter:
. rmse - Root mean square error
*/
static PetscErrorCode ComputeRMSE(Vec v1, Vec v2, PetscInt n, PetscReal *rmse)
{
  Vec       diff;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(v1, &diff));
  PetscCall(VecCopy(v1, diff));
  PetscCall(VecAXPY(diff, -1.0, v2));
  PetscCall(VecNorm(diff, NORM_2, &norm));
  *rmse = norm / PetscSqrtReal((PetscReal)n);
  PetscCall(VecDestroy(&diff));
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
  PetscDA      da_ctx;
  Vec          x0, x_mean, x_forecast;
  Vec          truth_state, truth_next;
  Vec          observation, obs_noise, obs_error_var;
  PetscRandom  rng;

  /* Statistics tracking */
  PetscReal rmse_forecast = 0.0, rmse_analysis = 0.0;
  PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscInt  n_stat_steps = 0;
  PetscInt  obs_count    = 0;
  PetscInt  step, progress_interval;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Parse command-line options */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Lorenz-96 ETKF Quick Example", NULL);
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

  /* Calculate progress reporting interval (avoid division by zero) */
  progress_interval = (steps >= PROGRESS_INTERVALS) ? (steps / PROGRESS_INTERVALS) : 1;

  /* Create 1D periodic DM for state space */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, n, 1, 2, NULL, &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));

  /* Create Lorenz96 context with reusable TS object */
  PetscCall(Lorenz96ContextCreate(da_state, n, F, dt, &l95_ctx));

  /* Initialize state vectors */
  PetscCall(DMCreateGlobalVector(da_state, &x0));
  PetscCall(VecSet(x0, F)); /* Begin from climatological equilibrium state */

  /* Initialize random number generator */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rng));
  PetscCall(PetscRandomSetSeed(rng, (unsigned long)random_seed));
  PetscCall(PetscRandomSetFromOptions(rng));
  PetscCall(PetscRandomSeed(rng));

  /* Initialize truth trajectory */
  PetscCall(VecDuplicate(x0, &truth_state));
  PetscCall(VecDuplicate(x0, &truth_next));
  PetscCall(VecCopy(x0, truth_state));

  /* Initialize observation vectors */
  PetscCall(VecDuplicate(x0, &observation));
  PetscCall(VecDuplicate(x0, &obs_noise));
  PetscCall(VecDuplicate(x0, &obs_error_var));
  PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));

  /* Initialize ensemble statistics vectors */
  PetscCall(VecDuplicate(x0, &x_mean));
  PetscCall(VecDuplicate(x0, &x_forecast));

  /* Create and configure PetscDA for ensemble data assimilation */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da_ctx));
  PetscCall(PetscDASetSizes(da_ctx, n, n, ensemble_size));
  PetscCall(PetscDASetFromOptions(da_ctx));
  PetscCall(PetscDASetUp(da_ctx));
  PetscCall(PetscDAViewFromOptions(da_ctx, NULL, "-petscda_view"));
  PetscCall(PetscDASetObsErrorVariance(da_ctx, obs_error_var));

  /* Initialize ensemble members */
  PetscCall(InitializeEnsemble(da_ctx, x0, ensemble_size, obs_error_std, rng));

  /* Print configuration summary */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Lorenz-96 ETKF Example\n"));
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
                        "  Random seed           : %" PetscInt_FMT "\n\n",
                        n, ensemble_size, (double)F, (double)dt, steps, burn, obs_freq, (double)obs_error_std, random_seed));

  /* Main assimilation cycle: forecast and analysis steps */
  for (step = 0; step <= steps; step++) {
    PetscReal time = step * dt;

    /* Forecast step: compute ensemble mean and forecast RMSE */
    PetscCall(PetscDAComputeMean(da_ctx, x_mean));
    PetscCall(VecCopy(x_mean, x_forecast));
    PetscCall(ComputeRMSE(x_forecast, truth_state, n, &rmse_forecast));
    rmse_analysis = rmse_forecast; /* Default to forecast RMSE if no analysis */

    /* Analysis step: assimilate observations when available */
    if (step % obs_freq == 0 && step > 0) {
      /* Generate synthetic noisy observations from truth */
      PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
      PetscCall(VecWAXPY(observation, 1.0, obs_noise, truth_state));

      /* Perform ETKF analysis */
      PetscCall(PetscDAAnalysis(da_ctx, observation, Lorenz96ObsIdentity, NULL));

      /* Compute analysis RMSE */
      PetscCall(PetscDAComputeMean(da_ctx, x_mean));
      PetscCall(ComputeRMSE(x_mean, truth_state, n, &rmse_analysis));
      obs_count++;
    }

    /* Accumulate statistics after burn-in period */
    if (step >= burn) {
      sum_rmse_forecast += rmse_forecast;
      sum_rmse_analysis += rmse_analysis;
      n_stat_steps++;
    }

    /* Progress reporting */
    if ((step % progress_interval == 0) || (step == steps) || (step == 0)) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %4" PetscInt_FMT ", time %6.3f  RMSE_forecast %.5f  RMSE_analysis %.5f%s\n", step, (double)time, (double)rmse_forecast, (double)rmse_analysis, (step < burn) ? " [burn-in]" : ""));
    }

    /* Propagate ensemble and truth trajectory */
    if (step < steps) {
      PetscCall(PetscDAApplyModel(da_ctx, Lorenz96Step, l95_ctx));
      PetscCall(Lorenz96Step(truth_state, truth_next, l95_ctx));
      PetscCall(VecCopy(truth_next, truth_state));
    }
  }

  /* Report final statistics */
  if (n_stat_steps > 0) {
    PetscReal avg_rmse_forecast = sum_rmse_forecast / n_stat_steps;
    PetscReal avg_rmse_analysis = sum_rmse_analysis / n_stat_steps;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStatistics (%" PetscInt_FMT " post-burn-in steps):\n", n_stat_steps));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==================================================\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (forecast) : %.6f\n", (double)avg_rmse_forecast));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (analysis) : %.6f\n", (double)avg_rmse_analysis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Observations used    : %" PetscInt_FMT "\n\n", obs_count));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nWarning: No post-burn-in statistics collected (burn >= steps)\n\n"));
  }

  /* Cleanup */
  PetscCall(VecDestroy(&x_forecast));
  PetscCall(VecDestroy(&x_mean));
  PetscCall(VecDestroy(&obs_error_var));
  PetscCall(VecDestroy(&obs_noise));
  PetscCall(VecDestroy(&observation));
  PetscCall(VecDestroy(&truth_next));
  PetscCall(VecDestroy(&truth_state));
  PetscCall(VecDestroy(&x0));
  PetscCall(PetscDADestroy(&da_ctx));
  PetscCall(DMDestroy(&da_state));
  PetscCall(Lorenz96ContextDestroy(&l95_ctx));
  PetscCall(PetscRandomDestroy(&rng));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: !complex
    args: -steps 120 -burn 10 -obs_freq 2 -obs_error 0.5 -petscda_view -ensemble_size 40

  test:
    suffix: chol
    args: -steps 120 -burn 10 -obs_freq 1 -obs_error 1.0 -petscdaetkf_sqrt_type cholesky

TEST*/
