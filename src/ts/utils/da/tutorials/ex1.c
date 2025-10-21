#include "petscda.h"
#include <petscdmda.h>
#include <petscts.h>
#include <petscvec.h>

static char help[] = "Deterministic ETKF example for the Lorenz-95 model. See "
                     "Algorithm 6.4 of \n"
                     "Asch, Bocquet, and Nodet (2016) \"Data Assimilation\" "
                     "(SIAM, doi:10.1137/1.9781611974546).\n\n";

// ./ex1 -steps 105000 -burn 5000 -obs_freq 1 -obs_error 1 -da_view
// -ensemble_size 30 : "Mean RMSE (analysis): 0.474040"

typedef struct {
  DM da;        /* 1D periodic DM storing the Lorenz-95 state */
  PetscInt n;   /* State dimension (number of grid points) */
  PetscReal F;  /* Constant forcing term in the Lorenz-95 equations */
  PetscReal dt; /* Integration time step size */
} Lorenz95Ctx;

static PetscErrorCode Lorenz95RHS(TS ts, PetscReal t, Vec X, Vec F_vec,
                                  void *ctx) {
  Lorenz95Ctx *l95 = (Lorenz95Ctx *)ctx;
  Vec X_local;
  const PetscScalar *x;
  PetscScalar *f;
  PetscInt xs, xm, i;

  PetscFunctionBeginUser;
  /* Work with a local (ghosted) vector so the Lorenz-95 stencil has the
   * required neighbors. */
  PetscCall(DMDAGetCorners(l95->da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(DMGetLocalVector(l95->da, &X_local));
  PetscCall(DMGlobalToLocalBegin(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(l95->da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecGetArray(l95->da, F_vec, &f));

  /* Standard Lorenz-95 tendency: (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F. */
  for (i = xs; i < xs + xm; i++) {
    f[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i] + l95->F;
  }

  PetscCall(DMDAVecRestoreArrayRead(l95->da, X_local, &x));
  PetscCall(DMDAVecRestoreArray(l95->da, F_vec, &f));
  PetscCall(DMRestoreLocalVector(l95->da, &X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Lorenz95Step(Vec x_in, Vec x_out, void *ctx) {
  Lorenz95Ctx *l95 = (Lorenz95Ctx *)ctx;
  TS ts;

  PetscFunctionBeginUser;
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts, NULL, Lorenz95RHS, l95));
  /* Configure a single explicit RK4 step to forecast one ensemble member
   * forward in time. */
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSRKSetType(ts, TSRK4));
  PetscCall(TSSetTimeStep(ts, l95->dt));
  PetscCall(TSSetMaxSteps(ts, 1));
  PetscCall(TSSetMaxTime(ts, l95->dt));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(VecCopy(x_in, x_out));
  PetscCall(TSSolve(ts, x_out));
  PetscCall(TSDestroy(&ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Lorenz95ObsIdentity(Vec x, Vec y, void *ctx) {
  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
  PetscInt ensemble_size = 30;
  Lorenz95Ctx l95_ctx;
  DM da_state;
  DA da_ctx;
  Vec x0, x_mean, x_forecast, diff;
  Vec truth_state, truth_next;
  Vec observation, obs_noise, obs_error_var;
  PetscRandom rng;
  PetscInt n = 40;
  PetscInt steps = 200;
  PetscInt burn = 100;
  PetscInt obs_freq = 5;
  PetscInt random_seed = 12345;
  PetscReal F = 8.0;
  PetscReal dt = 0.05;
  PetscReal obs_error_std = 1.0;
  PetscReal rmse_forecast = 0.0, rmse_analysis = 0.0;
  PetscReal sum_rmse_forecast = 0.0, sum_rmse_analysis = 0.0;
  PetscInt n_stat_steps = 0;
  PetscInt obs_count = 0;
  PetscInt step;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* Expose key configuration parameters so the experiment can be tuned from the
   * command line. */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Lorenz-95 ETKF Quick Example",
                    NULL);
  PetscCall(PetscOptionsInt("-n", "State dimension", "", n, &n, NULL));
  PetscCall(PetscOptionsInt("-steps", "Number of time steps", "", steps, &steps,
                            NULL));
  PetscCall(PetscOptionsInt("-burn", "Burn-in steps excluded from statistics",
                            "", burn, &burn, NULL));
  PetscCall(PetscOptionsInt("-obs_freq", "Observation frequency", "", obs_freq,
                            &obs_freq, NULL));
  PetscCall(PetscOptionsReal("-F", "Forcing parameter", "", F, &F, NULL));
  PetscCall(PetscOptionsReal("-dt", "Time step size", "", dt, &dt, NULL));
  PetscCall(PetscOptionsReal("-obs_error",
                             "Observation error standard deviation", "",
                             obs_error_std, &obs_error_std, NULL));
  PetscCall(PetscOptionsInt("-ensemble_size", "Number of ensemble members", "",
                            ensemble_size, &ensemble_size, NULL));
  PetscCall(PetscOptionsInt("-random_seed",
                            "Random seed for ensemble perturbations", "",
                            random_seed, &random_seed, NULL));
  PetscOptionsEnd();

  if (obs_freq < 1)
    obs_freq = 1;
  if (burn > steps)
    burn = steps;
  if (ensemble_size < 1)
    ensemble_size = 1;

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, n, 1, 2, NULL,
                         &da_state));
  PetscCall(DMSetFromOptions(da_state));
  PetscCall(DMSetUp(da_state));

  l95_ctx.da = da_state;
  l95_ctx.n = n;
  l95_ctx.F = F;
  l95_ctx.dt = dt;

  PetscCall(DMCreateGlobalVector(da_state, &x0));
  /* Begin from the climatological equilibrium state (all entries set to F). */
  PetscCall(VecSet(x0, F));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rng));
  PetscCall(PetscRandomSetType(rng, PETSCRAND48));
  PetscCall(PetscRandomSetSeed(rng, (unsigned long)random_seed));
  PetscCall(PetscRandomSeed(rng));

  /* Vec perturb; // not clear if this is correct but it hurts results
  PetscCall(VecDuplicate(x0, &perturb));
  PetscCall(VecSetRandomGaussian(perturb, rng, 0.0, 0.1));
  PetscCall(VecAXPY(x0, 1.0, perturb));
  PetscCall(VecDestroy(&perturb)); */

  PetscCall(VecDuplicate(x0, &truth_state));
  PetscCall(VecDuplicate(x0, &truth_next));
  /* Truth trajectory starts from the same background state as the ensemble
   * mean. */
  PetscCall(VecCopy(x0, truth_state));

  PetscCall(VecDuplicate(x0, &observation));
  PetscCall(VecDuplicate(x0, &obs_noise));
  PetscCall(VecDuplicate(x0, &obs_error_var));
  /* Observation error variance is uniform and diagonal in this simple example.
   */
  PetscCall(VecSet(obs_error_var, obs_error_std * obs_error_std));

  PetscCall(VecDuplicate(x0, &x_mean));
  PetscCall(VecDuplicate(x0, &x_forecast));
  PetscCall(VecDuplicate(x0, &diff));
  /* x_mean/x_forecast track ensemble statistics; diff is reused for RMSE
   * diagnostics. */

  PetscCall(DACreate(PETSC_COMM_WORLD, &da_ctx));
  /* DA stores ensemble members in a logical 3D layout: state_dim x state_dim x
   * ensemble_size. */
  PetscCall(DASetSizes(da_ctx, n, n, ensemble_size));
  PetscCall(DASetFromOptions(da_ctx));
  PetscCall(DASetUp(da_ctx));
  PetscCall(DAViewFromOptions(da_ctx, NULL, "-da_view"));
  PetscCall(DASetObsErrorVariance(da_ctx, obs_error_var));

  Vec member, spread;
  PetscCall(VecDuplicate(x0, &member));
  PetscCall(VecDuplicate(x0, &spread));
  /* Populate the ensemble by perturbing the background state with Gaussian
   * draws. */
  for (PetscInt i = 0; i < ensemble_size; i++) {
    PetscCall(VecCopy(x0, member));
    PetscCall(VecSetRandomGaussian(spread, rng, 0.0, obs_error_std));
    PetscCall(VecAXPY(member, 1.0, spread));
    PetscCall(DASetEnsembleMember(da_ctx, i, member));
  }
  PetscCall(VecDestroy(&member));
  PetscCall(VecDestroy(&spread));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Lorenz-95 ETKF quick example\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "  State dimension      : %D\n"
                        "  Ensemble size        : %D\n"
                        "  Time step             : %.4f\n"
                        "  Total steps           : %D\n"
                        "  Burn-in steps         : %D\n"
                        "  Observation frequency : %D\n"
                        "  Observation noise std : %.3f\n\n",
                        n, ensemble_size, (double)dt, steps, burn, obs_freq,
                        (double)obs_error_std));

  /* Cycle through forecast and analysis steps, tracking skill metrics along the
   * way. */
  for (step = 0; step <= steps; step++) {
    PetscReal time = step * dt;

    /* Forecast: advance every ensemble member and compute the ensemble mean. */
    PetscCall(DAComputeMean(da_ctx, x_mean));
    PetscCall(VecCopy(x_mean, x_forecast));
    PetscCall(VecCopy(x_forecast, diff));
    PetscCall(VecAXPY(diff, -1.0, truth_state));
    PetscCall(VecNorm(diff, NORM_2, &rmse_forecast));
    rmse_forecast /= PetscSqrtReal((PetscReal)n);
    rmse_analysis = rmse_forecast;

    if (step % obs_freq == 0) {
      /* Analysis: synthesize noisy observations and assimilate them with the
       * ETKF. */
      PetscCall(VecSetRandomGaussian(obs_noise, rng, 0.0, obs_error_std));
      PetscCall(VecCopy(truth_state, observation));
      PetscCall(VecAXPY(observation, 1.0, obs_noise));
      PetscCall(DAAnalysis(da_ctx, observation, Lorenz95ObsIdentity, NULL));

      PetscCall(DAComputeMean(da_ctx, x_mean));
      PetscCall(VecCopy(x_mean, diff));
      PetscCall(VecAXPY(diff, -1.0, truth_state));
      PetscCall(VecNorm(diff, NORM_2, &rmse_analysis));
      rmse_analysis /= PetscSqrtReal((PetscReal)n);
      obs_count++;
    }

    if (step >= burn) {
      /* Exclude the burn-in period when forming long-term statistics. */
      sum_rmse_forecast += rmse_forecast;
      sum_rmse_analysis += rmse_analysis;
      n_stat_steps++;
    }

    if ((step % (steps / 10) == 0) || (step == steps) || (step == 0)) {
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "Step %4D time %.3f  RMSE_forecast %.5f  RMSE_analysis %.5f%s\n",
          step, (double)time, (double)rmse_forecast, (double)rmse_analysis,
          (step < burn) ? " [burn-in]" : ""));
    }

    if (step < steps) {
      /* Propagate every ensemble member (in-place) and advance the truth
       * trajectory. */
      PetscCall(DAApplyModel(da_ctx, Lorenz95Step, &l95_ctx));
      PetscCall(Lorenz95Step(truth_state, truth_next, &l95_ctx));
      PetscCall(VecCopy(truth_next, truth_state));
    }
  }

  /* Report aggregate statistics only when enough post burn-in samples were
   * collected. */
  if (n_stat_steps > 0) {
    PetscReal avg_rmse_forecast = sum_rmse_forecast / n_stat_steps;
    PetscReal avg_rmse_analysis = sum_rmse_analysis / n_stat_steps;
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD,
                    "\nStatistics over %D assimilation steps (post burn-in):\n",
                    n_stat_steps));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (forecast): %.6f\n",
                          (double)avg_rmse_forecast));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Mean RMSE (analysis): %.6f\n",
                          (double)avg_rmse_analysis));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "  Observations assimilated: %D\n\n", obs_count));
  }

  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&x_forecast));
  PetscCall(VecDestroy(&x_mean));
  PetscCall(VecDestroy(&obs_noise));
  PetscCall(VecDestroy(&observation));
  PetscCall(VecDestroy(&truth_next));
  PetscCall(VecDestroy(&truth_state));
  PetscCall(VecDestroy(&obs_error_var));
  PetscCall(VecDestroy(&x0));
  PetscCall(DADestroy(&da_ctx));
  PetscCall(DMDestroy(&da_state));
  PetscCall(PetscRandomDestroy(&rng));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    args: -steps 120 -burn 10 -obs_freq 2 -obs_error 0.5 -da_view

  test:
    suffix: chol
    args: -steps 120 -burn 10 -obs_freq 1 -obs_error 1.0 -daetkf_sqrt_type
cholesky

TEST*/