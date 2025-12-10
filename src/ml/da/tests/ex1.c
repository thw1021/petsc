static char help[] = "Tests basic creation and destruction of PetscDA objects, and a simple ETKF analysis step.\n\n";

#include "petscda.h"

/*
  Simple linear observation operator: y = x
*/
static PetscErrorCode IdentityObservationOperator(Vec x, Vec y, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscDA     da;
  Vec         x_true, y_obs, obs_error_var;
  Vec         x_mean_forecast, x_mean_analysis;
  PetscInt    state_size = 10, obs_size = 10, ensemble_size = 20;
  PetscRandom rng;
  PetscInt    i;
  PetscReal   norm;

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Create the DAS object */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(PetscDASetType(da, PETSCDAETKF));
  PetscCall(PetscDASetSizes(da, state_size, obs_size, ensemble_size));
  PetscCall(PetscDASetFromOptions(da));
  PetscCall(PetscDASetUp(da));

  /* Initialize random number generator */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rng));
  PetscCall(PetscRandomSetFromOptions(rng));

  /* Create vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x_true));
  PetscCall(VecSetSizes(x_true, PETSC_DECIDE, state_size));
  PetscCall(VecSetFromOptions(x_true));
  PetscCall(VecSet(x_true, 1.0)); /* True state is all 1s */

  PetscCall(VecDuplicate(x_true, &y_obs));
  PetscCall(VecDuplicate(x_true, &obs_error_var));
  PetscCall(VecSet(obs_error_var, 0.1)); /* Observation error variance */
  PetscCall(PetscDASetObsErrorVariance(da, obs_error_var));

  /* Create synthetic observation: y = x_true (identity observation, no noise for this simple test) */
  PetscCall(VecCopy(x_true, y_obs));

  /* Initialize ensemble with some spread around 0 (far from truth 1.0) */
  for (i = 0; i < ensemble_size; i++) {
    Vec member;
    PetscCall(VecDuplicate(x_true, &member));
    PetscCall(VecSetRandom(member, rng)); /* Uniform random [0, 1] */
    /* Shift to be centered roughly around 0.5 */
    //PetscCall(VecShift(member, 0.0));
    PetscCall(PetscDASetEnsembleMember(da, i, member));
    PetscCall(VecDestroy(&member));
  }

  /* Compute forecast mean before analysis */
  PetscCall(VecDuplicate(x_true, &x_mean_forecast));
  PetscCall(PetscDAComputeMean(da, x_mean_forecast));

  /* Check forecast error */
  PetscCall(VecAXPY(x_mean_forecast, -1.0, x_true));
  PetscCall(VecNorm(x_mean_forecast, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Forecast error norm: %g\n", (double)norm));

  /* Perform Analysis Step */
  PetscCall(PetscDAAnalysis(da, y_obs, IdentityObservationOperator, NULL));

  /* Compute analysis mean */
  PetscCall(VecDuplicate(x_true, &x_mean_analysis));
  PetscCall(PetscDAComputeMean(da, x_mean_analysis));

  /* Check analysis error */
  PetscCall(VecAXPY(x_mean_analysis, -1.0, x_true));
  PetscCall(VecNorm(x_mean_analysis, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Analysis error norm: %g\n", (double)norm));

  /* The analysis should move the ensemble closer to the observation (truth) */
  /* Since observation error is small (0.1) and prior spread is ~0.08, it should pull towards observation */

  PetscCall(PetscDAViewFromOptions(da, NULL, "-das_view"));

  /* Cleanup */
  PetscCall(VecDestroy(&x_true));
  PetscCall(VecDestroy(&y_obs));
  PetscCall(VecDestroy(&obs_error_var));
  PetscCall(VecDestroy(&x_mean_forecast));
  PetscCall(VecDestroy(&x_mean_analysis));
  PetscCall(PetscRandomDestroy(&rng));
  PetscCall(PetscDADestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    requires: !complex
    args: -das_view
    requires: !complex

  test:
    suffix: chol
    requires: !complex
    args: -das_view -das_etkf_sqrt_type cholesky

TEST*/
