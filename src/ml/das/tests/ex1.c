static char help[] = "Tests PetscDAS ETKF implementation with unit tests and verification.\n\n";

/*
  This test performs:
  1. Unit test: Gaussian random number generation
  2. Unit test: Eigenvalue square root verification
  3. Unit test: Ensemble mean computation
  4. Verification test: Linear 2D system tracking
*/

#include <petscdas.h>
#include <petscblaslapack.h>

/* Test 1: Gaussian random number generation */
PetscErrorCode TestGaussianRandom(void)
{
  Vec         v;
  PetscRandom rctx;
  PetscInt    n           = 10000, i;
  PetscScalar mean_target = 0.0, stddev_target = 1.0;
  PetscReal   mean_computed, variance_computed, tolerance = 0.1;
  PetscScalar sum = 0.0, sum_sq = 0.0, *array;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test 1: Gaussian Random Number Generation\n"));

  /* Create vector and random context */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(VecSetSizes(v, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(v));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));

  /* Fill with Gaussian random numbers */
  PetscCall(VecSetGaussianRandom(v, rctx, mean_target, stddev_target));

  /* Compute sample mean and variance */
  PetscCall(VecGetArray(v, &array));
  for (i = 0; i < n; i++) {
    sum += array[i];
    sum_sq += array[i] * array[i];
  }
  PetscCall(VecRestoreArray(v, &array));

  mean_computed     = PetscRealPart(sum) / n;
  variance_computed = PetscRealPart(sum_sq) / n - mean_computed * mean_computed;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Target mean = %g, computed mean = %g\n", (double)mean_target, (double)mean_computed));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Target variance = %g, computed variance = %g\n", (double)(stddev_target * stddev_target), (double)variance_computed));

  /* Check if within tolerance */
  PetscCheck(PetscAbsReal(mean_computed - mean_target) < tolerance, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Mean error too large");
  PetscCheck(PetscAbsReal(variance_computed - stddev_target * stddev_target) < tolerance, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Variance error too large");

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n\n"));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Test 2: Eigenvalue square root verification */
PetscErrorCode TestEigenSquareRoot(void)
{
  Mat          A, V, A_dense;
  Vec          D, x, y;
  PetscInt     n = 5, i, j;
  PetscReal    error;
  KSP          ksp;
  PetscScalar *eig_vals, *eig_vecs;
  PetscBLASInt n_blas, lwork = -1, info_lapack;
  PetscScalar  work_query, *work;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test 2: Eigenvalue Square Root Verification\n"));

  /* Create a symmetric positive definite matrix A */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATSEQAIJ));
  PetscCall(MatSetUp(A));

  /* Fill with SPD matrix: A = I + 0.1 * ones */
  for (i = 0; i < n; i++) {
    PetscCall(MatSetValue(A, i, i, 1.1, INSERT_VALUES));
    for (j = 0; j < n; j++) {
      if (i != j) PetscCall(MatSetValue(A, i, j, 0.1, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* Convert to dense for eigenvalue decomposition */
  PetscCall(MatConvert(A, MATDENSE, MAT_INITIAL_MATRIX, &A_dense));
  PetscCall(MatDuplicate(A_dense, MAT_COPY_VALUES, &V));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &D));
  PetscCall(VecSetSizes(D, PETSC_DECIDE, n));
  PetscCall(VecSetUp(D));

  /* Compute eigenvalue decomposition using LAPACK */
  PetscCall(PetscBLASIntCast(n, &n_blas));
  PetscCall(MatDenseGetArray(V, &eig_vecs));
  PetscCall(VecGetArray(D, &eig_vals));

  /* Query optimal workspace */
  LAPACKsyev_("V", "U", &n_blas, eig_vecs, &n_blas, eig_vals, &work_query, &lwork, &info_lapack);
  lwork = (PetscBLASInt)PetscRealPart(work_query);
  PetscCall(PetscMalloc1(lwork, &work));

  /* Compute eigendecomposition */
  LAPACKsyev_("V", "U", &n_blas, eig_vecs, &n_blas, eig_vals, work, &lwork, &info_lapack);
  PetscCall(PetscFree(work));
  PetscCheck(info_lapack == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK syev failed with info = %d", (int)info_lapack);

  PetscCall(VecRestoreArray(D, &eig_vals));
  PetscCall(MatDenseRestoreArray(V, &eig_vecs));

  /* Verify: V * D^(-1) * V^T = A^(-1) */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecSet(x, 1.0));

  /* Compute A^(-1) * x using KSP */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, x, y));

  /* Compute V * D^(-1) * V^T * x using eigendecomposition */
  Vec temp1, temp2;
  PetscCall(VecDuplicate(x, &temp1));
  PetscCall(VecDuplicate(x, &temp2));

  /* temp1 = V^T * x */
  PetscCall(MatMultTranspose(V, x, temp1));

  /* Apply D^(-1) */
  PetscScalar *temp1_arr;
  PetscCall(VecGetArray(temp1, &temp1_arr));
  PetscCall(VecGetArray(D, &eig_vals));
  for (j = 0; j < n; j++) { temp1_arr[j] /= eig_vals[j]; }
  PetscCall(VecRestoreArray(D, &eig_vals));
  PetscCall(VecRestoreArray(temp1, &temp1_arr));

  /* temp2 = V * (D^(-1) * temp1) */
  PetscCall(MatMult(V, temp1, temp2));

  /* Compare y and temp2 */
  PetscCall(VecAXPY(temp2, -1.0, y));
  PetscCall(VecNorm(temp2, NORM_2, &error));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Error in eigenvalue inverse: %g\n", (double)error));
  PetscCheck(error < 1e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Eigenvalue inverse error too large");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n\n"));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&A_dense));
  PetscCall(MatDestroy(&V));
  PetscCall(VecDestroy(&D));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&temp1));
  PetscCall(VecDestroy(&temp2));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Test 3: Ensemble mean computation */
PetscErrorCode TestEnsembleMean(void)
{
  PetscDAS  das;
  Vec      *ensemble, mean, expected_mean;
  PetscInt  m = 5, n = 3, i;
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test 3: Ensemble Mean Computation\n"));

  /* Create DAS object */
  PetscCall(PetscDASCreate(PETSC_COMM_WORLD, &das));
  PetscCall(PetscDASSetType(das, PETSCDASETKF));
  PetscCall(PetscDASSetFromOptions(das));
  PetscCall(PetscDASSetEnsembleSize(das, m));
  PetscCall(PetscDASSetStateSize(das, n));

  /* Create ensemble */
  PetscCall(PetscMalloc1(m, &ensemble));
  for (i = 0; i < m; i++) {
    PetscCall(VecCreate(PETSC_COMM_WORLD, &ensemble[i]));
    PetscCall(VecSetSizes(ensemble[i], PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(ensemble[i]));
    PetscCall(VecSet(ensemble[i], (PetscReal)i));
  }
  PetscCall(PetscDASSetEnsemble(das, ensemble));

  /* Compute mean */
  mean = NULL;
  PetscCall(PetscDASGetEnsembleMean(das, &mean));

  /* Expected mean = (0 + 1 + 2 + 3 + 4) / 5 = 2.0 */
  PetscCall(VecDuplicate(mean, &expected_mean));
  PetscCall(VecSet(expected_mean, 2.0));

  /* Compare */
  PetscCall(VecAXPY(mean, -1.0, expected_mean));
  PetscCall(VecNorm(mean, NORM_2, &error));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Error in ensemble mean: %g\n", (double)error));
  PetscCheck(error < 1e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Ensemble mean error too large");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n\n"));

  for (i = 0; i < m; i++) PetscCall(VecDestroy(&ensemble[i]));
  PetscCall(PetscFree(ensemble));
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&expected_mean));
  PetscCall(PetscDASDestroy(&das));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Test 4: Linear 2D system verification */
PetscErrorCode TestLinearSystem(void)
{
  PetscDAS    das;
  Vec        *ensemble, truth, observation;
  Mat         A, H;
  PetscInt    m = 20, n = 2, p = 1, i, k, num_steps = 10;
  PetscRandom rctx;
  PetscReal   rmse, tolerance = 0.5;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test 4: Linear 2D System Verification\n"));

  /* Create DAS object */
  PetscCall(PetscDASCreate(PETSC_COMM_WORLD, &das));
  PetscCall(PetscDASSetType(das, PETSCDASETKF));
  PetscCall(PetscDASSetFromOptions(das));
  PetscCall(PetscDASSetEnsembleSize(das, m));
  PetscCall(PetscDASSetStateSize(das, n));

  /* Create model operator A = [0.95, 0.05; -0.05, 0.95] */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATSEQAIJ));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValue(A, 0, 0, 0.95, INSERT_VALUES));
  PetscCall(MatSetValue(A, 0, 1, 0.05, INSERT_VALUES));
  PetscCall(MatSetValue(A, 1, 0, -0.05, INSERT_VALUES));
  PetscCall(MatSetValue(A, 1, 1, 0.95, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* Create observation operator H = [1, 0] */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &H));
  PetscCall(MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE, p, n));
  PetscCall(MatSetType(H, MATSEQAIJ));
  PetscCall(MatSetUp(H));
  PetscCall(MatSetValue(H, 0, 0, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));

  /* Create observation error covariance diagonal R_diag = [1.0] */
  Vec R_diag;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &R_diag));
  PetscCall(VecSetSizes(R_diag, PETSC_DECIDE, p));
  PetscCall(VecSetFromOptions(R_diag));
  PetscCall(VecSet(R_diag, 1.5));

  PetscCall(PetscDASSetObservationOperator(das, H));
  PetscCall(PetscDASSetObservationErrorCovarianceDiagonal(das, R_diag));

  /* Create truth trajectory */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &truth));
  PetscCall(VecSetSizes(truth, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(truth));
  PetscCall(VecSet(truth, 1.0)); /* Initial condition */

  /* Create observation vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &observation));
  PetscCall(VecSetSizes(observation, PETSC_DECIDE, p));
  PetscCall(VecSetFromOptions(observation));

  /* Create ensemble with random perturbations around truth */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(PetscMalloc1(m, &ensemble));
  for (i = 0; i < m; i++) {
    PetscCall(VecDuplicate(truth, &ensemble[i]));
    PetscCall(VecSetGaussianRandom(ensemble[i], rctx, 1.0, 0.1));
  }
  PetscCall(PetscDASSetEnsemble(das, ensemble));
  PetscCall(PetscDASSetUp(das));

  /* Run assimilation cycle */
  for (k = 0; k < num_steps; k++) {
    /* Generate observation: y = H * truth + noise */
    PetscCall(MatMult(H, truth, observation));
    Vec noise;
    PetscCall(VecDuplicate(observation, &noise));
    PetscCall(VecSetGaussianRandom(noise, rctx, 0.0, .1));
    PetscCall(VecAXPY(observation, 1.0, noise));
    PetscCall(VecDestroy(&noise));

    /* Assimilate */
    PetscCall(PetscDASAssimilate(das, observation));

    /* Forecast */
    PetscCall(PetscDASForecast(das, A));

    /* Forecast truth */
    Vec truth_new;
    PetscCall(VecDuplicate(truth, &truth_new));
    PetscCall(MatMult(A, truth, truth_new));
    PetscCall(VecCopy(truth_new, truth));
    PetscCall(VecDestroy(&truth_new));
  }

  /* Compute RMSE between ensemble mean and truth */
  Vec mean, error_vec;
  mean = NULL;
  PetscCall(PetscDASGetEnsembleMean(das, &mean));
  PetscCall(VecDuplicate(mean, &error_vec));
  PetscCall(VecWAXPY(error_vec, -1.0, truth, mean));
  PetscCall(VecNorm(error_vec, NORM_2, &rmse));
  rmse = rmse / PetscSqrtReal((PetscReal)n);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  RMSE after %d steps: %g\n", num_steps, (double)rmse));
  PetscCheck(rmse < tolerance, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "RMSE too large, ETKF not tracking truth");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS\n\n"));

  /* Cleanup */
  for (i = 0; i < m; i++) PetscCall(VecDestroy(&ensemble[i]));
  PetscCall(PetscFree(ensemble));
  PetscCall(VecDestroy(&truth));
  PetscCall(VecDestroy(&observation));
  PetscCall(VecDestroy(&mean));
  PetscCall(VecDestroy(&error_vec));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&R_diag));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(PetscDASDestroy(&das));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "===== PetscDAS ETKF Tests =====\n\n"));

  /* Run all tests */
  PetscCall(TestGaussianRandom());
  PetscCall(TestEigenSquareRoot());
  PetscCall(TestEnsembleMean());
  PetscCall(TestLinearSystem());

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "===== All Tests Passed =====\n"));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single

  test:
    suffix: 1
    nsize: 1
    args: -das_view

TEST*/
