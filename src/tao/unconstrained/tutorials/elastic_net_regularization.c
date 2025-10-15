const char help[] = "Demonstration of elastic net regularization (https://en.wikipedia.org/wiki/Elastic_net_regularization) using TAO";

#include <petsctao.h>

int main(int argc, char **argv)
{
  /*
    This example demonstrates the solution of an elastic net regularized least squares problem

    (1/2) || Ax - b ||_W^2 + lambda_2 (1/2) || x ||_2^2 + lambda_1 || Dx - y ||_1
   */

  MPI_Comm    comm;
  Mat         A;                // data matrix
  Mat         D;                // dicionary matrix
  Mat         W;                // weight matrix
  Vec         w;                // observation vector
  Vec         b;                // observation vector
  Vec         y;                // dictionary vector
  Vec         x;                // solution vector
  PetscInt    m          = 100; // data size
  PetscInt    n          = 20;  // model size
  PetscInt    k          = 10;  // dicionary size
  PetscBool   set_prefix = PETSC_TRUE;
  TaoTerm     data_term;
  TaoTerm     l2_reg_term;
  TaoTerm     l1_reg_term;
  TaoTerm     full_objective;
  PetscRandom rand;
  PetscReal   lambda_1 = 0.1;
  PetscReal   lambda_2 = 0.1;
  Tao         tao;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsBoundedInt("-m", "data size", "", m, &m, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-n", "model size", "", n, &n, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-k", "dictionary size", "", k, &k, NULL, 0));
  PetscCall(PetscOptionsBool("-set_term_prefix", "Set prefix to subterms", NULL, set_prefix, &set_prefix, NULL));
  PetscOptionsEnd();

  PetscCall(TaoCreate(comm, &tao));

  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rand));

  // create the model data, A, W and b
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &A));
  PetscCall(MatSetRandom(A, rand));
  PetscCall(MatCreateVecs(A, NULL, &b));
  PetscCall(VecSetRandom(b, rand));
  PetscCall(VecDuplicate(b, &w));
  PetscCall(VecSetRandom(w, rand));
  PetscCall(VecAbs(w));
  PetscCall(VecShift(w, 1.0));
  PetscCall(MatCreateDiagonal(w, &W));
  PetscCall(VecDestroy(&w));

  // create the dictionary data, D and y
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, k, n, NULL, &D));
  PetscCall(MatSetRandom(D, rand));
  PetscCall(MatCreateVecs(D, NULL, &y));
  PetscCall(VecSetRandom(y, rand));

  // the model term,  (1/2) || Ax - b ||_W^2
  PetscCall(TaoTermCreateQuadratic(W, &data_term));
  if (set_prefix) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data_term, "data_"));
  PetscCall(TaoAddTerm(tao, "data_", 1.0, data_term, b, A));
  PetscCall(TaoTermDestroy(&data_term));

  // the L2 term,  (1/2) lambda_2 || x ||_2^2
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &l2_reg_term));
  if (set_prefix) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)l2_reg_term, "ridge_"));
  PetscCall(TaoAddTerm(tao, "ridge_", lambda_2, l2_reg_term, NULL, NULL)); // Note: no parameter vector, no map matrix needed
  PetscCall(TaoTermDestroy(&l2_reg_term));

  // the L1 term,  lambda_1 || Dx - y ||_1
  PetscCall(TaoTermCreateL1(comm, PETSC_DECIDE, k, 0.0, &l1_reg_term));
  if (set_prefix) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)l1_reg_term, "lasso_"));
  PetscCall(TaoAddTerm(tao, "lasso_", lambda_1, l1_reg_term, y, D));
  PetscCall(TaoTermDestroy(&l1_reg_term));

  PetscCall(TaoGetTerm(tao, NULL, &full_objective, NULL, NULL));
  PetscCall(TaoTermCreateVecs(full_objective, &x, NULL));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single

  test:
    suffix: 0
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls

  test:
    suffix: no_prefix
    args: -tao_monitor_short -tao_view -tao_term_l1_epsilon 0.1 -tao_type nls -set_term_prefix 0

  test:
    suffix: mask_failure
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls
    args: -tao_term_sum_ridge_mask objective -tao_term_sum_lasso_mask gradient
    args: -tao_view ::ascii_info_detail

  test:
    suffix: assembled
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -tao_term_hessian_mat_type dense -ridge_tao_term_hessian_mat_type constantdiagonal -lasso_tao_term_hessian_mat_type diagonal

  test:
    suffix: mffd
    requires: !single !__float128
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -tao_term_hessian_mat_type mffd

  test:
    suffix: separate_shell
    requires: !single !__float128
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -tao_term_hessian_pre_is_hessian 0 -tao_term_hessian_pre_mat_type shell

  test:
    suffix: separate_mffd
    requires: !single !__float128
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -tao_term_hessian_pre_is_hessian 0 -tao_term_hessian_pre_mat_type mffd

  test:
    suffix: snes
    requires: !single
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type snes

TEST*/
