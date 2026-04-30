const char help[] = "Demonstration of elastic net regularization (https://en.wikipedia.org/wiki/Elastic_net_regularization) using TAO";

#include <petsctao.h>

/* Context for the data term's mapped-Hessian callback registered with
   TaoTermSumSetTermMappedHessianFn(): the constant mapped Hessian A^T W A
   is pre-supplied to the summand via TaoTermSumSetTermHessianMatrices(),
   so the callback itself is a no-op and only records that it was reached. */
typedef struct {
  PetscInt call_count;
} DataMappedHessianCtx;

static PetscErrorCode DataMappedHessian(Vec x, Vec params, Mat H, Mat Hpre, PetscCtx ctx)
{
  DataMappedHessianCtx *m = (DataMappedHessianCtx *)ctx;

  PetscFunctionBeginUser;
  m->call_count++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  /*
    This example demonstrates the solution of an elastic net regularized least squares problem

    (1/2) || Ax - b ||_W^2 + lambda_2 (1/2) || x ||_2^2 + lambda_1 || Dx - y ||_1
   */

  MPI_Comm    comm;
  Mat         A;                // data matrix
  Mat         D;                // dictionary matrix
  Mat         W;                // weight matrix
  Vec         w;                // observation vector
  Vec         b;                // observation vector
  Vec         y;                // dictionary vector
  Vec         x;                // solution vector
  PetscInt    m          = 100; // data size
  PetscInt    n          = 20;  // model size
  PetscInt    k          = 10;  // dictionary size
  PetscBool             set_prefix                  = PETSC_TRUE;
  PetscBool             set_name                    = PETSC_FALSE;
  PetscBool             check_eps                   = PETSC_FALSE;
  PetscBool             test_mapping_hessian_setter = PETSC_FALSE;
  TaoTerm               data_term;
  TaoTerm               l2_reg_term;
  TaoTerm               l1_reg_term;
  TaoTerm               full_objective;
  PetscRandom           rand;
  PetscReal             lambda_1        = 0.1;
  PetscReal             lambda_2        = 0.1;
  Tao                   tao;
  DataMappedHessianCtx *mapped_hess_ctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsBoundedInt("-m", "data size", "", m, &m, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-n", "model size", "", n, &n, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-k", "dictionary size", "", k, &k, NULL, 0));
  PetscCall(PetscOptionsBool("-set_term_prefix", "Set prefix to terms", NULL, set_prefix, &set_prefix, NULL));
  PetscCall(PetscOptionsBool("-set_term_name", "Set name to terms", NULL, set_name, &set_name, NULL));
  PetscCall(PetscOptionsBool("-check_l1_eps", "Check epsilon of L1 term", NULL, check_eps, &check_eps, NULL));
  PetscCall(PetscOptionsBool("-test_mapping_hessian_setter", "Register a TaoTermSumSetTermMappedHessianFn callback for the data term", NULL, test_mapping_hessian_setter, &test_mapping_hessian_setter, NULL));
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
  if (set_prefix) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data_term, "data_"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)b, "bvec_"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)A, "Amat_"));
  }
  if (set_name) PetscCall(PetscObjectSetName((PetscObject)data_term, "Data TaoTerm"));
  PetscCall(TaoAddTerm(tao, "data_", 1.0, data_term, b, A));
  PetscCall(TaoTermDestroy(&data_term));

  // the L2 term,  (1/2) lambda_2 || x ||_2^2
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &l2_reg_term));
  if (set_prefix) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)l2_reg_term, "ridge_"));
  if (set_name) PetscCall(PetscObjectSetName((PetscObject)l2_reg_term, "Ridge TaoTerm"));
  PetscCall(TaoAddTerm(tao, "ridge_", lambda_2, l2_reg_term, NULL, NULL)); // Note: no parameter vector, no map matrix needed
  PetscCall(TaoTermDestroy(&l2_reg_term));

  // the L1 term,  lambda_1 || Dx - y ||_1
  PetscCall(TaoTermCreateL1(comm, PETSC_DECIDE, k, 0.0, &l1_reg_term));
  if (set_prefix) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)l1_reg_term, "lasso_"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)y, "yvec_"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)D, "Dmat_"));
  }
  if (set_name) PetscCall(PetscObjectSetName((PetscObject)l1_reg_term, "Lasso TaoTerm"));
  PetscCall(TaoAddTerm(tao, "lasso_", lambda_1, l1_reg_term, y, D));
  PetscCall(TaoTermDestroy(&l1_reg_term));

  PetscCall(TaoGetTerm(tao, NULL, &full_objective, NULL, NULL));
  if (test_mapping_hessian_setter) {
    Vec w_diag;
    Mat work, AtWA;

    /* Precompute the constant mapped Hessian A^T W A once using the same
       operation sequence the standard PtAP path takes when the inner Hessian
       is MATDIAGONAL (see the is_uH_diag branch in TaoTermMappingMatPtAP()),
       so the result is bitwise identical to the default path. Pre-supply it
       as the data term's _mapped_H so the registered callback can no-op:
       TaoTermMappingComputeHessian() forwards _mapped_H to the outer Hessian
       via MatCopy() (INSERT_VALUES branch) without re-running PtAP each
       Newton step. */
    PetscCall(PetscNew(&mapped_hess_ctx));
    PetscCall(MatDiagonalGetDiagonal(W, &w_diag));
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &work));
    PetscCall(MatDiagonalScale(work, w_diag, NULL));
    PetscCall(MatDiagonalRestoreDiagonal(W, &w_diag));
    PetscCall(MatTransposeMatMult(A, work, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &AtWA));
    PetscCall(MatDestroy(&work));
    PetscCall(TaoTermSumSetTermHessianMatrices(full_objective, 0, NULL, NULL, AtWA, AtWA));
    PetscCall(MatDestroy(&AtWA));
    PetscCall(TaoTermSumSetTermMappedHessianFn(full_objective, 0, DataMappedHessian, mapped_hess_ctx, PetscCtxDestroyDefault));
  }
  PetscCall(TaoTermCreateSolutionVec(full_objective, &x));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  if (test_mapping_hessian_setter) PetscCheck(mapped_hess_ctx->call_count > 0, comm, PETSC_ERR_PLIB, "data-term mapped-Hessian callback was never invoked");

  {
    PetscReal scale_get;
    TaoTerm   get_term;
    Vec       get_vec, p2, p1;
    Mat       get_mat;

    PetscCall(TaoGetTerm(tao, &scale_get, &get_term, &get_vec, &get_mat));
    PetscCall(VecNestGetTaoTermSumParameters(get_vec, 0, &p1));
    PetscCall(VecNestGetTaoTermSumParameters(get_vec, 1, &p2));
    PetscCheck(p1 == b, PETSC_COMM_SELF, PETSC_ERR_COR, "First parameter vector is not same as what was set");
    PetscCheck(p2 == NULL, PETSC_COMM_SELF, PETSC_ERR_COR, "Second parameter vector is not none");
  }

  if (check_eps) {
    PetscReal scale_get;
    TaoTerm   get_term;
    Vec       get_vec;
    Mat       get_mat;
    PetscInt  n_terms;
    PetscInt  last_index;
    PetscBool is_l1;
    TaoTerm   last_subterm;
    PetscReal epsilon;

    PetscCall(TaoGetTerm(tao, &scale_get, &get_term, &get_vec, &get_mat));
    PetscCall(TaoTermSumGetNumberTerms(get_term, &n_terms));
    last_index = n_terms - 1;
    PetscCall(TaoTermSumGetTerm(get_term, last_index, NULL, NULL, &last_subterm, NULL));
    PetscCall(PetscObjectTypeCompare((PetscObject)last_subterm, TAOTERML1, &is_l1));
    PetscCheck(is_l1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Last term is not L1");
    PetscCall(TaoTermL1GetEpsilon(last_subterm, &epsilon));
    PetscCheck(PetscAbsReal(epsilon - 0.1) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "L1 epsilon is not 0.1, got: %g", (double)epsilon);
  }

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
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: 0
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -check_l1_eps 1

  test:
    suffix: 1
    args: -tao_type nls -lasso_tao_term_hessian_mat_type aij -tao_view ::ascii_info_detail

  test:
    suffix: sum_hpre_is_not_h
    args: -tao_type nls -tao_view ::ascii_info_detail -tao_term_hessian_pre_is_hessian 0

  test:
    suffix: data_hpre_is_not_h
    args: -tao_type nls -tao_view ::ascii_info_detail -data_tao_term_hessian_pre_is_hessian 0

  test:
    suffix: ridge_hpre_is_not_h
    args: -tao_type nls -tao_view ::ascii_info_detail -ridge_tao_term_hessian_pre_is_hessian 0

  test:
    suffix: lasso_hpre_is_not_h
    args: -tao_type nls -tao_view ::ascii_info_detail -lasso_tao_term_hessian_pre_is_hessian 0

  test:
    suffix: hpre_is_not_h
    args: -tao_type nls -tao_view ::ascii_info_detail -lasso_tao_term_hessian_pre_is_hessian 0
    args: -ridge_tao_term_hessian_pre_is_hessian 0 -data_tao_term_hessian_pre_is_hessian 0

  test:
    suffix: data_ridge_hpre_is_not_h
    args: -tao_type nls -tao_view ::ascii_info_detail
    args: -ridge_tao_term_hessian_pre_is_hessian 0 -data_tao_term_hessian_pre_is_hessian 0

  test:
    suffix: no_prefix
    args: -tao_monitor_short -tao_view -tao_term_l1_epsilon 0.1 -tao_type nls -set_term_prefix 0

  test:
    suffix: no_prefix_yes_name
    args: -tao_monitor_short -tao_view -tao_term_l1_epsilon 0.1 -tao_type nls -set_term_prefix 0 -set_term_name 1

  test:
    suffix: yes_prefix_yes_name
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -set_term_prefix 1 -set_term_name 1

  test:
    suffix: mask_failure
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls
    args: -tao_term_sum_ridge_mask objective -tao_term_sum_lasso_mask gradient
    args: -tao_view ::ascii_info_detail

  test:
    suffix: assembled
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -ridge_tao_term_hessian_mat_type constantdiagonal -lasso_tao_term_hessian_mat_type diagonal

  test:
    suffix: snes
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type snes

  test:
    suffix: extra_info_view
    args: -tao_type nls -tao_add_terms extra_ -extra_tao_term_type halfl2squared -tao_term_sum_extra_scale 1.0 -tao_view

  # Exercises the outer-sum Hessian-shell path: TaoTermComputeHessianMult_Sum
  # dispatches MatMult on the MATSHELL outer Hessian through the per-summand
  # cache + MatMult fallback. The L1 epsilon is kept large enough that the
  # smoothed Hessian is well-conditioned for the unpreconditioned KSP used
  # when the outer Hessian is a MATSHELL.
  test:
    suffix: shell_hessian
    args: -tao_type nls -tao_term_hessian_mat_type shell -lasso_tao_term_l1_epsilon 0.1 -tao_monitor_short

  # Exercises TaoTermSumSetTermMappedHessianFn(): the data term's mapped
  # Hessian A^T W A is supplied by a user callback that bypasses the standard
  # PtAP assembly. Output mirrors test 0 except the data term reports zero
  # internal Hessian evaluations (its TaoTermComputeHessian is never invoked).
  test:
    suffix: mapping_hessian_setter
    args: -tao_monitor_short -tao_view -lasso_tao_term_l1_epsilon 0.1 -tao_type nls -check_l1_eps 1 -test_mapping_hessian_setter

TEST*/
