const char help[] = "Verify TaoTermSumFlatten() across two Tao objects using the elastic-net term mix.\n"
                    "Builds an inner Tao with two terms (data, ridge), retrieves the resulting TAOTERMSUM via TaoGetTerm(),\n"
                    "and adds it as a summand to an outer Tao that already holds a third term (lasso). Adding a sum-summand\n"
                    "after a non-sum triggers the wrapping path in TaoAddTerm(), so the outer TAOTERMSUM has a nested inner\n"
                    "TAOTERMSUM at index 1 before Flatten and three peer summands after Flatten. The result is compared\n"
                    "summand-by-summand against a flat baseline built in a single Tao with the same three terms.\n";

#include <petsctao.h>

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  Mat         A, W, D;
  Vec         b, w, y;
  PetscInt    m        = 20;
  PetscInt    n        = 8;
  PetscInt    k        = 5;
  PetscReal   lambda_1 = 0.1;
  PetscReal   lambda_2 = 0.1;
  TaoTerm     data_term, l2_reg_term, l1_reg_term;
  PetscRandom rand;
  Tao         tao_inner, tao_outer, tao_flat;
  TaoTerm     inner_sum, outer_sum, flat_sum, summand;
  Vec         inner_params;
  Mat         inner_map;
  PetscReal   inner_scale;
  PetscInt    n_terms;
  PetscBool   is_sum;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  /* --- Build the elastic-net data: same recipe as elastic_net_regularization.c --- */
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rand));

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

  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, k, n, NULL, &D));
  PetscCall(MatSetRandom(D, rand));
  PetscCall(MatCreateVecs(D, NULL, &y));
  PetscCall(VecSetRandom(y, rand));

  PetscCall(TaoTermCreateQuadratic(W, &data_term));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &l2_reg_term));
  PetscCall(TaoTermCreateL1(comm, PETSC_DECIDE, k, 0.0, &l1_reg_term));

  /* Options prefixes on the TaoTerm objects survive the wrap path in TaoAddTerm:
     when the first non-sum term gets wrapped, the wrapping pulls its options prefix
     to use as the new summand's prefix. The user-provided TaoAddTerm prefix is
     dropped on the empty-callback first-add path, so we must set it on the object. */
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data_term, "data_"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)l2_reg_term, "ridge_"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)l1_reg_term, "lasso_"));

  /* --- Inner Tao: build (data + ridge); inner_sum becomes a TAOTERMSUM with 2 summands. --- */
  PetscCall(TaoCreate(comm, &tao_inner));
  PetscCall(TaoAddTerm(tao_inner, "data_", 1.0, data_term, b, A));
  PetscCall(TaoAddTerm(tao_inner, "ridge_", lambda_2, l2_reg_term, NULL, NULL));
  PetscCall(TaoGetTerm(tao_inner, &inner_scale, &inner_sum, &inner_params, &inner_map));
  PetscCall(PetscObjectTypeCompare((PetscObject)inner_sum, TAOTERMSUM, &is_sum));
  PetscCheck(is_sum, comm, PETSC_ERR_PLIB, "tao_inner objective term should be TAOTERMSUM");
  PetscCall(TaoTermSumGetNumberTerms(inner_sum, &n_terms));
  PetscCheck(n_terms == 2, comm, PETSC_ERR_PLIB, "inner_sum should have 2 summands, got %" PetscInt_FMT, n_terms);
  PetscCheck(inner_scale == 1.0, comm, PETSC_ERR_PLIB, "expected outer scale 1.0 on tao_inner, got %g", (double)inner_scale);
  PetscCheck(!inner_map, comm, PETSC_ERR_PLIB, "expected NULL outer map on tao_inner");

  /* --- Outer Tao: lasso first (non-sum), then inner_sum (sum). The second add hits the
         wrapping path because objective_term.term is non-sum: lasso gets wrapped into a
         fresh outer TAOTERMSUM, and inner_sum is appended as summand 1. We pass NULL for
         the outer prefix so Flatten composes prefixes as (NULL + inner_prefix) = inner_prefix,
         which lets us compare against a flat baseline that uses the same per-term prefixes. --- */
  PetscCall(TaoCreate(comm, &tao_outer));
  PetscCall(TaoAddTerm(tao_outer, "lasso_", lambda_1, l1_reg_term, y, D));
  PetscCall(TaoAddTerm(tao_outer, NULL, inner_scale, inner_sum, inner_params, inner_map));

  PetscCall(TaoGetTerm(tao_outer, NULL, &outer_sum, NULL, NULL));
  PetscCall(TaoTermSumGetNumberTerms(outer_sum, &n_terms));
  PetscCheck(n_terms == 2, comm, PETSC_ERR_PLIB, "pre-Flatten: expected 2 outer summands, got %" PetscInt_FMT, n_terms);
  PetscCall(TaoTermSumGetTerm(outer_sum, 1, NULL, NULL, &summand, NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)summand, TAOTERMSUM, &is_sum));
  PetscCheck(is_sum, comm, PETSC_ERR_PLIB, "pre-Flatten: summand 1 should be the nested TAOTERMSUM");

  /* --- Flatten the nested structure. --- */
  PetscCall(TaoTermSumFlatten(outer_sum));
  PetscCall(TaoTermSumGetNumberTerms(outer_sum, &n_terms));
  PetscCheck(n_terms == 3, comm, PETSC_ERR_PLIB, "post-Flatten: expected 3 summands, got %" PetscInt_FMT, n_terms);
  for (PetscInt i = 0; i < 3; i++) {
    PetscCall(TaoTermSumGetTerm(outer_sum, i, NULL, NULL, &summand, NULL));
    PetscCall(PetscObjectTypeCompare((PetscObject)summand, TAOTERMSUM, &is_sum));
    PetscCheck(!is_sum, comm, PETSC_ERR_PLIB, "post-Flatten: summand %" PetscInt_FMT " is still TAOTERMSUM", i);
  }

  /* --- Flat baseline: same three terms added in the same order to one Tao. The result
         must match the flattened outer_sum summand-by-summand. --- */
  PetscCall(TaoCreate(comm, &tao_flat));
  PetscCall(TaoAddTerm(tao_flat, "lasso_", lambda_1, l1_reg_term, y, D));
  PetscCall(TaoAddTerm(tao_flat, "data_", 1.0, data_term, b, A));
  PetscCall(TaoAddTerm(tao_flat, "ridge_", lambda_2, l2_reg_term, NULL, NULL));
  PetscCall(TaoGetTerm(tao_flat, NULL, &flat_sum, NULL, NULL));
  PetscCall(TaoTermSumGetNumberTerms(flat_sum, &n_terms));
  PetscCheck(n_terms == 3, comm, PETSC_ERR_PLIB, "flat baseline: expected 3 summands, got %" PetscInt_FMT, n_terms);

  for (PetscInt i = 0; i < 3; i++) {
    const char *p_flat, *p_flat_post;
    PetscReal   s_flat, s_post;
    TaoTerm     t_flat, t_post;
    Mat         m_flat, m_post;
    PetscBool   prefix_eq;

    PetscCall(TaoTermSumGetTerm(flat_sum, i, &p_flat, &s_flat, &t_flat, &m_flat));
    PetscCall(TaoTermSumGetTerm(outer_sum, i, &p_flat_post, &s_post, &t_post, &m_post));

    if (p_flat && p_flat_post) PetscCall(PetscStrcmp(p_flat, p_flat_post, &prefix_eq));
    else prefix_eq = (PetscBool)(p_flat == p_flat_post);
    PetscCheck(prefix_eq, comm, PETSC_ERR_PLIB, "index %" PetscInt_FMT ": prefix mismatch flat='%s' flattened='%s'", i, p_flat ? p_flat : "(null)", p_flat_post ? p_flat_post : "(null)");
    PetscCheck(PetscAbsReal(s_flat - s_post) < PETSC_SMALL, comm, PETSC_ERR_PLIB, "index %" PetscInt_FMT ": scale mismatch flat=%g flattened=%g", i, (double)s_flat, (double)s_post);
    PetscCheck(t_flat == t_post, comm, PETSC_ERR_PLIB, "index %" PetscInt_FMT ": term pointer mismatch", i);
    PetscCheck(m_flat == m_post, comm, PETSC_ERR_PLIB, "index %" PetscInt_FMT ": map pointer mismatch", i);
  }

  PetscCall(TaoDestroy(&tao_flat));
  PetscCall(TaoDestroy(&tao_outer));
  PetscCall(TaoDestroy(&tao_inner));
  PetscCall(TaoTermDestroy(&data_term));
  PetscCall(TaoTermDestroy(&l2_reg_term));
  PetscCall(TaoTermDestroy(&l1_reg_term));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: 0
    output_file: output/empty.out

TEST*/
