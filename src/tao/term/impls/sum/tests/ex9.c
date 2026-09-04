const char help[] = "Tests automatic nested-sum flattening and parameter repacking in TaoAddTerm().\n";

#include <petsctao.h>

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  Mat         A, W, D;
  Vec         b, w, x, y;
  PetscInt    m = 20, n = 8, k = 5, nterms;
  PetscReal   lambda1 = 0.1, lambda2 = 0.1, fflat, fouter, inner_scale;
  TaoTerm     data, ridge, lasso, inner_sum, outer_sum, flat_sum, summand;
  PetscRandom rand;
  Tao         inner, outer, flat;
  Vec         flat_params, inner_params, outer_params;
  Mat         inner_map;
  PetscBool   is_sum;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &A));
  PetscCall(MatSetRandom(A, rand));
  PetscCall(MatCreateVecs(A, NULL, &b));
  PetscCall(MatCreateVecs(A, &x, NULL));
  PetscCall(VecSetRandom(b, rand));
  PetscCall(VecSetRandom(x, rand));
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

  PetscCall(TaoTermCreateQuadratic(W, &data));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &ridge));
  PetscCall(TaoTermCreateL1(comm, PETSC_DECIDE, k, 0.0, &lasso));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data, "data_"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ridge, "ridge_"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)lasso, "lasso_"));

  PetscCall(TaoCreate(comm, &inner));
  PetscCall(TaoAddTerm(inner, "data_", 1.0, data, b, A));
  PetscCall(TaoAddTerm(inner, "ridge_", lambda2, ridge, NULL, NULL));
  PetscCall(TaoGetTerm(inner, &inner_scale, &inner_sum, &inner_params, &inner_map));

  PetscCall(TaoCreate(comm, &outer));
  PetscCall(TaoAddTerm(outer, "lasso_", lambda1, lasso, y, D));
  PetscCall(TaoAddTerm(outer, "", inner_scale, inner_sum, inner_params, inner_map));
  PetscCall(TaoGetTerm(outer, NULL, &outer_sum, &outer_params, NULL));
  PetscCall(TaoTermSumGetNumberTerms(outer_sum, &nterms));
  PetscCheck(nterms == 3, comm, PETSC_ERR_PLIB, "Expected 3 automatically flattened terms, got %" PetscInt_FMT, nterms);
  for (PetscInt i = 0; i < nterms; i++) {
    PetscCall(TaoTermSumGetTerm(outer_sum, i, NULL, NULL, &summand, NULL));
    PetscCall(PetscObjectTypeCompare((PetscObject)summand, TAOTERMSUM, &is_sum));
    PetscCheck(!is_sum, comm, PETSC_ERR_PLIB, "Term %" PetscInt_FMT " is still a TAOTERMSUM after flattening", i);
  }

  PetscCall(TaoCreate(comm, &flat));
  PetscCall(TaoAddTerm(flat, "lasso_", lambda1, lasso, y, D));
  PetscCall(TaoAddTerm(flat, "data_", 1.0, data, b, A));
  PetscCall(TaoAddTerm(flat, "ridge_", lambda2, ridge, NULL, NULL));
  PetscCall(TaoGetTerm(flat, NULL, &flat_sum, &flat_params, NULL));

  for (PetscInt i = 0; i < nterms; i++) {
    const char *flat_prefix, *outer_prefix;
    PetscReal   flat_scale, outer_scale;
    TaoTerm     flat_term, outer_term;
    Mat         flat_map, outer_map;
    PetscBool   prefixes_equal;

    PetscCall(TaoTermSumGetTerm(flat_sum, i, &flat_prefix, &flat_scale, &flat_term, &flat_map));
    PetscCall(TaoTermSumGetTerm(outer_sum, i, &outer_prefix, &outer_scale, &outer_term, &outer_map));
    if (flat_prefix && outer_prefix) PetscCall(PetscStrcmp(flat_prefix, outer_prefix, &prefixes_equal));
    else prefixes_equal = (PetscBool)(flat_prefix == outer_prefix);
    PetscCheck(prefixes_equal, comm, PETSC_ERR_PLIB, "Term %" PetscInt_FMT " has a different prefix after flattening", i);
    PetscCheck(PetscAbsReal(flat_scale - outer_scale) < PETSC_SMALL, comm, PETSC_ERR_PLIB, "Term %" PetscInt_FMT " has a different scale after flattening", i);
    PetscCheck(flat_term == outer_term, comm, PETSC_ERR_PLIB, "Term %" PetscInt_FMT " has a different TaoTerm after flattening", i);
    PetscCheck(flat_map == outer_map, comm, PETSC_ERR_PLIB, "Term %" PetscInt_FMT " has a different map after flattening", i);
  }
  PetscCall(TaoTermComputeObjective(outer_sum, x, outer_params, &fouter));
  PetscCall(TaoTermComputeObjective(flat_sum, x, flat_params, &fflat));
  PetscCheck(PetscAbsReal(fouter - fflat) < 100.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, PetscAbsReal(fflat)), comm, PETSC_ERR_PLIB, "Objective mismatch after flattening and repacking: %g != %g", (double)fouter, (double)fflat);

  PetscCall(TaoDestroy(&flat));
  PetscCall(TaoDestroy(&outer));
  PetscCall(TaoDestroy(&inner));
  PetscCall(TaoTermDestroy(&data));
  PetscCall(TaoTermDestroy(&ridge));
  PetscCall(TaoTermDestroy(&lasso));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    output_file: output/empty.out

TEST*/
