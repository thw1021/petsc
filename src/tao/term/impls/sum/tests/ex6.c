const char help[] = "Regression test for TaoTermSumFlatten() early-return bug: a sum-no-map summand with exactly 1 inner term contributes 1 to n_new (matching its current 1-slot occupancy), so a count-balance short-circuit would skip the unwrap even though it is needed.";

#include <petsctaoterm.h>

int main(int argc, char **argv)
{
  MPI_Comm  comm;
  TaoTerm   outer, inner_a, inner_b, sub, summand_term;
  PetscInt  n = 4, n_terms;
  PetscBool is_sum;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  /* inner_a = [1.0 * sub_a], inner_b = [1.0 * sub_b]: each inner sum has exactly 1 term. */
  PetscCall(TaoTermCreate(comm, &inner_a));
  PetscCall(TaoTermSetType(inner_a, TAOTERMSUM));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &sub));
  PetscCall(TaoTermSumAddTerm(inner_a, NULL, 1.0, sub, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub));

  PetscCall(TaoTermCreate(comm, &inner_b));
  PetscCall(TaoTermSetType(inner_b, TAOTERMSUM));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &sub));
  PetscCall(TaoTermSumAddTerm(inner_b, NULL, 1.0, sub, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub));

  /* outer = [inner_a, inner_b]: 2 outer slots, each holding a 1-term inner sum.
     A naive count-balance check (n_new == sum->n_terms) would see 2 == 2 and skip
     the rebuild, leaving the inner wrappers in place. The fix tracks needs_flatten
     explicitly so the unwrap still happens. */
  PetscCall(TaoTermCreate(comm, &outer));
  PetscCall(TaoTermSetType(outer, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(outer, NULL, 1.0, inner_a, NULL, NULL));
  PetscCall(TaoTermDestroy(&inner_a));
  PetscCall(TaoTermSumAddTerm(outer, NULL, 1.0, inner_b, NULL, NULL));
  PetscCall(TaoTermDestroy(&inner_b));

  PetscCall(TaoTermSumFlatten(outer));
  PetscCall(TaoTermSumGetNumberTerms(outer, &n_terms));
  PetscCheck(n_terms == 2, comm, PETSC_ERR_PLIB, "expected 2 terms after flatten, got %" PetscInt_FMT, n_terms);

  /* Both summands must now be the unwrapped sub-terms, NOT TAOTERMSUM wrappers. */
  PetscCall(TaoTermSumGetTerm(outer, 0, NULL, NULL, &summand_term, NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)summand_term, TAOTERMSUM, &is_sum));
  PetscCheck(!is_sum, comm, PETSC_ERR_PLIB, "index 0 still TAOTERMSUM after Flatten (early-return regression)");
  PetscCall(TaoTermSumGetTerm(outer, 1, NULL, NULL, &summand_term, NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)summand_term, TAOTERMSUM, &is_sum));
  PetscCheck(!is_sum, comm, PETSC_ERR_PLIB, "index 1 still TAOTERMSUM after Flatten (early-return regression)");

  PetscCall(TaoTermDestroy(&outer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/empty.out

TEST*/
