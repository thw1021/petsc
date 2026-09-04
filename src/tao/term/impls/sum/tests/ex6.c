const char help[] = "Regression test for flattening nested one-term TaoTerm sums.\n";

#include <petsctaoterm.h>

int main(int argc, char **argv)
{
  MPI_Comm  comm;
  TaoTerm   outer, inner_a, inner_b, subterm, summand;
  PetscInt  n = 4, nterms;
  PetscBool is_sum;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(TaoTermCreate(comm, &inner_a));
  PetscCall(TaoTermSetType(inner_a, TAOTERMSUM));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &subterm));
  PetscCall(TaoTermSumAddTerm(inner_a, NULL, 1.0, subterm, NULL, NULL));
  PetscCall(TaoTermDestroy(&subterm));

  PetscCall(TaoTermCreate(comm, &inner_b));
  PetscCall(TaoTermSetType(inner_b, TAOTERMSUM));
  PetscCall(TaoTermCreateHalfL2Squared(comm, PETSC_DECIDE, n, &subterm));
  PetscCall(TaoTermSumAddTerm(inner_b, NULL, 1.0, subterm, NULL, NULL));
  PetscCall(TaoTermDestroy(&subterm));

  PetscCall(TaoTermCreate(comm, &outer));
  PetscCall(TaoTermSetType(outer, TAOTERMSUM));
  PetscCall(TaoTermSumAddTerm(outer, NULL, 1.0, inner_a, NULL, NULL));
  PetscCall(TaoTermDestroy(&inner_a));
  PetscCall(TaoTermSumAddTerm(outer, NULL, 1.0, inner_b, NULL, NULL));
  PetscCall(TaoTermDestroy(&inner_b));

  PetscCall(TaoTermSumFlatten(outer));
  PetscCall(TaoTermSumGetNumberTerms(outer, &nterms));
  PetscCheck(nterms == 2, comm, PETSC_ERR_PLIB, "Expected 2 terms after flattening, got %" PetscInt_FMT, nterms);
  for (PetscInt i = 0; i < nterms; i++) {
    PetscCall(TaoTermSumGetTerm(outer, i, NULL, NULL, &summand, NULL));
    PetscCall(PetscObjectTypeCompare((PetscObject)summand, TAOTERMSUM, &is_sum));
    PetscCheck(!is_sum, comm, PETSC_ERR_PLIB, "Term %" PetscInt_FMT " is still a TAOTERMSUM after flattening", i);
  }

  PetscCall(TaoTermDestroy(&outer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    output_file: output/empty.out

TEST*/
