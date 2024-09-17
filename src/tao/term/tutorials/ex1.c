const char help[] = "Basic TaoTerm usage";

#include <petsctao.h>

int main(int argc, char **argv)
{
  TaoTerm term;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &term));
  PetscCall(PetscObjectSetName((PetscObject)term, "example TaoTerm"));
  PetscCall(TaoTermSetFromOptions(term));
  PetscCall(TaoTermSetUp(term));
  PetscCall(TaoTermView(term, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

  test:
    suffix: 0_from_options
    output_file: output/ex1_0.out
    args: -taoterm_type shell

  test:
    suffix: 1
    args: -taoterm_type taocallbacks

  test:
    suffix: 2
    args: -taoterm_type sum -taoterm_sum_num_subterms 3 -subterm_0_taoterm_type halfl2squared -subterm_1_taoterm_type l1 -subterm_2_taoterm_type kl -taoterm_sum_subterm_0_scale 0.5 -taoterm_sum_subterm_2_scale 0.25

TEST*/
