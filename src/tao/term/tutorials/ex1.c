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
    args: -tao_term_type shell

  test:
    suffix: 1
    args: -tao_term_type tao

TEST*/
