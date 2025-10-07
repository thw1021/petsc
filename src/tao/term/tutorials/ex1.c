const char help[] = "Basic TaoTerm usage";

#include <petsctao.h>

int main(int argc, char **argv)
{
  TaoTerm     term;
  PetscViewer viewer;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerSetFromOptions(viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO_DETAIL));

  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &term));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, 10, 1));
  PetscCall(TaoTermSetParametersSizes(term, PETSC_DECIDE, 7, 1));
  PetscCall(PetscObjectSetName((PetscObject)term, "example TaoTerm"));
  PetscCall(TaoTermSetFromOptions(term));
  PetscCall(TaoTermSetUp(term));
  PetscCall(TaoTermView(term, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(PetscViewerDestroy(&viewer));
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
    args: -taoterm_type callbacks

  test:
    suffix: 2
    args: -taoterm_type sum -taoterm_sum_num_subterms 2 -subterm_0_taoterm_type halfl2squared -subterm_1_taoterm_type l1 -taoterm_sum_subterm_0_scale 0.5 -taoterm_sum_subterm_0_mask objective,gradient,hessian

TEST*/
