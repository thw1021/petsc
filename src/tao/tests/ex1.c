const char help[] = "Tests of TaoAddObjectiveTerm()";

#include <petsctao.h>

int main(int argc, char **argv)
{
  Tao      tao;
  TaoTerm  original_sum;
  TaoTerm  sub_0, sub_1, sub_2;
  Mat      sub_1_map;
  Mat      original_map;
  PetscInt n = 10, m = 11, k = 12;
  MPI_Comm comm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoTermCreate(comm, &original_sum));
  PetscCall(TaoTermSetType(original_sum, TAOTERMSUM));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &sub_0));
  PetscCall(TaoTermSetParametersMode(sub_0, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(sub_0, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermSumAddSubterm(original_sum, NULL, 1.0, sub_0, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub_0));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &sub_1));
  PetscCall(TaoTermSetParametersMode(sub_1, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(sub_1, PETSC_DECIDE, m, 1));
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &sub_1_map));
  PetscCall(TaoTermSumAddSubterm(original_sum, NULL, 2.0, sub_1, sub_1_map, NULL));
  PetscCall(MatDestroy(&sub_1_map));
  PetscCall(TaoTermDestroy(&sub_1));

  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, n, k, NULL, &original_map));
  PetscCall(TaoSetObjectiveTerm(tao, 1.5, original_sum, NULL, original_map));
  PetscCall(MatDestroy(&original_map));
  PetscCall(TaoTermDestroy(&original_sum));

  PetscCall(TaoView(tao, PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &sub_2));
  PetscCall(TaoTermSetParametersMode(sub_2, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(sub_2, PETSC_DECIDE, k, 1));

  PetscCall(TaoAddObjectiveTerm(tao, "subterm_2_", 3.0, sub_2, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub_2));

  PetscCall(TaoView(tao, PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 0

TEST*/
