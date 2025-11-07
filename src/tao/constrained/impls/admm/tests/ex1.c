const char help[] = "Test TAOADMM on quadratic problems";

#include <petsctao.h>

// test on alpha/2 || C x - u ||_2^2 + beta/2 || D x ||_2^2
static PetscErrorCode TestADMMUnconstrained(MPI_Comm comm, PetscRandom rand, PetscInt test_number, PetscBool *keep_going)
{
  PetscInt    N, M_C, M_D;
  Mat         C, D;
  Vec         u;
  PetscScalar alpha, beta;
  TaoTerm     term_f, term_g;
  Vec         solution, tmp;
  Tao         tao;
  PetscViewer viewer;

  PetscFunctionBegin;
  N      = 13;
  M_C    = 4;
  M_D    = 17;
  alpha  = 1.0;
  beta   = 0.4;
  viewer = PETSC_VIEWER_STDOUT_(comm);
  PetscOptionsBegin(comm, NULL, "Options for the unconstrained test", __FILE__);
  PetscCall(PetscOptionsInt("-n", "solution size", NULL, N, &N, NULL));
  PetscCall(PetscOptionsInt("-m_c", "number of C matrix rows", NULL, M_C, &M_C, NULL));
  PetscCall(PetscOptionsInt("-m_d", "number of D matrix rows", NULL, M_D, &M_D, NULL));
  PetscOptionsEnd();
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s, test %" PetscInt_FMT ":\n", PETSC_FUNCTION_NAME, test_number));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscRandomSetSeed(rand, 0));

  // C matrix
  PetscCall(MatCreate(comm, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, M_C, N));
  PetscCall(MatSetType(C, MATDENSE));
  PetscCall(PetscObjectSetName((PetscObject)C, "C"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)C, "C_"));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatSetRandom(C, rand));

  // D matrix
  PetscCall(MatCreate(comm, &D));
  PetscCall(MatSetSizes(D, PETSC_DECIDE, PETSC_DECIDE, M_D, N));
  PetscCall(MatSetType(D, MATDENSE));
  PetscCall(PetscObjectSetName((PetscObject)D, "D"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)D, "D_"));
  PetscCall(MatSetFromOptions(D));
  PetscCall(MatSetUp(D));
  PetscCall(MatSetRandom(D, rand));

  // u vector
  PetscCall(MatCreateVecs(C, NULL, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  PetscCall(VecSetRandom(u, rand));

  // first L2 term f
  PetscCall(TaoTermCreate(comm, &term_f));
  PetscCall(TaoTermSetSolutionTemplate(term_f, u));
  PetscCall(TaoTermSetType(term_f, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetParametersMode(term_f, TAOTERM_PARAMETERS_REQUIRED));
  PetscCall(PetscObjectSetName((PetscObject)term_f, "f"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term_f, "f_"));
  PetscCall(TaoTermSetFromOptions(term_f));

  // second L2 term g
  PetscCall(TaoTermCreate(comm, &term_g));
  PetscCall(MatCreateVecs(D, NULL, &tmp));
  PetscCall(TaoTermSetSolutionTemplate(term_g, tmp));
  PetscCall(VecDestroy(&tmp));
  PetscCall(TaoTermSetType(term_g, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetParametersMode(term_g, TAOTERM_PARAMETERS_NONE));
  PetscCall(PetscObjectSetName((PetscObject)term_g, "g"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term_g, "g_"));
  PetscCall(TaoTermSetFromOptions(term_g));

  // Tao
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoAddTerm(tao, NULL, alpha, term_f, u, C));
  PetscCall(TaoAddTerm(tao, NULL, beta, term_g, NULL, D));
  PetscCall(TaoSetType(tao, TAOADMM));

  *keep_going = PETSC_TRUE;
  switch (test_number) {
  case 0: // default behavior: min_{x,z} f(Ax) + g(Bz) s.t. x == z
    break;
  case 1: // request max_{x,z} f(Ax) + g(Bz) s.t. x == z
    PetscCall(TaoADMMSetTermGroups(tao, 1, NULL, PETSC_FALSE, 1, NULL, PETSC_FALSE));
    break;
  case 2: // max_{x,z} f(x) + g(Bz) s.t. x == Az
    PetscCall(TaoADMMSetTermGroups(tao, 1, NULL, PETSC_TRUE, 1, NULL, PETSC_FALSE));
    break;
  default: // max_{x,z} f(Ax) + g(z) s.t. z == Bx
    PetscCall(TaoADMMSetTermGroups(tao, 1, NULL, PETSC_FALSE, 1, NULL, PETSC_TRUE));
    *keep_going = PETSC_FALSE;
    break;
  }

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(MatCreateVecs(C, &solution, NULL));
  PetscCall(VecSetRandom(solution, rand));
  PetscCall(TaoSetSolution(tao, solution));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&term_g));
  PetscCall(TaoTermDestroy(&term_f));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&solution));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscRandom rand;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscRandomCreate(comm, &rand));
  for (PetscInt i = 0;; i++) {
    PetscBool keep_going;

    PetscCall(TestADMMUnconstrained(comm, rand, i, &keep_going));
    if (!keep_going) break;
  }
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: basic
      args: -tao_converged_reason -tao_view

   test:
      suffix: aradmm_exact
      args: -n 13 -m_c 13 -m_d 13 -C_mat_type constantdiagonal -D_mat_type constantdiagonal
      args: -tao_monitor_constraint_norm -tao_admm_update_type adaptive

TEST*/
