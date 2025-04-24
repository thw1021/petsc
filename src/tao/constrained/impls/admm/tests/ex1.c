const char help[] = "Test TAOADMM on quadratic problems";

#include <petsctao.h>

// test on alpha/2 || C x - u ||_2^2 + beta/2 || D x - v ||_2^2
static PetscErrorCode Test1(MPI_Comm comm, PetscRandom rand)
{
  PetscInt    N, M_C, M_D;
  Mat         C, D;
  Vec         u, v;
  PetscScalar alpha, beta;
  TaoTerm     term_f, term_g;
  Vec         solution;
  Tao         tao;

  PetscFunctionBegin;
  N     = 13;
  M_C   = 4;
  M_D   = 17;
  alpha = 0.3;
  beta  = 0.4;
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, M_C, N, NULL, &C));
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, M_D, N, NULL, &D));
  PetscCall(MatSetRandom(C, rand));
  PetscCall(MatSetRandom(D, rand));
  PetscCall(MatCreateVecs(C, &solution, &u));
  PetscCall(VecSetRandom(solution, rand));
  PetscCall(VecSetRandom(u, rand));
  PetscCall(MatCreateVecs(D, NULL, &v));
  PetscCall(VecSetRandom(v, rand));
  PetscCall(TaoTermCreate(comm, &term_f));
  PetscCall(TaoTermSetSolutionTemplate(term_f, u));
  PetscCall(TaoTermSetType(term_f, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermCreate(comm, &term_g));
  PetscCall(TaoTermSetSolutionTemplate(term_g, v));
  PetscCall(TaoTermSetType(term_g, TAOTERMHALFL2SQUARED));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetTerm(tao, alpha, term_f, u, C));
  PetscCall(TaoAddTerm(tao, NULL, beta, term_g, v, D));
  PetscCall(TaoSetType(tao, TAOADMM));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetSolution(tao, solution));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&term_g));
  PetscCall(TaoTermDestroy(&term_f));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&solution));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscRandom rand;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(Test1(comm, rand));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -tao_view -tao_converged_reason -tao_monitor_constraint_norm
    args: -admm_sub_0_tao_monitor_short -admm_sub_0_tao_converged_reason -admm_sub_0_tao_nls_ksp_monitor -admm_sub_0_tao_nls_ksp_converged_reason
    args: -admm_sub_1_tao_monitor_short -admm_sub_1_tao_converged_reason -admm_sub_1_tao_nls_ksp_monitor -admm_sub_1_tao_nls_ksp_converged_reason

TEST*/
