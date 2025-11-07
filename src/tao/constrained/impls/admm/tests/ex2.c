const char help[] = "Test TAOADMM on quadratic problems";

#include <petsctao.h>

typedef struct {
  Mat         C, D;
  PetscInt    N, M_C, M_D;
  PetscScalar alpha, beta;
  Vec         u, init_sol;
  MPI_Comm    comm;
  PetscViewer viewer;
  PetscRandom rand;
} AppCtx;

typedef struct {
  TaoTerm term_f, term_g;
  Vec     tmp;
  Tao     tao;
} ADMMTestCtx;

PetscErrorCode InitializeUser(MPI_Comm comm, AppCtx *user)
{
  PetscFunctionBegin;
  user->viewer = PETSC_VIEWER_STDOUT_(comm);
  user->N      = 13;
  user->M_C    = 4;
  user->M_D    = 17;
  user->alpha  = 1.0;
  user->beta   = 0.4;
  PetscOptionsBegin(comm, NULL, "Options for the unconstrained test", __FILE__);
  PetscCall(PetscOptionsInt("-n", "solution size", NULL, user->N, &user->N, NULL));
  PetscCall(PetscOptionsInt("-m_c", "number of C matrix rows", NULL, user->M_C, &user->M_C, NULL));
  PetscCall(PetscOptionsInt("-m_d", "number of D matrix rows", NULL, user->M_D, &user->M_D, NULL));
  PetscOptionsEnd();

  PetscCall(PetscRandomCreate(comm, &user->rand));
  PetscCall(PetscRandomSetSeed(user->rand, 0));

  // C matrix
  PetscCall(MatCreate(comm, &user->C));
  PetscCall(MatSetSizes(user->C, PETSC_DECIDE, PETSC_DECIDE, user->M_C, user->N));
  PetscCall(MatSetType(user->C, MATDENSE));
  PetscCall(PetscObjectSetName((PetscObject)user->C, "C"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->C, "C_"));
  PetscCall(MatSetFromOptions(user->C));
  PetscCall(MatSetUp(user->C));
  PetscCall(MatSetRandom(user->C, user->rand));

  // D matrix
  PetscCall(MatCreate(comm, &user->D));
  PetscCall(MatSetSizes(user->D, PETSC_DECIDE, PETSC_DECIDE, user->M_D, user->N));
  PetscCall(MatSetType(user->D, MATDENSE));
  PetscCall(PetscObjectSetName((PetscObject)user->D, "D"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->D, "D_"));
  PetscCall(MatSetFromOptions(user->D));
  PetscCall(MatSetUp(user->D));
  PetscCall(MatSetRandom(user->D, user->rand));

  // u vector
  PetscCall(MatCreateVecs(user->C, NULL, &user->u));
  PetscCall(PetscObjectSetName((PetscObject)user->u, "u"));
  PetscCall(VecSetRandom(user->u, user->rand));

  // initial solution
  PetscCall(MatCreateVecs(user->C, &user->init_sol, NULL));
  PetscCall(VecSetRandom(user->init_sol, user->rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DestroyUser(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(&user->C));
  PetscCall(MatDestroy(&user->D));
  PetscCall(VecDestroy(&user->u));
  PetscCall(VecDestroy(&user->init_sol));
  PetscCall(PetscRandomDestroy(&user->rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestADMMCtxCreate(MPI_Comm comm, ADMMTestCtx *actx, AppCtx *user)
{
  PetscFunctionBegin;
  // first L2 term f
  PetscCall(TaoTermCreate(comm, &actx->term_f));
  PetscCall(TaoTermSetSolutionTemplate(actx->term_f, user->u));
  PetscCall(TaoTermSetType(actx->term_f, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetParametersMode(actx->term_f, TAOTERM_PARAMETERS_REQUIRED));
  PetscCall(PetscObjectSetName((PetscObject)actx->term_f, "f"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)actx->term_f, "f_"));
  PetscCall(TaoTermSetFromOptions(actx->term_f));

  // second L2 term g
  PetscCall(TaoTermCreate(comm, &actx->term_g));
  PetscCall(MatCreateVecs(user->D, NULL, &actx->tmp));
  PetscCall(TaoTermSetSolutionTemplate(actx->term_g, actx->tmp));
  PetscCall(VecDestroy(&actx->tmp));
  PetscCall(TaoTermSetType(actx->term_g, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetParametersMode(actx->term_g, TAOTERM_PARAMETERS_NONE));
  PetscCall(PetscObjectSetName((PetscObject)actx->term_g, "g"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)actx->term_g, "g_"));
  PetscCall(TaoTermSetFromOptions(actx->term_g));

  // Tao
  PetscCall(TaoCreate(comm, &actx->tao));
  PetscCall(TaoAddTerm(actx->tao, NULL, user->alpha, actx->term_f, user->u, user->C));
  PetscCall(TaoAddTerm(actx->tao, NULL, user->beta, actx->term_g, NULL, user->D));
  PetscCall(TaoSetType(actx->tao, TAOADMM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// test on alpha/2 || C x - u ||_2^2 + beta/2 || D x ||_2^2
PetscErrorCode TestADMMCtxSolve(ADMMTestCtx *actx, AppCtx *user, Vec *sol)
{
  PetscFunctionBegin;
  PetscCall(TaoSetFromOptions(actx->tao));
  PetscCall(VecDuplicate(user->init_sol, sol));
  PetscCall(VecCopy(user->init_sol, *sol));
  PetscCall(TaoSetSolution(actx->tao, *sol));
  PetscCall(TaoSolve(actx->tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestADMMCtxDestoryReturnSol(ADMMTestCtx *actx, AppCtx *user, Vec *sol)
{
  PetscFunctionBegin;
  PetscCall(TaoDestroy(&actx->tao));
  PetscCall(TaoTermDestroy(&actx->term_g));
  PetscCall(TaoTermDestroy(&actx->term_f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestADMMGetter(MPI_Comm comm, ADMMTestCtx *actx, AppCtx *user, PetscBool fmap, PetscBool gmap)
{
  const PetscInt *fterms_get;
  const PetscInt *gterms_get;
  PetscInt        fnum, gnum;
  PetscBool       f_mapped_test, g_mapped_test;

  PetscFunctionBegin;
  PetscCall(TaoADMMGetTermGroups(actx->tao, &fnum, &fterms_get, &f_mapped_test, &gnum, &gterms_get, &g_mapped_test));
  PetscAssert(fnum == 1, comm, PETSC_ERR_ARG_CORRUPT, "fnum should equal to 1");
  PetscAssert(gnum == 1, comm, PETSC_ERR_ARG_CORRUPT, "gnum should equal to 1");
  PetscAssert(fterms_get[0] == 0, comm, PETSC_ERR_ARG_CORRUPT, "ftermidx should equal to 0");
  PetscAssert(gterms_get[0] == 1, comm, PETSC_ERR_ARG_CORRUPT, "gtermidx should equal to 1");
  PetscAssert(f_mapped_test == fmap, comm, PETSC_ERR_ARG_CORRUPT, "f_mapped should be false");
  PetscAssert(g_mapped_test == gmap, comm, PETSC_ERR_ARG_CORRUPT, "g_mapped should be false");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// min f(Ax) + g(Bz) s.t. x = z
PetscErrorCode TestADMMUnconstrained1(MPI_Comm comm, AppCtx *user)
{
  ADMMTestCtx   *actx;
  Vec            sol1, sol2, sol3, sol4, sol5;
  const PetscInt f_terms[1] = {0};
  const PetscInt g_terms[1] = {1};
  PetscReal      norm1, norm2, norm3, norm4;

  PetscFunctionBegin;
  PetscCall(PetscNew(&actx));

  //Total of five cases
  // case 1
  // default behavior: min_{x,z} f(Ax) + g(Bz) s.t. x == z
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(Bz) s.t. x == z, default \n"));
  PetscCall(TestADMMCtxSolve(actx, user, &sol1));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol1));

  // case 2
  // request min_{x,z} f(Ax) + g(Bz) s.t. x == z
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(Bz) s.t. x == z, with SetTermGroups \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, NULL, PETSC_FALSE, 1, NULL, PETSC_FALSE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol2));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol2));

  // case 3
  // min_{x,z} f(Ax) + g(Bz) s.t. x == z, with explicit f_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(Bz) s.t. x == z, with SetTermGroups and explicit f_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, f_terms, PETSC_FALSE, 1, NULL, PETSC_FALSE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol3));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol3));

  // case 4
  // min_{x,z} f(Ax) + g(Bz) s.t. x == z, with explicit g_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(Bz) s.t. x == z, with SetTermGroups and explicit g_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, NULL, PETSC_FALSE, 1, g_terms, PETSC_FALSE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol4));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol4));

  // case 5
  // min_{x,z} f(Ax) + g(Bz) s.t. x == z, with explicit f,g_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(Bz) s.t. x == z, with SetTermGroups and explicit f,g_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, f_terms, PETSC_FALSE, 1, g_terms, PETSC_FALSE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol5));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol5));

  // compare
  PetscCall(VecAXPY(sol2, -1., sol1));
  PetscCall(VecAXPY(sol3, -1., sol1));
  PetscCall(VecAXPY(sol4, -1., sol1));
  PetscCall(VecAXPY(sol5, -1., sol1));
  PetscCall(VecNorm(sol2, NORM_2, &norm1));
  PetscCall(VecNorm(sol3, NORM_2, &norm2));
  PetscCall(VecNorm(sol4, NORM_2, &norm3));
  PetscCall(VecNorm(sol5, NORM_2, &norm4));

  PetscCall(PetscViewerASCIIPrintf(user->viewer, "1-1, solution difference: %g\n", (double)norm1));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "1-2, solution difference: %g\n", (double)norm2));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "1-3, solution difference: %g\n", (double)norm3));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "1-4, solution difference: %g\n", (double)norm4));

  PetscCall(PetscFree(actx));
  PetscCall(VecDestroy(&sol1));
  PetscCall(VecDestroy(&sol2));
  PetscCall(VecDestroy(&sol3));
  PetscCall(VecDestroy(&sol4));
  PetscCall(VecDestroy(&sol5));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// min f(x) + g(Bz) s.t. x = Az
PetscErrorCode TestADMMUnconstrained2(MPI_Comm comm, AppCtx *user)
{
  ADMMTestCtx   *actx;
  Vec            sol1, sol2, sol3, sol4;
  const PetscInt f_terms[1] = {0};
  const PetscInt g_terms[1] = {1};
  PetscReal      norm1, norm2, norm3;

  PetscFunctionBegin;
  PetscCall(PetscNew(&actx));

  // Total of four cases
  // case 1
  // min f(x) + g(Bz) s.t. x = Az
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(x) + g(Bz) s.t. x == Az, with SetTermGroups \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, NULL, PETSC_TRUE, 1, NULL, PETSC_FALSE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol1));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_TRUE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol1));

  // case 3
  // min f(x) + g(Bz) s.t. x = Az, with explicit f_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(x) + g(Bz) s.t. x == Az, with SetTermGroups and explicit f_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, f_terms, PETSC_TRUE, 1, NULL, PETSC_FALSE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol2));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_TRUE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol2));

  // case 4
  // min f(x) + g(Bz) s.t. x = Az, with explicit g_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(x) + g(Bz) s.t. x == Az, with SetTermGroups and explicit g_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, NULL, PETSC_TRUE, 1, g_terms, PETSC_FALSE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol3));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_TRUE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol3));

  // case 5
  // min f(x) + g(Bz) s.t. x = Az, with explicit f,g_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(x) + g(Bz) s.t. x == Az, with SetTermGroups and explicit f,g_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, f_terms, PETSC_TRUE, 1, g_terms, PETSC_FALSE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol4));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_TRUE, PETSC_FALSE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol4));

  // compare
  PetscCall(VecAXPY(sol2, -1., sol1));
  PetscCall(VecAXPY(sol3, -1., sol1));
  PetscCall(VecAXPY(sol4, -1., sol1));
  PetscCall(VecNorm(sol2, NORM_2, &norm1));
  PetscCall(VecNorm(sol3, NORM_2, &norm2));
  PetscCall(VecNorm(sol4, NORM_2, &norm3));

  PetscCall(PetscViewerASCIIPrintf(user->viewer, "2-1, solution difference: %g\n", (double)norm1));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "2-2, solution difference: %g\n", (double)norm2));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "2-3, solution difference: %g\n", (double)norm3));

  PetscCall(PetscFree(actx));
  PetscCall(VecDestroy(&sol1));
  PetscCall(VecDestroy(&sol2));
  PetscCall(VecDestroy(&sol3));
  PetscCall(VecDestroy(&sol4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// min f(Ax) + g(z) s.t. z = Bx
PetscErrorCode TestADMMUnconstrained3(MPI_Comm comm, AppCtx *user)
{
  ADMMTestCtx   *actx;
  Vec            sol1, sol2, sol3, sol4;
  const PetscInt f_terms[1] = {0};
  const PetscInt g_terms[1] = {1};
  PetscReal      norm1, norm2, norm3;

  PetscFunctionBegin;
  PetscCall(PetscNew(&actx));

  // Total of three cases
  // case 1
  // min f(Ax) + g(z) s.t. z = Bx
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(z) s.t. z == Bx, with SetTermGroups \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, NULL, PETSC_FALSE, 1, NULL, PETSC_TRUE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol1));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_TRUE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol1));

  // case 3
  // min f(Ax) + g(z) s.t. z = Bx, with explicit f_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(z) s.t. z == Bx, with SetTermGroups and explicit f_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, f_terms, PETSC_FALSE, 1, NULL, PETSC_TRUE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol2));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_TRUE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol2));

  // case 4
  // min f(Ax) + g(z) s.t. z = Bx, with explicit g_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(z) s.t. z == Bx, with SetTermGroups and explicit g_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, NULL, PETSC_FALSE, 1, g_terms, PETSC_TRUE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol3));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_TRUE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol3));

  // case 5
  // min f(Ax) + g(z) s.t. z = Bx, with explicit f,g_term indexing
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(Ax) + g(z) s.t. z == Bx, with SetTermGroups and explicit f,g_term indexing \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, f_terms, PETSC_FALSE, 1, g_terms, PETSC_TRUE));
  PetscCall(TestADMMCtxSolve(actx, user, &sol4));
  PetscCall(TestADMMGetter(comm, actx, user, PETSC_FALSE, PETSC_TRUE));
  PetscCall(TestADMMCtxDestoryReturnSol(actx, user, &sol4));

  // compare
  PetscCall(VecAXPY(sol2, -1., sol1));
  PetscCall(VecAXPY(sol3, -1., sol1));
  PetscCall(VecAXPY(sol4, -1., sol1));
  PetscCall(VecNorm(sol2, NORM_2, &norm1));
  PetscCall(VecNorm(sol3, NORM_2, &norm2));
  PetscCall(VecNorm(sol4, NORM_2, &norm3));

  PetscCall(PetscViewerASCIIPrintf(user->viewer, "3-1, solution difference: %g\n", (double)norm1));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "3-2, solution difference: %g\n", (double)norm2));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "3-3, solution difference: %g\n", (double)norm3));

  PetscCall(PetscFree(actx));
  PetscCall(VecDestroy(&sol1));
  PetscCall(VecDestroy(&sol2));
  PetscCall(VecDestroy(&sol3));
  PetscCall(VecDestroy(&sol4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm comm;
  AppCtx  *user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscNew(&user));
  comm = PETSC_COMM_WORLD;
  PetscCall(InitializeUser(comm, user));
  // min f(Ax) + g(Bz) s.t. x = z
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "----------------------------\n"));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "Testing min f(Ax) + g(Bz) s.t. x = z\n"));
  PetscCall(TestADMMUnconstrained1(comm, user));
  // min f(x) + g(Bz) s.t. x = Az
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "----------------------------\n"));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "Testing min f(x) + g(Bz) s.t. x = Az\n"));
  PetscCall(TestADMMUnconstrained2(comm, user));
  // min f(Ax) + g(z) s.t. z = Bx
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "----------------------------\n"));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "Testing min f(Ax) + g(z) s.t. z = Bx\n"));
  PetscCall(TestADMMUnconstrained3(comm, user));
  PetscCall(DestroyUser(user));
  PetscCall(PetscFree(user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: basic

   test:
      suffix: aradmm_exact
      args: -n 13 -m_c 13 -m_d 13 -C_mat_type constantdiagonal -D_mat_type constantdiagonal
      args: -tao_monitor_constraint_norm -tao_admm_update_type adaptive

TEST*/
