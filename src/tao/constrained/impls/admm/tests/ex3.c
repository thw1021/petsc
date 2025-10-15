const char help[] = "Test TAOADMM on quadratic problems with equality constraints";

#include <petsctao.h>

typedef struct {
  TaoTerm term_f, term_g;
  Vec     tmp;
  Tao     tao;
} ADMMTestCtx;

typedef struct _n_TaoAffineEqConstraint {
  Mat J;
  Vec d;
} TaoAffineEqConstraint;

typedef struct {
  Mat                   C, D, E, F;
  Mat                   EqConstraints;
  Vec                   d, u, ce, init_sol;
  PetscInt              K, M_C, M_D;
  PetscScalar           alpha, beta;
  PetscRandom           rand;
  PetscViewer           viewer;
  TaoAffineEqConstraint constraints;
} AppCtx;

static PetscErrorCode CreateNamedRandomDenseMat(PetscRandom rand, PetscInt M, PetscInt N, const char name[], const char prefix[], Mat *mat)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)rand, &comm));
  PetscCall(MatCreate(comm, mat));
  PetscCall(MatSetSizes(*mat, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetType(*mat, MATDENSE));
  PetscCall(PetscObjectSetName((PetscObject)*mat, name));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*mat, prefix));
  PetscCall(MatSetFromOptions(*mat));
  PetscCall(MatSetUp(*mat));
  PetscCall(MatSetRandom(*mat, rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNestCreateRowMatScatters(Mat nest, Mat mat_scatters[])
{
  MPI_Comm comm;
  PetscInt n_row_blocks;
  IS      *allISs;
  Vec      full_vec;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(nest, &full_vec, NULL));
  PetscCall(PetscObjectGetComm((PetscObject)nest, &comm));
  PetscCall(MatNestGetSize(nest, NULL, &n_row_blocks));
  PetscCall(PetscMalloc1(n_row_blocks, &allISs));
  PetscCall(MatNestGetISs(nest, NULL, allISs));
  for (PetscInt i = 0; i < n_row_blocks; i++) {
    IS         is = allISs[i];
    VecScatter scatter;
    Mat        sub_mat;
    Vec        sub_vec;

    PetscCall(MatNestGetSubMat(nest, 0, i, &sub_mat));
    PetscCall(MatCreateVecs(sub_mat, &sub_vec, NULL));
    PetscCall(VecScatterCreate(full_vec, is, sub_vec, NULL, &scatter));
    PetscCall(MatCreateScatter(comm, scatter, &mat_scatters[i]));
    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&sub_vec));
  }
  PetscCall(VecDestroy(&full_vec));
  PetscCall(PetscFree(allISs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoComputeAffineEqualityConstraints(Tao tao, Vec x, Vec r, void *ctx)
{
  TaoAffineEqConstraint *c = (TaoAffineEqConstraint *)ctx;

  PetscFunctionBegin;
  PetscCall(MatMult(c->J, x, r));
  if (c->d) PetscCall(VecAXPY(r, 1.0, c->d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoComputeAffineEqualityJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *ctx)
{
  TaoAffineEqConstraint *c = (TaoAffineEqConstraint *)ctx;

  PetscFunctionBegin;
  if (J) PetscCall(MatCopy(c->J, J, SAME_NONZERO_PATTERN));
  if (Jpre && Jpre != J) PetscCall(MatCopy(c->J, Jpre, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InitializeUser(MPI_Comm comm, AppCtx *user)
{
  PetscFunctionBegin;
  user->viewer = PETSC_VIEWER_STDOUT_(comm);
  user->K      = 17;
  user->M_C    = 11;
  user->M_D    = 13;
  user->alpha  = 1.0;
  user->beta   = 0.4;
  PetscOptionsBegin(comm, NULL, "Options for the unconstrained test", __FILE__);
  PetscCall(PetscOptionsInt("-n", "solution size", NULL, user->K, &user->K, NULL));
  PetscCall(PetscOptionsInt("-m_c", "number of C matrix rows", NULL, user->M_C, &user->M_C, NULL));
  PetscCall(PetscOptionsInt("-m_d", "number of D matrix rows", NULL, user->M_D, &user->M_D, NULL));
  PetscOptionsEnd();

  PetscCall(PetscRandomCreate(comm, &user->rand));
  PetscCall(PetscRandomSetSeed(user->rand, 0));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "%s:\n", PETSC_FUNCTION_NAME));

  PetscCall(CreateNamedRandomDenseMat(user->rand, user->K, user->M_C, "E", "E_", &user->E));
  PetscCall(CreateNamedRandomDenseMat(user->rand, user->K, user->M_D, "F", "F_", &user->F));

  {
    Mat EF[2];
    Mat CD[2];

    EF[0] = user->E;
    EF[1] = user->F;
    PetscCall(MatCreateNest(comm, 1, NULL, 2, NULL, EF, &user->EqConstraints));
    PetscCall(MatNestCreateRowMatScatters(user->EqConstraints, CD));
    user->C = CD[0];
    user->D = CD[1];
    PetscCall(PetscObjectSetName((PetscObject)user->C, "P_x"));
    PetscCall(PetscObjectSetName((PetscObject)user->D, "P_y"));
  }

  // d vector
  PetscCall(MatCreateVecs(user->EqConstraints, NULL, &user->d));
  PetscCall(PetscObjectSetName((PetscObject)user->d, "d"));
  PetscCall(VecSetRandom(user->d, user->rand));

  user->constraints.J = user->EqConstraints;
  user->constraints.d = user->d;

  // u vector
  PetscCall(MatCreateVecs(user->C, NULL, &user->u));
  PetscCall(PetscObjectSetName((PetscObject)user->u, "u"));
  PetscCall(VecSetRandom(user->u, user->rand));

  // Initial solution
  PetscCall(MatCreateVecs(user->EqConstraints, &user->init_sol, NULL));
  PetscCall(VecSetRandom(user->init_sol, user->rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestADMMCtxCreate(MPI_Comm comm, ADMMTestCtx *actx, AppCtx *user)
{
  Vec tmp;

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
  PetscCall(MatCreateVecs(user->D, NULL, &tmp));
  PetscCall(TaoTermSetSolutionTemplate(actx->term_g, tmp));
  PetscCall(VecDestroy(&tmp));
  PetscCall(TaoTermSetType(actx->term_g, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetParametersMode(actx->term_g, TAOTERM_PARAMETERS_NONE));
  PetscCall(PetscObjectSetName((PetscObject)actx->term_g, "g"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)actx->term_g, "g_"));
  PetscCall(TaoTermSetFromOptions(actx->term_g));

  // Tao
  PetscCall(TaoCreate(comm, &actx->tao));
  PetscCall(TaoAddTerm(actx->tao, NULL, user->alpha, actx->term_f, user->u, user->C));
  PetscCall(TaoAddTerm(actx->tao, NULL, user->beta, actx->term_g, NULL, user->D));
  PetscCall(VecDuplicate(user->d, &user->ce));
  PetscCall(TaoSetEqualityConstraintsRoutine(actx->tao, user->ce, TaoComputeAffineEqualityConstraints, (void *)&user->constraints));
  PetscCall(TaoSetJacobianEqualityRoutine(actx->tao, user->EqConstraints, NULL, TaoComputeAffineEqualityJacobian, (void *)&user->constraints));

  PetscCall(TaoSetType(actx->tao, TAOADMM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestADMMCtxSolveDestoryReturnSol(ADMMTestCtx *actx, AppCtx *user, Vec *sol)
{
  PetscFunctionBegin;
  PetscCall(TaoSetFromOptions(actx->tao));
  PetscCall(VecDuplicate(user->init_sol, sol));
  PetscCall(VecCopy(user->init_sol, *sol));
  PetscCall(TaoSolve(actx->tao));
  PetscCall(TaoDestroy(&actx->tao));
  PetscCall(TaoTermDestroy(&actx->term_g));
  PetscCall(TaoTermDestroy(&actx->term_f));
  PetscCall(VecDestroy(&user->ce));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DestroyUser(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&user->ce));
  PetscCall(VecDestroy(&user->u));
  PetscCall(VecDestroy(&user->d));
  PetscCall(VecDestroy(&user->init_sol));
  PetscCall(MatDestroy(&user->EqConstraints));
  PetscCall(MatDestroy(&user->F));
  PetscCall(MatDestroy(&user->E));
  PetscCall(MatDestroy(&user->D));
  PetscCall(MatDestroy(&user->C));
  PetscCall(PetscRandomDestroy(&user->rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestADMMConstrained(MPI_Comm comm, AppCtx *user)
{
  ADMMTestCtx   *actx;
  Vec            sol1, sol2, sol3, sol4;
  const PetscInt f_terms[1] = {0};
  const PetscInt g_terms[1] = {1};
  PetscReal      norm1, norm2, norm3;

  PetscFunctionBegin;
  PetscCall(PetscNew(&actx));

  // solve f(P_x(u)) + g(P_y(u)) s.t. EqConstraints * u + d == 0
  //
  // because EqConstraints = [E * P_x(u) | F * P_y(u)]
  // this is equivalent to
  //
  // solve f(x) + g(y) s.t. E * x + F * y + d == 0
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(x) + g(z) s.t. Ax + Bz + d = 0, default \n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, NULL, PETSC_TRUE, 1, NULL, PETSC_TRUE));
  PetscCall(TestADMMCtxSolveDestoryReturnSol(actx, user, &sol1));

  // case 2
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(x) + g(z) s.t. Ax + Bz + d = 0, with SetTermGroups and explicit f_term indexing\n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, f_terms, PETSC_TRUE, 1, NULL, PETSC_TRUE));
  PetscCall(TestADMMCtxSolveDestoryReturnSol(actx, user, &sol2));

  // case 3
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(x) + g(z) s.t. Ax + Bz + d = 0, with SetTermGroups and explicit g_term indexing\n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, NULL, PETSC_TRUE, 1, g_terms, PETSC_TRUE));
  PetscCall(TestADMMCtxSolveDestoryReturnSol(actx, user, &sol3));

  // case 4
  PetscCall(TestADMMCtxCreate(comm, actx, user));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "min f(x) + g(z) s.t. Ax + Bz + d = 0, with SetTermGroups and explicit f,g_term indexing\n"));
  PetscCall(TaoADMMSetTermGroups(actx->tao, 1, f_terms, PETSC_TRUE, 1, g_terms, PETSC_TRUE));
  PetscCall(TestADMMCtxSolveDestoryReturnSol(actx, user, &sol4));

  PetscCall(VecAXPY(sol2, -1., sol1));
  PetscCall(VecAXPY(sol3, -1., sol1));
  PetscCall(VecAXPY(sol4, -1., sol1));
  PetscCall(VecNorm(sol2, NORM_2, &norm1));
  PetscCall(VecNorm(sol3, NORM_2, &norm2));
  PetscCall(VecNorm(sol4, NORM_2, &norm3));

  PetscCall(PetscViewerASCIIPrintf(user->viewer, "1-1, solution difference: %g\n", (double)norm1));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "1-2, solution difference: %g\n", (double)norm2));
  PetscCall(PetscViewerASCIIPrintf(user->viewer, "1-3, solution difference: %g\n", (double)norm3));

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
  PetscCall(TestADMMConstrained(comm, user));
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
      args: -tao_converged_reason -tao_view

   test:
      suffix: aradmm_exact
      args: -n 13 -m_c 13 -m_d 13 -E_mat_type constantdiagonal -F_mat_type constantdiagonal
      args: -tao_monitor_constraint_norm -tao_admm_update_type adaptive -tao_max_it 10

TEST*/
