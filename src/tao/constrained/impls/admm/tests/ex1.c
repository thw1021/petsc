const char help[] = "Test TAOADMM on quadratic problems";

#include <petsctao.h>

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

typedef struct _n_TaoAffineEqConstraint {
  Mat J;
  Vec d;
} TaoAffineEqConstraint;

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
  if (J) { PetscCall(MatCopy(c->J, J, SAME_NONZERO_PATTERN)); }
  if (Jpre && Jpre != J) { PetscCall(MatCopy(c->J, Jpre, SAME_NONZERO_PATTERN)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// test on alpha/2 || C x - u ||_2^2 + beta/2 || D x ||_2^2
static PetscErrorCode TestADMMUnconstrained(PetscRandom rand, PetscInt test_number, PetscBool *keep_going)
{
  MPI_Comm    comm;
  PetscInt    N, M_C, M_D;
  Mat         C, D;
  Vec         u;
  PetscScalar alpha, beta;
  TaoTerm     term_f, term_g;
  Vec         solution, tmp;
  Tao         tao;
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)rand, &comm));
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

  PetscCall(CreateNamedRandomDenseMat(rand, M_C, N, "C", "C_", &C));
  PetscCall(CreateNamedRandomDenseMat(rand, M_D, N, "D", "D_", &D));

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
  PetscCall(TaoSetTerm(tao, alpha, term_f, u, C));
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

static PetscErrorCode TestADMMConstrained(PetscRandom rand)
{
  MPI_Comm              comm;
  PetscInt              N_C, N_D, K;
  Mat                   C, D, E, F;
  Mat                   EqConstraints;
  Vec                   d, u, ce;
  PetscScalar           alpha, beta;
  TaoTerm               term_f, term_g;
  Vec                   solution, tmp;
  Tao                   tao;
  PetscViewer           viewer;
  TaoAffineEqConstraint constraints;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)rand, &comm));
  N_C    = 11;
  N_D    = 13;
  K      = 17;
  alpha  = 1.0;
  beta   = 0.4;
  viewer = PETSC_VIEWER_STDOUT_(comm);
  PetscOptionsBegin(comm, NULL, "Options for the unconstrained test", __FILE__);
  PetscCall(PetscOptionsInt("-n_e", "number of E matrix columns", NULL, N_C, &N_C, NULL));
  PetscCall(PetscOptionsInt("-n_f", "number of F matrix columns", NULL, N_D, &N_D, NULL));
  PetscCall(PetscOptionsInt("-k", "number of constraint matrix rows", NULL, K, &K, NULL));
  PetscOptionsEnd();
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s:\n", PETSC_FUNCTION_NAME));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscRandomSetSeed(rand, 0));

  PetscCall(CreateNamedRandomDenseMat(rand, K, N_C, "E", "E_", &E));
  PetscCall(CreateNamedRandomDenseMat(rand, K, N_D, "F", "F_", &F));

  {
    Mat EF[2];
    Mat CD[2];

    EF[0] = E;
    EF[1] = F;
    PetscCall(MatCreateNest(comm, 1, NULL, 2, NULL, EF, &EqConstraints));
    PetscCall(MatNestCreateRowMatScatters(EqConstraints, CD));
    C = CD[0];
    D = CD[1];
    PetscCall(PetscObjectSetName((PetscObject)C, "P_x"));
    PetscCall(PetscObjectSetName((PetscObject)D, "P_y"));
  }

  // d vector
  PetscCall(MatCreateVecs(EqConstraints, NULL, &d));
  PetscCall(PetscObjectSetName((PetscObject)d, "d"));
  PetscCall(VecSetRandom(d, rand));

  constraints.J = EqConstraints;
  constraints.d = d;

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
  PetscCall(TaoSetTerm(tao, alpha, term_f, u, C));
  PetscCall(TaoAddTerm(tao, NULL, beta, term_g, NULL, D));
  PetscCall(VecDuplicate(d, &ce));
  PetscCall(TaoSetEqualityConstraintsRoutine(tao, ce, TaoComputeAffineEqualityConstraints, (void *)&constraints));
  PetscCall(TaoSetJacobianEqualityRoutine(tao, EqConstraints, NULL, TaoComputeAffineEqualityJacobian, (void *)&constraints));

  PetscCall(TaoSetType(tao, TAOADMM));

  // solve f(P_x(u)) + g(P_y(u)) s.t. EqConstraints * u + d == 0
  //
  // because EqConstraints = [E * P_x(u) | F * P_y(u)]
  // this is equivalent to
  //
  // solve f(x) + g(y) s.t. E * x + F * y + d == 0
  PetscCall(TaoADMMSetTermGroups(tao, 1, NULL, PETSC_TRUE, 1, NULL, PETSC_TRUE));

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(MatCreateVecs(EqConstraints, &solution, NULL));
  PetscCall(VecSetRandom(solution, rand));
  PetscCall(TaoSetSolution(tao, solution));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&term_g));
  PetscCall(TaoTermDestroy(&term_f));
  PetscCall(VecDestroy(&ce));
  PetscCall(VecDestroy(&solution));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&d));
  PetscCall(MatDestroy(&EqConstraints));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&E));
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

    PetscCall(TestADMMUnconstrained(rand, i, &keep_going));
    if (!keep_going) break;
  }
  PetscCall(TestADMMConstrained(rand));
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
