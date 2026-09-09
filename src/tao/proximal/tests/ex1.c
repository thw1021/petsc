static const char help[] = "TAOFB on problems with known solutions: soft thresholding and a linear objective over a box.\n";

#include <petsctao.h>

typedef struct {
  Vec c;
} AppCtx;

/* f(x) = 0.5 |x - c|^2, so that one forward-backward step from x = 0 with step 1 gives prox_g(c) */
static PetscErrorCode QuadraticObjGrad(TaoTerm term, Vec X, Vec param, PetscReal *f, Vec G)
{
  AppCtx     *user;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(VecWAXPY(G, -1.0, user->c, X));
  PetscCall(VecDot(G, G, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* f(x) = <c, x>, whose gradient never changes between iterates */
static PetscErrorCode LinearObjGrad(TaoTerm term, Vec X, Vec param, PetscReal *f, Vec G)
{
  AppCtx     *user;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(VecCopy(user->c, G));
  PetscCall(VecDot(user->c, X, &dot));
  *f = PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSolution(Vec x, Vec expected, const char name[])
{
  PetscReal error;

  PetscFunctionBeginUser;
  PetscCall(VecAXPY(expected, -1.0, x));
  PetscCall(VecNorm(expected, NORM_INFINITY, &error));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)x), "%s: %s\n", name, error < 1.e-6 ? "PASS" : "FAIL"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao                tao;
  TaoTerm            fterm, gterm;
  Vec                x, expected;
  AppCtx             user;
  PetscInt           nsolves = 1, i, niter1 = 0;
  PetscReal          lambda = 10.0, f1 = 0.0;
  PetscBool          linear    = PETSC_FALSE;
  const PetscScalar  c_quad[4] = {3.0, -3.0, 0.5, 20.0}, c_lin[4] = {1.0, -2.0, 0.5, 3.0};
  const PetscScalar  soft[4] = {0.0, 0.0, 0.0, 10.0}, box[4] = {-1.0, 1.0, -1.0, -1.0};
  const PetscScalar *cvals, *xvals;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-linear", &linear, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsolves", &nsolves, NULL));
  cvals = linear ? c_lin : c_quad;
  xvals = linear ? box : soft;

  PetscCall(VecCreate(PETSC_COMM_WORLD, &user.c));
  PetscCall(VecSetSizes(user.c, PETSC_DECIDE, 4));
  PetscCall(VecSetFromOptions(user.c));
  PetscCall(VecDuplicate(user.c, &x));
  PetscCall(VecDuplicate(user.c, &expected));
  for (i = 0; i < 4; i++) {
    PetscCall(VecSetValue(user.c, i, cvals[i], INSERT_VALUES));
    PetscCall(VecSetValue(expected, i, xvals[i], INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(user.c));
  PetscCall(VecAssemblyEnd(user.c));
  PetscCall(VecAssemblyBegin(expected));
  PetscCall(VecAssemblyEnd(expected));

  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &fterm));
  PetscCall(TaoTermSetType(fterm, TAOTERMSHELL));
  PetscCall(TaoTermShellSetContext(fterm, &user));
  PetscCall(TaoTermSetParametersMode(fterm, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermShellSetObjectiveAndGradient(fterm, linear ? LinearObjGrad : QuadraticObjGrad));
  PetscCall(TaoTermSetLipschitz(fterm, 1.0));
  PetscCall(TaoTermSetSolutionTemplate(fterm, x));
  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &gterm));
  if (linear) {
    PetscCall(TaoTermSetType(gterm, TAOTERMBOX));
    PetscCall(TaoTermBoxSetBounds(gterm, -1.0, 1.0, NULL, NULL));
    lambda = 1.0;
  } else PetscCall(TaoTermSetType(gterm, TAOTERML1));
  PetscCall(TaoTermSetSolutionTemplate(gterm, x));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetType(tao, TAOFB));
  PetscCall(TaoAddTerm(tao, "f_", 1.0, fterm, NULL, NULL));
  PetscCall(TaoAddTerm(tao, "g_", lambda, gterm, NULL, NULL));
  PetscCall(TaoSetFromOptions(tao));
  /* Repeated solves from the same initial point must not depend on state left by an earlier solve */
  for (i = 0; i < nsolves; i++) {
    PetscInt  niter;
    PetscReal f;

    PetscCall(VecSet(x, 0.0));
    PetscCall(TaoSolve(tao));
    PetscCall(TaoGetIterationNumber(tao, &niter));
    PetscCall(TaoGetSolutionStatus(tao, NULL, &f, NULL, NULL, NULL, NULL));
    if (i == 0) {
      niter1 = niter;
      f1     = f;
    } else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "solve %" PetscInt_FMT " matches solve 1: %s\n", i + 1, (niter == niter1 && f == f1) ? "PASS" : "FAIL"));
  }
  PetscCall(CheckSolution(x, expected, linear ? "linear objective over box" : "soft thresholding"));

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&fterm));
  PetscCall(TaoTermDestroy(&gterm));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&expected));
  PetscCall(VecDestroy(&user.c));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex

  testset:
    args: -tao_max_it 200
    output_file: output/ex1_soft.out
    test:
      suffix: soft_fixed
      args: -tao_fb_accel 0 -tao_ls_max_funcs 0
    test:
      suffix: soft_fista
      args: -tao_fb_accel 1 -tao_ls_max_funcs 0
    test:
      suffix: soft_backtrack
      args: -tao_fb_accel 0 -tao_ls_max_funcs 30 -tao_fb_ls_scale 1.05
    test:
      suffix: soft_adaptive
      args: -tao_fb_accel 0 -tao_fb_adaptive 1
    test:
      suffix: soft_parallel
      nsize: 2
      args: -tao_fb_accel 1 -tao_ls_max_funcs 30 -tao_fb_ls_scale 1.05

  test:
    suffix: soft_nonmonotone_resolve
    args: -tao_max_it 200 -nsolves 3 -tao_fb_accel 0 -tao_ls_max_funcs 30 -tao_ls_ps_memory_size 5 -tao_fb_ls_scale 1.05
    output_file: output/ex1_soft_resolve.out

  test:
    suffix: linear_box_adaptive
    args: -linear -tao_max_it 50 -tao_fb_accel 0 -tao_fb_adaptive 1
    output_file: output/ex1_linear.out

TEST*/
