#include <petsctao.h>

static char help[] = "Tests TAOTERMSUM matrix-free dispatch across mapped assembled fallback, direct HessianMult, and separate Hpre paths.\n";

typedef struct {
  PetscBool provide_mult;
  PetscBool split_hpre;
  PetscInt  hessian_calls;
  PetscInt  mult_calls;
} TermCtx;

static PetscErrorCode FormObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  PetscFunctionBeginUser;
  *f = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  ctx->hessian_calls++;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(H, 2.0));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(Hpre, ctx->split_hpre ? 5.0 : 2.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianMult(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  TermCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &ctx));
  ctx->mult_calls++;
  PetscCall(VecCopy(v, Hv));
  PetscCall(VecScale(Hv, 2.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTerm(MPI_Comm comm, PetscInt n, TermCtx *ctx, TaoTerm *term)
{
  PetscFunctionBeginUser;
  PetscCall(TaoTermCreateShell(comm, ctx, NULL, term));
  PetscCall(TaoTermSetParametersMode(*term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(*term, PETSC_DECIDE, n, 1));
  PetscCall(TaoTermShellSetObjectiveAndGradient(*term, FormObjectiveAndGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(*term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(*term, ctx->split_hpre ? PETSC_FALSE : PETSC_TRUE, MATAIJ, ctx->split_hpre ? MATAIJ : NULL));
  PetscCall(TaoTermShellSetHessian(*term, FormHessian));
  if (ctx->provide_mult) PetscCall(TaoTermShellSetHessianMult(*term, FormHessianMult));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMap(MPI_Comm comm, PetscInt n, Mat *map)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 1, NULL, map));
  PetscCall(MatGetOwnershipRange(*map, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(MatSetValue(*map, i, i, i + 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(*map, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*map, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckAction(Mat H, Vec v, Vec work, PetscBool mapped_fallback, PetscBool mixed)
{
  PetscScalar *a;
  PetscReal    norm;
  PetscInt     rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatMult(H, v, work));
  PetscCall(VecGetOwnershipRange(work, &rstart, &rend));
  PetscCall(VecGetArray(work, &a));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscReal expected = 2.0 + 2.0 * (i + 1.0) * (i + 1.0);

    if (!mapped_fallback && !mixed) expected = 2.0;
    a[i - rstart] -= expected;
  }
  PetscCall(VecRestoreArray(work, &a));
  PetscCall(VecNorm(work, NORM_2, &norm));
  PetscCheck(norm <= 1.e-10, PetscObjectComm((PetscObject)H), PETSC_ERR_PLIB, "Matrix-free Hessian action is incorrect (norm of error %g)", (double)norm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RunCase(MPI_Comm comm, PetscBool mapped_fallback, PetscBool mixed, PetscBool split_hpre, PetscBool direct_only)
{
  const PetscInt n = 4;
  TermCtx       ctx[2] = {{0}};
  Tao           tao;
  TaoTerm       term[2];
  Mat           H, Hpre, map = NULL;
  Vec           x, v, work;
  PetscReal     norm;

  PetscFunctionBeginUser;
  ctx[0].provide_mult = mixed || split_hpre || direct_only;
  ctx[0].split_hpre   = split_hpre;
  ctx[1].provide_mult = split_hpre || direct_only;
  PetscCall(CreateTerm(comm, n, &ctx[0], &term[0]));
  PetscCall(CreateTerm(comm, n, &ctx[1], &term[1]));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term[0], "direct_"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)term[1], "fallback_"));
  if (mapped_fallback || mixed) PetscCall(CreateMap(comm, n, &map));
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, n, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecDuplicate(x, &v));
  PetscCall(VecDuplicate(x, &work));
  PetscCall(VecSet(v, 1.0));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAONLS));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoAddTerm(tao, "direct_", 1.0, term[0], NULL, NULL));
  PetscCall(TaoAddTerm(tao, "fallback_", direct_only ? 2.0 : 1.0, term[1], NULL, split_hpre ? NULL : map));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoGetHessianMatrices(tao, &H, &Hpre));
  PetscCall(TaoComputeHessian(tao, x, H, Hpre));

  if (direct_only) {
    PetscCall(MatMult(H, v, work));
    PetscCall(VecShift(work, -6.0));
    PetscCall(VecNorm(work, NORM_2, &norm));
    PetscCheck(norm <= 1.e-10, comm, PETSC_ERR_PLIB, "Direct-only HessianMult sum is incorrect (norm of error %g)", (double)norm);
    PetscCheck(ctx[0].mult_calls > 0 && ctx[0].hessian_calls == 0 && ctx[1].mult_calls > 0 && ctx[1].hessian_calls == 0, comm, PETSC_ERR_PLIB, "Direct-only shell sum did not use HessianMult exclusively for every summand");
  } else if (split_hpre) {
    PetscCall(MatMult(H, v, work));
    PetscCall(VecShift(work, -4.0));
    PetscCall(VecNorm(work, NORM_2, &norm));
    PetscCheck(norm <= 1.e-10, comm, PETSC_ERR_PLIB, "Direct HessianMult action is incorrect (norm of error %g)", (double)norm);
    PetscCall(MatMult(Hpre, v, work));
    PetscCall(VecShift(work, -7.0));
    PetscCall(VecNorm(work, NORM_2, &norm));
    PetscCheck(norm <= 1.e-10, comm, PETSC_ERR_PLIB, "Separate assembled Hpre is incorrect (norm of error %g)", (double)norm);
    PetscCheck(ctx[0].mult_calls > 0 && ctx[0].hessian_calls > 0 && ctx[1].mult_calls > 0 && ctx[1].hessian_calls > 0, comm, PETSC_ERR_PLIB, "Direct HessianMult plus separate-Hpre path did not invoke both callbacks");
  } else {
    PetscCall(CheckAction(H, v, work, mapped_fallback, mixed));
    if (mixed) PetscCheck(ctx[0].mult_calls > 0 && ctx[0].hessian_calls == 0 && ctx[1].hessian_calls > 0, comm, PETSC_ERR_PLIB, "Mixed shell sum did not dispatch direct and fallback summands independently");
    else PetscCheck(ctx[1].hessian_calls > 0 && ctx[1].mult_calls == 0, comm, PETSC_ERR_PLIB, "Mapped assembled fallback did not use the Hessian callback");
  }

  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&term[0]));
  PetscCall(TaoTermDestroy(&term[1]));
  PetscCall(MatDestroy(&map));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscBool mapped_fallback = PETSC_FALSE, mixed = PETSC_FALSE, split_hpre = PETSC_FALSE, direct_only = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mapped_fallback", &mapped_fallback, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mixed", &mixed, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-split_hpre", &split_hpre, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-direct_only", &direct_only, NULL));
  PetscCheck((PetscInt)mapped_fallback + (PetscInt)mixed + (PetscInt)split_hpre + (PetscInt)direct_only == 1, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Select exactly one test case");
  PetscCall(RunCase(PETSC_COMM_WORLD, mapped_fallback, mixed, split_hpre, direct_only));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "TAOTERMSUM matrix-free dispatch is correct\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/taotermtest9.out
    requires: !complex

    test:
      suffix: mapped_fallback
      args: -mapped_fallback -tao_term_hessian_mat_type shell

    test:
      suffix: mixed
      args: -mixed -tao_term_hessian_mat_type shell

    test:
      suffix: split_hpre
      args: -split_hpre -tao_term_hessian_mat_type shell -tao_term_hessian_pre_is_hessian false -tao_term_hessian_pre_mat_type aij

    test:
      suffix: direct_only
      args: -direct_only -tao_term_hessian_mat_type shell

TEST*/
