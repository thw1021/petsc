const char help[] = "Solve a least-squares problem with regularizers added through TaoAddTerm().\n";

#include <petsctao.h>
#include "taotermtestclassic.h"

typedef struct {
  MPI_Comm  comm;
  PetscInt  n;
  Vec       target;
  PetscBool use_term1;
  PetscBool term1_has_A;
  PetscBool term1_has_params;
  PetscBool term1_shell;
  PetscBool term1_hessian_mult;
  PetscReal term1_scale;
  PetscInt  map_row_size;
  PetscBool separate_hpre;
  PetscBool callback_hessian_mult;
  PetscBool repeat_setfromoptions;
} TestCtx;

/* Forward declarations */
static PetscErrorCode TestCtxInitialize(MPI_Comm, TestCtx *);
static PetscErrorCode TestCtxFinalize(TestCtx *);
static PetscErrorCode CreateTaoTermWithOptions(TestCtx *, TaoTerm *, Vec *, Mat *, const char *, const char *, PetscBool, PetscBool, PetscBool, PetscBool);
static PetscErrorCode FormFunctionGradient_Callback(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_Callback(Tao, Vec, Mat, Mat, void *);
static PetscErrorCode FormHessianMult_Callback(Tao, Vec, Vec, Vec, void *);
static PetscErrorCode FormObjectiveGradient_Shell(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian_Shell(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode FormHessianMult_Shell(TaoTerm, Vec, Vec, Vec, Vec);
static PetscErrorCode SetClassicLeaf(TaoTerm, Mat, Vec, PetscReal, ExampleClassicLeaf *);

int main(int argc, char **argv)
{
  TestCtx           ctx;
  Tao               tao_term, ctao;
  Vec               x_term, cx;
  Mat               H_term, Hpre_term, cH, cHpre;
  TaoTerm           term1        = NULL, objective;
  ExampleClassicCtx cctx         = {0};
  Vec               term1_params = NULL;
  Mat               term1_A;
  MPI_Comm          comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(TestCtxInitialize(comm, &ctx));
  PetscCall(TaoCreate(comm, &tao_term));
  PetscCall(TaoSetOptionsPrefix(tao_term, "t_"));
  PetscCall(TaoSetType(tao_term, TAOLMVM));

  /* Create a least-squares objective using the traditional TaoSet interface */
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, ctx.n, ctx.n, 1, NULL, 0, NULL, &H_term));
  if (ctx.separate_hpre) {
    PetscCall(MatAssemblyBegin(H_term, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H_term, MAT_FINAL_ASSEMBLY));
    PetscCall(MatDuplicate(H_term, MAT_COPY_VALUES, &Hpre_term));
  } else Hpre_term = H_term;
  PetscCall(MatCreateVecs(H_term, &x_term, NULL));
  PetscCall(VecZeroEntries(x_term));
  PetscCall(TaoSetSolution(tao_term, x_term));
  PetscCall(TaoSetObjectiveAndGradient(tao_term, NULL, FormFunctionGradient_Callback, &ctx));
  PetscCall(TaoSetHessian(tao_term, H_term, Hpre_term, FormHessian_Callback, &ctx));
  if (ctx.callback_hessian_mult) PetscCall(TaoSetHessianMult(tao_term, FormHessianMult_Callback, &ctx));

  /* Add term 1 if requested */
  if (ctx.use_term1) {
    PetscCall(CreateTaoTermWithOptions(&ctx, &term1, &term1_params, &term1_A, "reg1_", "A1_", ctx.term1_has_A, ctx.term1_has_params, ctx.term1_shell, ctx.term1_hessian_mult));
    PetscCall(TaoAddTerm(tao_term, "reg1_", ctx.term1_scale, term1, term1_params, term1_A));
  }
  PetscCall(TaoSetFromOptions(tao_term));
  cctx.nleaves              = 1 + (PetscInt)ctx.use_term1;
  cctx.leaves[0].type       = EXAMPLE_CLASSIC_LEAST_SQUARES;
  cctx.leaves[0].parameters = ctx.target;
  cctx.leaves[0].scale      = 1.0;
  if (ctx.use_term1) PetscCall(SetClassicLeaf(term1, term1_A, term1_params, ctx.term1_scale, &cctx.leaves[1]));
  if (cctx.nleaves > 1) {
    PetscCall(TaoGetTerm(tao_term, NULL, &objective, NULL, NULL));
    for (PetscInt i = 0; i < cctx.nleaves; i++) PetscCall(TaoTermSumGetTermMask(objective, i, &cctx.leaves[i].mask));
  }
  PetscCall(VecDuplicate(x_term, &cx));
  PetscCall(VecCopy(x_term, cx));
  PetscCall(ExampleClassicCreateTao(comm, tao_term, cx, &cctx, &ctao, &cH, &cHpre));
  PetscCall(TaoSolve(tao_term));
  PetscCall(TaoSolve(ctao));
  PetscCall(ExampleClassicCompareResults(tao_term, x_term, ctao, cx));

  if (ctx.repeat_setfromoptions) {
    TaoTerm     objective;
    TaoTermMask mask_before = TAOTERM_MASK_NONE, mask_after;
    PetscInt    max_it;

    if (ctx.use_term1) {
      PetscCall(TaoGetTerm(tao_term, NULL, &objective, NULL, NULL));
      PetscCall(TaoTermSumGetTermMask(objective, 1, &mask_before));
      PetscCall(PetscOptionsSetValue(NULL, "-t_tao_term_sum_reg1_mask", "none"));
    }
    PetscCall(PetscOptionsSetValue(NULL, "-t_tao_max_it", "7"));
    PetscCall(TaoSetFromOptions(tao_term));
    PetscCall(TaoGetMaximumIterations(tao_term, &max_it));
    PetscCheck(max_it == 7, comm, PETSC_ERR_PLIB, "Repeated TaoSetFromOptions() did not update -t_tao_max_it");
    if (ctx.use_term1) {
      PetscCall(TaoTermSumGetTermMask(objective, 1, &mask_after));
      PetscCheck(mask_after == mask_before, comm, PETSC_ERR_PLIB, "Repeated TaoSetFromOptions() changed structural TaoTerm options after setup");
    }
    PetscCall(TaoSolve(tao_term));
    PetscCall(TaoSolve(ctao));
    PetscCall(ExampleClassicCompareResults(tao_term, x_term, ctao, cx));
    PetscCall(PetscPrintf(comm, "Repeated TaoSetFromOptions() check passed\n"));
  }

  PetscCall(TaoDestroy(&ctao));
  if (cHpre != cH) PetscCall(MatDestroy(&cHpre));
  PetscCall(MatDestroy(&cH));
  PetscCall(VecDestroy(&cctx.hessian_x));
  PetscCall(VecDestroy(&cx));
  if (ctx.use_term1) {
    PetscCall(VecDestroy(&term1_params));
    PetscCall(MatDestroy(&term1_A));
  }
  PetscCall(TaoDestroy(&tao_term));
  PetscCall(TaoTermDestroy(&term1));
  PetscCall(VecDestroy(&x_term));
  PetscCall(MatDestroy(&H_term));
  if (ctx.separate_hpre) PetscCall(MatDestroy(&Hpre_term));
  PetscCall(TestCtxFinalize(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode TestCtxInitialize(MPI_Comm comm, TestCtx *ctx)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(TestCtx)));

  /* Default configuration */
  ctx->comm                  = comm;
  ctx->n                     = 10;
  ctx->use_term1             = PETSC_FALSE;
  ctx->term1_has_A           = PETSC_FALSE;
  ctx->term1_has_params      = PETSC_FALSE;
  ctx->term1_shell           = PETSC_FALSE;
  ctx->term1_hessian_mult    = PETSC_TRUE;
  ctx->term1_scale           = 0.1;
  ctx->map_row_size          = ctx->n - 1;
  ctx->separate_hpre         = PETSC_FALSE;
  ctx->callback_hessian_mult = PETSC_FALSE;
  ctx->repeat_setfromoptions = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "TaoTerm Coverage Test Options", "TAO");
  PetscCall(PetscOptionsBool("-use_term1", "Use first additional term", "", ctx->use_term1, &ctx->use_term1, NULL));
  PetscCall(PetscOptionsBool("-term1_has_A", "Term 1 has a map matrix A", "", ctx->term1_has_A, &ctx->term1_has_A, NULL));
  PetscCall(PetscOptionsBool("-term1_has_params", "Term 1 has parameters", "", ctx->term1_has_params, &ctx->term1_has_params, NULL));
  PetscCall(PetscOptionsBool("-term1_shell", "Create term 1 as a least-squares shell term", "", ctx->term1_shell, &ctx->term1_shell, NULL));
  PetscCall(PetscOptionsBool("-term1_hessian_mult", "Register HessianMult on shell term 1", "", ctx->term1_hessian_mult, &ctx->term1_hessian_mult, NULL));
  PetscCall(PetscOptionsReal("-term1_scale", "Scaling for term 1", "", ctx->term1_scale, &ctx->term1_scale, NULL));
  PetscCall(PetscOptionsInt("-map_row_size", "Row size of mapping matrix", "", ctx->map_row_size, &ctx->map_row_size, NULL));
  PetscCall(PetscOptionsBool("-separate_hpre", "Use a separate preconditioning matrix for the legacy callback term", "", ctx->separate_hpre, &ctx->separate_hpre, NULL));
  PetscCall(PetscOptionsBool("-callback_hessian_mult", "Register HessianMult on the legacy callback term", "", ctx->callback_hessian_mult, &ctx->callback_hessian_mult, NULL));
  PetscCall(PetscOptionsBool("-repeat_setfromoptions", "Call TaoSetFromOptions() again after TaoSolve()", "", ctx->repeat_setfromoptions, &ctx->repeat_setfromoptions, NULL));
  PetscOptionsEnd();
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, ctx->n, &ctx->target));
  PetscCall(VecGetOwnershipRange(ctx->target, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(VecSetValue(ctx->target, i, 1.0 + 0.05 * (i + 1), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(ctx->target));
  PetscCall(VecAssemblyEnd(ctx->target));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Finalize test context */
static PetscErrorCode TestCtxFinalize(TestCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&ctx->target));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTaoTermWithOptions(TestCtx *ctx, TaoTerm *term, Vec *params, Mat *A, const char *term_prefix, const char *A_prefix, PetscBool has_A, PetscBool has_params, PetscBool shell, PetscBool hessian_mult)
{
  MPI_Comm    comm = ctx->comm;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  *term   = NULL;
  *params = NULL;
  *A      = NULL;

  PetscCallMPI(MPI_Comm_size(comm, &size));
  /* Create parameters if requested */
  if (has_params) {
    PetscCall(VecCreate(comm, params));
    PetscCall(VecSetSizes(*params, PETSC_DECIDE, has_A ? ctx->map_row_size : ctx->n));
    PetscCall(VecSetFromOptions(*params));
    PetscCall(VecSetRandom(*params, NULL));
  }

  /* Create map matrix A if requested */
  if (has_A) {
    PetscCall(MatCreate(comm, A));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*A, A_prefix));
    PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, ctx->map_row_size, ctx->n));
    PetscCall(MatSetType(*A, MATAIJ)); /* Set default type before SetFromOptions */
    PetscCall(MatSetFromOptions(*A));
    /* Check matrix type and set up accordingly */
    if (size == 1) PetscCall(MatSeqAIJSetPreallocation(*A, PETSC_DEFAULT, NULL));
    else PetscCall(MatMPIAIJSetPreallocation(*A, 5, NULL, 5, NULL));
    PetscCall(MatSetUp(*A));
    PetscCall(MatSetRandom(*A, NULL));
    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  }
  /* Create TaoTerm, set prefix, and configure from options */
  PetscCall(TaoTermCreate(comm, term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*term, term_prefix));
  PetscCall(TaoTermSetSolutionSizes(*term, PETSC_DECIDE, has_A ? ctx->map_row_size : ctx->n, 1));
  if (shell) {
    PetscCall(TaoTermSetType(*term, TAOTERMSHELL));
    PetscCall(TaoTermSetParametersSizes(*term, PETSC_DECIDE, has_A ? ctx->map_row_size : ctx->n, 1));
    PetscCall(TaoTermShellSetObjectiveAndGradient(*term, FormObjectiveGradient_Shell));
    PetscCall(TaoTermShellSetCreateHessianMatrices(*term, TaoTermCreateHessianMatricesDefault));
    PetscCall(TaoTermSetCreateHessianMode(*term, PETSC_TRUE, MATAIJ, NULL));
    PetscCall(TaoTermShellSetHessian(*term, FormHessian_Shell));
    if (hessian_mult) PetscCall(TaoTermShellSetHessianMult(*term, FormHessianMult_Shell));
  }
  PetscCall(TaoTermSetFromOptions(*term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormFunctionGradient_Callback(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TestCtx    *ctx = (TestCtx *)ptr;
  Vec         diff;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(X, &diff));
  PetscCall(VecWAXPY(diff, -1.0, ctx->target, X));
  PetscCall(VecDot(diff, diff, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscCall(VecCopy(diff, G));
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian_Callback(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  if (ctx->separate_hpre) PetscCheck(!Hpre || Hpre != H, PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Distinct callback Hessian storage was aliased during evaluation");
  PetscCall(MatZeroEntries(H));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(H, 1.0));
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianMult_Callback(Tao tao, Vec X, Vec V, Vec HV, void *ptr)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(V, HV));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormObjectiveGradient_Shell(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  if (params) PetscCall(VecWAXPY(g, -1.0, params, x));
  else PetscCall(VecCopy(x, g));
  PetscCall(VecDot(g, g, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian_Shell(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(H));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(H, 1.0));
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessianMult_Shell(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetClassicLeaf(TaoTerm term, Mat map, Vec parameters, PetscReal scale, ExampleClassicLeaf *leaf)
{
  TaoTermType type;
  PetscBool   is_l1, is_l2, is_shell;

  PetscFunctionBeginUser;
  PetscCall(TaoTermGetType(term, &type));
  PetscCall(PetscStrcmp(type, TAOTERML1, &is_l1));
  PetscCall(PetscStrcmp(type, TAOTERMHALFL2SQUARED, &is_l2));
  PetscCall(PetscStrcmp(type, TAOTERMSHELL, &is_shell));
  PetscCheck(is_l1 || is_l2 || is_shell, PetscObjectComm((PetscObject)term), PETSC_ERR_SUP, "Classic reference supports only L1, half-L2, and shell least-squares added terms, not %s", type);
  leaf->type       = is_l1 ? EXAMPLE_CLASSIC_L1 : (is_l2 ? EXAMPLE_CLASSIC_HALF_L2 : EXAMPLE_CLASSIC_LEAST_SQUARES);
  leaf->map        = map;
  leaf->parameters = parameters;
  leaf->scale      = scale;
  if (is_l1) PetscCall(TaoTermL1GetEpsilon(term, &leaf->epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: l1_mapped
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -t_tao_type nls -use_term1 -term1_has_params -term1_has_A -term1_scale 0.012
    args: -reg1_tao_term_type l1 -t_tao_view ::ascii_info_detail

  test:
    suffix: shell
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_hessian_mat_type shell -t_tao_view ::ascii_info_detail

  test:
    suffix: shell_mapped
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A
    args: -t_tao_term_hessian_mat_type shell -t_tao_view ::ascii_info_detail

  test:
    suffix: shell_separate_hpre
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_hessian_mat_type shell -t_tao_term_hessian_pre_is_hessian false
    args: -t_tao_term_hessian_pre_mat_type aij -t_tao_view ::ascii_info_detail

  test:
    suffix: shell_mapped_separate_hpre
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A
    args: -t_tao_term_hessian_mat_type shell -t_tao_term_hessian_pre_is_hessian false
    args: -t_tao_term_hessian_pre_mat_type aij -t_tao_view ::ascii_info_detail

  test:
    suffix: masked_hessian
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -t_tao_type nls -use_term1 -reg1_tao_term_type halfl2squared
    args: -t_tao_term_sum_reg1_mask hessian -t_tao_view ::ascii_info_detail

  test:
    suffix: promoted_separate_hpre
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -t_tao_type nls -separate_hpre -use_term1 -reg1_tao_term_type halfl2squared
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: promoted_separate_hpre_shell
    filter: grep "Solution converged"
    args: -t_tao_type nls -separate_hpre -use_term1 -reg1_tao_term_type halfl2squared
    args: -t_tao_term_hessian_mat_type shell -t_tao_view ::ascii_info_detail
    output_file: output/taotermtest1_promoted_separate_hpre_shell.out

  test:
    suffix: halfl2_parameters
    filter: grep -E "parameter vector space|rows=.*cols=|Solution converged"
    args: -t_tao_type nls -use_term1 -term1_has_params -reg1_tao_term_type halfl2squared
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: halfl2_mapped_parameters
    filter: grep -E "parameter vector space|rows=.*cols=|Solution converged"
    args: -t_tao_type nls -use_term1 -term1_has_params -term1_has_A
    args: -reg1_tao_term_type halfl2squared -t_tao_view ::ascii_info_detail

  test:
    suffix: repeat_setfromoptions
    args: -repeat_setfromoptions -t_tao_fd_gradient
    filter: grep "Repeated TaoSetFromOptions"

  test:
    suffix: repeat_setfromoptions_sum
    args: -t_tao_type nls -use_term1 -repeat_setfromoptions
    args: -reg1_tao_term_type halfl2squared -t_tao_term_sum_reg1_mask hessian
    filter: grep "Repeated TaoSetFromOptions"
    output_file: output/taotermtest1_repeat_setfromoptions.out

  test:
    suffix: promoted_shell_fd_direct
    filter: grep "Classic callback comparison passed"
    output_file: output/taotermtest1_classic_comparison.out
    args: -t_tao_type nls -use_term1 -term1_shell -t_tao_fd_hessian
    args: -term1_hessian_mult -t_tao_term_hessian_mat_type shell

  test:
    suffix: promoted_shell_all_direct
    filter: grep "Classic callback comparison passed"
    output_file: output/taotermtest1_classic_comparison.out
    args: -t_tao_type nls -use_term1 -term1_shell -callback_hessian_mult
    args: -term1_hessian_mult -t_tao_term_hessian_mat_type shell

  test:
    suffix: promoted_shell_mapped_fallback
    filter: grep "Classic callback comparison passed"
    output_file: output/taotermtest1_classic_comparison.out
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A
    args: -term1_hessian_mult false -t_tao_term_sum_t_callbacks_mask objective,gradient,hessian
    args: -t_tao_term_hessian_mat_type shell

  test:
    suffix: promoted_mapped_mffd
    filter: grep "Classic callback comparison passed"
    output_file: output/taotermtest1_classic_comparison.out
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A
    args: -t_tao_term_hessian_mat_type mffd

  testset:
    filter: grep -E "Classic callback comparison passed|unused database options|Option left"
    output_file: output/taotermtest1_classic_comparison.out
    args: -t_tao_type nls -c_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_hessian_mat_type mffd -options_left

    test:
      suffix: r087_mffd_added
      args: -term1_scale 100 -t_tao_term_sum_t_callbacks_mask objective,gradient,hessian

    test:
      suffix: r088_mffd_mapped_added
      args: -term1_has_A -map_row_size 11 -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask objective,gradient,hessian

    test:
      suffix: r089_mffd_callback
      args: -t_tao_term_sum_reg1_mask objective,gradient,hessian

    test:
      suffix: r090_mffd_both

  test:
    suffix: callback_fd
    filter: grep "Classic callback comparison passed"
    output_file: output/taotermtest1_classic_comparison.out
    args: -t_tao_type nls -t_tao_fd_hessian

  test:
    suffix: callback_fd_separate_hpre
    filter: grep "Classic callback comparison passed"
    output_file: output/taotermtest1_classic_comparison.out
    args: -t_tao_type nls -t_tao_fd_hessian -separate_hpre

  test:
    suffix: r021_added_analytic
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_sum_t_callbacks_mask hessian -t_tao_view ::ascii_info_detail

  test:
    suffix: r022_added_fd
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -reg1_tao_term_hessian_use_fd true
    args: -t_tao_term_sum_t_callbacks_mask hessian -t_tao_view ::ascii_info_detail

  test:
    suffix: r025_callback_analytic
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_sum_reg1_mask hessian -t_tao_view ::ascii_info_detail

  test:
    suffix: r026_callback_fd
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -t_callbacks_tao_term_hessian_use_fd true
    args: -t_tao_term_sum_reg1_mask hessian -t_tao_view ::ascii_info_detail

  test:
    suffix: r028_analytic_fd
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -reg1_tao_term_hessian_use_fd true
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r029_fd_fd
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_callbacks_tao_term_hessian_use_fd true -reg1_tao_term_hessian_use_fd true
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r033_added_analytic_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_sum_t_callbacks_mask hessian
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r034_added_fd_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_sum_t_callbacks_mask hessian -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r037_callback_analytic_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -separate_hpre
    args: -t_tao_term_sum_reg1_mask hessian
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r038_callback_fd_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -separate_hpre
    args: -t_tao_term_sum_reg1_mask hessian -t_callbacks_tao_term_hessian_use_fd true
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r040_analytic_fd_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -separate_hpre
    args: -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r041_fd_fd_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -separate_hpre
    args: -t_callbacks_tao_term_hessian_use_fd true -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r023_mapped_added_analytic
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10
    args: -t_tao_term_sum_t_callbacks_mask hessian -t_tao_view ::ascii_info_detail

  test:
    suffix: r024_mapped_added_fd
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 11 -term1_scale 100
    args: -reg1_tao_term_hessian_use_fd true
    args: -t_tao_term_sum_t_callbacks_mask hessian -t_tao_view ::ascii_info_detail

  test:
    suffix: r031_analytic_mapped_fd
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10
    args: -reg1_tao_term_hessian_use_fd true -t_tao_view ::ascii_info_detail

  test:
    suffix: r032_fd_mapped_fd
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10
    args: -t_callbacks_tao_term_hessian_use_fd true -reg1_tao_term_hessian_use_fd true
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r035_mapped_added_analytic_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10
    args: -t_tao_term_sum_t_callbacks_mask hessian
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r036_mapped_added_fd_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10
    args: -t_tao_term_sum_t_callbacks_mask hessian -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r042_mapped_analytic_analytic_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10 -separate_hpre
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r043_analytic_mapped_fd_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10 -separate_hpre
    args: -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  test:
    suffix: r044_fd_mapped_fd_separate_hpre
    filter: grep -E "Mask \\(|Using finite differences for Hessian computation|Hessian preconditioning MatType|Classic callback comparison passed" | sed -E "s/^[[:space:]]+//"
    args: -t_tao_type nls -use_term1 -term1_shell -term1_has_A -map_row_size 10 -separate_hpre
    args: -t_callbacks_tao_term_hessian_use_fd true -reg1_tao_term_hessian_use_fd true
    args: -reg1_tao_term_hessian_pre_is_hessian false -reg1_tao_term_hessian_pre_mat_type aij
    args: -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij
    args: -t_tao_view ::ascii_info_detail

  testset:
    filter: grep -E "Classic callback comparison passed|unused database options|Option left"
    output_file: output/taotermtest1_classic_comparison.out
    args: -t_tao_type nls -c_tao_type nls -use_term1 -term1_shell
    args: -t_tao_term_hessian_mat_type shell -options_left

    test:
      suffix: r045_added_analytic
      args: -term1_hessian_mult false -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r046_added_fd
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r047_added_hessianmult
      args: -term1_scale 100 -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r049_mapped_added_fd
      args: -term1_has_A -map_row_size 11 -term1_scale 100 -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r050_mapped_added_hessianmult
      args: -term1_has_A -map_row_size 11 -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian

    test:
      suffix: r051_callback_analytic
      args: -t_tao_term_sum_reg1_mask hessian

    test:
      suffix: r052_callback_fd
      args: -t_callbacks_tao_term_hessian_use_fd true -t_tao_term_sum_reg1_mask hessian

    test:
      suffix: r053_callback_hessianmult
      args: -callback_hessian_mult -t_tao_term_sum_reg1_mask hessian

    test:
      suffix: r054_analytic_analytic
      args: -term1_hessian_mult false

    test:
      suffix: r055_analytic_fd
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true

    test:
      suffix: r057_fd_fd
      args: -t_callbacks_tao_term_hessian_use_fd true -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true

    test:
      suffix: r060_mapped_analytic_analytic
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false

    test:
      suffix: r061_mapped_analytic_fd
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true

    test:
      suffix: r063_mapped_fd_fd
      args: -term1_has_A -map_row_size 11 -t_callbacks_tao_term_hessian_use_fd true
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true

    test:
      suffix: r064_mapped_fd_hessianmult
      args: -term1_has_A -map_row_size 11 -t_callbacks_tao_term_hessian_use_fd true

    test:
      suffix: r065_mapped_hessianmult_hessianmult
      args: -term1_has_A -map_row_size 11 -callback_hessian_mult

    test:
      suffix: r066_added_analytic_separate_hpre
      args: -term1_hessian_mult false -term1_scale 100 -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r067_added_fd_separate_hpre
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r068_added_hessianmult_separate_hpre
      args: -term1_scale 100 -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r069_mapped_added_analytic_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r070_mapped_added_fd_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false -term1_scale 100
      args: -reg1_tao_term_hessian_use_fd true -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r071_mapped_added_hessianmult_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_scale 100
      args: -t_tao_term_sum_t_callbacks_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r072_callback_analytic_separate_hpre
      args: -t_tao_term_sum_reg1_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r073_callback_fd_separate_hpre
      args: -t_callbacks_tao_term_hessian_use_fd true -t_tao_term_sum_reg1_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r074_callback_hessianmult_separate_hpre
      args: -callback_hessian_mult -t_tao_term_sum_reg1_mask hessian
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r075_analytic_analytic_separate_hpre
      args: -term1_hessian_mult false
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r076_analytic_fd_separate_hpre
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r078_fd_fd_separate_hpre
      args: -t_callbacks_tao_term_hessian_use_fd true -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r079_fd_hessianmult_separate_hpre
      args: -t_callbacks_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r081_mapped_analytic_analytic_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r082_mapped_analytic_fd_separate_hpre
      args: -term1_has_A -map_row_size 11 -term1_hessian_mult false
      args: -reg1_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r084_mapped_fd_fd_separate_hpre
      args: -term1_has_A -map_row_size 11 -t_callbacks_tao_term_hessian_use_fd true
      args: -term1_hessian_mult false -reg1_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r085_mapped_fd_hessianmult_separate_hpre
      args: -term1_has_A -map_row_size 11 -t_callbacks_tao_term_hessian_use_fd true
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

    test:
      suffix: r086_mapped_hessianmult_hessianmult_separate_hpre
      args: -term1_has_A -map_row_size 11 -callback_hessian_mult
      args: -separate_hpre -t_tao_term_hessian_pre_is_hessian false -t_tao_term_hessian_pre_mat_type aij

TEST*/
