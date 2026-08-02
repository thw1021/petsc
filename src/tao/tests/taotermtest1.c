const char help[] = "Solve a Rosenbrock problem with regularizers added through TaoAddTerm().\n";

#include <petsctao.h>
#include "../unconstrained/tutorials/rosenbrock4.h"

typedef struct {
  AppCtx    user; /* Note: AppCtx is a pointer type in rosenbrock4.h */
  PetscBool use_term1;
  PetscBool term1_has_A;
  PetscBool term1_has_params;
  PetscReal term1_scale;
  PetscBool use_term2;
  PetscBool term2_has_A;
  PetscBool term2_has_params;
  PetscReal term2_scale;
  PetscInt  map_row_size;
  PetscBool separate_hpre;
} TestCtx;

/* Forward declarations */
static PetscErrorCode TestCtxInitialize(MPI_Comm, TestCtx *);
static PetscErrorCode TestCtxFinalize(TestCtx *);
static PetscErrorCode CreateTaoTermWithOptions(TestCtx *, TaoTerm *, Vec *, Mat *, const char *, const char *, PetscBool, PetscBool);
static PetscErrorCode FormFunctionGradient_TaoTerm(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_TaoTerm(Tao, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  TestCtx  ctx;
  Tao      tao_term;
  Vec      x_term;
  Mat      H_term, Hpre_term;
  TaoTerm  term1, term2;
  Vec      term1_params = NULL;
  Vec      term2_params = NULL;
  Mat      term1_A, term2_A;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(TestCtxInitialize(comm, &ctx));
  PetscCall(TaoCreate(comm, &tao_term));
  PetscCall(TaoSetType(tao_term, TAOLMVM));

  /* Create Rosenbrock objective using traditional TaoSet interface with user context */
  PetscCall(CreateHessian(ctx.user, &H_term));
  if (ctx.separate_hpre) {
    PetscCall(MatAssemblyBegin(H_term, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H_term, MAT_FINAL_ASSEMBLY));
    PetscCall(MatDuplicate(H_term, MAT_COPY_VALUES, &Hpre_term));
  } else Hpre_term = H_term;
  PetscCall(CreateVectors(ctx.user, H_term, &x_term, NULL));
  PetscCall(VecZeroEntries(x_term));
  PetscCall(TaoSetSolution(tao_term, x_term));
  PetscCall(TaoSetObjectiveAndGradient(tao_term, NULL, FormFunctionGradient_TaoTerm, &ctx));
  PetscCall(TaoSetHessian(tao_term, H_term, Hpre_term, FormHessian_TaoTerm, &ctx));

  /* Add term 1 if requested */
  if (ctx.use_term1) {
    PetscCall(CreateTaoTermWithOptions(&ctx, &term1, &term1_params, &term1_A, "reg1_", "A1_", ctx.term1_has_A, ctx.term1_has_params));
    PetscCall(TaoAddTerm(tao_term, "reg1_", ctx.term1_scale, term1, term1_params, term1_A));
    PetscCall(TaoTermDestroy(&term1));
  }
  /* Add term 2 if requested */
  if (ctx.use_term2) {
    PetscCall(CreateTaoTermWithOptions(&ctx, &term2, &term2_params, &term2_A, "reg2_", "A2_", ctx.term2_has_A, ctx.term2_has_params));
    PetscCall(TaoAddTerm(tao_term, "reg2_", ctx.term2_scale, term2, term2_params, term2_A));
    PetscCall(TaoTermDestroy(&term2));
  }

  PetscCall(TaoSetFromOptions(tao_term));
  PetscCall(TaoSolve(tao_term));

  if (ctx.use_term1) {
    PetscCall(VecDestroy(&term1_params));
    PetscCall(MatDestroy(&term1_A));
  }
  if (ctx.use_term2) {
    PetscCall(VecDestroy(&term2_params));
    PetscCall(MatDestroy(&term2_A));
  }
  PetscCall(TaoDestroy(&tao_term));
  PetscCall(VecDestroy(&x_term));
  PetscCall(MatDestroy(&H_term));
  if (ctx.separate_hpre) PetscCall(MatDestroy(&Hpre_term));
  PetscCall(TestCtxFinalize(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode TestCtxInitialize(MPI_Comm comm, TestCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(TestCtx)));

  /* Initialize Rosenbrock contexts */
  PetscCall(AppCtxCreate(comm, &ctx->user));

  /* Default configuration */
  ctx->use_term1        = PETSC_FALSE;
  ctx->use_term2        = PETSC_FALSE;
  ctx->term1_has_A      = PETSC_FALSE;
  ctx->term1_has_params = PETSC_FALSE;
  ctx->term2_has_A      = PETSC_FALSE;
  ctx->term2_has_params = PETSC_FALSE;
  ctx->term1_scale      = 0.1;
  ctx->term2_scale      = 0.05;
  ctx->map_row_size     = ctx->user->n - 1;
  ctx->separate_hpre    = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "TaoTerm Coverage Test Options", "TAO");
  PetscCall(PetscOptionsBool("-use_term1", "Use first additional term", "", ctx->use_term1, &ctx->use_term1, NULL));
  PetscCall(PetscOptionsBool("-use_term2", "Use second additional term", "", ctx->use_term2, &ctx->use_term2, NULL));
  PetscCall(PetscOptionsBool("-term1_has_A", "Term 1 has a map matrix A", "", ctx->term1_has_A, &ctx->term1_has_A, NULL));
  PetscCall(PetscOptionsBool("-term1_has_params", "Term 1 has parameters", "", ctx->term1_has_params, &ctx->term1_has_params, NULL));
  PetscCall(PetscOptionsBool("-term2_has_A", "Term 2 has a map matrix A", "", ctx->term2_has_A, &ctx->term2_has_A, NULL));
  PetscCall(PetscOptionsBool("-term2_has_params", "Term 2 has parameters", "", ctx->term2_has_params, &ctx->term2_has_params, NULL));
  PetscCall(PetscOptionsReal("-term1_scale", "Scaling for term 1", "", ctx->term1_scale, &ctx->term1_scale, NULL));
  PetscCall(PetscOptionsReal("-term2_scale", "Scaling for term 2", "", ctx->term2_scale, &ctx->term2_scale, NULL));
  PetscCall(PetscOptionsInt("-map_row_size", "Row size of mapping matrix", "", ctx->map_row_size, &ctx->map_row_size, NULL));
  PetscCall(PetscOptionsBool("-separate_hpre", "Use a separate preconditioning matrix for the legacy callback term", "", ctx->separate_hpre, &ctx->separate_hpre, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Finalize test context */
static PetscErrorCode TestCtxFinalize(TestCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(AppCtxDestroy(&ctx->user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTaoTermWithOptions(TestCtx *ctx, TaoTerm *term, Vec *params, Mat *A, const char *term_prefix, const char *A_prefix, PetscBool has_A, PetscBool has_params)
{
  MPI_Comm    comm = ctx->user->comm;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  *term   = NULL;
  *params = NULL;
  *A      = NULL;

  PetscCallMPI(MPI_Comm_size(comm, &size));
  /* Create parameters if requested */
  if (has_params) {
    PetscCall(VecCreate(comm, params));
    PetscCall(VecSetSizes(*params, PETSC_DECIDE, has_A ? ctx->map_row_size : ctx->user->n));
    PetscCall(VecSetFromOptions(*params));
    PetscCall(VecSetRandom(*params, NULL));
  }

  /* Create map matrix A if requested */
  if (has_A) {
    PetscCall(MatCreate(comm, A));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*A, A_prefix));
    PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, ctx->map_row_size, ctx->user->n));
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
  PetscCall(TaoTermSetSolutionSizes(*term, PETSC_DECIDE, has_A ? ctx->map_row_size : ctx->user->n, 1));
  PetscCall(TaoTermSetFromOptions(*term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormFunctionGradient_TaoTerm(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(FormObjectiveGradient(tao, X, f, G, ctx->user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormHessian_TaoTerm(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(FormHessian(tao, X, H, Hpre, ctx->user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: l1_mapped
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -tao_type nls -use_term1 -term1_has_params -term1_has_A -term1_scale 0.012
    args: -reg1_tao_term_type l1 -tao_view ::ascii_info_detail

  test:
    suffix: two_terms
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -tao_type nls -use_term1 -use_term2 -term1_scale 0.075 -term2_scale 0.098
    args: -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared -tao_view ::ascii_info_detail

  test:
    suffix: shell
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -tao_type nls -use_term1 -use_term2 -term2_has_A
    args: -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
    args: -tao_term_hessian_mat_type shell -tao_view ::ascii_info_detail

  test:
    suffix: shell_separate_hpre
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -tao_type nls -use_term1 -use_term2 -term2_has_A
    args: -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
    args: -tao_term_hessian_mat_type shell -tao_term_hessian_pre_is_hessian false
    args: -tao_term_hessian_pre_mat_type aij -tao_view ::ascii_info_detail

  test:
    suffix: masked_hessian
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -tao_type nls -use_term1 -use_term2
    args: -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
    args: -tao_term_sum_reg2_mask hessian -tao_view ::ascii_info_detail

  test:
    suffix: promoted_separate_hpre
    filter: grep -E "Hessian (preconditioning )?MatType|HessianMult evaluations|Mask \(|rows=.*cols=|Solution converged" | grep -v "MatType.*undefined"
    args: -tao_type nls -separate_hpre -use_term1 -reg1_tao_term_type halfl2squared
    args: -tao_term_hessian_pre_is_hessian false -tao_term_hessian_pre_mat_type aij
    args: -tao_view ::ascii_info_detail

  test:
    suffix: halfl2_parameters
    filter: grep -E "parameter vector space|rows=.*cols=|Solution converged"
    args: -tao_type nls -use_term1 -term1_has_params -reg1_tao_term_type halfl2squared
    args: -tao_view ::ascii_info_detail

  test:
    suffix: halfl2_mapped_parameters
    filter: grep -E "parameter vector space|rows=.*cols=|Solution converged"
    args: -tao_type nls -use_term1 -term1_has_params -term1_has_A
    args: -reg1_tao_term_type halfl2squared -tao_view ::ascii_info_detail

TEST*/
