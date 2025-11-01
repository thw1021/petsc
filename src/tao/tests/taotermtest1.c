const char help[] = "TaoTerm coverage test comparing TaoTerm interface with traditional callbacks for Rosenbrock problem.\n\
Tests different TaoTerm configurations including L1, HALFL2SQUARED, and QUADRATIC types with various matrix and parameter options.\n";

#include <petsctao.h>
#include "../unconstrained/tutorials/rosenbrock4.h"

typedef struct {
  AppCtx         user; /* Note: AppCtx is a pointer type in rosenbrock4.h */
  /* Configuration for term 1 */
  PetscBool      use_term1;
  PetscBool      term1_has_A;
  PetscBool      term1_has_params;
  PetscReal      term1_scale;
  /* Configuration for term 2 */
  PetscBool      use_term2;
  PetscBool      term2_has_A;
  PetscBool      term2_has_params;
  PetscReal      term2_scale;
  /* Callback-only scale options (for testing equivalence with TaoTerm scales) */
  PetscReal      term1_scale_callback;
  PetscReal      term2_scale_callback;
  /* Common sizes */
  PetscInt       term_size;
  /* Random number generator */
  PetscRandom    rand;
  /* Stored terms data for callback implementation */
  TaoTerm        term1;
  Vec            term1_params;
  Mat            term1_A;
  Mat            term1_A_callback; /* AIJ version for callbacks */
  TaoTerm        term2;
  Vec            term2_params;
  Mat            term2_A;
  Mat            term2_A_callback; /* AIJ version for callbacks */
} TestCtx;

/* Forward declarations */
static PetscErrorCode TestCtxInitialize(MPI_Comm, TestCtx *);
static PetscErrorCode TestCtxFinalize(TestCtx *);
static PetscErrorCode CreateTaoTermWithOptions(TestCtx *, TaoTerm *, Vec *, Mat *, const char *, const char *, const char *, PetscBool, PetscBool);
static PetscErrorCode FormFunctionGradient_TaoTerm(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_TaoTerm(Tao, Vec, Mat, Mat, void *);
static PetscErrorCode FormFunctionGradient_Callbacks(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian_Callbacks(Tao, Vec, Mat, Mat, void *);
static PetscErrorCode CompareSolutions(Tao, Tao);

int main(int argc, char **argv)
{
  TestCtx     ctx;
  Tao         tao_term, tao_callback;
  Vec         x_term, x_callback;
  Mat         H_term, H_callback;
  TaoTerm     term1, term2;
  Vec         term1_params, term2_params;
  Mat         term1_A, term2_A;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  /* Initialize test context */

  PetscCall(TestCtxInitialize(comm, &ctx));

  /* ===================== Setup Tao with TaoTerm interface ===================== */
  PetscCall(TaoCreate(comm, &tao_term));
  PetscCall(TaoSetType(tao_term, TAOLMVM));

  /* Create Rosenbrock objective using traditional TaoSet interface */
  PetscCall(CreateHessian(ctx.user, &H_term));
  PetscCall(CreateVectors(ctx.user, H_term, &x_term, NULL));
  PetscCall(VecZeroEntries(x_term));
  PetscCall(TaoSetSolution(tao_term, x_term));
  PetscCall(TaoSetObjectiveAndGradient(tao_term, NULL, FormFunctionGradient_TaoTerm, &ctx));
  PetscCall(TaoSetHessian(tao_term, H_term, H_term, FormHessian_TaoTerm, &ctx));

  /* Add term 1 if requested */
  if (ctx.use_term1) {
    PetscCall(CreateTaoTermWithOptions(&ctx, &term1, &term1_params, &term1_A, "reg1_", "A1_", "Q1_", ctx.term1_has_A, ctx.term1_has_params));
    PetscCall(TaoAddTerm(tao_term, "reg1_", ctx.term1_scale, term1, term1_params, term1_A));
    /* Store for callback implementation */
    ctx.term1        = term1;
    ctx.term1_params = term1_params;
    ctx.term1_A      = term1_A;
    /* Create AIJ version for callbacks if matrix exists */
    if (term1_A) {
      PetscCall(MatConvert(term1_A, MATAIJ, MAT_INITIAL_MATRIX, &ctx.term1_A_callback));
    }
  }

  /* Add term 2 if requested */
  if (ctx.use_term2) {
    PetscCall(CreateTaoTermWithOptions(&ctx, &term2, &term2_params, &term2_A, "reg2_", "A2_", "Q2_", ctx.term2_has_A, ctx.term2_has_params));
    PetscCall(TaoAddTerm(tao_term, "reg2_", ctx.term2_scale, term2, term2_params, term2_A));
    /* Store for callback implementation */
    ctx.term2        = term2;
    ctx.term2_params = term2_params;
    ctx.term2_A      = term2_A;
    /* Create AIJ version for callbacks if matrix exists */
    if (term2_A) {
      PetscCall(MatConvert(term2_A, MATAIJ, MAT_INITIAL_MATRIX, &ctx.term2_A_callback));
    }
  }

  PetscCall(TaoSetFromOptions(tao_term));

  /* Solve with TaoTerm interface */
  PetscCall(PetscPrintf(comm, "================ Solving with TaoTerm interface ================\n"));
  PetscCall(TaoSolve(tao_term));

  MPI_Barrier(comm);
  /* ===================== Setup Tao with traditional callbacks ===================== */
  PetscCall(TaoCreate(comm, &tao_callback));
  PetscCall(TaoSetType(tao_callback, TAOLMVM));

  /* Create Rosenbrock objective using callbacks that manually add term evaluations */
  PetscCall(CreateHessian(ctx.user, &H_callback));
  PetscCall(CreateVectors(ctx.user, H_callback, &x_callback, NULL));
  PetscCall(VecZeroEntries(x_callback));
  PetscCall(TaoSetSolution(tao_callback, x_callback));
  PetscCall(TaoSetObjectiveAndGradient(tao_callback, NULL, FormFunctionGradient_Callbacks, &ctx));
  PetscCall(TaoSetHessian(tao_callback, H_callback, H_callback, FormHessian_Callbacks, &ctx));

  PetscCall(TaoSetFromOptions(tao_callback));

  /* Solve with traditional callback interface */
  PetscCall(PetscPrintf(comm, "================ Solving with traditional callbacks ================\n"));
  PetscCall(TaoSolve(tao_callback));

  /* ===================== Compare solutions ===================== */
  PetscCall(CompareSolutions(tao_term, tao_callback));

  /* Clean up */
  if (ctx.use_term1) {
    PetscCall(TaoTermDestroy(&term1));
    PetscCall(VecDestroy(&term1_params));
    PetscCall(MatDestroy(&term1_A));
    PetscCall(MatDestroy(&ctx.term1_A_callback));
  }
  if (ctx.use_term2) {
    PetscCall(TaoTermDestroy(&term2));
    PetscCall(VecDestroy(&term2_params));
    PetscCall(MatDestroy(&term2_A));
    PetscCall(MatDestroy(&ctx.term2_A_callback));
  }
  PetscCall(TaoDestroy(&tao_term));
  PetscCall(TaoDestroy(&tao_callback));
  PetscCall(VecDestroy(&x_term));
  PetscCall(VecDestroy(&x_callback));
  PetscCall(MatDestroy(&H_term));
  PetscCall(MatDestroy(&H_callback));
  PetscCall(TestCtxFinalize(&ctx));

  PetscCall(PetscFinalize());
  return 0;
}

/* Initialize test context with command line options */
static PetscErrorCode TestCtxInitialize(MPI_Comm comm, TestCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(TestCtx)));

  /* Initialize Rosenbrock context */
  PetscCall(AppCtxCreate(comm, &ctx->user));

  /* Default configuration */
  ctx->use_term1          = PETSC_FALSE;
  ctx->use_term2          = PETSC_FALSE;
  ctx->term1_has_A        = PETSC_FALSE;
  ctx->term1_has_params   = PETSC_FALSE;
  ctx->term2_has_A        = PETSC_FALSE;
  ctx->term2_has_params   = PETSC_FALSE;
  ctx->term1_scale        = 0.1;
  ctx->term2_scale        = 0.05;
  ctx->term1_scale_callback = 0.1;  /* Default same as term1_scale */
  ctx->term2_scale_callback = 0.05; /* Default same as term2_scale */
  ctx->term_size          = ctx->user->n;

  PetscOptionsBegin(comm, "", "TaoTerm Coverage Test Options", "TAO");
  PetscCall(PetscOptionsBool("-use_term1", "Use first additional term", "", ctx->use_term1, &ctx->use_term1, NULL));
  PetscCall(PetscOptionsBool("-use_term2", "Use second additional term", "", ctx->use_term2, &ctx->use_term2, NULL));
  PetscCall(PetscOptionsBool("-term1_has_A", "Term 1 has a map matrix A", "", ctx->term1_has_A, &ctx->term1_has_A, NULL));
  PetscCall(PetscOptionsBool("-term1_has_params", "Term 1 has parameters", "", ctx->term1_has_params, &ctx->term1_has_params, NULL));
  PetscCall(PetscOptionsBool("-term2_has_A", "Term 2 has a map matrix A", "", ctx->term2_has_A, &ctx->term2_has_A, NULL));
  PetscCall(PetscOptionsBool("-term2_has_params", "Term 2 has parameters", "", ctx->term2_has_params, &ctx->term2_has_params, NULL));
  PetscCall(PetscOptionsReal("-term1_scale", "Scaling for term 1", "", ctx->term1_scale, &ctx->term1_scale, NULL));
  PetscCall(PetscOptionsReal("-term2_scale", "Scaling for term 2", "", ctx->term2_scale, &ctx->term2_scale, NULL));
  PetscCall(PetscOptionsReal("-term1_scale_callback", "Scaling for term 1 in callback version", "", ctx->term1_scale_callback, &ctx->term1_scale_callback, NULL));
  PetscCall(PetscOptionsReal("-term2_scale_callback", "Scaling for term 2 in callback version", "", ctx->term2_scale_callback, &ctx->term2_scale_callback, NULL));
  PetscCall(PetscOptionsInt("-term_size", "Size of term domain", "", ctx->term_size, &ctx->term_size, NULL));
  PetscOptionsEnd();

  ctx->term1 = NULL;
  ctx->term2 = NULL;
  ctx->term1_params = NULL;
  ctx->term2_params = NULL;
  ctx->term1_A = NULL;
  ctx->term2_A = NULL;
  ctx->term1_A_callback = NULL;
  ctx->term2_A_callback = NULL;

  /* Create random number generator */
  PetscCall(PetscRandomCreate(comm, &ctx->rand));
  PetscCall(PetscRandomSetFromOptions(ctx->rand));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Finalize test context */
static PetscErrorCode TestCtxFinalize(TestCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscRandomDestroy(&ctx->rand));
  PetscCall(AppCtxDestroy(&ctx->user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create TaoTerm with all necessary setup */
static PetscErrorCode CreateTaoTermWithOptions(TestCtx *ctx, TaoTerm *term, Vec *params, Mat *A, const char *term_prefix, const char *A_prefix, const char *Q_prefix, PetscBool has_A, PetscBool has_params)
{
  MPI_Comm      comm = ctx->user->comm;
  PetscBool     is_quad;

  PetscFunctionBeginUser;
  *term   = NULL;
  *params = NULL;
  *A      = NULL;

  /* Create parameters if requested */
  if (has_params) {
    PetscCall(VecCreate(comm, params));
    PetscCall(VecSetSizes(*params, PETSC_DECIDE, ctx->term_size));
    PetscCall(VecSetFromOptions(*params));
    PetscCall(VecSetRandom(*params, ctx->rand));
  }

  /* Create map matrix A if requested */
  if (has_A) {
    PetscCall(MatCreate(comm, A));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*A, A_prefix));
    PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, ctx->term_size, ctx->user->n));
    PetscCall(MatSetType(*A, MATAIJ)); /* Set default type before SetFromOptions */
    PetscCall(MatSetFromOptions(*A));
    PetscCall(MatSeqAIJSetPreallocation(*A, 5, NULL));
    PetscCall(MatMPIAIJSetPreallocation(*A, 5, NULL, 5, NULL));
    PetscCall(MatSetUp(*A));
    PetscCall(MatSetRandom(*A, ctx->rand));
    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  }

  /* Create TaoTerm, set prefix, and configure from options */
  PetscCall(TaoTermCreate(comm, term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*term, term_prefix));
  PetscCall(TaoTermSetSolutionSizes(*term, PETSC_DECIDE, ctx->term_size, 1));
  PetscCall(TaoTermSetFromOptions(*term));

  /* For quadratic terms, set up the matrix after type is set */
  PetscCall(PetscObjectTypeCompare((PetscObject)*term, TAOTERMQUADRATIC, &is_quad));
  if (is_quad) {
    Mat Aquad;
    /* Create matrix for quadratic term */
    PetscCall(MatCreate(comm, &Aquad));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)Aquad, Q_prefix));
    PetscCall(MatSetSizes(Aquad, PETSC_DECIDE, PETSC_DECIDE, ctx->term_size, ctx->term_size));
    PetscCall(MatSetType(Aquad, MATAIJ)); /* Set default type before SetFromOptions */
    PetscCall(MatSetFromOptions(Aquad));
    PetscCall(MatSeqAIJSetPreallocation(Aquad, 5, NULL));
    PetscCall(MatMPIAIJSetPreallocation(Aquad, 5, NULL, 5, NULL));
    PetscCall(MatSetUp(Aquad));
    PetscCall(MatSetRandom(Aquad, ctx->rand));
    PetscCall(MatAssemblyBegin(Aquad, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Aquad, MAT_FINAL_ASSEMBLY));

    PetscCall(TaoTermQuadraticSetMat(*term, Aquad));
    PetscCall(MatDestroy(&Aquad));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Form function and gradient for TaoTerm version (just Rosenbrock) */
static PetscErrorCode FormFunctionGradient_TaoTerm(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(FormObjectiveGradient(tao, X, f, G, ctx->user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Form Hessian for TaoTerm version (just Rosenbrock) */
static PetscErrorCode FormHessian_TaoTerm(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  TestCtx *ctx = (TestCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(FormHessian(tao, X, H, Hpre, ctx->user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Form function and gradient for callback version (Rosenbrock + terms manually) */
static PetscErrorCode FormFunctionGradient_Callbacks(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  TestCtx   *ctx = (TestCtx *)ptr;
  PetscReal f_term;
  Vec       X_mapped1 = NULL, X_mapped2 = NULL;
  Vec       G_mapped1 = NULL, G_mapped2 = NULL;
  PetscReal gnorm;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK] FormFunctionGradient called\n"));
  
  /* Compute Rosenbrock part */
  PetscCall(FormObjectiveGradient(tao, X, f, G, ctx->user));
  PetscCall(VecNorm(G, NORM_2, &gnorm));
  PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]   After Rosenbrock: f = %.16e, ||G|| = %.16e\n", (double)*f, (double)gnorm));
  /* Add term 1 contribution */
  if (ctx->use_term1 && ctx->term1) {
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]   Adding term1 (scale_callback = %.16e)\n", (double)ctx->term1_scale_callback));
    /* Map X if needed */
    if (ctx->term1_A_callback) {
      PetscCall(MatCreateVecs(ctx->term1_A_callback, NULL, &X_mapped1));
      PetscCall(MatMult(ctx->term1_A_callback, X, X_mapped1));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     X mapped through A1\n"));
    } else {
      X_mapped1 = X;
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     No A1 mapping\n"));
    }

    /* Compute term objective and gradient */
    if (ctx->term1_A_callback) {
      Vec       G_add;
      PetscReal gnorm_term, gnorm_add;
      PetscCall(VecDuplicate(X_mapped1, &G_mapped1));
      PetscCall(TaoTermComputeObjectiveAndGradient(ctx->term1, X_mapped1, ctx->term1_params, &f_term, G_mapped1));
      PetscCall(VecNorm(G_mapped1, NORM_2, &gnorm_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     Term1 raw: f_term = %.16e, ||G_term|| = %.16e\n", (double)f_term, (double)gnorm_term));
      /* Map gradient back and add to G */
      PetscCall(VecDuplicate(X, &G_add));
      PetscCall(MatMultTranspose(ctx->term1_A_callback, G_mapped1, G_add));
      PetscCall(VecNorm(G_add, NORM_2, &gnorm_add));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     After A1^T mapping: ||G_add|| = %.16e\n", (double)gnorm_add));
      PetscCall(VecAXPY(G, ctx->term1_scale_callback, G_add));
      PetscCall(VecDestroy(&G_add));
      PetscCall(VecDestroy(&X_mapped1));
      PetscCall(VecDestroy(&G_mapped1));
    } else {
      Vec       G_term;
      PetscReal gnorm_term;
      PetscCall(VecDuplicate(X, &G_term));
      PetscCall(TaoTermComputeObjectiveAndGradient(ctx->term1, X_mapped1, ctx->term1_params, &f_term, G_term));
      PetscCall(VecNorm(G_term, NORM_2, &gnorm_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     Term1 raw: f_term = %.16e, ||G_term|| = %.16e\n", (double)f_term, (double)gnorm_term));
      PetscCall(VecAXPY(G, ctx->term1_scale_callback, G_term));
      PetscCall(VecDestroy(&G_term));
    }
    *f += ctx->term1_scale_callback * f_term;
    PetscCall(VecNorm(G, NORM_2, &gnorm));
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     After adding scaled term1: f = %.16e, ||G|| = %.16e\n", (double)*f, (double)gnorm));
  }

  /* Add term 2 contribution */
  if (ctx->use_term2 && ctx->term2) {
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]   Adding term2 (scale_callback = %.16e)\n", (double)ctx->term2_scale_callback));
    /* Map X if needed */
    if (ctx->term2_A_callback) {
      PetscCall(MatCreateVecs(ctx->term2_A_callback, NULL, &X_mapped2));
      PetscCall(MatMult(ctx->term2_A_callback, X, X_mapped2));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     X mapped through A2\n"));
    } else {
      X_mapped2 = X;
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     No A2 mapping\n"));
    }

    /* Compute term objective and gradient */
    if (ctx->term2_A_callback) {
      Vec       G_add;
      PetscReal gnorm_term, gnorm_add;
      PetscCall(VecDuplicate(X_mapped2, &G_mapped2));
      PetscCall(TaoTermComputeObjectiveAndGradient(ctx->term2, X_mapped2, ctx->term2_params, &f_term, G_mapped2));
      PetscCall(VecNorm(G_mapped2, NORM_2, &gnorm_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     Term2 raw: f_term = %.16e, ||G_term|| = %.16e\n", (double)f_term, (double)gnorm_term));
      /* Map gradient back and add to G */
      PetscCall(VecDuplicate(X, &G_add));
      PetscCall(MatMultTranspose(ctx->term2_A_callback, G_mapped2, G_add));
      PetscCall(VecNorm(G_add, NORM_2, &gnorm_add));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     After A2^T mapping: ||G_add|| = %.16e\n", (double)gnorm_add));
      PetscCall(VecAXPY(G, ctx->term2_scale_callback, G_add));
      PetscCall(VecDestroy(&G_add));
      PetscCall(VecDestroy(&X_mapped2));
      PetscCall(VecDestroy(&G_mapped2));
    } else {
      Vec       G_term;
      PetscReal gnorm_term;
      PetscCall(VecDuplicate(X, &G_term));
      PetscCall(TaoTermComputeObjectiveAndGradient(ctx->term2, X_mapped2, ctx->term2_params, &f_term, G_term));
      PetscCall(VecNorm(G_term, NORM_2, &gnorm_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     Term2 raw: f_term = %.16e, ||G_term|| = %.16e\n", (double)f_term, (double)gnorm_term));
      PetscCall(VecAXPY(G, ctx->term2_scale_callback, G_term));
      PetscCall(VecDestroy(&G_term));
    }
    *f += ctx->term2_scale_callback * f_term;
    PetscCall(VecNorm(G, NORM_2, &gnorm));
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     After adding scaled term2: f = %.16e, ||G|| = %.16e\n", (double)*f, (double)gnorm));
  }

  PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]   FINAL: f = %.16e, ||G|| = %.16e\n", (double)*f, (double)gnorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Form Hessian for callback version (Rosenbrock + terms manually) */
static PetscErrorCode FormHessian_Callbacks(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  TestCtx   *ctx = (TestCtx *)ptr;
  Mat       H_term;
  Vec       X_mapped;
  PetscReal hnorm;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK] FormHessian called\n"));
  
  /* Compute Rosenbrock Hessian */
  PetscCall(FormHessian(tao, X, H, Hpre, ctx->user));
  PetscCall(MatNorm(H, NORM_FROBENIUS, &hnorm));
  PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]   After Rosenbrock: ||H||_F = %.16e\n", (double)hnorm));

  /* Add term 1 Hessian contribution */
  if (ctx->use_term1 && ctx->term1) {
    PetscInt  m, n;
    PetscReal hnorm_term;
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]   Adding term1 Hessian (scale_callback = %.16e)\n", (double)ctx->term1_scale_callback));
    if (ctx->term1_A_callback) {
      PetscCall(MatCreateVecs(ctx->term1_A_callback, NULL, &X_mapped));
      PetscCall(MatMult(ctx->term1_A_callback, X, X_mapped));
      PetscCall(VecGetSize(X_mapped, &m));
      PetscCall(MatCreate(ctx->user->comm, &H_term));
      PetscCall(MatSetSizes(H_term, PETSC_DECIDE, PETSC_DECIDE, m, m));
      PetscCall(MatSetType(H_term, MATAIJ));
      PetscCall(MatSetUp(H_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     X mapped through A1, H_term created (%d x %d)\n", (int)m, (int)m));
    } else {
      X_mapped = X;
      PetscCall(VecGetSize(X, &n));
      PetscCall(MatCreate(ctx->user->comm, &H_term));
      PetscCall(MatSetSizes(H_term, PETSC_DECIDE, PETSC_DECIDE, n, n));
      PetscCall(MatSetType(H_term, MATAIJ));
      PetscCall(MatSetUp(H_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     No A1 mapping, H_term created (%d x %d)\n", (int)n, (int)n));
    }

    PetscCall(TaoTermComputeHessian(ctx->term1, X_mapped, ctx->term1_params, H_term, NULL));
    PetscCall(MatNorm(H_term, NORM_FROBENIUS, &hnorm_term));
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     Term1 raw Hessian: ||H_term||_F = %.16e\n", (double)hnorm_term));

    if (ctx->term1_A_callback) {
      Mat H_mapped;
      /* H = H + scale * A^T * H_term * A */
      PetscCall(MatMatMult(H_term, ctx->term1_A_callback, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H_mapped));
      PetscCall(MatDestroy(&H_term));
      PetscCall(MatTransposeMatMult(ctx->term1_A_callback, H_mapped, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H_term));
      PetscCall(MatDestroy(&H_mapped));
      PetscCall(MatNorm(H_term, NORM_FROBENIUS, &hnorm_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     After A1^T * H_term * A1: ||H_term||_F = %.16e\n", (double)hnorm_term));
      PetscCall(MatScale(H_term, ctx->term1_scale_callback));
      PetscCall(MatAXPY(H, 1.0, H_term, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&H_term));
      PetscCall(VecDestroy(&X_mapped));
    } else {
      PetscCall(MatScale(H_term, ctx->term1_scale_callback));
      PetscCall(MatAXPY(H, 1.0, H_term, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&H_term));
    }
    PetscCall(MatNorm(H, NORM_FROBENIUS, &hnorm));
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     After adding scaled term1: ||H||_F = %.16e\n", (double)hnorm));
  }

  /* Add term 2 Hessian contribution */
  if (ctx->use_term2 && ctx->term2) {
    PetscInt  m, n;
    PetscReal hnorm_term;
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]   Adding term2 Hessian (scale_callback = %.16e)\n", (double)ctx->term2_scale_callback));
    if (ctx->term2_A_callback) {
      PetscCall(MatCreateVecs(ctx->term2_A_callback, NULL, &X_mapped));
      PetscCall(MatMult(ctx->term2_A_callback, X, X_mapped));
      PetscCall(VecGetSize(X_mapped, &m));
      PetscCall(MatCreate(ctx->user->comm, &H_term));
      PetscCall(MatSetSizes(H_term, PETSC_DECIDE, PETSC_DECIDE, m, m));
      PetscCall(MatSetType(H_term, MATAIJ));
      PetscCall(MatSetUp(H_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     X mapped through A2, H_term created (%d x %d)\n", (int)m, (int)m));
    } else {
      X_mapped = X;
      PetscCall(VecGetSize(X, &n));
      PetscCall(MatCreate(ctx->user->comm, &H_term));
      PetscCall(MatSetSizes(H_term, PETSC_DECIDE, PETSC_DECIDE, n, n));
      PetscCall(MatSetType(H_term, MATAIJ));
      PetscCall(MatSetUp(H_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     No A2 mapping, H_term created (%d x %d)\n", (int)n, (int)n));
    }

    PetscCall(TaoTermComputeHessian(ctx->term2, X_mapped, ctx->term2_params, H_term, NULL));
    PetscCall(MatNorm(H_term, NORM_FROBENIUS, &hnorm_term));
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     Term2 raw Hessian: ||H_term||_F = %.16e\n", (double)hnorm_term));

    if (ctx->term2_A_callback) {
      Mat H_mapped;
      /* H = H + scale * A^T * H_term * A */
      PetscCall(MatMatMult(H_term, ctx->term2_A_callback, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H_mapped));
      PetscCall(MatDestroy(&H_term));
      PetscCall(MatTransposeMatMult(ctx->term2_A_callback, H_mapped, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H_term));
      PetscCall(MatDestroy(&H_mapped));
      PetscCall(MatNorm(H_term, NORM_FROBENIUS, &hnorm_term));
      PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     After A2^T * H_term * A2: ||H_term||_F = %.16e\n", (double)hnorm_term));
      PetscCall(MatScale(H_term, ctx->term2_scale_callback));
      PetscCall(MatAXPY(H, 1.0, H_term, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&H_term));
      PetscCall(VecDestroy(&X_mapped));
    } else {
      PetscCall(MatScale(H_term, ctx->term2_scale_callback));
      PetscCall(MatAXPY(H, 1.0, H_term, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&H_term));
    }
    PetscCall(MatNorm(H, NORM_FROBENIUS, &hnorm));
    PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]     After adding scaled term2: ||H||_F = %.16e\n", (double)hnorm));
  }

  PetscCall(PetscPrintf(ctx->user->comm, "[CALLBACK]   FINAL: ||H||_F = %.16e\n", (double)hnorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compare solutions from both methods */
static PetscErrorCode CompareSolutions(Tao tao_term, Tao tao_callback)
{
  Vec         x_term, x_callback, diff;
  PetscReal   norm_term, norm_callback, norm_diff, rel_diff;
  PetscReal   f_term, f_callback;
  TaoConvergedReason reason_term, reason_callback;

  PetscFunctionBeginUser;
  PetscCall(TaoGetSolution(tao_term, &x_term));
  PetscCall(TaoGetSolution(tao_callback, &x_callback));

  /* Compute difference */
  PetscCall(VecDuplicate(x_term, &diff));
  PetscCall(VecCopy(x_term, diff));
  PetscCall(VecAXPY(diff, -1.0, x_callback));

  PetscCall(VecNorm(x_term, NORM_2, &norm_term));
  PetscCall(VecNorm(x_callback, NORM_2, &norm_callback));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));

  rel_diff = norm_diff / PetscMax(norm_term, 1.0e-10);

  /* Get objective values */
  PetscCall(TaoGetSolutionStatus(tao_term, NULL, &f_term, NULL, NULL, NULL, NULL));
  PetscCall(TaoGetSolutionStatus(tao_callback, NULL, &f_callback, NULL, NULL, NULL, NULL));

  /* Get convergence reasons */
  PetscCall(TaoGetConvergedReason(tao_term, &reason_term));
  PetscCall(TaoGetConvergedReason(tao_callback, &reason_callback));

  /* Print comparison */
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "\n============== Solution Comparison ==============\n"));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "TaoTerm interface:\n"));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Final objective: %.6e\n", (double)f_term));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Solution norm:   %.6e\n", (double)norm_term));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Converged reason: %d\n", (int)reason_term));

  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "\nTraditional callbacks:\n"));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Final objective: %.6e\n", (double)f_callback));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Solution norm:   %.6e\n", (double)norm_callback));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Converged reason: %d\n", (int)reason_callback));

  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "\nDifference:\n"));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Absolute difference norm: %.6e\n", (double)norm_diff));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Relative difference:      %.6e\n", (double)rel_diff));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "  Objective difference:     %.6e\n", (double)PetscAbsReal(f_term - f_callback)));

  /* Check if solutions are close */
  if (rel_diff < 1.0e-6) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "\n✓ Solutions match (relative difference < 1e-6)\n"));
  } else if (rel_diff < 1.0e-3) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "\n⚠ Solutions approximately match (relative difference < 1e-3)\n"));
  } else {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "\n✗ Solutions differ significantly (relative difference >= 1e-3)\n"));
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao_term), "=================================================\n"));

  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: !complex

   test:
      suffix: 0
      args: -tao_monitor_short

#TODO outputfile name prob wrong due to params separate output
   testset:
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -term1_has_params {{0 1}separate output} -tao_monitor_short -term1_has_A -A1_mat_type {{aij dense diagonal}} -tao_term_sum_hessian_mat_type {{aij dense}}
      test:
         suffix: term1_only_l1_A
         args: -reg1_tao_term_type l1 -term1_scale 1.234
      test:
         suffix: term1_only_l1_2_A
         output_file: output/taotermtest1_term1_only_l1_A.out
         args: -reg1_tao_term_type l1 -tao_term_sum_reg1_scale 1.234 -term1_scale_callback 1.234
      test:
         suffix: term1_only_l2_A
         args: -reg1_tao_term_type halfl2squared -term1_scale 1.234
      test:
         suffix: term1_only_l2_2_A
         output_file: output/taotermtest1_term1_only_l2_A.out
         args: -reg1_tao_term_type halfl2squared -tao_term_sum_reg1_scale 1.234 -term1_scale_callback 1.234
      test:
         suffix: term1_only_quad_A
         args: -reg1_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -term1_scale 1.234
      test:
         suffix: term1_only_quad_2_A
         output_file: output/taotermtest1_term1_only_quad_A.out
         args: -reg1_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -tao_term_sum_reg1_scale 1.234 -term1_scale_callback 1.234

   testset:
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -term1_has_params {{0 1}separate output} -tao_monitor_short -term1_has_A 0
      test:
         suffix: term1_only_l1_no_A
         args: -reg1_tao_term_type l1 -term1_scale 1.234
      test:
         suffix: term1_only_l1_2_no_A
         output_file: output/taotermtest1_term1_only_l1_no_A.out
         args: -reg1_tao_term_type l1 -tao_term_sum_reg1_scale 1.234 -term1_scale_callback 1.234
      test:
         suffix: term1_only_l2_no_A
         args: -reg1_tao_term_type halfl2squared -term1_scale 1.234
      test:
         suffix: term1_only_l2_2_no_A
         output_file: output/taotermtest1_term1_only_l2_no_A.out
         args: -reg1_tao_term_type halfl2squared -tao_term_sum_reg1_scale 1.234 -term1_scale_callback 1.234
      test:
         suffix: term1_only_quad_no_A
         args: -reg1_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -term1_scale 1.234
      test:
         suffix: term1_only_quad_2_no_A
         output_file: output/taotermtest1_term1_only_quad_no_A.out
         args: -reg1_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -tao_term_sum_reg1_scale 1.234 -term1_scale_callback 1.234

# Two terms, with no mapping
   testset:
      suffix: l1_l1_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type l1 -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l1_l2_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l1_quad_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type l1 -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_l1_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type halfl2squared -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_l2_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type halfl2squared -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_quad_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type halfl2squared -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_l1_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_l2_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_quad_no_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 0 -reg1_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} 
      args: -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852


# Two terms: term1 no A, term2 has A
   testset:
      suffix: l1_l1_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l1_l2_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l1_quad_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_l1_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_l2_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_quad_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_l1_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type l1 -Q1_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_l2_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type halfl2squared -Q1_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_quad_no_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 0 -term2_has_A 1 -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

# Two terms: term1 has A, term2 no A
   testset:
      suffix: l1_l1_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l1_l2_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l1_quad_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_l1_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_l2_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_quad_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_l1_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type l1 -Q1_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_l2_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type halfl2squared -Q1_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_quad_A1_no_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 0 -A1_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

# Two terms: term1 has A, term2 has A
   testset:
      suffix: l1_l1_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l1_l2_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l1_quad_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type l1 -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_l1_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type l1
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_l2_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type halfl2squared
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: l2_quad_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type halfl2squared -reg2_tao_term_type quadratic -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_l1_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type l1 -Q1_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_l2_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type halfl2squared -Q1_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

   testset:
      suffix: quad_quad_A1_A2
      nsize: {{1 2 3 4}}
      args: -tao_type nls -use_term1 -use_term2 -term1_has_params {{0 1}separate output} -term2_has_params {{0 1}separate output} -tao_monitor_short
      args: -term1_has_A 1 -term2_has_A 1 -A1_mat_type {{aij dense diagonal}} -A2_mat_type {{aij dense diagonal}} -reg1_tao_term_type quadratic -reg2_tao_term_type quadratic -Q1_mat_type {{aij dense diagonal}} -Q2_mat_type {{aij dense diagonal}}
      test:
         args: -term1_scale 0.123 -term2_scale 1.852
      test:
         args: -term1_scale 0.123 -term2_scale_callback 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale 1.852
      test:
         args: -term1_scale_callback 0.123 -term2_scale_callback 1.852

TEST*/

