#pragma once

#include <petsctao.h>

typedef enum {
  EXAMPLE_LEAST_SQUARES,
  EXAMPLE_HALF_L2,
  EXAMPLE_L1
} ExampleLeafType;

typedef struct {
  ExampleLeafType type;
  Mat             map;
  Vec             parameters;
  PetscReal       scale;
  PetscReal       epsilon;
  TaoTermMask     mask;
} ExampleLeaf;

typedef struct {
  PetscInt    nleaves;
  ExampleLeaf leaves[3];
  Vec         hessian_x;
} ExampleCtx;

typedef struct {
  Tao         tao;
  Mat         H;
  Mat         Hpre;
  Vec         x;
  ExampleCtx *ctx;
} ExampleReference;

static PETSC_UNUSED PetscErrorCode ExampleIdentityLeastSquaresObjective(TaoTerm term, Vec x, Vec parameters, PetscReal *f)
{
  Vec         work;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &work));
  if (parameters) PetscCall(VecWAXPY(work, -1.0, parameters, x));
  else PetscCall(VecCopy(x, work));
  PetscCall(VecDot(work, work, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleIdentityLeastSquaresGradient(TaoTerm term, Vec x, Vec parameters, Vec g)
{
  PetscFunctionBeginUser;
  if (parameters) PetscCall(VecWAXPY(g, -1.0, parameters, x));
  else PetscCall(VecCopy(x, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleIdentityLeastSquaresObjectiveGradient(TaoTerm term, Vec x, Vec parameters, PetscReal *f, Vec g)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(ExampleIdentityLeastSquaresGradient(term, x, parameters, g));
  PetscCall(VecDot(g, g, &dot));
  *f = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PETSC_UNUSED PetscErrorCode ExampleIdentityLeastSquaresHessian(TaoTerm term, Vec x, Vec parameters, Mat H, Mat Hpre)
{
  PetscFunctionBeginUser;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(H, 1.0));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PETSC_UNUSED PetscErrorCode ExampleIdentityLeastSquaresHessianMult(TaoTerm term, Vec x, Vec parameters, Vec v, Vec Hv)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleGetTermVector(ExampleLeaf *leaf, Vec x, Vec *y, PetscBool *destroy)
{
  PetscFunctionBeginUser;
  if (leaf->map) {
    PetscCall(MatCreateVecs(leaf->map, NULL, y));
    PetscCall(MatMult(leaf->map, x, *y));
    *destroy = PETSC_TRUE;
  } else {
    *y       = x;
    *destroy = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleComputeLeaf(ExampleLeaf *leaf, Vec x, PetscBool need_f, PetscBool need_g, PetscBool need_h, PetscReal *f, Vec gx, Mat Hx)
{
  Mat         D = NULL, contribution = NULL;
  Vec         y, diff, gy = NULL, diag = NULL;
  PetscScalar dot, sum;
  PetscBool   destroy_y;
  PetscInt    n;

  PetscFunctionBeginUser;
  PetscCall(ExampleGetTermVector(leaf, x, &y, &destroy_y));
  PetscCall(VecDuplicate(y, &diff));
  if (leaf->parameters) PetscCall(VecWAXPY(diff, -1.0, leaf->parameters, y));
  else PetscCall(VecCopy(y, diff));
  if (need_f) {
    if (leaf->type == EXAMPLE_L1) {
      if (leaf->epsilon == 0.0) PetscCall(VecNorm(diff, NORM_1, f));
      else {
        PetscCall(VecDuplicate(diff, &diag));
        PetscCall(VecPointwiseMult(diag, diff, diff));
        PetscCall(VecShift(diag, leaf->epsilon * leaf->epsilon));
        PetscCall(VecSqrtAbs(diag));
        PetscCall(VecSum(diag, &sum));
        PetscCall(VecGetSize(diag, &n));
        *f = PetscRealPart(sum) - n * leaf->epsilon;
        PetscCall(VecDestroy(&diag));
      }
    } else {
      PetscCall(VecDot(diff, diff, &dot));
      *f = 0.5 * PetscRealPart(dot);
    }
    *f *= leaf->scale;
  }
  if (need_g) {
    PetscCall(VecDuplicate(diff, &gy));
    if (leaf->type == EXAMPLE_L1) {
      if (leaf->epsilon == 0.0) PetscCall(VecPointwiseSign(gy, diff, VEC_SIGN_ZERO_TO_ZERO));
      else {
        PetscCall(VecDuplicate(diff, &diag));
        PetscCall(VecPointwiseMult(diag, diff, diff));
        PetscCall(VecShift(diag, leaf->epsilon * leaf->epsilon));
        PetscCall(VecSqrtAbs(diag));
        PetscCall(VecPointwiseDivide(gy, diff, diag));
        PetscCall(VecDestroy(&diag));
      }
    } else PetscCall(VecCopy(diff, gy));
    if (leaf->map) PetscCall(MatMultTranspose(leaf->map, gy, gx));
    else PetscCall(VecCopy(gy, gx));
    PetscCall(VecScale(gx, leaf->scale));
    PetscCall(VecDestroy(&gy));
  }
  if (need_h) {
    PetscCall(VecDuplicate(diff, &diag));
    PetscCall(VecSet(diag, 1.0));
    if (leaf->type == EXAMPLE_L1) {
      if (leaf->epsilon == 0.0) PetscCall(VecZeroEntries(diag));
      else {
        PetscCall(VecPointwiseMult(diag, diff, diff));
        PetscCall(VecShift(diag, leaf->epsilon * leaf->epsilon));
        PetscCall(VecPow(diag, -1.5));
        PetscCall(VecScale(diag, leaf->epsilon * leaf->epsilon));
      }
    }
    PetscCall(MatCreateDiagonal(diag, &D));
    if (leaf->map) PetscCall(MatPtAP(D, leaf->map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &contribution));
    else {
      PetscCall(MatDuplicate(D, MAT_COPY_VALUES, &contribution));
    }
    PetscCall(MatAXPY(Hx, leaf->scale, contribution, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&contribution));
    PetscCall(MatDestroy(&D));
    PetscCall(VecDestroy(&diag));
  }
  PetscCall(VecDestroy(&diff));
  if (destroy_y) PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleFormObjectiveGradient(Tao tao, Vec x, PetscReal *f, Vec g, void *vctx)
{
  ExampleCtx *ctx = (ExampleCtx *)vctx;
  Vec         work;
  PetscReal   leaf_f;

  PetscFunctionBeginUser;
  *f = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscCall(VecDuplicate(g, &work));
  for (PetscInt i = 0; i < ctx->nleaves; i++) {
    ExampleLeaf *leaf   = &ctx->leaves[i];
    PetscBool    need_f = !(leaf->mask & TAOTERM_MASK_OBJECTIVE);
    PetscBool    need_g = !(leaf->mask & TAOTERM_MASK_GRADIENT);

    leaf_f = 0.0;
    PetscCall(ExampleComputeLeaf(leaf, x, need_f, need_g, PETSC_FALSE, &leaf_f, work, NULL));
    if (need_f) *f += leaf_f;
    if (need_g) PetscCall(VecAXPY(g, 1.0, work));
  }
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleAssembleHessian(ExampleCtx *ctx, Vec x, Mat H)
{
  Vec       work;
  PetscReal unused;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(H));
  PetscCall(MatCreateVecs(H, &work, NULL));
  for (PetscInt i = 0; i < ctx->nleaves; i++) {
    ExampleLeaf *leaf = &ctx->leaves[i];

    if (!(leaf->mask & TAOTERM_MASK_HESSIAN)) PetscCall(ExampleComputeLeaf(leaf, x, PETSC_FALSE, PETSC_FALSE, PETSC_TRUE, &unused, work, H));
  }
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleApplyHessian(ExampleCtx *ctx, Vec x, Vec v, Vec Hv)
{
  Vec       y, diff, Av, HAv, contribution;
  PetscBool destroy_y;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(Hv));
  PetscCall(VecDuplicate(Hv, &contribution));
  for (PetscInt i = 0; i < ctx->nleaves; i++) {
    ExampleLeaf *leaf = &ctx->leaves[i];

    if (leaf->mask & TAOTERM_MASK_HESSIAN) continue;
    PetscCall(ExampleGetTermVector(leaf, x, &y, &destroy_y));
    PetscCall(VecDuplicate(y, &diff));
    PetscCall(VecDuplicate(y, &Av));
    PetscCall(VecDuplicate(y, &HAv));
    if (leaf->parameters) PetscCall(VecWAXPY(diff, -1.0, leaf->parameters, y));
    else PetscCall(VecCopy(y, diff));
    if (leaf->map) PetscCall(MatMult(leaf->map, v, Av));
    else PetscCall(VecCopy(v, Av));
    if (leaf->type == EXAMPLE_L1) {
      if (leaf->epsilon == 0.0) PetscCall(VecZeroEntries(HAv));
      else {
        PetscCall(VecPointwiseMult(HAv, diff, diff));
        PetscCall(VecShift(HAv, leaf->epsilon * leaf->epsilon));
        PetscCall(VecPow(HAv, -1.5));
        PetscCall(VecScale(HAv, leaf->epsilon * leaf->epsilon));
        PetscCall(VecPointwiseMult(HAv, HAv, Av));
      }
    } else PetscCall(VecCopy(Av, HAv));
    if (leaf->map) PetscCall(MatMultTranspose(leaf->map, HAv, contribution));
    else PetscCall(VecCopy(HAv, contribution));
    PetscCall(VecAXPY(Hv, leaf->scale, contribution));
    PetscCall(VecDestroy(&HAv));
    PetscCall(VecDestroy(&Av));
    PetscCall(VecDestroy(&diff));
    if (destroy_y) PetscCall(VecDestroy(&y));
  }
  PetscCall(VecDestroy(&contribution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleMatMult(Mat H, Vec v, Vec Hv)
{
  ExampleCtx *ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(H, &ctx));
  PetscCheck(ctx->hessian_x, PetscObjectComm((PetscObject)H), PETSC_ERR_ARG_WRONGSTATE, "Reference shell Hessian has not been evaluated at a solution");
  PetscCall(ExampleApplyHessian(ctx, ctx->hessian_x, v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleFormHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *vctx)
{
  ExampleCtx *ctx = (ExampleCtx *)vctx;
  PetscBool   is_shell;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)H, MATSHELL, &is_shell));
  if (is_shell) {
    PetscCall(PetscObjectReference((PetscObject)x));
    PetscCall(VecDestroy(&ctx->hessian_x));
    ctx->hessian_x = x;
    /* Clear shifts and scales left by the previous TAONLS iteration before publishing the Hessian at x. */
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    if (Hpre && Hpre != H) PetscCall(ExampleAssembleHessian(ctx, x, Hpre));
  } else {
    PetscCall(ExampleAssembleHessian(ctx, x, H));
    if (Hpre && Hpre != H) PetscCall(MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleCreateTao(MPI_Comm comm, Tao source, Vec x, ExampleCtx *ctx, Tao *tao, Mat *H, Mat *Hpre)
{
  TaoType   type;
  Mat       source_H, source_Hpre;
  PetscInt  n, nlocal;
  PetscInt  max_it, max_funcs;
  PetscReal gatol, grtol, gttol;
  PetscBool is_nls, is_mffd, is_shell;

  PetscFunctionBeginUser;
  PetscCall(VecGetSize(x, &n));
  PetscCall(VecGetLocalSize(x, &nlocal));
  PetscCall(TaoSetUp(source));
  PetscCall(TaoGetHessianMatrices(source, &source_H, &source_Hpre));
  PetscCall(PetscObjectTypeCompare((PetscObject)source_H, MATSHELL, &is_shell));
  PetscCall(PetscObjectTypeCompare((PetscObject)source_H, MATMFFD, &is_mffd));
  if (is_mffd) {
    PetscCall(MatCreateMFFD(comm, nlocal, nlocal, n, n, H));
    PetscCall(MatSetOption(*H, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(*H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  } else if (is_shell) {
    PetscCall(MatCreateShell(comm, nlocal, nlocal, n, n, ctx, H));
    PetscCall(MatShellSetOperation(*H, MATOP_MULT, (PetscErrorCodeFn *)ExampleMatMult));
    PetscCall(MatSetOption(*H, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(*H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  } else {
    PetscCall(MatCreateAIJ(comm, nlocal, nlocal, n, n, n, NULL, n, NULL, H));
    PetscCall(MatSetOption(*H, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  }
  if (source_Hpre != source_H) {
    PetscCall(MatCreateAIJ(comm, nlocal, nlocal, n, n, n, NULL, n, NULL, Hpre));
    PetscCall(MatSetOption(*Hpre, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatAssemblyBegin(*Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*Hpre, MAT_FINAL_ASSEMBLY));
  } else *Hpre = *H;
  PetscCall(TaoCreate(comm, tao));
  PetscCall(TaoSetOptionsPrefix(*tao, "c_"));
  PetscCall(TaoGetType(source, &type));
  PetscCall(TaoSetType(*tao, type));
  PetscCall(TaoGetTolerances(source, &gatol, &grtol, &gttol));
  PetscCall(TaoSetTolerances(*tao, gatol, grtol, gttol));
  PetscCall(TaoGetMaximumIterations(source, &max_it));
  PetscCall(TaoSetMaximumIterations(*tao, max_it));
  PetscCall(TaoGetMaximumFunctionEvaluations(source, &max_funcs));
  PetscCall(TaoSetMaximumFunctionEvaluations(*tao, max_funcs));
  PetscCall(TaoSetSolution(*tao, x));
  PetscCall(TaoSetObjectiveAndGradient(*tao, NULL, ExampleFormObjectiveGradient, ctx));
  PetscCall(TaoSetFromOptions(*tao));
  PetscCall(PetscObjectTypeCompare((PetscObject)*tao, TAONLS, &is_nls));
  if (is_nls) PetscCall(TaoSetHessian(*tao, *H, *Hpre, is_mffd ? TaoDefaultComputeHessianMFFD : ExampleFormHessian, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleCompareResults(Tao ttao, Vec tx, Tao ctao, Vec cx)
{
  const PetscReal solution_rtol = 1.e-3, objective_rtol = 1.e-4, gradient_rtol = 1.e-3;
  Vec             diff;
  PetscReal       tnorm, cnorm, error, scale, tf, cf, tg, cg;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(tx, &diff));
  PetscCall(VecWAXPY(diff, -1.0, cx, tx));
  PetscCall(VecNorm(diff, NORM_2, &error));
  PetscCall(VecNorm(tx, NORM_2, &tnorm));
  PetscCall(VecNorm(cx, NORM_2, &cnorm));
  scale = 1.0 + PetscMax(tnorm, cnorm);
  PetscCheck(error <= solution_rtol * scale, PetscObjectComm((PetscObject)ttao), PETSC_ERR_PLIB, "TaoTerm and reference solutions differ by %g relative to scale %g", (double)error, (double)scale);
  PetscCall(TaoGetSolutionStatus(ttao, NULL, &tf, &tg, NULL, NULL, NULL));
  PetscCall(TaoGetSolutionStatus(ctao, NULL, &cf, &cg, NULL, NULL, NULL));
  scale = 1.0 + PetscMax(PetscAbsReal(tf), PetscAbsReal(cf));
  PetscCheck(PetscAbsReal(tf - cf) <= objective_rtol * scale, PetscObjectComm((PetscObject)ttao), PETSC_ERR_PLIB, "TaoTerm and reference final objectives differ by %g relative to scale %g", (double)PetscAbsReal(tf - cf), (double)scale);
  scale = 1.0 + PetscMax(PetscAbsReal(tg), PetscAbsReal(cg));
  PetscCheck(PetscAbsReal(tg - cg) <= gradient_rtol * scale, PetscObjectComm((PetscObject)ttao), PETSC_ERR_PLIB, "TaoTerm and reference final gradient norms differ by %g relative to scale %g", (double)PetscAbsReal(tg - cg), (double)scale);
  PetscCall(VecDestroy(&diff));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ttao), "Reference callback comparison passed\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleReferenceCreate(MPI_Comm comm, Tao source, Vec source_x, ExampleCtx *ctx, ExampleReference *reference)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(reference, sizeof(*reference)));
  reference->ctx = ctx;
  PetscCall(VecDuplicate(source_x, &reference->x));
  PetscCall(VecCopy(source_x, reference->x));
  PetscCall(ExampleCreateTao(comm, source, reference->x, ctx, &reference->tao, &reference->H, &reference->Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleReferenceSolveAndCompare(Tao source, Vec source_x, ExampleReference *reference)
{
  PetscFunctionBeginUser;
  PetscCall(TaoSolve(source));
  PetscCall(TaoSolve(reference->tao));
  PetscCall(ExampleCompareResults(source, source_x, reference->tao, reference->x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleReferenceDestroy(ExampleReference *reference)
{
  PetscFunctionBeginUser;
  PetscCall(TaoDestroy(&reference->tao));
  if (reference->Hpre != reference->H) PetscCall(MatDestroy(&reference->Hpre));
  PetscCall(MatDestroy(&reference->H));
  PetscCall(VecDestroy(&reference->x));
  if (reference->ctx) PetscCall(VecDestroy(&reference->ctx->hessian_x));
  reference->ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
