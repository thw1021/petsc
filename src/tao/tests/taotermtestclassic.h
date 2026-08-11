#pragma once

#include <petsctao.h>

typedef enum {
  EXAMPLE_CLASSIC_LEAST_SQUARES,
  EXAMPLE_CLASSIC_HALF_L2,
  EXAMPLE_CLASSIC_L1
} ExampleClassicLeafType;

typedef struct {
  ExampleClassicLeafType type;
  Mat                    map;
  Vec                    parameters;
  PetscReal              scale;
  PetscReal              epsilon;
  TaoTermMask            mask;
} ExampleClassicLeaf;

typedef struct {
  PetscInt           nleaves;
  ExampleClassicLeaf leaves[3];
} ExampleClassicCtx;

static PetscErrorCode ExampleClassicGetTermVector(ExampleClassicLeaf *leaf, Vec x, Vec *y, PetscBool *destroy)
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

static PetscErrorCode ExampleClassicComputeLeaf(ExampleClassicLeaf *leaf, Vec x, PetscBool need_f, PetscBool need_g, PetscBool need_h, PetscReal *f, Vec gx, Mat Hx)
{
  Mat         D = NULL, contribution = NULL;
  Vec         y, diff, gy = NULL, diag = NULL;
  PetscScalar dot, sum;
  PetscBool   destroy_y;
  PetscInt    n;

  PetscFunctionBeginUser;
  PetscCall(ExampleClassicGetTermVector(leaf, x, &y, &destroy_y));
  PetscCall(VecDuplicate(y, &diff));
  if (leaf->parameters) PetscCall(VecWAXPY(diff, -1.0, leaf->parameters, y));
  else PetscCall(VecCopy(y, diff));
  if (need_f) {
    if (leaf->type == EXAMPLE_CLASSIC_L1) {
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
    if (leaf->type == EXAMPLE_CLASSIC_L1) {
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
    if (leaf->type == EXAMPLE_CLASSIC_L1) {
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

static PetscErrorCode ExampleClassicFormObjectiveGradient(Tao tao, Vec x, PetscReal *f, Vec g, void *vctx)
{
  ExampleClassicCtx *ctx = (ExampleClassicCtx *)vctx;
  Vec                work;
  PetscReal          leaf_f;

  PetscFunctionBeginUser;
  *f = 0.0;
  PetscCall(VecZeroEntries(g));
  PetscCall(VecDuplicate(g, &work));
  for (PetscInt i = 0; i < ctx->nleaves; i++) {
    ExampleClassicLeaf *leaf   = &ctx->leaves[i];
    PetscBool           need_f = !(leaf->mask & TAOTERM_MASK_OBJECTIVE);
    PetscBool           need_g = !(leaf->mask & TAOTERM_MASK_GRADIENT);

    leaf_f = 0.0;
    PetscCall(ExampleClassicComputeLeaf(leaf, x, need_f, need_g, PETSC_FALSE, &leaf_f, work, NULL));
    if (need_f) *f += leaf_f;
    if (need_g) PetscCall(VecAXPY(g, 1.0, work));
  }
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleClassicFormHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *vctx)
{
  ExampleClassicCtx *ctx = (ExampleClassicCtx *)vctx;
  Vec                work;
  PetscReal          unused;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(H));
  PetscCall(MatCreateVecs(H, &work, NULL));
  for (PetscInt i = 0; i < ctx->nleaves; i++) {
    ExampleClassicLeaf *leaf = &ctx->leaves[i];

    if (!(leaf->mask & TAOTERM_MASK_HESSIAN)) PetscCall(ExampleClassicComputeLeaf(leaf, x, PETSC_FALSE, PETSC_FALSE, PETSC_TRUE, &unused, work, H));
  }
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  if (Hpre && Hpre != H) PetscCall(MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN));
  PetscCall(VecDestroy(&work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExampleClassicCreateTao(MPI_Comm comm, Tao source, Vec x, ExampleClassicCtx *ctx, Tao *tao, Mat *H)
{
  TaoType   type;
  PetscInt  n, nlocal;
  PetscBool is_nls;

  PetscFunctionBeginUser;
  PetscCall(VecGetSize(x, &n));
  PetscCall(VecGetLocalSize(x, &nlocal));
  PetscCall(MatCreateAIJ(comm, nlocal, nlocal, n, n, n, NULL, n, NULL, H));
  PetscCall(MatSetOption(*H, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatAssemblyBegin(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*H, MAT_FINAL_ASSEMBLY));
  PetscCall(TaoCreate(comm, tao));
  PetscCall(TaoGetType(source, &type));
  PetscCall(TaoSetType(*tao, type));
  PetscCall(TaoSetSolution(*tao, x));
  PetscCall(TaoSetObjectiveAndGradient(*tao, NULL, ExampleClassicFormObjectiveGradient, ctx));
  PetscCall(PetscObjectTypeCompare((PetscObject)*tao, TAONLS, &is_nls));
  if (is_nls) PetscCall(TaoSetHessian(*tao, *H, *H, ExampleClassicFormHessian, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
