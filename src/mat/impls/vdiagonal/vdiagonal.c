
#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/

typedef struct {
  Vec       diag;
  PetscBool diag_valid;
  Vec       inv_diag;
  PetscBool inv_diag_valid;
} Mat_VecDiagonal;

static PetscErrorCode MatVecDiagonalValidate(Mat A)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)A->data;

  PetscFunctionBegin;
  if (!ctx->diag_valid) {
    PetscAssert(ctx->inv_diag_valid, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Neither diagonal nor inverse valid");
    PetscCall(VecCopy(ctx->inv_diag, ctx->diag));
    PetscCall(VecReciprocal(ctx->diag));
    ctx->diag_valid = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatVecDiagonalValidateInverse(Mat A)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)A->data;
  PetscFunctionBegin;
  if (!ctx->inv_diag_valid) {
    PetscAssert(ctx->diag_valid, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Neither diagonal nor inverse valid");
    PetscCall(VecCopy(ctx->diag, ctx->inv_diag));
    PetscCall(VecReciprocal(ctx->inv_diag));
    ctx->inv_diag_valid = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAXPY_VecDiagonal(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_VecDiagonal *yctx = (Mat_VecDiagonal *)Y->data;
  Mat_VecDiagonal *xctx = (Mat_VecDiagonal *)X->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(Y));
  PetscCall(MatVecDiagonalValidate(X));
  PetscCall(VecAXPY(yctx->diag, a, xctx->diag));
  yctx->inv_diag_valid = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_VecDiagonal(Mat A, Vec x, Vec y)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(A));
  PetscCall(VecPointwiseMult(y, ctx->diag, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_VecDiagonal(Mat mat, Vec v1, Vec v2, Vec v3)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(mat));
  if (v2 != v3) {
    PetscCall(VecPointwiseMult(v3, ctx->diag, v1));
    PetscCall(VecAXPY(v3, 1.0, v2));
  } else {
    Vec w;
    PetscCall(VecDuplicate(v3, &w));
    PetscCall(VecPointwiseMult(w, ctx->diag, v1));
    PetscCall(VecAXPY(v3, 1.0, w));
    PetscCall(VecDestroy(&w));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNorm_VecDiagonal(Mat A, NormType type, PetscReal *nrm)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(A));
  type = (type == NORM_FROBENIUS) ? NORM_2 : NORM_INFINITY;
  PetscCall(VecNorm(ctx->diag, type, nrm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_VecDiagonal(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_VecDiagonal *actx = (Mat_VecDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(MatSetBlockSizesFromMats(*B, A, A));
  PetscCall(MatSetType(*B, MATVECDIAGONAL));
  PetscCall(PetscLayoutReference(A->rmap, &(*B)->rmap));
  PetscCall(PetscLayoutReference(A->cmap, &(*B)->cmap));
  if (op == MAT_COPY_VALUES) {
    Mat_VecDiagonal *bctx = (Mat_VecDiagonal *)(*B)->data;
    PetscCall(VecCopy(actx->diag, bctx->diag));
    PetscCall(VecCopy(actx->inv_diag, bctx->inv_diag));
    bctx->diag_valid     = actx->diag_valid;
    bctx->inv_diag_valid = actx->inv_diag_valid;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatVecDiagonalGetDiagonal - Get the diagonal of a `MATVECDIAGONAL`

  Input Parameter:
. A - the `MATVECDIAGONAL`

  Output Parameter:
. diag - the `Vec` that defines the diagonal

  Level: developer

  Note:

  Although this returns a reference rather than a copy, the user must call
  `MatVecDiagonalRestoreDiagonal()` before using the matrix again.

  For a copy of the diagonal values, rather than a reference, use `MatGetDiagonal()`

.seealso: [](chapter_matrices), `MATVECDIAGONAL`, `MatCreateVecDiagonal()`, `MatVecDiagonalRestoreDiagonal()`, `MatVecDiagonalGetInverse()`, `MatGetDiagonal()`
@*/
PetscErrorCode MatVecDiagonalGetDiagonal(Mat A, Vec *diag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(diag, 2);
  *diag = NULL;
  PetscTryMethod((PetscObject)A, "MatVecDiagonalGetDiagonal_C", (Mat, Vec *), (A, diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatVecDiagonalGetDiagonal_VecDiagonal(Mat A, Vec *diag)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(A));
  *diag = ctx->diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatVecDiagonalRestoreDiagonal - Restore the diagonal of a `MATVECDIAGONAL`

  Input Parameters:
+ A - the `MATVECDIAGONAL`
- diag - the `Vec` obtained from `MatVecDiagonalGetDiagonal()`

  Level: developer

  Note:

  Use `MatDiagonalSet()` to change the values by copy, rather than reference.

.seealso: [](chapter_matrices), `MATVECDIAGONAL`, `MatCreateVecDiagonal()`, `MatVecDiagonalGetDiagonal()`
@*/
PetscErrorCode MatVecDiagonalRestoreDiagonal(Mat A, Vec *diag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(diag, 2);
  PetscTryMethod((PetscObject)A, "MatVecDiagonalRestoreDiagonal_C", (Mat, Vec *), (A, diag));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  *diag = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatVecDiagonalRestoreDiagonal_VecDiagonal(Mat A, Vec *diag)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCheck(ctx->diag == *diag, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Restored a different diagonal vector");
  ctx->diag_valid     = PETSC_TRUE;
  ctx->inv_diag_valid = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatVecDiagonalGetInverse - Get the inverse diagonal of a `MATVECDIAGONAL`

  Input Parameter:
. A - the `MATVECDIAGONAL`

  Output Parameter:
. inv_diag - the `Vec` that defines the inverse diagonal

  Level: developer

  Note:

  Although this returns a reference rather than a copy, the user must call
  `MatVecDiagonalRestoreInverse()` before using the matrix again.

  If a matrix is created only to call `MatSolve()` (which happens for `MATLMVMDIAGBROYDEN`),
  using `MatVecDiagonalGetInverse()` and `MatVecDiagonalRestoreInverse()` avoids copies
  and avoids any call to `VecReciprocal()`.

.seealso: [](chapter_matrices), `MATVECDIAGONAL`, `MatCreateVecDiagonal()`, `MatVecDiagonalRestoreInverse()`, `MatVecDiagonalGetDiagonal()`, `MATLMVMBROYDEN`, `MatSolve()`
@*/
PetscErrorCode MatVecDiagonalGetInverse(Mat A, Vec *inv_diag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(inv_diag, 2);
  *inv_diag = NULL;
  PetscTryMethod((PetscObject)A, "MatVecDiagonalGetInverse_C", (Mat, Vec *), (A, inv_diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatVecDiagonalGetInverse_VecDiagonal(Mat A, Vec *inv_diag)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidateInverse(A));
  *inv_diag = ctx->inv_diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatVecDiagonalRestoreInverse - Restore the inverse diagonal of a `MATVECDIAGONAL`

  Input Parameters:
+ A - the `MATVECDIAGONAL`
- inv_diag - the `Vec` obtained from `MatVecDiagonalGetInverse()`

  Level: developer

.seealso: [](chapter_matrices), `MATVECDIAGONAL`, `MatCreateVecDiagonal()`, `MatVecDiagonalGetInverse()`
@*/
PetscErrorCode MatVecDiagonalRestoreInverse(Mat A, Vec *inv_diag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(inv_diag, 2);
  PetscTryMethod((PetscObject)A, "MatVecDiagonalRestoreInverse_C", (Mat, Vec *), (A, inv_diag));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  *inv_diag = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatVecDiagonalRestoreInverse_VecDiagonal(Mat A, Vec *inv_diag)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCheck(ctx->inv_diag == *inv_diag, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Restored a different diagonal vector");
  ctx->inv_diag_valid = PETSC_TRUE;
  ctx->diag_valid     = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_VecDiagonal(Mat mat)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)mat->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&ctx->diag));
  PetscCall(VecDestroy(&ctx->inv_diag));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatVecDiagonalGetDiagonal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatVecDiagonalRestoreDiagonal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatVecDiagonalGetInverse_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatVecDiagonalRestoreInverse_C", NULL));
  PetscCall(PetscFree(mat->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_VecDiagonal(Mat J, PetscViewer viewer)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)J->data;
  PetscBool        iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(MatVecDiagonalValidate(J));
    PetscCall(VecView(ctx->diag, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_VecDiagonal(Mat J, Vec x)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)J->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(J));
  PetscCall(VecCopy(ctx->diag, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDiagonalSet_VecDiagonal(Mat J, Vec D, InsertMode is)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)J->data;

  PetscFunctionBegin;
  switch (is) {
  case ADD_VALUES:
  case ADD_ALL_VALUES:
  case ADD_BC_VALUES:
    PetscCall(MatVecDiagonalValidate(J));
    PetscCall(VecAXPY(ctx->diag, 1.0, D));
    break;
  case INSERT_VALUES:
  case INSERT_BC_VALUES:
  case INSERT_ALL_VALUES:
    PetscCall(VecCopy(D, ctx->diag));
    ctx->diag_valid = PETSC_TRUE;
    break;
  case MAX_VALUES:
    PetscCall(MatVecDiagonalValidate(J));
    PetscCall(VecPointwiseMax(ctx->diag, D, ctx->diag));
    break;
  case MIN_VALUES:
    PetscCall(MatVecDiagonalValidate(J));
    PetscCall(VecPointwiseMin(ctx->diag, D, ctx->diag));
    break;
  case NOT_SET_VALUES:
    break;
  }
  ctx->inv_diag_valid = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatShift_VecDiagonal(Mat Y, PetscScalar a)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)Y->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(Y));
  PetscCall(VecShift(ctx->diag, a));
  ctx->inv_diag_valid = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_VecDiagonal(Mat Y, PetscScalar a)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)Y->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(Y));
  if (ctx->diag_valid) PetscCall(VecScale(ctx->diag, a));
  ctx->inv_diag_valid = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDiagonalScale_VecDiagonal(Mat Y, Vec l, Vec r)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)Y->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidate(Y));
  if (l) {
    PetscCall(VecPointwiseMult(ctx->diag, ctx->diag, l));
    ctx->inv_diag_valid = PETSC_FALSE;
  }
  if (r) {
    PetscCall(VecPointwiseMult(ctx->diag, ctx->diag, r));
    ctx->inv_diag_valid = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_VecDiagonal(Mat Y)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)Y->data;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(ctx->diag));
  ctx->inv_diag_valid = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_VecDiagonal(Mat matin, Vec b, Vec x)
{
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)matin->data;

  PetscFunctionBegin;
  PetscCall(MatVecDiagonalValidateInverse(matin));
  PetscCall(VecPointwiseMult(x, b, ctx->inv_diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetInfo_VecDiagonal(Mat A, MatInfoType flag, MatInfo *info)
{
  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = A->cmap->N;
  info->nz_used           = A->cmap->N;
  info->nz_unneeded       = 0.0;
  info->assemblies        = A->num_ass;
  info->mallocs           = 0.0;
  info->memory            = 0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatCreateVecDiagonal - Creates a matrix with given vector on its diagonal.

   Collective

   Input Parameter:
.  diag - vector for the diagonal

   Output Parameter:
.  J - the diagonal matrix

   Level: advanced

   Notes:
    Only supports square matrices with the same number of local rows and columns.

    The input vector diag will be referenced internally: any changes to diag
    will affect the matrix.

.seealso: [](chapter_matrices), `Mat`, `MatDestroy()`, `MATCONSTANTDIAGONAL`, `MatScale()`, `MatShift()`, `MatMult()`, `MatGetDiagonal()`, `MatSolve()`
@*/
PetscErrorCode MatCreateVecDiagonal(Vec diag, Mat *J)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(diag, VEC_CLASSID, 1);
  PetscCall(MatCreate(PetscObjectComm((PetscObject)diag), J));
  PetscInt m, M;
  PetscCall(VecGetLocalSize(diag, &m));
  PetscCall(VecGetSize(diag, &M));
  PetscCall(MatSetSizes(*J, m, m, M, M));
  PetscCall(MatSetType(*J, MATVECDIAGONAL));

  PetscLayout map;
  PetscCall(VecGetLayout(diag, &map));
  PetscCall(MatSetLayouts(*J, map, map));
  Mat_VecDiagonal *ctx = (Mat_VecDiagonal *)(*J)->data;
  PetscCall(PetscObjectReference((PetscObject)diag));
  PetscCall(VecDestroy(&ctx->diag));
  PetscCall(VecDestroy(&ctx->inv_diag));
  ctx->diag           = diag;
  ctx->diag_valid     = PETSC_TRUE;
  ctx->inv_diag_valid = PETSC_FALSE;
  VecType type;
  PetscCall(VecDuplicate(ctx->diag, &ctx->inv_diag));
  PetscCall(VecGetType(diag, &type));
  PetscCall(PetscFree((*J)->defaultvectype));
  PetscCall(PetscStrallocpy(type, &(*J)->defaultvectype));
  PetscCall(MatSetUp(*J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATVECDIAGONAL - MATVECDIAGONAL = "vecdiagonal" - A diagonal matrix type with the diagonal implemented as a `Vec`.  Useful for
   cases where `VecPointwiseMult()` or `VecPointwiseDivide()` should be thought of as the actions of a linear operator.

  Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MatCreateVecDiagonal()`
M*/
PETSC_INTERN PetscErrorCode MatCreate_VecDiagonal(Mat A)
{
  PetscInt         m, M;
  Mat_VecDiagonal *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatGetSize(A, &M, NULL));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)A), &ctx->diag));
  PetscCall(VecSetSizes(ctx->diag, m, M));
  PetscCall(VecSetType(ctx->diag, A->defaultvectype));
  PetscCall(VecDuplicate(ctx->diag, &ctx->inv_diag));

  PetscCall(VecSetLayout(ctx->diag, A->rmap));

  A->data = (void *)ctx;

  A->assembled                   = PETSC_TRUE;
  A->preallocated                = PETSC_TRUE;
  A->structurally_symmetric      = PETSC_BOOL3_TRUE;
  A->structural_symmetry_eternal = PETSC_TRUE;
  A->symmetry_eternal            = PETSC_TRUE;
  A->symmetric                   = PETSC_BOOL3_TRUE;
  if (!PetscDefined(USE_COMPLEX)) A->hermitian = PETSC_BOOL3_TRUE;

  A->ops->mult             = MatMult_VecDiagonal;
  A->ops->multadd          = MatMultAdd_VecDiagonal;
  A->ops->multtranspose    = MatMult_VecDiagonal;
  A->ops->multtransposeadd = MatMultAdd_VecDiagonal;
  A->ops->norm             = MatNorm_VecDiagonal;
  A->ops->duplicate        = MatDuplicate_VecDiagonal;
  A->ops->solve            = MatSolve_VecDiagonal;
  A->ops->solvetranspose   = MatSolve_VecDiagonal;
  A->ops->shift            = MatShift_VecDiagonal;
  A->ops->scale            = MatScale_VecDiagonal;
  A->ops->diagonalscale    = MatDiagonalScale_VecDiagonal;
  A->ops->getdiagonal      = MatGetDiagonal_VecDiagonal;
  A->ops->diagonalset      = MatDiagonalSet_VecDiagonal;
  A->ops->view             = MatView_VecDiagonal;
  A->ops->zeroentries      = MatZeroEntries_VecDiagonal;
  A->ops->destroy          = MatDestroy_VecDiagonal;
  A->ops->getinfo          = MatGetInfo_VecDiagonal;
  A->ops->axpy             = MatAXPY_VecDiagonal;

  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatVecDiagonalGetDiagonal_C", MatVecDiagonalGetDiagonal_VecDiagonal));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatVecDiagonalRestoreDiagonal_C", MatVecDiagonalRestoreDiagonal_VecDiagonal));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatVecDiagonalGetInverse_C", MatVecDiagonalGetInverse_VecDiagonal));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatVecDiagonalRestoreInverse_C", MatVecDiagonalRestoreInverse_VecDiagonal));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATVECDIAGONAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
