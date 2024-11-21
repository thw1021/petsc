#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Quadratic TaoTerm_Quadratic;

struct _n_TaoTerm_Quadratic {
  Mat A;
  Vec _diff;
  Vec _Adiff;
};

static PetscErrorCode TaoTermDestroy_Quadratic(TaoTerm term)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&quad->A));
  PetscCall(VecDestroy(&quad->_diff));
  PetscCall(VecDestroy(&quad->_Adiff));
  PetscCall(PetscFree(quad));
  term->data = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermQuadraticSetMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermQuadraticGetMat_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Quadratic(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  if (quad->A) {
    PetscBool iascii;

    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));

    if (iascii) {
      PetscViewerFormat format;
      PetscBool         pop = PETSC_FALSE;

      PetscCall(PetscViewerGetFormat(viewer, &format));
      if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
        pop = PETSC_TRUE;
      }
      PetscCall(MatView(quad->A, viewer));
      if (pop) PetscCall(PetscViewerPopFormat(viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermQuadraticDiff(TaoTerm term, Vec x, Vec params, Vec *diff)
{
  PetscFunctionBegin;
  *diff = x;
  if (params) {
    TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
    if (!quad->_diff) PetscCall(VecDuplicate(x, &quad->_diff));
    PetscCall(VecCopy(x, quad->_diff));
    PetscCall(VecAXPY(quad->_diff, -1.0, params));
    *diff = quad->_diff;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_Quadratic(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
  Vec                diff;
  PetscScalar        sval;

  PetscFunctionBegin;
  PetscCall(TaoTermQuadraticDiff(term, x, params, &diff));
  if (!quad->_Adiff) PetscCall(VecDuplicate(diff, &quad->_Adiff));
  PetscCall(MatMult(quad->A, diff, quad->_Adiff));
  PetscCall(VecDot(diff, quad->_Adiff, &sval));
  *value = 0.5 * PetscRealPart(sval);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_Quadratic(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
  Vec                diff;

  PetscFunctionBegin;
  PetscCall(TaoTermQuadraticDiff(term, x, params, &diff));
  PetscCall(MatMult(quad->A, diff, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_Quadratic(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
  Vec                diff;
  PetscScalar        sval;

  PetscFunctionBegin;
  PetscCall(TaoTermQuadraticDiff(term, x, params, &diff));
  PetscCall(MatMult(quad->A, diff, g));
  PetscCall(VecDot(diff, g, &sval));
  *value = 0.5 * PetscRealPart(sval);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermHessian_Quadratic(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  if (H) PetscCall(MatCopy(quad->A, H, UNKNOWN_NONZERO_PATTERN));
  if (Hpre && Hpre != H) PetscCall(MatCopy(quad->A, Hpre, UNKNOWN_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermHessianMult_Quadratic(TaoTerm term, Vec x, Vec Params, Vec v, Vec Hv)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  PetscCall(MatMult(quad->A, v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateHessianMatrices_Quadratic(TaoTerm term, Mat *H, Mat *Hpre)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  if (H) PetscCall(MatDuplicate(quad->A, MAT_DO_NOT_COPY_VALUES, H));
  if (Hpre) {
    if (H) {
      PetscCall(PetscObjectReference((PetscObject)*H));
      *Hpre = *H;
    } else PetscCall(MatDuplicate(quad->A, MAT_DO_NOT_COPY_VALUES, H));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermQuadraticGetMat - Get the matrix defining a `TaoTerm` of type `TAOTERMQUADRATIC`

  Not collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERMQUADRATIC`

  Output Parameter:
. A - the matrix

  Level: intermediate

  Note:
  This function will return `NULL` if the term is not a `TAOTERMQUADRATIC`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMQUADRATIC`,
          `TaoTermQuadraticSetMat()`,
@*/
PetscErrorCode TaoTermQuadraticGetMat(TaoTerm term, Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (A) PetscAssertPointer(A, 2);
  *A = NULL;
  PetscTryMethod(term, "TaoTermQuadraticGetMat_C", (TaoTerm, Mat *), (term, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermQuadraticGetMat_Quadratic(TaoTerm term, Mat *A)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  *A = quad->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermQuadraticSetMat - Set the matrix defining a `TaoTerm` of type `TAOTERMQUADRATIC`

  Collective

  Input Parameters:
+ term - a `TaoTerm` of type `TAOTERMQUADRATIC`
- A    - the matrix

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMQUADRATIC`,
          `TaoTermQuadraticGetMat()`,
@*/
PetscErrorCode TaoTermQuadraticSetMat(TaoTerm term, Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 2);
  PetscCheckSameComm(term, 1, A, 2);
  PetscTryMethod(term, "TaoTermQuadraticSetMat_C", (TaoTerm, Mat), (term, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermQuadraticSetMat_Quadratic(TaoTerm term, Mat A)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
  PetscLayout        rmap, cmap;
  VecType            vec_type;
  PetscBool          is_square;

  PetscFunctionBegin;
  if (A != quad->A) {
    MatType mat_type;
    PetscCall(MatGetLayouts(A, &rmap, &cmap));
    PetscCall(PetscLayoutCompare(rmap, cmap, &is_square));
    PetscCheck(is_square, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_SIZ, "Matrix for quadratic term must be square");
    PetscCall(PetscObjectReference((PetscObject)A));
    PetscCall(MatDestroy(&quad->A));
    PetscCall(VecDestroy(&quad->_diff));
    PetscCall(VecDestroy(&quad->_Adiff));
    quad->A = A;
    PetscCall(MatGetVecType(A, &vec_type));
    PetscCall(TaoTermSetVecTypes(term, vec_type, vec_type));
    PetscCall(TaoTermSetLayouts(term, rmap, rmap));
    PetscCall(PetscFree(term->H_mattype));
    PetscCall(PetscFree(term->Hpre_mattype));
    PetscCall(MatGetType(A, &mat_type));
    PetscCall(PetscStrallocpy(mat_type, &term->H_mattype));
    PetscCall(PetscStrallocpy(mat_type, &term->Hpre_mattype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMQUADRATIC - A `TaoTerm` that computes $\tfrac{1}{2}(x - p)^T A (x - p)$, for a fixed matrix $A$, solution $x$ and parameters $p$.

  Level: intermediate

  Notes:
  This term is `TAOTERM_PARAMETERS_OPTIONAL`.  If the paramters argument is `NULL` for
  evaluation routines the term computes $\tfrac{1}{2}x^T A x$.

  The default Hessian creation mode (see `TaoTermGetCreateHessianMode()`) is `H == Hpre` and `TaoTermCreateHessianMatrices()`
  will create a matrix with the same type as $A$.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreateQuadratic()`,
          `TAOTERMHALFL2SQUARED`,
          `TAOTERMHALFL1`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Quadratic(TaoTerm term)
{
  TaoTerm_Quadratic *quad;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&quad));
  term->data = (void *)quad;

  term->ops->destroy               = TaoTermDestroy_Quadratic;
  term->ops->view                  = TaoTermView_Quadratic;
  term->ops->objective             = TaoTermObjective_Quadratic;
  term->ops->gradient              = TaoTermGradient_Quadratic;
  term->ops->objectiveandgradient  = TaoTermObjectiveAndGradient_Quadratic;
  term->ops->hessian               = TaoTermHessian_Quadratic;
  term->ops->hessianmult           = TaoTermHessianMult_Quadratic;
  term->ops->createhessianmatrices = TaoTermCreateHessianMatrices_Quadratic;

  term->Hpre_is_H = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermQuadraticGetMat_C", TaoTermQuadraticGetMat_Quadratic));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermQuadraticSetMat_C", TaoTermQuadraticSetMat_Quadratic));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateQuadratic - Create a `TAOTERMQUADRATIC` for a given matrix

  Collective

  Input Parameter:
. A - a square matrix

  Output Parameter:
. term - a `TaoTerm` that impements $\tfrac{1}{2}(x - p)^T A (x - p)$

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMQUADRATIC`,
          `TaoTermCreateHalfL2Squared()`,
          `TaoTermCreateL1()`,
@*/
PetscErrorCode TaoTermCreateQuadratic(Mat A, TaoTerm *term)
{
  TaoTerm _term;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)A), &_term));
  PetscCall(TaoTermSetType(_term, TAOTERMQUADRATIC));
  PetscCall(TaoTermQuadraticSetMat(_term, A));
  *term = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}
