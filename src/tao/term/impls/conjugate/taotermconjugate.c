#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Conjugate TaoTerm_Conjugate;

struct _n_TaoTerm_Conjugate {
  TaoTerm orig;
  Vec     workvec;
};

static PetscErrorCode TaoTermDestroy_Conjugate(TaoTerm term)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  term->data = NULL;
  PetscCall(VecDestroy(&cj->workvec));
  PetscCall(TaoTermDestroy(&cj->orig));
  PetscCall(PetscFree(cj));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermConjugateGetOriginalTerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermConjugateGetOriginalType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermConjugateSetOriginalTaoTerm_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_Conjugate(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  PetscTryTypeMethod(cj->orig, conjugate_objective, x, params, value);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_Conjugate(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  PetscTryTypeMethod(cj->orig, conjugate_objectiveandgradient, x, params, value, g);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_Conjugate(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  PetscTryTypeMethod(cj->orig, conjugate_gradient, x, params, g);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_Conjugate(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  PetscTryTypeMethod(cj->orig, conjugate_hessian, x, params, H, Hpre);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessianMult_Conjugate(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  PetscTryTypeMethod(cj->orig, conjugate_hessianmult, x, params, v, Hv);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Conjugate(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;
  PetscBool          is_ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &is_ascii));
  if (is_ascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "TaoTermConjugate original term:\n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(TaoTermView(cj->orig, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO write detailed docstring
 * In this case, there are two possibilities:
   1. Translation of Conjugate, or
   2. Conjugate of Translation.
   Most algorithms think about second case - therefore, we
   only support conjugate of translation.
   (I don't think, with current API, Translation of Conjugate is possible.TODO)

   g(x) = f(x+p), \lambda = \alpha / \beta
   Translation of conjugate:
   prox_{\lambda, g*}(y) = y - \lambda prox_{1/\lambda, f^*} ((y+p)/\lambda)

   Conjugate of Translation: (Current implementation)
   prox_{\lambda, g*}(y) = y - \lambda prox_{1/\lambda, g}(y / \lambda)

   prox_{\lambda, g}(z) = prox_{\lambda, f}(z + a) - a                        */
static PetscErrorCode TaoTermProximalMap_Conjugate(TaoTerm term, Vec p, PetscReal alpha, TaoTerm g, Vec q, PetscReal beta, Vec x)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;
  PetscBool          is_hl2s, workvec_compat;
  PetscReal          lambda;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)g, TAOTERMHALFL2SQUARED, &is_hl2s));

  if (is_hl2s || (g == NULL)) {
    lambda = alpha / beta;

    if (!cj->workvec) PetscCall(VecDuplicate(x, &cj->workvec));
    else {
      PetscCall(TaoTermWorkvecTestCompatibility(x, cj->workvec, &workvec_compat));
      if (!workvec_compat) {
        PetscCall(VecDestroy(&cj->workvec));
        PetscCall(VecDuplicate(x, &cj->workvec));
      }
    }
    PetscCall(VecCopy(q, cj->workvec));
    PetscCall(VecScale(cj->workvec, 1. / lambda));
    //Doing prox_(1/step), so switch alpha and beta. If param vector is given, its handled internally as translation, if applicable.
    PetscCall(TaoTermProximalMap(cj->orig, p, beta, g, cj->workvec, alpha, x));
    PetscCall(VecAYPX(x, -lambda, q));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoTermProximalMap for conjugate currently only supports TAOTERMHALFL2SQUARED regularizer");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermConjugateGetOriginalType - Get the type of a original `TaoTerm`
  of `TAOTERMCONJUGATE`

  Not collective

  Input Parameter:
. term - a `TaoTerm` of `TAOTERMCONJUGATE` type

  Output Parameter:
. type - the `TaoTermType` of original `TaoTerm`

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermType`, `TAOTERMCONJUGATE`
@*/
PetscErrorCode TaoTermConjugateGetOriginalType(TaoTerm term, TaoTermType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscUseMethod(term, "TaoTermConjugateGetOriginalType_C", (TaoTerm, TaoTermType *), (term, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermConjugateGetOriginalType_Conjugate(TaoTerm term, TaoTermType *type)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  PetscCall(TaoTermGetType(cj->orig, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermConjugateGetOriginalTerm - Get the type of a original `TaoTerm`
  of `TAOTERMCONJUGATE`

  Not collective

  Input Parameter:
. cj_term - a `TaoTerm` of `TAOTERMCONJUGATE` type

  Output Parameter:
. orig_term - the original `TaoTerm`

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMCONJUGATE`
@*/
PetscErrorCode TaoTermConjugateGetOriginalTerm(TaoTerm cj_term, TaoTerm *orig_term)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cj_term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(orig_term, 2);
  PetscUseMethod(cj_term, "TaoTermConjugateGetOriginalTerm_C", (TaoTerm, TaoTerm *), (cj_term, orig_term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermConjugateGetOriginalTerm_Conjugate(TaoTerm cj_term, TaoTerm *orig_term)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)cj_term->data;

  PetscFunctionBegin;
  *orig_term = cj->orig;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermConjugateSetOriginalTaoTerm - Sets a original `TaoTerm` to a
  `TaoTerm` with `TAOTERMCONJUGATE` type.

  Collective

  Input Parameter:
+ cj   - the `TaoTerm` of `TAOTERMCONJUGATE` type
- orig - the original `TaoTerm` to be set inside of conjugate term

  Level: advanced

  Note: This is virtual setting - no copying

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreateConjugate()`
@*/
PetscErrorCode TaoTermConjugateSetOriginalTaoTerm(TaoTerm cj, TaoTerm orig)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cj, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(orig, TAOTERM_CLASSID, 2);
  PetscTryMethod(cj, "TaoTermConjugateSetOriginalTaoTerm_C", (TaoTerm, TaoTerm), (cj, orig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermConjugateSetOriginalTaoTerm_Conjugate(TaoTerm cj, TaoTerm orig)
{
  TaoTerm_Conjugate *cjctx = (TaoTerm_Conjugate *)cj->data;
  PetscLayout        sol_layout, param_layout;

  PetscFunctionBegin;
  cjctx->orig = orig;
  PetscCall(TaoTermGetLayouts(orig, &sol_layout, &param_layout));
  PetscCall(TaoTermSetLayouts(cj, sol_layout, param_layout));
  PetscCall(PetscObjectReference((PetscObject)cjctx->orig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMCONJUGATE - A `TaoTerm` that is convex conjugate of an original term.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreateConjugate`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Conjugate(TaoTerm term)
{
  TaoTerm_Conjugate *cj;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&cj));
  term->data = (void *)cj;

  term->ops->destroy               = TaoTermDestroy_Conjugate;
  term->ops->view                  = TaoTermView_Conjugate;
  term->ops->objective             = TaoTermObjective_Conjugate;
  term->ops->gradient              = TaoTermGradient_Conjugate;
  term->ops->objectiveandgradient  = TaoTermObjectiveAndGradient_Conjugate;
  term->ops->hessian               = TaoTermHessian_Conjugate;
  term->ops->hessianmult           = TaoTermHessianMult_Conjugate;
  term->ops->createhessianmatrices = TaoTermCreateHessianMatricesDefault;
  term->ops->proximalmap           = TaoTermProximalMap_Conjugate;

  if (!term->H_mattype) PetscCall(PetscStrallocpy(MATSHELL, &term->H_mattype));
  if (!term->Hpre_mattype) PetscCall(PetscStrallocpy(MATCONSTANTDIAGONAL, &term->Hpre_mattype));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermConjugateGetOriginalTerm_C", TaoTermConjugateGetOriginalTerm_Conjugate));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermConjugateGetOriginalType_C", TaoTermConjugateGetOriginalType_Conjugate));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermConjugateSetOriginalTaoTerm_C", TaoTermConjugateSetOriginalTaoTerm_Conjugate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateConjugate - Create a convex conjugate version of `TaoTerm`

  Collective

  Input Parameter:
. term - the original `TaoTerm`

  Output Parameter:
. cc_term - a new `TaoTerm`, that is convex conjugate of input term

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`
@*/
PetscErrorCode TaoTermCreateConjugate(TaoTerm term, TaoTerm *cc_term)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(cc_term, 2);
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)term), cc_term));
  PetscCall(TaoTermSetType(*cc_term, TAOTERMCONJUGATE));
  PetscCall(TaoTermConjugateSetOriginalTaoTerm(*cc_term, term));
  PetscFunctionReturn(PETSC_SUCCESS);
}
