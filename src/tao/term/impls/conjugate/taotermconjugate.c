#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Conjugate TaoTerm_Conjugate;

struct _n_TaoTerm_Conjugate {
  TaoTerm   orig;
  PetscBool is_virtual;
  Vec       workvec;
};

static PetscErrorCode TaoTermDestroy_Conjugate(TaoTerm term)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  term->data = NULL;
  //TODO virtual data copy etc
  //prob need to do some ref counting etc
  PetscCall(VecDestroy(&cj->workvec));
  PetscCall(TaoTermDestroy(&cj->orig));
  PetscCall(PetscFree(cj));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermConjugateGetOriginalType_C", NULL));
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
//f(x-p)
// TaoTermType = TAOTERMTRANSLATE, or something else?
// TaoTermParametersTranslate = TRUE -> then this,
// I couldn't think of case where, we have f(x;p), and want f*(y;p)?
// if paramtrans = false, then will just pass param along,
//
// KL?
static PetscErrorCode TaoTermProximalMap_Conjugate(TaoTerm term, Vec p, PetscReal alpha, TaoTerm g, Vec q, PetscReal beta, Vec x)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;
  PetscBool          is_hl2s, workvec_compat;
  PetscReal          lambda;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)g, TAOTERMHALFL2SQUARED, &is_hl2s));

  //TODO this version assumes that p is translation vector. For else??
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
    if (p) PetscCall(VecAXPY(cj->workvec, 1., p));
    PetscCall(VecScale(cj->workvec, 1. / lambda));
    //TODO for L1, at least, since we are pre-processing it, we dont need to pass p.
    //but for other methods, how can we tell??
    //Doing prox_(1/step), so switch alpha and beta
    //PetscCall(TaoTermProximalMap(cj->orig, p, alpha, g, cj->workvec, 1./beta, x));
    PetscCall(TaoTermProximalMap(cj->orig, NULL, beta, g, cj->workvec, alpha, x));
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
  PetscBool is_cj;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMHALFL2SQUARED, &is_cj));
  PetscCheck(is_cj, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "Input TaoTerm needs to be of TAOTERMCONJUGATE type");
  *type = ((PetscObject)term)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermConjugateGetOriginalType_Conjugate(TaoTerm term, TaoTermType *type)
{
  TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)term->data;

  PetscFunctionBegin;
  PetscCall(TaoTermGetType(cj->orig, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMCONJUGATE - A `TaoTerm` that is convex conjugate of...

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Conjugate(TaoTerm term)
{
  TaoTerm_Conjugate *cj;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&cj));
  term->data = (void *)cj;

  term->ops->destroy               = TaoTermDestroy_Conjugate;
  term->ops->objective             = TaoTermObjective_Conjugate;
  term->ops->gradient              = TaoTermGradient_Conjugate;
  term->ops->objectiveandgradient  = TaoTermObjectiveAndGradient_Conjugate;
  term->ops->hessian               = TaoTermHessian_Conjugate;
  term->ops->hessianmult           = TaoTermHessianMult_Conjugate;
  term->ops->createhessianmatrices = TaoTermCreateHessianMatricesDefault;
  term->ops->proximalmap           = TaoTermProximalMap_Conjugate;

  if (!term->H_mattype) PetscCall(PetscStrallocpy(MATSHELL, &term->H_mattype));
  if (!term->Hpre_mattype) PetscCall(PetscStrallocpy(MATCONSTANTDIAGONAL, &term->Hpre_mattype));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermConjugateGetOriginalType_C", TaoTermConjugateGetOriginalType_Conjugate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO what if cc_Term is already created? destroy and create again?
/*@
  TaoTermCreateConjugate - Create a convex conjugate version of `TaoTerm`

  Collective

  Input Parameter:
. term - the original `TaoTerm`

  Output Parameter:
. cc_term - a new TaoTerm, that is convex conjugate of input term

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreateConjugateVirtual()`
@*/
PetscErrorCode TaoTermCreateConjugate(TaoTerm term, TaoTerm *cc_term)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(term, 2);
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)term), cc_term));
  PetscCall(TaoTermSetType(*cc_term, TAOTERMCONJUGATE));

  {
    TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)(*cc_term)->data;

    cj->orig       = term;
    cj->is_virtual = PETSC_FALSE;
    //TODO actualy copying things?
    PetscCall(PetscObjectReference((PetscObject)cj->orig));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateConjugateVirtual - Create a convex conjugate version of `TaoTerm` virtually

  Collective

  Input Parameter:
. term - the original `TaoTerm`

  Output Parameter:
. cc_term - a new TaoTerm, that is convex conjugate of input term

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreateConjugate()`
@*/
PetscErrorCode TaoTermCreateConjugateVirtual(TaoTerm term, TaoTerm *cc_term)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(term, 2);
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)term), cc_term));
  PetscCall(TaoTermSetType(*cc_term, TAOTERMCONJUGATE));
  {
    TaoTerm_Conjugate *cj = (TaoTerm_Conjugate *)(*cc_term)->data;

    cj->orig       = term;
    cj->is_virtual = PETSC_TRUE;

    PetscCall(PetscObjectReference((PetscObject)cj->orig));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
