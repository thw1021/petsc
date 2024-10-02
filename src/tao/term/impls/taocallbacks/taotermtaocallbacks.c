#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_TaoCallbacks TaoTerm_TaoCallbacks;

struct _n_TaoTerm_TaoCallbacks {
  Tao tao;
  PetscErrorCode (*objective)(Tao, Vec, PetscReal *, void *);
  PetscErrorCode (*gradient)(Tao, Vec, Vec, void *);
  PetscErrorCode (*objectiveandgradient)(Tao, Vec, PetscReal *, Vec, void *);
  PetscErrorCode (*hessian)(Tao, Vec, Mat, Mat, void *);
  void *obj_ctx;
  void *grad_ctx;
  void *objgrad_ctx;
  void *hess_ctx;
  char *obj_name;
  char *grad_name;
  char *objgrad_name;
  char *hess_name;
  char *set_obj_name;
  char *set_grad_name;
  char *set_objgrad_name;
  char *set_hess_name;
};

#define PetscCheckTaoTermCallbacksValid(term, tt, params) \
  do { \
    PetscCheck((params) == NULL, PetscObjectComm((PetscObject)(term)), PETSC_ERR_ARG_INCOMP, "TAOTERMTAOCALLBACKS does not accept a vector of parameters"); \
    PetscCheck((tt)->tao != NULL, PetscObjectComm((PetscObject)(term)), PETSC_ERR_ARG_WRONGSTATE, "TAOTERMTAOCALLBACKS does not have an outer Tao"); \
  } while (0)

static PetscErrorCode TaoTermDestroy_TaoCallbacks(TaoTerm term)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  // tt->tao is a weak reference, we do not destroy it
  PetscCall(PetscFree(tt->obj_name));
  PetscCall(PetscFree(tt->grad_name));
  PetscCall(PetscFree(tt->objgrad_name));
  PetscCall(PetscFree(tt->hess_name));
  PetscCall(PetscFree(tt->set_obj_name));
  PetscCall(PetscFree(tt->set_grad_name));
  PetscCall(PetscFree(tt->set_objgrad_name));
  PetscCall(PetscFree(tt->set_hess_name));
  PetscCall(PetscFree(tt));
  term->data = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksSetObjective_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksGetObjective_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksSetGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksGetGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksSetObjAndGrad_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksGetObjAndGrad_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksSetHessian_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksGetHessian_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_TaoCallbacks(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  if (tt->objective) {
    PetscCallBack(tt->obj_name, (*tt->objective)(tt->tao, x, value, tt->obj_ctx));
  } else if (tt->objectiveandgradient) {
    Vec dummy;

    PetscCall(PetscInfo(term, "Duplicating variable vector in order to call func/grad routine\n"));
    PetscCall(VecDuplicate(x, &dummy));
    PetscCallBack(tt->objgrad_name, (*tt->objectiveandgradient)(tt->tao, x, value, dummy, tt->objgrad_ctx));
    PetscCall(VecDestroy(&dummy));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "Objective routine not set: call %s", tt->set_obj_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_TaoCallbacks(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  if (tt->gradient) {
    PetscCallBack(tt->grad_name, (*tt->gradient)(tt->tao, x, g, tt->grad_ctx));
  } else if (tt->objectiveandgradient) {
    PetscReal dummy;

    PetscCallBack(tt->objgrad_name, (*tt->objectiveandgradient)(tt->tao, x, &dummy, g, tt->objgrad_ctx));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "Gradient routine not set: call %s", tt->set_grad_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_TaoCallbacks(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  if (tt->objectiveandgradient) {
    PetscCallBack(tt->objgrad_name, (*tt->objectiveandgradient)(tt->tao, x, value, g, tt->objgrad_ctx));
  } else if (tt->objective && tt->gradient) {
    PetscCallBack(tt->obj_name, (*tt->objective)(tt->tao, x, value, tt->obj_ctx));
    PetscCallBack(tt->grad_name, (*tt->gradient)(tt->tao, x, g, tt->grad_ctx));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "Objective/gradient routine not set: call %s", tt->set_objgrad_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_TaoCallbacks(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  PetscCheck(tt->hessian, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "Hessian routine not set: call %s", tt->set_hess_name);
  PetscCallBack(tt->hess_name, (*tt->hessian)(tt->tao, x, H, Hpre, tt->hess_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_TaoCallbacks(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;
  PetscBool             iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    if (!tt->tao) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "not attached to a Tao\n"));
    } else {
      const char *name = "[name omitted]";
      const char *prefix;

      if (!PetscCIEnabled) PetscCall(PetscObjectGetName((PetscObject)tt->tao, &name));
      else if (tt->tao->hdr.name) name = tt->tao->hdr.name;
      PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tt->tao, &prefix));
      if (prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "attached to Tao %s (%s)\n", name, prefix));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "attached to Tao %s\n", name));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermTaoCallbacksSetObjective(TaoTerm term, PetscErrorCode (*tao_obj)(Tao, Vec, PetscReal *, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermTaoCallbacksSetObjective_C", (TaoTerm, PetscErrorCode (*)(Tao, Vec, PetscReal *, void *), void *), (term, tao_obj, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermTaoCallbacksSetObjective_TaoCallbacks(TaoTerm term, PetscErrorCode (*tao_obj)(Tao, Vec, PetscReal *, void *), void *ctx)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  tt->objective = tao_obj;
  tt->obj_ctx   = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermTaoCallbacksGetObjective(TaoTerm term, PetscErrorCode (**tao_obj)(Tao, Vec, PetscReal *, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao_obj) *tao_obj = NULL;
  if (ctx) *ctx = NULL;
  PetscTryMethod(term, "TaoTermTaoCallbacksSetObjective_C", (TaoTerm, PetscErrorCode (**)(Tao, Vec, PetscReal *, void *), void **), (term, tao_obj, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermTaoCallbacksGetObjective_TaoCallbacks(TaoTerm term, PetscErrorCode (**tao_obj)(Tao, Vec, PetscReal *, void *), void **ctx)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  if (tao_obj) *tao_obj = tt->objective;
  if (ctx) *ctx = tt->obj_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermTaoCallbacksSetGradient(TaoTerm term, PetscErrorCode (*tao_grad)(Tao, Vec, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermTaoCallbacksSetGradient_C", (TaoTerm, PetscErrorCode (*)(Tao, Vec, Vec, void *), void *), (term, tao_grad, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermTaoCallbacksSetGradient_TaoCallbacks(TaoTerm term, PetscErrorCode (*tao_grad)(Tao, Vec, Vec, void *), void *ctx)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  tt->gradient = tao_grad;
  tt->grad_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermTaoCallbacksGetGradient(TaoTerm term, PetscErrorCode (**tao_grad)(Tao, Vec, Vec, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao_grad) *tao_grad = NULL;
  if (ctx) *ctx = NULL;
  PetscTryMethod(term, "TaoTermTaoCallbacksSetGradient_C", (TaoTerm, PetscErrorCode (**)(Tao, Vec, Vec, void *), void **), (term, tao_grad, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermTaoCallbacksGetGradient_TaoCallbacks(TaoTerm term, PetscErrorCode (**tao_grad)(Tao, Vec, Vec, void *), void **ctx)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  if (tao_grad) *tao_grad = tt->gradient;
  if (ctx) *ctx = tt->grad_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermTaoCallbacksSetObjAndGrad(TaoTerm term, PetscErrorCode (*tao_objgrad)(Tao, Vec, PetscReal *, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermTaoCallbacksSetObjAndGrad_C", (TaoTerm, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, void *), void *), (term, tao_objgrad, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermTaoCallbacksSetObjAndGrad_TaoCallbacks(TaoTerm term, PetscErrorCode (*tao_objgrad)(Tao, Vec, PetscReal *, Vec, void *), void *ctx)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  tt->objectiveandgradient = tao_objgrad;
  tt->objgrad_ctx          = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermTaoCallbacksGetObjAndGrad(TaoTerm term, PetscErrorCode (**tao_objgrad)(Tao, Vec, PetscReal *, Vec, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao_objgrad) *tao_objgrad = NULL;
  if (ctx) *ctx = NULL;
  PetscTryMethod(term, "TaoTermTaoCallbacksSetObjAndGrad_C", (TaoTerm, PetscErrorCode (**)(Tao, Vec, PetscReal *, Vec, void *), void **), (term, tao_objgrad, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermTaoCallbacksGetObjAndGrad_TaoCallbacks(TaoTerm term, PetscErrorCode (**tao_objgrad)(Tao, Vec, PetscReal *, Vec, void *), void **ctx)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  if (tao_objgrad) *tao_objgrad = tt->objectiveandgradient;
  if (ctx) *ctx = tt->objgrad_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermTaoCallbacksSetHessian(TaoTerm term, PetscErrorCode (*tao_hess)(Tao, Vec, Mat, Mat, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermTaoCallbacksSetHessian_C", (TaoTerm, PetscErrorCode (*)(Tao, Vec, Mat, Mat, void *), void *), (term, tao_hess, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermTaoCallbacksSetHessian_TaoCallbacks(TaoTerm term, PetscErrorCode (*tao_hess)(Tao, Vec, Mat, Mat, void *), void *ctx)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  tt->hessian  = tao_hess;
  tt->hess_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermTaoCallbacksGetHessian(TaoTerm term, PetscErrorCode (**tao_hess)(Tao, Vec, Mat, Mat, void *), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao_hess) *tao_hess = NULL;
  if (ctx) *ctx = NULL;
  PetscTryMethod(term, "TaoTermTaoCallbacksSetHessian_C", (TaoTerm, PetscErrorCode (**)(Tao, Vec, Mat, Mat, void *), void **), (term, tao_hess, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermTaoCallbacksGetHessian_TaoCallbacks(TaoTerm term, PetscErrorCode (**tao_hess)(Tao, Vec, Mat, Mat, void *), void **ctx)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  if (tao_hess) *tao_hess = tt->hessian;
  if (ctx) *ctx = tt->hess_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsObjectiveDefined_TaoCallbacks(TaoTerm term, PetscBool *flg)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  *flg = (tt->objective != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsGradientDefined_TaoCallbacks(TaoTerm term, PetscBool *flg)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  *flg = (tt->gradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsObjectiveAndGradientDefined_TaoCallbacks(TaoTerm term, PetscBool *flg)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  *flg = (tt->objectiveandgradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsHessianDefined_TaoCallbacks(TaoTerm term, PetscBool *flg)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  *flg = (tt->hessian != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreate_TaoCallbacks_Internal(TaoTerm term, const char obj[], const char set_obj[], const char grad[], const char set_grad[], const char objgrad[], const char set_objgrad[], const char hess[], const char set_hess[])
{
  TaoTerm_TaoCallbacks *tt;
  char                  buf[256];
  size_t                len = PETSC_STATIC_ARRAY_LENGTH(buf);

  PetscFunctionBegin;
  term->parameters_mode = TAOTERM_PARAMETERS_NONE;

  PetscCall(PetscNew(&tt));
  term->data = (void *)tt;

  term->ops->destroy                       = TaoTermDestroy_TaoCallbacks;
  term->ops->objective                     = TaoTermObjective_TaoCallbacks;
  term->ops->gradient                      = TaoTermGradient_TaoCallbacks;
  term->ops->objectiveandgradient          = TaoTermObjectiveAndGradient_TaoCallbacks;
  term->ops->hessian                       = TaoTermHessian_TaoCallbacks;
  term->ops->view                          = TaoTermView_TaoCallbacks;
  term->ops->isobjectivedefined            = TaoTermIsObjectiveDefined_TaoCallbacks;
  term->ops->isgradientdefined             = TaoTermIsGradientDefined_TaoCallbacks;
  term->ops->isobjectiveandgradientdefined = TaoTermIsObjectiveAndGradientDefined_TaoCallbacks;
  term->ops->ishessiandefined              = TaoTermIsHessianDefined_TaoCallbacks;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksSetObjective_C", TaoTermTaoCallbacksSetObjective_TaoCallbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksGetObjective_C", TaoTermTaoCallbacksGetObjective_TaoCallbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksSetGradient_C", TaoTermTaoCallbacksSetGradient_TaoCallbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksGetGradient_C", TaoTermTaoCallbacksGetGradient_TaoCallbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksSetObjAndGrad_C", TaoTermTaoCallbacksSetObjAndGrad_TaoCallbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksGetObjAndGrad_C", TaoTermTaoCallbacksGetObjAndGrad_TaoCallbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksSetHessian_C", TaoTermTaoCallbacksSetHessian_TaoCallbacks));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermTaoCallbacksGetHessian_C", TaoTermTaoCallbacksGetHessian_TaoCallbacks));

  PetscCall(PetscSNPrintf(buf, len, "%s callback", obj ? obj : "unknown objective"));
  PetscCall(PetscStrallocpy(buf, &tt->obj_name));
  PetscCall(PetscStrallocpy(set_obj, &tt->set_obj_name));

  PetscCall(PetscSNPrintf(buf, len, "%s callback", grad ? grad : "unknown gradient"));
  PetscCall(PetscStrallocpy(buf, &tt->grad_name));
  PetscCall(PetscStrallocpy(set_grad, &tt->set_grad_name));

  PetscCall(PetscSNPrintf(buf, len, "%s callback", objgrad ? objgrad : "unknown objective/gradient"));
  PetscCall(PetscStrallocpy(buf, &tt->objgrad_name));
  PetscCall(PetscStrallocpy(set_objgrad, &tt->set_objgrad_name));

  PetscCall(PetscSNPrintf(buf, len, "%s callback", hess ? hess : "unknown hessian"));
  PetscCall(PetscStrallocpy(buf, &tt->hess_name));
  PetscCall(PetscStrallocpy(set_objgrad, &tt->set_hess_name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMTAOCALLBACKS - A `TaoTerm` implementation that accesses the callbacks that have been set in `TaoSetObective()`, `TaoSetGradient()`,
  `TaoSetObjectiveAndGradient()`, and `TaoSetHessian()`.

  Level: developer

  Notes:
  If you are interested in creating you own term, you shouldn't use this. Use
  `TAOTERMSHELL` or create your own implementation of `TaoTerm` with
  `TaoTermRegister()`.

  A `TAOTERMTAOCALLBACKS` is always `TAOTERM_PARAMETERS_NONE`, so the `params`
  argument of `TaoTerm` evaluation routines should always be `NULL`.

  A `TAOTERMTAOCALLBACKS` cannot create Hessian matrices, the user needs to pass the
  Hessian matrices used in algorithms in `TaoSetHessian()`.

  Developer Note:
  Internally each `Tao` has a `TaoTerm` of type `TAOTERMTAOCALLBACKS` that is updated
  by the `Tao` callback routines (`TaoSetObective()`, `TaoSetGradient()`,
  `TaoSetObjectiveAndGradient()`, and `TaoSetHessian()`).

  The routines that get the user-defined `Tao` callback functions
  (`TaoGetObjective()`, `TaoGetObjectiveAndGradient()`, `TaoGetGradient()`,
  `TaoGetHessian()`) will always return those original callbacks, even if the
  objective function has been changed by `TaoSetObjectiveTerm()` or `TaoAddObjectiveTerm()`,
  so PETSc/TAO should not assume that those callbacks are valid in any library code.

  A `TAOTERMTAOCALLBACKS` has a weak-reference to the `Tao` that created it,
  which may not be the `Tao` currently using it because the term could have been shared
  using `TaoGetObjectiveTerm()` and `TaoSetObjectiveTerm()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TAOTERMBRGNREGULARIZER`,
          `TAOTERMADMMREGULARIZER`,
          `TAOTERMADMMMISFIT`,
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_TaoCallbacks(TaoTerm term)
{
  PetscFunctionBegin;
  // clang-format off
  PetscCall(TaoTermCreate_TaoCallbacks_Internal(term,
        "TaoComputeObjective()",            "TaoSetObjective()",
        "TaoComputeGradient()",             "TaoSetComputeGradient()",
        "TaoComputeObjectiveAndGradient()", "TaoSetObjectiveAndGradient()",
        "TaoComputeHessian()",              "TaoSetHessian()"));
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateTaoCallbacks(Tao tao, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), term));
  PetscCall(TaoTermSetType(*term, TAOTERMTAOCALLBACKS));
  {
    TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)((*term)->data);

    tt->tao = tao; // weak reference, do not increment reference count
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMBRGNREGULARIZER - A `TaoTerm` implementation that accesses the callbacks that have been set in
  `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()` and
  `TaoBRGNSetRegularizerHessianRoutine()`.

  Level: developer

  Note:
  This implementation has the same restrictions os `TAOTERMTAOCALLBACKS`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TAOBRGN`,
          `TAOTERMTAOCALLBACKS`,
          `TAOTERMADMMREGULARIZER`,
          `TAOTERMADMMMISFIT`,
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_BRGNRegularizer(TaoTerm term)
{
  PetscFunctionBegin;
  // clang-format off
  PetscCall(TaoTermCreate_TaoCallbacks_Internal(term,
        NULL, "TaoBRGNSetRegularizerObjectiveAndGradientRoutine()",
        NULL, "TaoBRGNSetRegularizerObjectiveAndGradientRoutine()",
        "BRGN regularizer objective/gradient", "TaoBRGNSetRegularizerObjectiveAndGradientRoutine()",
        "BRGN regularizer hessian",            "TaoBRGNSetRegularizerHessianRoutine()"));
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateBRGNRegularizer(Tao tao, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), term));
  PetscCall(TaoTermSetType(*term, TAOTERMBRGNREGULARIZER));
  {
    TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)((*term)->data);

    tt->tao = tao; // weak reference, do not increment reference count
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMADMMREGULARIZER - A `TaoTerm` implementation that accesses the callbacks that have been set in
  `TaoADMMSetRegularizerObjectiveAndGradientRoutine()` and
  `TaoADMMSetRegularizerHessianRoutine()`.

  Level: developer

  Note:
  This implementation has the same restrictions os `TAOTERMTAOCALLBACKS`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TAOADMM`,
          `TAOTERMTAOCALLBACKS`
          `TAOTERMBRGNREGULARIZER`,
          `TAOTERMADMMMISFIT`,
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_ADMMRegularizer(TaoTerm term)
{
  PetscFunctionBegin;
  // clang-format off
  PetscCall(TaoTermCreate_TaoCallbacks_Internal(term,
        NULL, "TaoADMMSetRegularizerObjectiveAndGradientRoutine()",
        NULL, "TaoADMMSetRegularizerObjectiveAndGradientRoutine()",
        "ADMM regularizer objective/gradient", "TaoADMMSetRegularizerObjectiveAndGradientRoutine()",
        "ADMM regularizer hessian",            "TaoADMMSetRegularizerHessianRoutine()"));
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateADMMRegularizer(Tao tao, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), term));
  PetscCall(TaoTermSetType(*term, TAOTERMADMMREGULARIZER));
  {
    TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)((*term)->data);

    tt->tao = tao; // weak reference, do not increment reference count
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMADMMMISFIT - A `TaoTerm` implementation that accesses the callbacks that have been set in
  `TaoADMMSetMisfitObjectiveAndGradientRoutine()` and
  `TaoADMMSetMisfitHessianRoutine()`.

  Level: developer

  Note:
  This implementation has the same restrictions os `TAOTERMTAOCALLBACKS`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TAOADMM`,
          `TAOTERMTAOCALLBACKS`
          `TAOTERMBRGNREGULARIZER`,
          `TAOTERMADMMREGULARIZER`,
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_ADMMMisfit(TaoTerm term)
{
  PetscFunctionBegin;
  // clang-format off
  PetscCall(TaoTermCreate_TaoCallbacks_Internal(term,
        NULL, "TaoADMMSetMisfitObjectiveAndGradientRoutine()",
        NULL, "TaoADMMSetMisfitObjectiveAndGradientRoutine()",
        "ADMM misfit objective/gradient", "TaoADMMSetMisfitObjectiveAndGradientRoutine()",
        "ADMM misfit hessian",            "TaoADMMSetMisfitHessianRoutine()"));
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateADMMMisfit(Tao tao, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), term));
  PetscCall(TaoTermSetType(*term, TAOTERMADMMMISFIT));
  {
    TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)((*term)->data);

    tt->tao = tao; // weak reference, do not increment reference count
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
