#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Tao TaoTerm_Tao;

struct _n_TaoTerm_Tao {
  Tao tao;
};

static PetscErrorCode TaoTermDestroy_Tao(TaoTerm term)
{
  TaoTerm_Tao *tt = (TaoTerm_Tao *)term->data;

  PetscFunctionBegin;
  // tt->tao is a weak reference, we do not destroy it
  PetscCall(PetscFree(tt));
  term->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_Tao(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_Tao *tt = (TaoTerm_Tao *)term->data;

  PetscFunctionBegin;
  PetscCheck(params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "TAOTERMTAO does not accept a vector of parameters");
  PetscCheck(tt->tao != NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TAOTERMTAO does not have an outer Tao");
  if (tt->tao->ops->computeobjective) {
    PetscUseTypeMethod(tt->tao, computeobjective, x, value, tt->tao->user_objP);
  } else if (tt->tao->ops->computeobjectiveandgradient) {
    Vec dummy;

    PetscCall(PetscInfo(tt->tao, "Duplicating variable vector in order to call func/grad routine\n"));
    PetscCall(VecDuplicate(x, &dummy));
    PetscUseTypeMethod(tt->tao, computeobjectiveandgradient, x, value, dummy, tt->tao->user_objgradP);
    PetscCall(VecDestroy(&dummy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_Tao(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_Tao *tt = (TaoTerm_Tao *)term->data;

  PetscFunctionBegin;
  PetscCheck(params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "TAOTERMTAO does not accept a vector of parameters");
  PetscCheck(tt->tao != NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TAOTERMTAO does not have an outer Tao");
  if (tt->tao->ops->computegradient) PetscUseTypeMethod(tt->tao, computegradient, x, g, tt->tao->user_gradP);
  else if (tt->tao->ops->computeobjectiveandgradient) {
    PetscReal dummy;

    PetscUseTypeMethod(tt->tao, computeobjectiveandgradient, x, &dummy, g, tt->tao->user_objgradP);
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoSetGradient() has not been called");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_Tao(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_Tao *tt = (TaoTerm_Tao *)term->data;

  PetscFunctionBegin;
  PetscCheck(params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "TAOTERMTAO does not accept a vector of parameters");
  PetscCheck(tt->tao != NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TAOTERMTAO does not have an outer Tao");
  if (tt->tao->ops->computeobjectiveandgradient) {
    PetscUseTypeMethod(tt->tao, computeobjectiveandgradient, x, value, g, tt->tao->user_objgradP);
  } else if (tt->tao->ops->computeobjective && tt->tao->ops->computegradient) {
    PetscUseTypeMethod(tt->tao, computeobjective, x, value, tt->tao->user_objP);
    PetscUseTypeMethod(tt->tao, computegradient, x, g, tt->tao->user_gradP);
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoSetObjective() or TaoSetGradient() not set");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_Tao(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_Tao *tt = (TaoTerm_Tao *)term->data;

  PetscFunctionBegin;
  PetscCheck(params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "TAOTERMTAO does not accept a vector of parameters");
  PetscCheck(tt->tao != NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TAOTERMTAO does not have an outer Tao");
  PetscUseTypeMethod(tt->tao, computehessian, x, H, Hpre, tt->tao->user_hessP);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Tao(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Tao *tt = (TaoTerm_Tao *)term->data;
  PetscBool    iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    if (!tt->tao) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "TaoTerm is not attached to a Tao\n"));
    } else {
      const char *name;

      PetscCall(PetscObjectGetName((PetscObject)tt->tao, &name));
      PetscCall(PetscViewerASCIIPrintf(viewer, "TaoTerm is attached to Tao %s\n", name));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMTAO - A `TaoTerm` implementation that accesses the callbacks that have been set to an outer `Tao`, e.g. `TaoSetObjective()`.

  Level: developer

  Notes:
  If you are interested in creating you own term, you shouldn't use this. Use `TAOTERMSHELL` or create your own
  implementation of `TaoTerm` with `TaoTermRegister()`.

  The `params` argument of `TaoTerm` functions will always be `NULL` for a `TAOTERMTAO`.

  Developer Note: a `TAOTERMTAO` cannot be shared between `Tao`s.

.seealso: [](ch_tao), `Tao`, `TaoTerm`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Tao(TaoTerm term)
{
  TaoTerm_Tao *tt;

  PetscFunctionBegin;
  PetscCall(PetscNew(&tt));
  term->data                      = (void *)tt;
  term->ops->destroy              = TaoTermDestroy_Tao;
  term->ops->objective            = TaoTermObjective_Tao;
  term->ops->gradient             = TaoTermGradient_Tao;
  term->ops->objectiveandgradient = TaoTermObjectiveAndGradient_Tao;
  term->ops->hessian              = TaoTermHessian_Tao;
  term->ops->view                 = TaoTermView_Tao;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateTao(Tao tao, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), term));
  PetscCall(TaoTermSetType(*term, TAOTERMTAO));
  {
    TaoTerm_Tao *tt = (TaoTerm_Tao *)((*term)->data);
    tt->tao         = tao; // weak reference, do not increment reference count
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
