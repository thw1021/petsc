#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_TaoCallbacks TaoTerm_TaoCallbacks;

struct _n_TaoTerm_TaoCallbacks {
  Tao tao;
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
  PetscCall(PetscFree(tt));
  term->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_TaoCallbacks(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
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

static PetscErrorCode TaoTermGradient_TaoCallbacks(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  if (tt->tao->ops->computegradient) PetscUseTypeMethod(tt->tao, computegradient, x, g, tt->tao->user_gradP);
  else if (tt->tao->ops->computeobjectiveandgradient) {
    PetscReal dummy;

    PetscUseTypeMethod(tt->tao, computeobjectiveandgradient, x, &dummy, g, tt->tao->user_objgradP);
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoSetGradient() has not been called");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_TaoCallbacks(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  if (tt->tao->ops->computeobjectiveandgradient) {
    PetscUseTypeMethod(tt->tao, computeobjectiveandgradient, x, value, g, tt->tao->user_objgradP);
  } else if (tt->tao->ops->computeobjective && tt->tao->ops->computegradient) {
    PetscUseTypeMethod(tt->tao, computeobjective, x, value, tt->tao->user_objP);
    PetscUseTypeMethod(tt->tao, computegradient, x, g, tt->tao->user_gradP);
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoSetObjective() or TaoSetGradient() not set");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_TaoCallbacks(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)term->data;

  PetscFunctionBegin;
  PetscCheckTaoTermCallbacksValid(term, tt, params);
  PetscUseTypeMethod(tt->tao, computehessian, x, H, Hpre, tt->tao->user_hessP);
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
  TAOTERMTAOCALLBACKS - A `TaoTerm` implementation that accesses the callbacks that have been set to an outer `Tao`, e.g. `TaoSetObjective()`.

  Level: developer

  Notes:
  If you are interested in creating you own term, you shouldn't use this. Use `TAOTERMSHELL` or create your own
  implementation of `TaoTerm` with `TaoTermRegister()`.

  The `params` argument of `TaoTerm` functions will always be `NULL` for a `TAOTERMTAOCALLBACKS`.

  Developer Note: a `TAOTERMTAOCALLBACKS` cannot be shared between `Tao`s.

.seealso: [](ch_tao), `Tao`, `TaoTerm`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_TaoCallbacks(TaoTerm term)
{
  TaoTerm_TaoCallbacks *tt;

  PetscFunctionBegin;
  PetscCall(PetscNew(&tt));
  term->data                      = (void *)tt;
  term->ops->destroy              = TaoTermDestroy_TaoCallbacks;
  term->ops->objective            = TaoTermObjective_TaoCallbacks;
  term->ops->gradient             = TaoTermGradient_TaoCallbacks;
  term->ops->objectiveandgradient = TaoTermObjectiveAndGradient_TaoCallbacks;
  term->ops->hessian              = TaoTermHessian_TaoCallbacks;
  term->ops->view                 = TaoTermView_TaoCallbacks;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateTaoCallbacks(Tao tao, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), term));
  PetscCall(TaoTermSetType(*term, TAOTERMTAOCALLBACKS));
  {
    TaoTerm_TaoCallbacks *tt = (TaoTerm_TaoCallbacks *)((*term)->data);
    tt->tao                  = tao; // weak reference, do not increment reference count
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
