#include <../src/tao/unconstrained/impls/prox/prox.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>

PetscErrorCode TaoProxSetSoftThresholdContext(Tao tao, PetscReal lb, PetscReal ub)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->type   = TAOPROX_L1;
  proxP->L1->lb = lb;
  proxP->L1->ub = ub;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoProxSetAffineContext(Tao tao, Vec a)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->type      = TAOPROX_L1;
  proxP->affine->a = a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoApplyProximalMap_Affine(Tao tao, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  TaoRegularizerType reg_type;
  TaoRegularizer     reg;
  PetscBool          is_prox, is_l2;

  PetscFunctionBegin;
  PetscCall(TaoGetRegularizer(tao, &reg));
  PetscCall(TaoRegularizerGetType(reg, &reg_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
  PetscCall(PetscStrcmp(reg_type, TAOREGULARIZERL2, &is_l2));
  /* TODO diag reg */

  if (is_l2) {
    TAO_PROX *proxP = (TAO_PROX *)tao->data;
    Vec       a;
    a = proxP->affine->a;
    PetscCall(VecWAXPY(x, -1., a, y));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_Affine only supports L2, and Diag type regularizers. ");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoApplyProximalMap_L1(Tao tao, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  TaoRegularizerType reg_type;
  TaoRegularizer     reg;
  PetscBool          is_prox, is_l2;

  PetscFunctionBegin;
  PetscCall(TaoGetRegularizer(tao, &reg));
  PetscCall(TaoRegularizerGetType(reg, &reg_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
  PetscCall(PetscStrcmp(reg_type, TAOREGULARIZERL2, &is_l2));
  /* TODO diag reg */
  if (is_l2) {
    TAO_PROX *proxP = (TAO_PROX *)tao->data;
    if (&(proxP->L1->lb) == NULL || &(proxP->L1->ub) == NULL) {
      PetscCall(TaoSoftThreshold(y, -lambda, lambda, x));
    } else {
      PetscCall(TaoSoftThreshold(y, proxP->L1->lb, proxP->L1->ub, x));
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_L1 only supports L2 and diag Regularizer types.");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoApplyProximalMap_Simplex(Tao tao, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  TaoRegularizerType reg_type;
  TaoRegularizer     reg;
  PetscBool          is_prox, is_l2;

  PetscFunctionBegin;
  PetscCall(TaoGetRegularizer(tao, &reg));
  PetscCall(TaoRegularizerGetType(reg, &reg_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
  PetscCall(PetscStrcmp(reg_type, TAOREGULARIZERL2, &is_l2));
  /* TODO diag reg */
  /*TODO implement simplex proj */
  PetscFunctionReturn(PETSC_SUCCESS);
}
