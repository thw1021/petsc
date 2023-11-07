#include <../src/tao/unconstrained/impls/prox/prox.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>

static PetscErrorCode TaoProxContextDestroy_L1(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(proxP->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxCreate_L1(Tao tao)
{
  TAO_PROX    *proxP = (TAO_PROX *)tao->data;
  TAO_PROX_L1 *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  /* Default is 0,0 - i.e., no-op */
  ctx->lb                    = 0;
  ctx->ub                    = 0;
  proxP->data                = (void *)ctx;
  tao->ops->applyproximalmap = TaoApplyProximalMap_L1;
  proxP->ops->destroy        = TaoProxContextDestroy_L1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoProxL1SetContext(Tao tao, PetscReal lb, PetscReal ub)
{
  TAO_PROX    *proxP = (TAO_PROX *)tao->data;
  TAO_PROX_L1 *ctx   = (TAO_PROX_L1 *)(proxP->data);
  PetscBool    is_l1;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(proxP->type, TAOPROX_L1, &is_l1));
  if (is_l1) PetscFunctionReturn(PETSC_SUCCESS);
  ctx->lb = lb;
  ctx->ub = ub;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoApplyProximalMap_L1(Tao tao, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  TAO_PROX          *proxP = (TAO_PROX *)tao->data;
  TAO_PROX_L1       *ctx   = (TAO_PROX_L1 *)(proxP->data);
  TaoRegularizerType reg_type;
  TaoRegularizer     reg;
  PetscBool          is_prox, is_l2;

  PetscFunctionBegin;
  PetscCall(TaoGetRegularizer(tao, &reg));
  PetscCall(TaoRegularizerGetType(reg, &reg_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
  PetscCall(PetscStrcmp(reg_type, TAOREGULARIZERL2, &is_l2));
  PetscCheck(is_prox, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoType is not TaoProx.");
  if (is_l2) {
    //Check whats lb ub if unset? TODO
    if (ctx->lb == 0 && ctx->ub == 0) {
      PetscCall(TaoSoftThreshold(y, -lambda, lambda, x));
    } else {
      PetscCall(TaoSoftThreshold(y, ctx->lb, ctx->ub, x));
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_L1 only supports L2 Regularizer types.");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
