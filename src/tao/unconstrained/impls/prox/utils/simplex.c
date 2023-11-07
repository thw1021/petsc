#include <../src/tao/unconstrained/impls/prox/prox.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>

static PetscErrorCode TaoProxContextDestroy_Simplex(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(proxP->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoProxSetFromOptions_Simplex(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_PROX         *proxP = (TAO_PROX *)tao->data;
  TAO_PROX_SIMPLEX *ctx   = (TAO_PROX_SIMPLEX *)(proxP->data);

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoProx Simplex options");
  PetscCall(PetscOptionsReal("-tao_prox_simplex_tol", "simplex tolerance", "", ctx->tol, &ctx->tol, NULL));
  PetscCall(PetscOptionsReal("-tao_prox_simplex_size", "size of simplex", "", ctx->size, &ctx->size, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoProxSimplexSetContext(Tao tao, PetscReal size)
{
  TAO_PROX         *proxP = (TAO_PROX *)tao->data;
  TAO_PROX_SIMPLEX *ctx   = (TAO_PROX_SIMPLEX *)(proxP->data);
  PetscBool         is_s;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(proxP->type, TAOPROX_SIMPLEX, &is_s));
  if (is_s) PetscFunctionReturn(PETSC_SUCCESS);
  ctx->size = size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscReal Clip_Internal(PetscReal in, PetscReal tmax)
{
  return PetscMax(0, in - tmax);
}

/* https://arxiv.org/pdf/1101.6081.pdf */
PetscErrorCode TaoApplyProximalMap_Simplex(Tao tao, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  TAO_PROX          *proxP = (TAO_PROX *)tao->data;
  TAO_PROX_SIMPLEX  *ctx   = (TAO_PROX_SIMPLEX *)(proxP->data);
  TaoRegularizerType reg_type;
  TaoRegularizer     reg;
  PetscBool          is_prox, is_l2, bget = PETSC_FALSE;
  PetscReal         *xarray, *yarray, tmax, min, sum, size, cumsum = 0;
  PetscInt           len, i;

  PetscFunctionBegin;
  PetscCall(TaoGetRegularizer(tao, &reg));
  PetscCall(TaoRegularizerGetType(reg, &reg_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPROX, &is_prox));
  PetscCall(PetscStrcmp(reg_type, TAOREGULARIZERL2, &is_l2));
  PetscCheck(is_prox, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoType is not TaoProx.");
  PetscCheck(is_l2, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoApplyProximalMap_Simplex only upports L2 regularizer type..");

  size = ctx->size;
  PetscCall(VecSum(y, &sum));
  PetscCall(VecMin(y, NULL, &min));
  if (PetscAbsReal(sum - size) < ctx->tol) { PetscFunctionReturn(PETSC_SUCCESS); }

  PetscCall(VecGetSize(y, &len));
  PetscCall(VecGetArray(y, &yarray));
  PetscCall(PetscSortReal(len, yarray));

  /* Not thinking about parallelism right now... */
  /* Scalar Version. Technically one can du cumulative sum,
   * but VecCumSum/PrefixSum isn't there yet. */
  for (i = len - 1; i >= 0; i--) {
    cumsum += yarray[i];
    tmax = (cumsum - size) / (len - i);
    if (tmax >= yarray[i - 1]) {
      bget = PETSC_TRUE;
      break;
    }
  }
  if (!bget) { tmax = (cumsum - size) / len; }

  /* Ideally, VecShift(y, -tmax), and set all neg to 0 in two kernel calls... */
  PetscCall(VecGetArray(x, &xarray));
  for (i = 0; i < len; i++) { xarray[i] = Clip_Internal(yarray[i], tmax); }
  PetscCall(VecRestoreArray(y, &yarray));
  PetscCall(VecRestoreArray(x, &xarray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxCreate_Simplex(Tao tao)
{
  TAO_PROX         *proxP = (TAO_PROX *)tao->data;
  TAO_PROX_SIMPLEX *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  /* Default is unit simplex */
  ctx->size = 1;
#if defined(PETSC_USE_REAL_SINGLE)
  ctx->tol = 1.e-5;
#else
  ctx->tol = 1.e-5;
#endif
  proxP->data                = (void *)ctx;
  tao->ops->applyproximalmap = TaoApplyProximalMap_Simplex;
  proxP->ops->destroy        = TaoProxContextDestroy_Simplex;
  proxP->ops->setfromoptions = TaoProxSetFromOptions_Simplex;
  PetscFunctionReturn(PETSC_SUCCESS);
}
