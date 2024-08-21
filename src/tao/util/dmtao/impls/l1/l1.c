#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/
#include <../src/tao/util/dmtao/impls/l1/l1.h>

static PetscErrorCode DMTaoContextDestroy_L1(DMTao dm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoSetFromOptions_L1(DMTao dm, PetscOptionItems *PetscOptionsObject)
{
  DMTao_L1 *ctx = (DMTao_L1 *)dm->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMTao L1options");
  PetscCall(PetscOptionsReal("-dmtao_l1_scale", "DMTao SoftThreshold scale.", "", ctx->lam, &ctx->lam, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoView_L1(DMTao dm, PetscViewer pv)
{
  DMTao_L1 *ctx = (DMTao_L1 *)dm->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  DMTao L1"));
    PetscCall(PetscViewerASCIIPrintf(pv, ": scale=%g \n", (double)ctx->lam));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoComputeObjective_L1(DM dm, Vec X, PetscReal *f, void *ctx)
{
  Vec   y;
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(DMTaoGetCentralVector(dm, &y));

  if (y) {
    PetscCall(DMTaoSetWorkVec(dm, X, 1));
    PetscCall(VecWAXPY(tdm->workvec, -1., y, X));
    PetscCall(VecNorm(tdm->workvec, NORM_1, f));
  } else PetscCall(VecNorm(X, NORM_1, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoL1SetContext - sets lambda scaling constant for `DMTAOL1`.
  It should be noted that this lambda scale is \lambda * |x|_1, NOT
  |\lambda x|_1. For the latter, see `DMTaoSetScaling()`.

  Logically Collective

  Input Parameters:
+ dm  - the `DM` containing `DMTAOL1`
- lam - lambda scale

  Level: advanced

.seealso: `DMTao`, `DMTAOL1`
@*/
PetscErrorCode DMTaoL1SetContext(DM dm, PetscReal lam)
{
  DMTao     tdm;
  PetscBool is_l1;

  PetscFunctionBegin;
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, DMTAOL1, &is_l1));
  PetscCheck(is_l1, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "DMTaoL1SetContext requires DMTaoType to be of DMTAOL1");

  DMTao_L1 *ctx = (DMTao_L1 *)tdm->data;

  ctx->lam = lam;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* @ DMTaoApplyProximalMap_L1 - This routine performs a soft threshold operation.

   Input Parameters:
+  dm0  - L1 DMTao context
.  dm1  - Regularizer DMTao context
.  step - scale for the regularizer
.  y    - input vector
-  flg  - denotes conjugate. If true, computes prox of conjugate

   Output parameters:
.  x - output vector

@ */
static PetscErrorCode DMTaoApplyProximalMap_L1(DMTao tdm0, DMTao tdm1, PetscReal step, Vec y, Vec x, PetscBool flg)
{
  PetscBool is_0_l1, is_1_l2 = PETSC_TRUE;
  PetscReal scale;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm0, DMTAOL1, &is_0_l1));
  PetscCheck(is_0_l1, PetscObjectComm((PetscObject)tdm0), PETSC_ERR_USER, "DMTaoApplyProximalMap_L1 requires first DMTao to be of L1 type");
  if (tdm1) {
    PetscCall(PetscObjectTypeCompare((PetscObject)tdm1, DMTAOL2, &is_1_l2));
    PetscCheck(is_1_l2, PetscObjectComm((PetscObject)tdm1), PETSC_ERR_USER, "DMTaoApplyProximalMap_L1 requires second DMTao to be of L2 type");
  }
  /* dm1 == NULL assumes L2 type */

  DMTao_L1 *ctx = (DMTao_L1 *)tdm0->data;

  scale = (ctx->lam == 0) ? step : ctx->lam * step;
  PetscCall(TaoSoftThreshold(y, -scale, scale, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   DMTAOL1 - L1 Norm DMTao object. This represents |x-y|_1, and has
   soft-thresholding for a proximal map.

  Level: beginner

.seealso: `DMTao`
M*/
PETSC_EXTERN PetscErrorCode DMTaoCreate_L1_Private(DMTao dm)
{
  DMTao_L1 *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));

  ctx->lam                  = 1.;
  dm->ops->applyproximalmap = DMTaoApplyProximalMap_L1;
  dm->ops->computeobjective = DMTaoComputeObjective_L1;
  dm->data                  = (void *)ctx;
  dm->ops->setup            = NULL;
  dm->ops->destroy          = DMTaoContextDestroy_L1;
  dm->ops->view             = DMTaoView_L1;
  dm->ops->setfromoptions   = DMTaoSetFromOptions_L1;
  dm->ops->reset            = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
