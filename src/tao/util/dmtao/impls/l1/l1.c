#include <petsc/private/taoimpl.h>
#include <petsc/private/dmimpl.h>
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
  PetscCall(PetscOptionsReal("-dmtao_l1_lb", "DMTao SoftThreshold lower bound.", "", ctx->lb, &ctx->lb, NULL));
  PetscCall(PetscOptionsReal("-dmtao_l1_ub", "DMTao SoftThreshold upper bound.", "", ctx->ub, &ctx->ub, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTaoL1SetContext(DM dm, PetscReal lb, PetscReal ub)
{
  DMTao     tdm;
  PetscBool is_l2;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, DMTAOL2, &is_l2));

  DMTao_L1 *ctx = (DMTao_L1 *)tdm->data;

  ctx->lb = lb;
  ctx->ub = ub;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoApplyProximalMap_L1(DMTao dm0, DMTao dm1, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  PetscBool is_0_l1, is_1_l2;
  DMTao_L1 *ctx = (DMTao_L1 *)dm0->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)dm0, DMTAOL1, &is_0_l1));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm1, DMTAOL2, &is_1_l2));
  if (is_1_l2) {
    //Check whats lb ub if unset? TODO
    //TODO VM L2 reg
    if (ctx->lb == 0 && ctx->ub == 0) {
      PetscCall(TaoSoftThreshold(y, -lambda, lambda, x));
    } else {
      PetscCall(TaoSoftThreshold(y, ctx->lb, ctx->ub, x));
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)dm0), PETSC_ERR_USER, "DMTaoApplyProximalMap_L1 only supports L2 Regularizer types.");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Internal Register-Dispatch Version */
PETSC_EXTERN PetscErrorCode DMTaoCreate_L1_Private(DMTao dm)
{
  DMTao_L1 *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  /* Default is 0,0 - i.e., no-op */
  ctx->lb                   = 0.;
  ctx->ub                   = 0.;
  dm->data                  = (void *)ctx;
  dm->ops->applyproximalmap = DMTaoApplyProximalMap_L1;
  dm->ops->destroy          = DMTaoContextDestroy_L1;
  dm->ops->setfromoptions   = DMTaoSetFromOptions_L1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* External Version for User to create L1 in lump-sum fashion TODO add manual `*/
PetscErrorCode DMTaoCreate_L1(MPI_Comm comm, DM *dm, Mat vm, Vec y, PetscReal lb, PetscReal ub)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMGetDMTao(*dm, &tdm));
  PetscCall(DMTaoSetType(*dm, DMTAOL1));
  PetscCall(DMTaoSetCentralVector(*dm, y));
  PetscCall(DMTaoSetVM(*dm, vm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
