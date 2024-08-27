#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/

/* L2 Norm |x|_2^2 DMTao */
static PetscErrorCode DMTaoDestroy_L2(DMTao dm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoView_L2(DMTao dm, PetscViewer pv)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) { PetscCall(PetscViewerASCIIPrintf(pv, "  DMTao L2\n")); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DMTaoCompute... routines are only if they are Regularizer */
static PetscErrorCode DMTaoComputeObjective_L2(DM dm, Vec X, PetscReal *f, void *ctx)
{
  Vec   y;
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(DMTaoGetCentralVector(dm, &y));

  if (y) {
    PetscCall(DMTaoSetWorkVec(dm, X, 1));
    PetscCall(VecWAXPY(tdm->workvec, -1., y, X));
    PetscCall(VecNorm(tdm->workvec, NORM_2, f));
  } else PetscCall(VecNorm(X, NORM_2, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoComputeGradient_L2(DM dm, Vec X, Vec G, void *ctx)
{
  Vec   y;
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(DMTaoGetCentralVector(dm, &y));

  if (y) PetscCall(VecAXPBYPCZ(G, 2, -2, 0, X, y));
  else PetscCall(VecAXPBY(G, 2, 0., X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoComputeObjectiveAndGradient_L2(DM dm, Vec X, PetscReal *f, Vec G, void *ctx)
{
  Vec   y;
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(DMTaoGetCentralVector(dm, &y));

  if (y) {
    PetscCall(VecAXPBYPCZ(G, 1, -1, 0, X, y));
    PetscCall(VecTDot(G, G, f));
    PetscCall(VecScale(G, 2.));
  } else {
    PetscCall(VecTDot(X, X, f));
    PetscCall(VecAXPBY(G, 2, 0., X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoApplyProximalMap_L2(DMTao tdm0, DMTao tdm1, PetscReal lambda, Vec y, Vec x, PetscBool flg)
{
  PetscBool is_0_l2;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm0, DMTAOL2, &is_0_l2));
  PetscAssert(is_0_l2, PetscObjectComm((PetscObject)tdm0), PETSC_ERR_USER, "L2 Square does not have proximal map.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO Hessian Routine */

/*MC
     DMTAOL2 - L2 norm DMTao. This DMTao object represents |x-y|_2^2.

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode DMTaoCreate_L2_Private(DMTao dm)
{
  PetscFunctionBegin;
  PetscAssertPointer(dm, 1);
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);

  dm->data = NULL;

  dm->ops->applyproximalmap            = DMTaoApplyProximalMap_L2;
  dm->ops->setup                       = NULL;
  dm->ops->destroy                     = DMTaoDestroy_L2;
  dm->ops->view                        = DMTaoView_L2;
  dm->ops->setfromoptions              = NULL;
  dm->ops->reset                       = NULL;
  dm->ops->computeobjective            = DMTaoComputeObjective_L2;
  dm->ops->computegradient             = DMTaoComputeGradient_L2;
  dm->ops->computeobjectiveandgradient = DMTaoComputeObjectiveAndGradient_L2;
  PetscFunctionReturn(PETSC_SUCCESS);
}
