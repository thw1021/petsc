#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/

static PetscErrorCode DMTaoContextDestroy_Zero(DMTao dm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoSetFromOptions_Zero(DMTao dm, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMTao Zero options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoView_Zero(DMTao tdm, PetscViewer pv)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) { PetscCall(PetscViewerASCIIPrintf(pv, "  DMTao Zero \n")); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoApplyProximalMap_Zero(DMTao tdm0, DMTao tdm1, PetscReal lambda, Vec y, Vec x, PetscBool flg)
{
  PetscFunctionBegin;
  PetscCall(VecSet(x, 0.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   DMTAOZERO - Zero DMTao object.
   Indicator function of the set containing the origin.
   Its proximal mapping returns zero vector.

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode DMTaoCreate_Zero_Private(DMTao dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  dm->ops->applyproximalmap = DMTaoApplyProximalMap_Zero;
  dm->data                  = NULL;
  dm->ops->setup            = NULL;
  dm->ops->destroy          = DMTaoContextDestroy_Zero;
  dm->ops->view             = DMTaoView_Zero;
  dm->ops->setfromoptions   = DMTaoSetFromOptions_Zero;
  dm->ops->reset            = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
