#include <petsc/private/taoregularizerimpl.h>

static PetscErrorCode TaoRegularizerDestroy_L1(TaoRegularizer reg)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(reg->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerSetFromOptions_L1(TaoRegularizer reg, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "L1 Regularizer options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerView_L1(TaoRegularizer reg, PetscViewer pv)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(pv, "  L1 regularizer"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerComputeObjectiveAndGradient_L1(TaoRegularizer reg, Vec X, PetscReal *f, Vec G, void *ctx)
{
  Vec       y;
  PetscReal temp;

  PetscFunctionBegin;
  y = reg->y;

  /*TODO waxpy with null vec ?*/
  PetscCall(VecWAXPY(G, -1., y, X));
  PetscCall(VecNorm(G, NORM_1, &temp));
  temp = PetscPowReal(temp, 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoRegularizerCreate_L1(TaoRegularizer reg)
{
  PetscFunctionBegin;

  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);

  reg->ops->view           = TaoRegularizerView_L1;
  reg->ops->destroy        = TaoRegularizerDestroy_L1;
  reg->ops->setfromoptions = TaoRegularizerSetFromOptions_L1;

  reg->ops->computeobjectiveandgradient = TaoRegularizerComputeObjectiveAndGradient_L1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
