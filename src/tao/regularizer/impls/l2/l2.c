#include <petsc/private/taoregularizerimpl.h>

static PetscErrorCode TaoRegularizerDestroy_L2(TaoRegularizer reg)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(reg->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerSetFromOptions_L2(TaoRegularizer reg, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "L2 Regularizer options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerComputeObjective_L2(TaoRegularizer reg, Vec X, PetscReal *f, void *ctx)
{
  Vec       y;
  PetscReal scale;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetCentralVector(reg, &y));
  PetscCall(TaoRegularizerGetScale(reg, &scale));
  if (y) {
    PetscCall(VecWAXPY(reg->workvec, -1., y, X));
    PetscCall(VecNorm(reg->workvec, NORM_2, f));
    *f *= scale;
  } else {
    PetscCall(VecNorm(X, NORM_2, f));
    *f *= scale;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerComputeGradient_L2(TaoRegularizer reg, Vec X, Vec G, void *ctx)
{
  Vec       y;
  PetscReal scale;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetCentralVector(reg, &y));
  PetscCall(TaoRegularizerGetScale(reg, &scale));
  if (y) {
    PetscCall(VecAXPBYPCZ(G, 2 * scale, -2 * scale, 0, X, y));
  } else {
    PetscCall(VecAXPBY(G, 2 * scale, 0., X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerComputeObjectiveAndGradient_L2(TaoRegularizer reg, Vec X, PetscReal *f, Vec G, void *ctx)
{
  Vec       y;
  PetscReal temp, scale;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetCentralVector(reg, &y));
  PetscCall(TaoRegularizerGetScale(reg, &scale));
  if (y) {
    PetscCall(VecAXPBYPCZ(G, 2 * scale, -2 * scale, 0, X, y));
    PetscCall(VecTDot(G, G, &temp));
    temp *= scale;
    *f = temp;
  } else {
    PetscCall(VecTDot(X, X, &temp));
    temp *= scale;
    *f = temp;
    PetscCall(VecAXPBY(G, 2 * scale, 0., X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO Need to implement Hessian Routine */

PETSC_EXTERN PetscErrorCode TaoRegularizerCreate_L2(TaoRegularizer reg)
{
  PetscFunctionBegin;

  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);

  reg->ops->destroy        = TaoRegularizerDestroy_L2;
  reg->ops->setfromoptions = TaoRegularizerSetFromOptions_L2;

  reg->ops->computeobjective            = TaoRegularizerComputeObjective_L2;
  reg->ops->computegradient             = TaoRegularizerComputeGradient_L2;
  reg->ops->computeobjectiveandgradient = TaoRegularizerComputeObjectiveAndGradient_L2;

  PetscFunctionReturn(PETSC_SUCCESS);
}
