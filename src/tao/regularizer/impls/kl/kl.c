#include <petsctao.h>
#include <petsc/private/taoregularizerimpl.h>

static PetscErrorCode TaoRegularizerDestroy_KL(TaoRegularizer reg)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(reg->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerSetFromOptions_KL(TaoRegularizer reg, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KL Regularizer options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerView_KL(TaoRegularizer reg, PetscViewer pv)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) { PetscCall(PetscViewerASCIIPrintf(pv, "  KL regularizer")); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective: \sum_i x_i \log(x_i/y_i) */
static PetscErrorCode TaoRegularizerComputeObjective_KL(TaoRegularizer reg, Vec X, PetscReal *f, void *ctx)
{
  Vec       y;
  PetscReal temp;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetCentralVector(reg, &y));
  PetscCheck(y, PetscObjectComm((PetscObject)reg), PETSC_ERR_ARG_WRONG, "TaoRegularizer Central Vector not set! Central Vector is needed for KL divergence.");
  PetscCall(VecPointwiseDivide(reg->workvec, X, y));
  PetscCall(VecLog(reg->workvec));
  PetscCall(VecPointwiseMult(reg->workvec, reg->workvec, X));
  PetscCall(VecSum(reg->workvec, &temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Gradient: \log(x_i/y_i) + 1 */
static PetscErrorCode TaoRegularizerComputeGradient_KL(TaoRegularizer reg, Vec X, Vec G, void *ctx)
{
  Vec y;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetCentralVector(reg, &y));
  PetscCheck(y, PetscObjectComm((PetscObject)reg), PETSC_ERR_ARG_WRONG, "TaoRegularizer Central Vector not set! Central Vector is needed for KL divergence.");
  PetscCall(VecPointwiseDivide(G, X, y));
  PetscCall(VecLog(G));
  PetscCall(VecPointwiseMult(reg->workvec, G, X));
  PetscCall(VecShift(G, 1.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerComputeObjectiveAndGradient_KL(TaoRegularizer reg, Vec X, PetscReal *f, Vec G, void *ctx)
{
  Vec       y;
  PetscReal temp;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetCentralVector(reg, &y));
  PetscCheck(y, PetscObjectComm((PetscObject)reg), PETSC_ERR_ARG_WRONG, "TaoRegularizer Central Vector not set! Central Vector is needed for KL divergence.");
  PetscCall(VecPointwiseDivide(G, X, y));
  PetscCall(VecLog(G));
  PetscCall(VecPointwiseMult(reg->workvec, G, X));
  PetscCall(VecSum(reg->workvec, &temp));
  PetscCall(VecShift(G, 1.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoRegularizerComputeHessian_KL(TaoRegularizer reg, Vec X, Mat H, Mat Hpre, void *ctx)
{
  Vec y;

  PetscFunctionBegin;
  PetscCall(TaoRegularizerGetCentralVector(reg, &y));
  PetscCheck(y, PetscObjectComm((PetscObject)reg), PETSC_ERR_ARG_WRONG, "TaoRegularizer Central Vector not set! Central Vector is needed for KL divergence.");
  /* TODO Fisher Information Matrix */
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoRegularizerCreate_KL(TaoRegularizer reg)
{
  PetscFunctionBegin;

  PetscValidHeaderSpecific(reg, TAOREGULARIZER_CLASSID, 1);

  reg->ops->view           = TaoRegularizerView_KL;
  reg->ops->destroy        = TaoRegularizerDestroy_KL;
  reg->ops->setfromoptions = TaoRegularizerSetFromOptions_KL;

  reg->ops->computeobjective            = TaoRegularizerComputeObjective_KL;
  reg->ops->computegradient             = TaoRegularizerComputeGradient_KL;
  reg->ops->computeobjectiveandgradient = TaoRegularizerComputeObjectiveAndGradient_KL;
  reg->ops->computehessian              = TaoRegularizerComputeHessian_KL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
