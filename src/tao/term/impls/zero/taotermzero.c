#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

static PetscErrorCode TaoTermDestroy_Zero(TaoTerm term)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(term->data));
  term->data = NULL;
  PetscCall(TaoTermDestroy_ElementwiseDivergence_Internal(term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_Zero(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  PetscFunctionBegin;
  *value = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermProximalMap_Zero(TaoTerm term, Vec p, PetscReal alpha, TaoTerm g, Vec q, PetscReal beta, Vec x)
{
  TaoTermProxMapL2Op l2ops;
  PetscBool          is_zero, is_l2;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMZERO, &is_zero));
  PetscCheck(is_zero, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoTermProximalMap_Zero requires first TaoTerm to be of Zero type");
  /* If Regularizer term is given. If not given, assume HALFL2SQUARED */
  if (g) {
    PetscCall(PetscObjectTypeCompare((PetscObject)g, TAOTERMHALFL2SQUARED, &is_l2));
    PetscCheck(is_l2, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoTermProximalMap_Zero: TAOTERMZERO only supports TAOTERMHALFL2SQUARED as regularizer");
  }
  PetscCall(TaoTermProxL2FindOps_Internal(q, p, beta, alpha, &l2ops));

  //tODO
  switch (l2ops) {
  case TAOTERM_PROX_NO_OP:
    break;
  case TAOTERM_PROX_ZERO:
  case TAOTERM_PROX_SOLVE:
  case TAOTERM_PROX_SOLVE_COMPOSITE:
    PetscCall(VecZeroEntries(x));
    break;
  case TAOTERM_PROX_Q:
    PetscCall(VecCopy(q, x));
    break;
  case TAOTERM_PROX_PROX:
    PetscCall(VecSet(x, 0.));
    break;
  case TAOTERM_PROX_PROX_TRANS:
    PetscCall(VecAXPBYPCZ(x, 1., 1., 0., p, q));
    PetscCall(VecSet(x, 0.));
    PetscCall(VecAXPY(x, -1., p));
    break;
  case TAOTERM_PROX_SOLVE_PARAM:
    PetscCall(VecCopy(p, x)); //TODO is this corect?
    break;
  case TAOTERM_PROX_SOLVE_COMPOSITE_TRANS:
    PetscCall(TaoSoftThreshold(p, -alpha / beta, alpha / beta, x)); //?????
    // what the hell is argmin_y a*IndZero(y+p) + b/2 * ||y||_2^2 ?? dont really care tho
    // maybe error out
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "Invalid problem formulation type.");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Zero(TaoTerm term, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) { PetscCall(PetscViewerASCIIPrintf(viewer, "  TaoTerm Zero")); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMZERO - Zero TaoTerm object.
   Indicator function of the set containing the origin.
   Its proximal mapping returns zero vector.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMHALFL2SQUARED`
MC*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Zero(TaoTerm term)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  term->ops->destroy = TaoTermDestroy_Zero;
  term->ops->view    = TaoTermView_Zero;
  term->data         = NULL;
  //TODO does making it NULL will error out for TAOTERMSUM?
  //For these, maybe having empty routine that doesnt do anything
  //but merely log petscinfo saying nothing is done, is better?
  term->ops->objective             = TaoTermObjective_Zero;
  term->ops->gradient              = NULL;
  term->ops->objectiveandgradient  = NULL;
  term->ops->hessian               = NULL;
  term->ops->hessianmult           = NULL;
  term->ops->createhessianmatrices = NULL;
  term->ops->proximalmap           = TaoTermProximalMap_Zero;
  PetscFunctionReturn(PETSC_SUCCESS);
}
