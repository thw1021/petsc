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

static PetscErrorCode TaoTermProximalMap_Zero(TaoTerm term, Vec p, PetscReal alpha, TaoTerm g, Vec q, PetscReal beta, Vec x)
{
  PetscBool          is_zero;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMZERO, &is_zero));
  PetscCheck(is_zero, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoTermProximalMap_Zero requires first TaoTerm to be of Zero type");
  PetscCall(VecSet(x, 0.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Zero(TaoTerm term, PetscViewer viewer)
{
  PetscBool    isascii;

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
  term->ops->destroy               = TaoTermDestroy_Zero;
  term->ops->view                  = TaoTermView_Zero;
  term->data                       = NULL;
  //TODO does making it NULL will error out for TAOTERMSUM?
  //For these, maybe having empty routine that doesnt do anything
  //but merely log petscinfo saying nothing is done, is better?
  term->ops->objective             = NULL;
  term->ops->gradient              = NULL;
  term->ops->objectiveandgradient  = NULL;
  term->ops->hessian               = NULL;
  term->ops->hessianmult           = NULL;
  term->ops->createhessianmatrices = NULL;
  term->ops->proximalmap           = TaoTermProximalMap_Zero;
  PetscFunctionReturn(PETSC_SUCCESS);
}
