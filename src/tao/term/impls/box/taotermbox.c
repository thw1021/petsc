#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

typedef struct {
  PetscReal lb, ub;
  Vec       lbvec, ubvec;
  Vec       work;
} TaoTerm_Box;

static PetscErrorCode TaoTermDestroy_Box(TaoTerm term)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&box->lbvec));
  PetscCall(VecDestroy(&box->ubvec));
  PetscCall(VecDestroy(&box->work));
  PetscCall(PetscFree(box));
  term->data = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermBoxSetBounds_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBoxProject(TaoTerm term, Vec q, Vec x)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;
  Vec          bound;

  PetscFunctionBegin;
  if (q != x) PetscCall(VecCopy(q, x));
  if (box->lbvec) {
    PetscCall(VecPointwiseMax(x, x, box->lbvec));
  } else if (box->lb > PETSC_NINFINITY) {
    PetscCall(VecIfNotCongruentGetSameLayoutVec(x, &box->work));
    bound = box->work;
    PetscCall(VecSet(bound, box->lb));
    PetscCall(VecPointwiseMax(x, x, bound));
  }
  if (box->ubvec) {
    PetscCall(VecPointwiseMin(x, x, box->ubvec));
  } else if (box->ub < PETSC_INFINITY) {
    PetscCall(VecIfNotCongruentGetSameLayoutVec(x, &box->work));
    bound = box->work;
    PetscCall(VecSet(bound, box->ub));
    PetscCall(VecPointwiseMin(x, x, bound));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_Box(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;
  PetscInt     nviolations;
  PetscReal    extremum;
  IS           is;

  PetscFunctionBegin;
  *value = 0.0;
  if (box->lbvec) {
    PetscCall(VecWhichGreaterThan(box->lbvec, x, &is));
    PetscCall(ISGetSize(is, &nviolations));
    PetscCall(ISDestroy(&is));
    if (nviolations) *value = PETSC_INFINITY;
  } else if (box->lb > PETSC_NINFINITY) {
    PetscCall(VecMin(x, NULL, &extremum));
    if (extremum < box->lb) *value = PETSC_INFINITY;
  }
  if (*value == 0.0) {
    if (box->ubvec) {
      PetscCall(VecWhichGreaterThan(x, box->ubvec, &is));
      PetscCall(ISGetSize(is, &nviolations));
      PetscCall(ISDestroy(&is));
      if (nviolations) *value = PETSC_INFINITY;
    } else if (box->ub < PETSC_INFINITY) {
      PetscCall(VecMax(x, NULL, &extremum));
      if (extremum > box->ub) *value = PETSC_INFINITY;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermProximalMap_Box(TaoTerm term, Vec p, PetscReal alpha, TaoTerm reg, Vec q, PetscReal beta, Vec x)
{
  PetscBool is_l2;

  PetscFunctionBegin;
  if (reg) {
    PetscCall(PetscObjectTypeCompare((PetscObject)reg, TAOTERMHALFL2SQUARED, &is_l2));
    PetscCheck(is_l2, PetscObjectComm((PetscObject)term), PETSC_ERR_SUP, "TAOTERMBOX only supports TAOTERMHALFL2SQUARED as its proximal regularizer");
  }
  PetscCheck(!p, PetscObjectComm((PetscObject)term), PETSC_ERR_SUP, "TAOTERMBOX does not support parameterized proximal maps");
  if (q) {
    if (alpha == 0.0) PetscCall(VecCopy(q, x));
    else PetscCall(TaoTermBoxProject(term, q, x));
  } else {
    PetscCall(VecZeroEntries(x));
    if (alpha != 0.0) PetscCall(TaoTermBoxProject(term, x, x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBoxValidateBounds(TaoTerm term, PetscReal lb, PetscReal ub, Vec lbvec, Vec ubvec)
{
  PetscInt  nviolations;
  PetscReal extremum;
  IS        is;

  PetscFunctionBegin;
  if (lbvec && ubvec) {
    PetscCheckSameTypeAndComm(lbvec, 4, ubvec, 5);
    VecCheckSameSize(lbvec, 4, ubvec, 5);
    PetscCall(VecWhichGreaterThan(lbvec, ubvec, &is));
    PetscCall(ISGetSize(is, &nviolations));
    PetscCall(ISDestroy(&is));
    PetscCheck(!nviolations, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Lower bound vector exceeds upper bound vector at %" PetscInt_FMT " entries", nviolations);
  } else if (lbvec) {
    PetscCall(VecMax(lbvec, NULL, &extremum));
    PetscCheck(extremum <= ub, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Lower bound vector has maximum %g greater than scalar upper bound %g", (double)extremum, (double)ub);
  } else if (ubvec) {
    PetscCall(VecMin(ubvec, NULL, &extremum));
    PetscCheck(lb <= extremum, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Scalar lower bound %g is greater than upper bound vector minimum %g", (double)lb, (double)extremum);
  } else PetscCheck(lb <= ub, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_INCOMP, "Scalar lower bound %g is greater than scalar upper bound %g", (double)lb, (double)ub);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBoxSetBounds_Box(TaoTerm term, PetscReal lb, PetscReal ub, Vec lbvec, Vec ubvec)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;

  PetscFunctionBegin;
  PetscCall(TaoTermBoxValidateBounds(term, lb, ub, lbvec, ubvec));
  PetscCall(PetscObjectReference((PetscObject)lbvec));
  PetscCall(PetscObjectReference((PetscObject)ubvec));
  PetscCall(VecDestroy(&box->lbvec));
  PetscCall(VecDestroy(&box->ubvec));
  box->lb    = lb;
  box->ub    = ub;
  box->lbvec = lbvec;
  box->ubvec = ubvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermBoxSetBounds - Set componentwise lower and upper bounds for a `TAOTERMBOX`.

  Logically Collective

  Input Parameters:
+ term  - a `TaoTerm` of type `TAOTERMBOX`
. lb    - scalar lower bound
. ub    - scalar upper bound
. lbvec - (optional) vector of lower bounds
- ubvec - (optional) vector of upper bounds

  Level: intermediate

  Notes:
  A bound vector, when provided, takes precedence over the corresponding scalar bound. Passing
  `NULL` for a bound vector selects the scalar bound and clears any vector set previously.

  Every active lower bound must be less than or equal to the corresponding active upper bound.

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMBOX`, `TaoTermProximalMap()`
@*/
PetscErrorCode TaoTermBoxSetBounds(TaoTerm term, PetscReal lb, PetscReal ub, Vec lbvec, Vec ubvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(term, lb, 2);
  PetscValidLogicalCollectiveReal(term, ub, 3);
  if (lbvec) {
    PetscValidHeaderSpecific(lbvec, VEC_CLASSID, 4);
    PetscCheckSameComm(term, 1, lbvec, 4);
  }
  if (ubvec) {
    PetscValidHeaderSpecific(ubvec, VEC_CLASSID, 5);
    PetscCheckSameComm(term, 1, ubvec, 5);
  }
  PetscTryMethod(term, "TaoTermBoxSetBounds_C", (TaoTerm, PetscReal, PetscReal, Vec, Vec), (term, lb, ub, lbvec, ubvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Box(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;
  PetscBool    isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    if (box->lbvec) PetscCall(PetscViewerASCIIPrintf(viewer, "lower bound: vector\n"));
    else PetscCall(PetscViewerASCIIPrintf(viewer, "lower bound: %g\n", (double)box->lb));
    if (box->ubvec) PetscCall(PetscViewerASCIIPrintf(viewer, "upper bound: vector\n"));
    else PetscCall(PetscViewerASCIIPrintf(viewer, "upper bound: %g\n", (double)box->ub));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetFromOptions_Box(TaoTerm term, PetscOptionItems PetscOptionsObject)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;
  PetscReal    lb = box->lb, ub = box->ub;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoTerm box options");
  PetscCall(PetscOptionsReal("-tao_term_box_lb", "Scalar lower bound", "TaoTermBoxSetBounds", lb, &lb, NULL));
  PetscCall(PetscOptionsReal("-tao_term_box_ub", "Scalar upper bound", "TaoTermBoxSetBounds", ub, &ub, NULL));
  PetscOptionsHeadEnd();
  PetscCall(TaoTermBoxSetBounds_Box(term, lb, ub, box->lbvec, box->ubvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMBOX - A `TaoTerm` representing the indicator function of componentwise lower and upper bounds.

  Level: intermediate

  Options Database Keys:
+ -tao_term_box_lb real - scalar lower bound
- -tao_term_box_ub real - scalar upper bound

  Notes:
  The objective value is zero at feasible points and `PETSC_INFINITY` at infeasible points.
  Its proximal map with `TAOTERMHALFL2SQUARED` is componentwise projection onto the bounds.

  Vector bounds can be set with `TaoTermBoxSetBounds()`.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermType`, `TaoTermBoxSetBounds()`, `TaoTermProximalMap()`, `TAOTERMHALFL2SQUARED`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Box(TaoTerm term)
{
  TaoTerm_Box *box;

  PetscFunctionBegin;
  PetscCall(PetscNew(&box));
  term->data                = (void *)box;
  term->parameters_mode     = TAOTERM_PARAMETERS_NONE;
  box->lb                   = PETSC_NINFINITY;
  box->ub                   = PETSC_INFINITY;
  term->ops->destroy        = TaoTermDestroy_Box;
  term->ops->view           = TaoTermView_Box;
  term->ops->setfromoptions = TaoTermSetFromOptions_Box;
  term->ops->objective      = TaoTermComputeObjective_Box;
  term->ops->proximalmap    = TaoTermProximalMap_Box;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermBoxSetBounds_C", TaoTermBoxSetBounds_Box));
  PetscFunctionReturn(PETSC_SUCCESS);
}
