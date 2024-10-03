#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

typedef struct _n_TaoTerm_Box TaoTerm_Box;

struct _n_TaoTerm_Box {
  PetscReal lb_real, ub_real;
  Vec       lb_vec, ub_vec;
};

static PetscErrorCode TaoTermDestroy_Box(TaoTerm term)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;

  PetscFunctionBegin;
  term->data = NULL;
  PetscCall(VecDestroy(&box->lb_vec));
  PetscCall(VecDestroy(&box->ub_vec));
  PetscCall(PetscFree(box));
  PetscCall(TaoTermDestroy_ElementwiseDivergence_Internal(term));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermBoxSetContext_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//y: input, x: output
//TODO what if input and output vec are the same?
static PetscErrorCode TaoTermBoxProx_Internal(TaoTerm term, Vec y, Vec x)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;
  PetscInt     n, i;
  PetscReal   *larray, *uarray, *yarray, *xarray;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArray(x, &xarray));
  PetscCall(VecGetArray(y, &yarray));
  if (box->lb_vec) PetscCall(VecGetArray(box->lb_vec, &larray));
  if (box->ub_vec) PetscCall(VecGetArray(box->ub_vec, &uarray));

  /* Four cases:
   * 1: lb_vec,  ub_vec,
   * 2: lb_vec,  ub_real,
   * 3: lb_real, ub_vec,
   * 4: lb_real, ub_real */
  if (box->lb_vec && box->ub_vec) {
    /* Case 1 */
    PetscCall(VecPointwiseMax(x, y, box->lb_vec));
    PetscCall(VecPointwiseMin(x, x, box->ub_vec));
  } else if (box->lb_vec) {
    /* Case 2. Ignore lb_real */
    PetscCall(VecGetArray(x, &xarray));
    PetscCall(VecGetArray(y, &yarray));
    PetscCall(VecGetArray(box->lb_vec, &larray));
    for (i = 0; i < n; i++) xarray[i] = PetscMin(PetscMax(yarray[i], larray[i]), box->ub_real);
    PetscCall(VecRestoreArray(x, &xarray));
    PetscCall(VecRestoreArray(y, &yarray));
    PetscCall(VecRestoreArray(box->lb_vec, &larray));
  } else if (box->ub_vec) {
    /* Case 3. Ignore ub_real */
    PetscCall(VecGetArray(x, &xarray));
    PetscCall(VecGetArray(y, &yarray));
    PetscCall(VecGetArray(box->ub_vec, &uarray));
    for (i = 0; i < n; i++) xarray[i] = PetscMin(PetscMax(yarray[i], box->lb_real), uarray[i]);
    PetscCall(VecRestoreArray(x, &xarray));
    PetscCall(VecRestoreArray(y, &yarray));
    PetscCall(VecRestoreArray(box->ub_vec, &uarray));
  } else {
    /* Case 4 */
    PetscCall(VecGetArray(x, &xarray));
    PetscCall(VecGetArray(y, &yarray));
    for (i = 0; i < n; i++) xarray[i] = PetscMin(PetscMax(yarray[i], box->lb_real), box->ub_real);
    PetscCall(VecRestoreArray(x, &xarray));
    PetscCall(VecRestoreArray(y, &yarray));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermProximalMap_Box(TaoTerm term, Vec p, PetscReal alpha, TaoTerm g, Vec q, PetscReal beta, Vec x)
{
  TaoTermProxMapL2Op l2ops;
  PetscBool          is_box, is_l2;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMBOX, &is_box));
  PetscCheck(is_box, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoTermProximalMap_Box requires first TaoTerm to be of BOX type");
  /* If Regularizer term is given. If not given, assume HALFL2SQUARED */
  if (g) {
    PetscCall(PetscObjectTypeCompare((PetscObject)g, TAOTERMHALFL2SQUARED, &is_l2));
    PetscCheck(is_l2, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoTermProximalMap_Box: TAOTERMBOX only supports TAOTERMHALFL2SQUARED as regularizer");
  }
  PetscCall(TaoTermProxL2FindOps_Internal(q, p, beta, alpha, &l2ops));

  //Note: For Indicator functions, alpha is ignored, and for box function, beta doesn't matter
  switch (l2ops) {
  case TAOTERM_PROX_NO_OP:
    break;
  case TAOTERM_PROX_ZERO:
    PetscCall(VecZeroEntries(x));
    break;
  case TAOTERM_PROX_Q:
    PetscCall(VecCopy(q, x));
    break;
  case TAOTERM_PROX_SOLVE:
  case TAOTERM_PROX_SOLVE_PARAM:
    SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "No unique solution available for this formulation.");
  case TAOTERM_PROX_PROX:
    PetscCall(TaoTermBoxProx_Internal(term, q, x));
    break;
  case TAOTERM_PROX_PROX_TRANS:
    PetscCall(VecAXPBYPCZ(x, -1., 1., 0., p, q));
    PetscCall(TaoTermBoxProx_Internal(term, x, x));
    PetscCall(VecAXPY(x, 1., p));
    break;
  case TAOTERM_PROX_SOLVE_COMPOSITE:
  case TAOTERM_PROX_SOLVE_COMPOSITE_TRANS:
    PetscCall(VecCopy(p, x));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "Invalid problem formulation type.");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermBoxSetContext - sets the upperbound and lowerbound context for
  `TAOTERMBOX` such that lowerbound <= upperbound.

  Logically Collective

  Input Parameters:
+ dm      - the `TaoTerm` of type `TAOTERMBOX`
. lb_real - lowerbound, real number
. ub_real - upperbound, real number
. lb_vec  - lowerbound, vector
- ub_vec  - upperbound, vector

  Level: advanced

  Notes:
  One can set either real number bounds, or vector bounds.
  In all cases, condition lb <= ub needs to be strictly satisfied,
  and if vectors are given, sizes of lb and ub needs to match.
  If both lb_real and lb_vec are provided  lb_real is ignored.
  Likewise for ub_real and ub_vec.

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMBOX`
@*/
PetscErrorCode TaoTermBoxSetContext(TaoTerm term, PetscReal lb_real, PetscReal ub_real, Vec lb_vec, Vec ub_vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(term, lb_real, 2);
  PetscValidLogicalCollectiveReal(term, ub_real, 3);
  PetscTryMethod(term, "TaoTermBoxSetContext_C", (TaoTerm, PetscReal, PetscReal, Vec, Vec), (term, lb_real, ub_real, lb_vec, ub_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBoxSetContext_Box(TaoTerm term, PetscReal lb_real, PetscReal ub_real, Vec lb_vec, Vec ub_vec)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;
  PetscReal    l_max, u_min;

  PetscFunctionBegin;
  if (lb_vec) {
    PetscValidHeaderSpecific(lb_vec, VEC_CLASSID, 4);
    PetscCall(PetscObjectReference((PetscObject)lb_vec));
    PetscCall(VecDestroy(&box->lb_vec));
    PetscCall(VecMax(lb_vec, NULL, &l_max));

    box->lb_vec = lb_vec;
  }
  if (ub_vec) {
    PetscValidHeaderSpecific(ub_vec, VEC_CLASSID, 5);
    PetscCall(PetscObjectReference((PetscObject)ub_vec));
    PetscCall(VecDestroy(&box->ub_vec));
    PetscCall(VecMin(ub_vec, NULL, &u_min));

    box->ub_vec = ub_vec;
  }
  /* Case 1. Ignore lb_real and ub_real */
  if (lb_vec && ub_vec) {
    PetscCheckSameTypeAndComm(lb_vec, 4, ub_vec, 5);
    VecCheckSameSize(lb_vec, 4, ub_vec, 5);
    PetscCheck(l_max <= u_min, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "lower bound needs to be equal or less than upper bound");
  }
  /* Case 2. Ignore lb_real */
  else if (lb_vec)
    PetscCheck(l_max <= ub_real, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "lower bound needs to be equal or less than upper bound");
  /* Case 3. Ignore ub_real */
  else if (ub_vec) PetscCheck(lb_real <= u_min, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "lower bound needs to be equal or less than upper bound");
  /* Case 4 */
  else PetscCheck(lb_real <= ub_real, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "lower bound needs to be equal or less than upper bound");

  box->lb_real = lb_real;
  box->ub_real = ub_real;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Box(TaoTerm term, PetscViewer viewer)
{
  PetscBool    isascii;
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  TaoTerm Box"));
    PetscCall(PetscViewerASCIIPrintf(viewer, ": lb=%g ub=%g\n", (double)box->lb_real, (double)box->ub_real));
    if (box->lb_vec) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "TaoTermBox Lower bound vector\n"));
      PetscCall(VecView(box->lb_vec, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    if (box->ub_vec) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "TaoTermBox Upper bound vector\n"));
      PetscCall(VecView(box->ub_vec, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetFromOptions_Box(TaoTerm term, PetscOptionItems *PetscOptionsObject)
{
  TaoTerm_Box *box = (TaoTerm_Box *)term->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoTerm box options");
  PetscCall(PetscOptionsReal("-taoterm_box_lb_real", "DMTao Box lower bound double.", "", box->lb_real, &box->lb_real, NULL));
  PetscCall(PetscOptionsReal("-taoterm_box_ub_real", "DMTao Box upper bound double.", "", box->ub_real, &box->ub_real, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMBOX - A `TaoTerm` that computes $\|x - p\|_1$, for solution $x$ and parameters $p$.

  Level: intermediate

  Options Database Keys:
. -taoterm_box_epsilon <real> - (default 0.0) a smoothing parameter (see `TaoTermBoxSetEpsilon()`)

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMHALFL2SQUARED`
MC*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Box(TaoTerm term)
{
  TaoTerm_Box *box;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&box));
  term->data = (void *)box;

  box->lb_real = 0.;
  box->ub_real = 0.;
  box->lb_vec  = NULL;
  box->ub_vec  = NULL;

  term->ops->destroy               = TaoTermDestroy_Box;
  term->ops->view                  = TaoTermView_Box;
  term->ops->setfromoptions        = TaoTermSetFromOptions_Box;
  //TODO does making it NULL will error out for TAOTERMSUM?
  //For these, maybe having empty routine that doesnt do anything
  //but merely log petscinfo saying nothing is done, is better?
  term->ops->objective             = NULL;
  term->ops->gradient              = NULL;
  term->ops->objectiveandgradient  = NULL;
  term->ops->hessian               = NULL;
  term->ops->hessianmult           = NULL;
  term->ops->createhessianmatrices = NULL;
  term->ops->proximalmap           = TaoTermProximalMap_Box;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermBoxSetContext_C", TaoTermBoxSetContext_Box));
  PetscFunctionReturn(PETSC_SUCCESS);
}
