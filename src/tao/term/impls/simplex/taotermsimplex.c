#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

typedef struct _n_TaoTerm_Simplex TaoTerm_Simplex;

struct _n_TaoTerm_Simplex {
  PetscReal  size, tol;
  VecScatter vscat;
  Vec        yseq, workvec;
};

static PetscErrorCode TaoTermDestroy_Simplex(TaoTerm term)
{
  TaoTerm_Simplex *simplex = (TaoTerm_Simplex *)term->data;

  PetscFunctionBegin;
  PetscCall(VecScatterDestroy(&simplex->vscat));
  PetscCall(VecDestroy(&simplex->yseq));
  PetscCall(VecDestroy(&simplex->workvec));
  term->data = NULL;
  PetscCall(PetscFree(simplex));
  PetscCall(TaoTermDestroy_ElementwiseDivergence_Internal(term));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSimplexSetSize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSimplexSetTolerance_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscReal Clip_Internal(PetscReal in, PetscReal tmax)
{
  return PetscMax(0, in - tmax);
}

static PetscErrorCode TaoTermSimplexProx_Internal(TaoTerm term, Vec y, Vec x)
{
  TaoTerm_Simplex *simplex = (TaoTerm_Simplex *)term->data;
  PetscBool        bget    = PETSC_FALSE, workvec_compat;
  PetscReal       *xarray, *yarray, min, max, sum, size, cumsum = 0, tmax = 0;
  PetscMPIInt      mpi_size, rank;
  MPI_Comm         comm;
  PetscInt         len, i, N;

  PetscFunctionBegin;
  size = simplex->size;
  PetscCall(VecSum(y, &sum));
  PetscCall(VecMin(y, NULL, &min));
  PetscCall(VecMax(y, NULL, &max));
  if (PetscAbsReal(sum - size) < simplex->tol && min >= 0 && max <= size) { PetscFunctionReturn(PETSC_SUCCESS); }

  /* Note: DMTaoSetWorkVec is done in parent DMTaoApplyProximalMap */
  PetscCall(PetscObjectGetComm((PetscObject)y, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &mpi_size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (!simplex->workvec) PetscCall(VecDuplicate(x, &simplex->workvec));
  else {
    PetscCall(TaoTermWorkvecTestCompatibility(x, simplex->workvec, &workvec_compat));
    if (!workvec_compat) {
      PetscCall(VecDestroy(&simplex->workvec));
      PetscCall(VecDuplicate(x, &simplex->workvec));
    }
  }
  PetscCall(VecCopy(y, simplex->workvec));
  if (mpi_size == 1) {
    PetscCall(VecGetSize(simplex->workvec, &len));
    PetscCall(VecGetArray(simplex->workvec, &yarray));
    PetscCall(PetscSortReal(len, yarray));

    /* Not thinking about parallelism right now... */
    /* Scalar Version. Technically one can do cumulative sum,
     * but VecCumSum/PrefixSum isn't there yet. */
    for (i = len - 1; i >= 0; i--) {
      cumsum += yarray[i];
      tmax = (cumsum - size) / (len - i);
      if (tmax >= yarray[i - 1]) {
        bget = PETSC_TRUE;
        break;
      }
    }
    PetscCall(VecRestoreArray(simplex->workvec, &yarray));
    if (!bget) { tmax = (cumsum - size) / len; }

    /* Ideally, VecShift(y, -tmax), and set all neg to 0 in two kernel calls... */
    PetscCall(VecGetArray(x, &xarray));
    PetscCall(VecGetArray(y, &yarray));
    for (i = 0; i < len; i++) { xarray[i] = Clip_Internal(yarray[i], tmax); }
    PetscCall(VecRestoreArray(x, &xarray));
    PetscCall(VecRestoreArray(y, &yarray));
  } else {
    /* Parallel case */
    /* Gather x to yseq on rank 0. yseq on other ranks are actually empty */
    if (!simplex->vscat) PetscCall(VecScatterCreateToZero(y, &simplex->vscat, &simplex->yseq));
    else {
      /* Check whether existing VecScatter is same size as vector input */
      //TODO ideally i would want something like VecScatterCheckCompat(vscat, y);
      //and then decide whether to destroy-create again...
      PetscCall(VecScatterDestroy(&simplex->vscat));
      PetscCall(VecDestroy(&simplex->yseq));
      PetscCall(VecScatterCreateToZero(y, &simplex->vscat, &simplex->yseq));
    }
    PetscCall(VecScatterBegin(simplex->vscat, y, simplex->yseq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(simplex->vscat, y, simplex->yseq, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecGetLocalSize(simplex->yseq, &N));
    if (N > 0) {
      PetscCall(VecGetArray(simplex->yseq, &yarray));
      PetscCall(PetscSortReal(N, yarray));

      for (i = N - 1; i >= 0; i--) {
        cumsum += yarray[i];
        tmax = (cumsum - size) / (N - i);
        if (tmax >= yarray[i - 1]) {
          bget = PETSC_TRUE;
          break;
        }
      }
      PetscCall(VecRestoreArray(simplex->yseq, &yarray));
    }
    if (rank == 0 && !bget) tmax = (cumsum - size) / N;

    PetscCallMPI(MPI_Bcast(&tmax, 1, MPIU_REAL, 0, comm));
    /* Ideally, VecShift(y, -tmax), and set all neg to 0 in two kernel calls... */
    PetscCall(VecGetArray(x, &xarray));
    PetscCall(VecGetArray(y, &yarray));
    PetscCall(VecGetLocalSize(y, &len));
    for (i = 0; i < len; i++) { xarray[i] = Clip_Internal(yarray[i], tmax); }
    PetscCall(VecRestoreArray(x, &xarray));
    PetscCall(VecRestoreArray(y, &yarray));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermProximalMap_Simplex(TaoTerm term, Vec p, PetscReal alpha, TaoTerm g, Vec q, PetscReal beta, Vec x)
{
  TaoTermProxMapL2Op l2ops;
  PetscBool          is_simplex, is_l2;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMSIMPLEX, &is_simplex));
  PetscCheck(is_simplex, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoTermProximalMap_Simplex requires first TaoTerm to be of Simplex type");
  /* If Regularizer term is given. If not given, assume HALFL2SQUARED */
  if (g) {
    PetscCall(PetscObjectTypeCompare((PetscObject)g, TAOTERMHALFL2SQUARED, &is_l2));
    PetscCheck(is_l2, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoTermProximalMap_Simplex: TAOTERMBOX only supports TAOTERMHALFL2SQUARED as regularizer");
  }
  PetscCall(TaoTermProxL2FindOps_Internal(q, p, beta, alpha, &l2ops));

  //Note: For Indicator functions, alpha is ignored
  //TODO check whether beta matters for simplex?
  //TODO diag L2 simplex projection
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
    PetscCall(TaoTermSimplexProx_Internal(term, q, x));
    break;
  case TAOTERM_PROX_PROX_TRANS:
    PetscCall(VecAXPBYPCZ(x, 1., 1., 0., p, q));
    PetscCall(TaoTermSimplexProx_Internal(term, x, x));
    PetscCall(VecAXPY(x, -1., p));
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
  TaoTermSimplexSetTolerance - sets the tolerance for `TAOTERMSIMPLEX`

  Logically Collective

  Input Parameters:
+ term - the `TaoTerm` of type `TAOTERMSIMPLEX`
- tol  - tolerance

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSIMPLEX`
@*/
PetscErrorCode TaoTermSimplexSetTolerance(TaoTerm term, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(term, tol, 2);
  PetscTryMethod(term, "TaoTermSimplexSetTolerance_C", (TaoTerm, PetscReal), (term, tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSimplexSetTolerance_Simplex(TaoTerm term, PetscReal tol)
{
  TaoTerm_Simplex *simplex = (TaoTerm_Simplex *)term->data;

  PetscFunctionBegin;
  PetscCheck(tol >= 0, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_OUTOFRANGE, "Simplex tolerance (%g) cannot be negative", (double)tol);
  simplex->tol = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSimplexSetSize - sets the size for `TAOTERMSIMPLEX`

  Logically Collective

  Input Parameters:
+ term - the `TaoTerm` of type `TAOTERMSIMPLEX`
- size - the size of simplex

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSIMPLEX`
@*/
PetscErrorCode TaoTermSimplexSetSize(TaoTerm term, PetscReal size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(term, size, 2);
  PetscTryMethod(term, "TaoTermSimplexSetSize_C", (TaoTerm, PetscReal), (term, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSimplexSetSize_Simplex(TaoTerm term, PetscReal size)
{
  TaoTerm_Simplex *simplex = (TaoTerm_Simplex *)term->data;

  PetscFunctionBegin;
  PetscCheck(size > 0, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_OUTOFRANGE, "Simplex size (%g) must be positive", (double)size);
  simplex->size = size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Simplex(TaoTerm term, PetscViewer viewer)
{
  PetscBool        isascii;
  TaoTerm_Simplex *simplex = (TaoTerm_Simplex *)term->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) { PetscCall(PetscViewerASCIIPrintf(viewer, "Simplex tolerance: %g, size: %g \n", (double)simplex->tol, (double)simplex->size)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetFromOptions_Simplex(TaoTerm term, PetscOptionItems *PetscOptionsObject)
{
  TaoTerm_Simplex *simplex = (TaoTerm_Simplex *)term->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoTerm simplex options");
  PetscCall(PetscOptionsBoundedReal("-taoterm_simplex_tol", "simplex tolerance", "", simplex->tol, &simplex->tol, NULL, 0.0));
  PetscCall(PetscOptionsBoundedReal("-taoterm_simplex_size", "size of simplex", "", simplex->size, &simplex->size, NULL, 0.0));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMSIMPLEX - Simplex TaoTerm object.
  Simplex is defined by x_i >= 0, \sum x = size.
  If size=1, it is a unit simplex. Proximal mapping is an orthogonal
  projection on to this simplex.

  Level: intermediate

  Options Database Keys:
+ -taoterm_simplex_tol <real> - (default 0.0) a tolerance parameter (see `TaoTermSimplexSetTolerance()`)
- -taoterm_simplex_size <real> - (default 1.0) a size parameter (see `TaoTermSimplexSetSize()`)

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMHALFL2SQUARED`
MC*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Simplex(TaoTerm term)
{
  TaoTerm_Simplex *simplex;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&simplex));
  term->data = (void *)simplex;

  /* Default is unit simplex */
  simplex->size  = 1;
  simplex->yseq  = NULL;
  simplex->vscat = NULL;
#if defined(PETSC_USE_REAL_SINGLE)
  simplex->tol = 1.e-5;
#else
  simplex->tol = 1.e-6;
#endif

  term->ops->destroy               = TaoTermDestroy_Simplex;
  term->ops->view                  = TaoTermView_Simplex;
  term->ops->setfromoptions        = TaoTermSetFromOptions_Simplex;
  term->ops->objective             = NULL;
  term->ops->gradient              = NULL;
  term->ops->objectiveandgradient  = NULL;
  term->ops->hessian               = NULL;
  term->ops->hessianmult           = NULL;
  term->ops->createhessianmatrices = NULL;
  term->ops->proximalmap           = TaoTermProximalMap_Simplex;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSimplexSetSize_C", TaoTermSimplexSetSize_Simplex));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSimplexSetTolerance_C", TaoTermSimplexSetTolerance_Simplex));
  PetscFunctionReturn(PETSC_SUCCESS);
}
