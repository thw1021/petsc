#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/
#include <../src/tao/util/dmtao/impls/simplex/simplex.h>

static PetscErrorCode DMTaoContextDestroy_Simplex(DMTao dm)
{
  DMTao_Simplex *ctx = (DMTao_Simplex *)dm->data;

  PetscFunctionBegin;
  PetscCall(VecScatterDestroy(&ctx->vscat));
  PetscCall(VecDestroy(&ctx->yseq));
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoSetFromOptions_Simplex(DMTao dm, PetscOptionItems *PetscOptionsObject)
{
  DMTao_Simplex *ctx = (DMTao_Simplex *)dm->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMTao Simplex options");
  PetscCall(PetscOptionsReal("-dmtao_simplex_tol", "simplex tolerance", "", ctx->tol, &ctx->tol, NULL));
  PetscCall(PetscOptionsReal("-dmtao_simplex_size", "size of simplex", "", ctx->size, &ctx->size, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoView_Simplex(DMTao tdm, PetscViewer pv)
{
  DMTao_Simplex *ctx = (DMTao_Simplex *)tdm->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  DMTao Simplex"));
    PetscCall(PetscViewerASCIIPrintf(pv, ": tol=%g size=%g \n", (double)ctx->tol, (double)ctx->size));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscReal Clip_Internal(PetscReal in, PetscReal tmax)
{
  return PetscMax(0, in - tmax);
}

//TODO PetscParallelSortReal
/* https://arxiv.org/pdf/1101.6081.pdf */
static PetscErrorCode DMTaoApplyProximalMap_Simplex(DMTao tdm0, DMTao tdm1, PetscReal lambda, Vec y, Vec x, PetscBool flg)
{
  DMTao_Simplex *sctx = (DMTao_Simplex *)tdm0->data;
  PetscBool      is_0_s, is_1_l2 = PETSC_TRUE;
  PetscBool      bget = PETSC_FALSE;
  PetscReal     *xarray, *yarray, tmax, min, max, sum, size, cumsum = 0;
  PetscMPIInt    mpi_size, rank;
  MPI_Comm       comm;
  PetscInt       len, i, N;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm0, DMTAOSIMPLEX, &is_0_s));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm1, DMTAOL2, &is_1_l2));

  PetscCheck(is_0_s, PetscObjectComm((PetscObject)tdm0), PETSC_ERR_USER, "DMTaoApplyProximalMap_Simplex requires first DMTao to be of DMTAOSIMPLEX type.");
  /* dm1 == NULL assumes L2 type */
  if (tdm1) PetscCheck(is_1_l2, PetscObjectComm((PetscObject)tdm1), PETSC_ERR_USER, "DMTaoApplyProximalMap_Simplex only upports L2 regularizer type.");

  size = sctx->size;
  PetscCall(VecSum(y, &sum));
  PetscCall(VecMin(y, NULL, &min));
  PetscCall(VecMax(y, NULL, &max));
  if (PetscAbsReal(sum - size) < sctx->tol && min >= 0 && max <= size) { PetscFunctionReturn(PETSC_SUCCESS); }

  /* Note: DMTaoSetWorkVec is done in parent DMTaoApplyProximalMap */
  PetscCall(PetscObjectGetComm((PetscObject)y, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &mpi_size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecCopy(y, tdm0->workvec));
  if (mpi_size == 1) {
    PetscCall(VecGetSize(tdm0->workvec, &len));
    PetscCall(VecGetArray(tdm0->workvec, &yarray));
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
    PetscCall(VecRestoreArray(tdm0->workvec, &yarray));
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
    if (!sctx->vscat) PetscCall(VecScatterCreateToZero(y, &sctx->vscat, &sctx->yseq));
    else {
      /* Check whether existing VecScatter is same size as vector input */
      //TODO ideally i would want something like VecScatterCheckCompat(vscat, y);
      //and then decide whether to destroy-create again...
      PetscCall(VecScatterDestroy(&sctx->vscat));
      PetscCall(VecDestroy(&sctx->yseq));
      PetscCall(VecScatterCreateToZero(y, &sctx->vscat, &sctx->yseq));
    }
    PetscCall(VecScatterBegin(sctx->vscat, y, sctx->yseq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sctx->vscat, y, sctx->yseq, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecGetLocalSize(sctx->yseq, &N));
    if (N > 0) {
      PetscCall(VecGetArray(sctx->yseq, &yarray));
      PetscCall(PetscSortReal(N, yarray));

      for (i = N - 1; i >= 0; i--) {
        cumsum += yarray[i];
        tmax = (cumsum - size) / (N - i);
        if (tmax >= yarray[i - 1]) {
          bget = PETSC_TRUE;
          break;
        }
      }
      PetscCall(VecRestoreArray(sctx->yseq, &yarray));
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

/*@
  DMTaoSimplexSetContext - sets the size and tolerance context for simplex DMTao

  Logically Collective

  Input Parameters:
+ dm   - the `DM` containing `DMTAOSIMPLEX`
. size - size of simplex
- tol  - tolerance

  Level: advanced

  Fortran Notes:
  The context can only be an integer or a `PetscObject`

.seealso: `DMTao`, `DMTAOSIMPLEX`
@*/
PetscErrorCode DMTaoSimplexSetContext(DM dm, PetscReal size, PetscReal tol)
{
  PetscBool is_s;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, DMTAOSIMPLEX, &is_s));

  if (!is_s) PetscFunctionReturn(PETSC_SUCCESS);

  DMTao_Simplex *ctx = (DMTao_Simplex *)tdm->data;

  ctx->size = size;
  ctx->tol  = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   DMTAOSIMPLEX - Simplex DMTao object.
Simplex is defined by x_i >= 0, \sum x = size.
If size=1, it is a unit simplex. Proximal mapping is an orthogonal
projection on to this simplex.

   Options Database Keys:
+      -dmtao_simplex_tol <r>  - tolerance
-      -dmtao_simplex_size <r> - size of simplex

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode DMTaoCreate_Simplex_Private(DMTao dm)
{
  DMTao_Simplex *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  /* Default is unit simplex */
  ctx->size  = 1;
  ctx->yseq  = NULL;
  ctx->vscat = NULL;
#if defined(PETSC_USE_REAL_SINGLE)
  ctx->tol = 1.e-5;
#else
  ctx->tol = 1.e-6;
#endif
  dm->ops->applyproximalmap = DMTaoApplyProximalMap_Simplex;
  dm->data                  = (void *)ctx;
  dm->ops->setup            = NULL;
  dm->ops->destroy          = DMTaoContextDestroy_Simplex;
  dm->ops->view             = DMTaoView_Simplex;
  dm->ops->setfromoptions   = DMTaoSetFromOptions_Simplex;
  dm->ops->reset            = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
