#include <petsc/private/taoimpl.h>
#include <petsc/private/dmimpl.h>
#include <../src/tao/util/dmtao/impls/simplex/simplex.h>

static PetscErrorCode DMTaoContextDestroy_Simplex(DMTao dm)
{
  PetscFunctionBegin;
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

static inline PetscReal Clip_Internal(PetscReal in, PetscReal tmax)
{
  return PetscMax(0, in - tmax);
}

/* https://arxiv.org/pdf/1101.6081.pdf */
static PetscErrorCode DMTaoApplyProximalMap_Simplex(DMTao dm0, DMTao dm1, PetscReal lambda, Vec y, Vec x, void *ptr)
{
  PetscBool  is_0_s, is_1_l2, is_vm_diag, bget = PETSC_FALSE;
  PetscReal *xarray, *yarray, tmax, min, sum, size, cumsum = 0;
  PetscInt   len, i;
  Mat        vm;
  DM         pdm;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)dm0, DMTAOSIMPLEX, &is_0_s));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm1, DMTAOL2, &is_1_l2));
  PetscCheck(is_1_l2, PetscObjectComm((PetscObject)dm0), PETSC_ERR_USER, "DMTaoApplyProximalMap_Simplex only upports L2 regularizer type.");

  DMTao_Simplex *sctx = (DMTao_Simplex *)dm0->data;

  size = sctx->size;
  PetscCall(VecSum(y, &sum));
  PetscCall(VecMin(y, NULL, &min));
  if (PetscAbsReal(sum - size) < sctx->tol) { PetscFunctionReturn(PETSC_SUCCESS); }

  PetscCall(DMTaoGetParentDM(dm1, &pdm));
  PetscCall(DMTaoGetVM(pdm, &vm));
  if (vm) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vm, MATDIAGONAL, &is_vm_diag));
    PetscCheck(is_vm_diag, PetscObjectComm((PetscObject)dm0), PETSC_ERR_USER, "DMTaoApplyProximalMap_Simplex only supports diagonal VM matrix.");
  }
  PetscCall(VecGetSize(y, &len));
  PetscCall(VecGetArray(y, &yarray));
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
  if (!bget) { tmax = (cumsum - size) / len; }

  /* Ideally, VecShift(y, -tmax), and set all neg to 0 in two kernel calls... */
  PetscCall(VecGetArray(x, &xarray));
  for (i = 0; i < len; i++) { xarray[i] = Clip_Internal(yarray[i], tmax); }
  PetscCall(VecRestoreArray(y, &yarray));
  PetscCall(VecRestoreArray(x, &xarray));
  //TODO support diag vm metric
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTaoSimplexSetContext(DM dm, PetscReal size, PetscReal tol)
{
  PetscBool is_s;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, DMTAOSIMPLEX, &is_s));

  DMTao_Simplex *ctx = (DMTao_Simplex *)tdm->data;

  if (!is_s) PetscFunctionReturn(PETSC_SUCCESS);
  ctx->size = size;
  ctx->tol  = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMTaoCreate_Simplex_Private(DMTao dm)
{
  DMTao_Simplex *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  /* Default is unit simplex */
  ctx->size = 1;
#if defined(PETSC_USE_REAL_SINGLE)
  ctx->tol = 1.e-5;
#else
  ctx->tol = 1.e-6;
#endif
  dm->data                  = (void *)ctx;
  dm->ops->applyproximalmap = DMTaoApplyProximalMap_Simplex;
  dm->ops->destroy          = DMTaoContextDestroy_Simplex;
  dm->ops->setfromoptions   = DMTaoSetFromOptions_Simplex;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMTaoCreate_Simplex(MPI_Comm comm, DM *dm, Mat vm, Vec y, PetscReal size, PetscReal tol)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMGetDMTao(*dm, &tdm));
  PetscCall(DMTaoSetType(*dm, DMTAOSIMPLEX));
  PetscCall(DMTaoSetCentralVector(*dm, y));
  PetscCall(DMTaoSetVM(*dm, vm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
