#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/
#include <../src/tao/util/dmtao/impls/box/box.h>

static PetscErrorCode DMTaoContextDestroy_Box(DMTao dm)
{
  DMTao_Box *ctx = (DMTao_Box *)dm->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&ctx->lb_vec));
  PetscCall(VecDestroy(&ctx->ub_vec));
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoSetFromOptions_Box(DMTao dm, PetscOptionItems *PetscOptionsObject)
{
  DMTao_Box *ctx = (DMTao_Box *)dm->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMTao Box options");
  PetscCall(PetscOptionsReal("-dmtao_box_lb_real", "DMTao Box lower bound double.", "", ctx->lb_real, &ctx->lb_real, NULL));
  PetscCall(PetscOptionsReal("-dmtao_box_ub_real", "DMTao Box upper bound double.", "", ctx->ub_real, &ctx->ub_real, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoView_Box(DMTao dm, PetscViewer pv)
{
  DMTao_Box *ctx = (DMTao_Box *)dm->data;
  PetscBool  isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  DMTao Box"));
    PetscCall(PetscViewerASCIIPrintf(pv, ": lb=%g ub=%g\n", (double)ctx->lb_real, (double)ctx->ub_real));
    if (ctx->lb_vec) {
      PetscCall(PetscViewerASCIIPushTab(pv));
      PetscCall(PetscViewerASCIIPrintf(pv, "DMTao Box Lower bound vector\n"));
      PetscCall(VecView(ctx->lb_vec, pv));
      PetscCall(PetscViewerASCIIPopTab(pv));
    }
    if (ctx->ub_vec) {
      PetscCall(PetscViewerASCIIPushTab(pv));
      PetscCall(PetscViewerASCIIPrintf(pv, "DMTao Box Upper bound vector\n"));
      PetscCall(VecView(ctx->ub_vec, pv));
      PetscCall(PetscViewerASCIIPopTab(pv));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoBoxSetContext - sets the upperbound and lowerbound context for
  `DMTAOBox` such that lowerbound <= upperbound.

  Logically Collective

  Input Parameters:
+ dm - the `DM` containing `DMTAOBOX`
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

.seealso: `DMTao`, `DMTAOBox`
@*/
PetscErrorCode DMTaoBoxSetContext(DM dm, PetscReal lb_real, PetscReal ub_real, Vec lb_vec, Vec ub_vec)
{
  DMTao     tdm;
  PetscReal l_max, u_min;

  PetscFunctionBegin;
  /* Four cases:
   * 1: lb_vec,  ub_vec,
   * 2: lb_vec,  ub_real,
   * 3: lb_real, ub_vec,
   * 4: lb_real, ub_real */
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(dm, lb_real, 2);
  PetscValidLogicalCollectiveReal(dm, ub_real, 3);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));

  DMTao_Box *ctx = (DMTao_Box *)tdm->data;

  if (lb_vec) {
    PetscValidHeaderSpecific(lb_vec, VEC_CLASSID, 4);
    PetscCall(PetscObjectReference((PetscObject)lb_vec));
    PetscCall(VecDestroy(&ctx->lb_vec));
    PetscCall(VecMax(lb_vec, NULL, &l_max));

    ctx->lb_vec = lb_vec;
  }
  if (ub_vec) {
    PetscValidHeaderSpecific(ub_vec, VEC_CLASSID, 4);
    PetscCall(PetscObjectReference((PetscObject)ub_vec));
    PetscCall(VecDestroy(&ctx->ub_vec));
    PetscCall(VecMin(ub_vec, NULL, &u_min));

    ctx->ub_vec = ub_vec;
  }
  /* Case 1. Ignore lb_real and ub_real */
  if (lb_vec && ub_vec) {
    PetscCheckSameTypeAndComm(lb_vec, 4, ub_vec, 5);
    VecCheckSameSize(lb_vec, 4, ub_vec, 5);
    PetscCheck(l_max <= u_min, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "lower bound needs to be equal or less than upper bound");
  }
  /* Case 2. Ignore lb_real */
  else if (lb_vec)
    PetscCheck(l_max <= ub_real, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "lower bound needs to be equal or less than upper bound");
  /* Case 3. Ignore ub_real */
  else if (ub_vec) PetscCheck(lb_real <= u_min, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "lower bound needs to be equal or less than upper bound");
  /* Case 4 */
  else PetscCheck(lb_real <= ub_real, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "lower bound needs to be equal or less than upper bound");

  ctx->lb_real = lb_real;
  ctx->ub_real = ub_real;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* @ DMTaoApplyProximalMap_Box - This routine performs a projection onto a box constraint

   Input Parameters:
+  dm0  - Box DMTao context
.  dm1  - Regularizer DMTao context
.  step - scale for the regularizer
.  y    - input vector
-  flg  - denotes conjugate. If true, computes prox of conjugate

   Output parameters:
.  x - output vector

@ */
static PetscErrorCode DMTaoApplyProximalMap_Box(DMTao tdm0, DMTao tdm1, PetscReal step, Vec y, Vec x, PetscBool flg)
{
  PetscBool  is_0_box, is_1_l2 = PETSC_TRUE;
  PetscReal *larray, *uarray, *yarray, *xarray;
  PetscInt   i, n;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm0, DMTAOBOX, &is_0_box));
  PetscCheck(is_0_box, PetscObjectComm((PetscObject)tdm0), PETSC_ERR_USER, "DMTaoApplyProximalMap_Box requires first DMTao to be of Box type");
  if (tdm1) {
    PetscCall(PetscObjectTypeCompare((PetscObject)tdm1, DMTAOL2, &is_1_l2));
    PetscCheck(is_1_l2, PetscObjectComm((PetscObject)tdm1), PETSC_ERR_USER, "DMTaoApplyProximalMap_Box requires second DMTao to be of L2 type");
  }
  /* dm1 == NULL assumes L2 type */

  DMTao_Box *ctx = (DMTao_Box *)tdm0->data;

  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArray(x, &xarray));
  PetscCall(VecGetArray(y, &yarray));
  if (ctx->lb_vec) PetscCall(VecGetArray(ctx->lb_vec, &larray));
  if (ctx->ub_vec) PetscCall(VecGetArray(ctx->ub_vec, &uarray));

  /* Four cases:
   * 1: lb_vec,  ub_vec,
   * 2: lb_vec,  ub_real,
   * 3: lb_real, ub_vec,
   * 4: lb_real, ub_real */
  if (ctx->lb_vec && ctx->ub_vec) {
    /* Case 1 */
    PetscCall(VecPointwiseMax(x, y, ctx->lb_vec));
    PetscCall(VecPointwiseMin(x, x, ctx->ub_vec));
  } else if (ctx->lb_vec) {
    /* Case 2. Ignore lb_real */
    PetscCall(VecGetArray(x, &xarray));
    PetscCall(VecGetArray(y, &yarray));
    PetscCall(VecGetArray(ctx->lb_vec, &larray));
    for (i = 0; i < n; i++) xarray[i] = PetscMin(PetscMax(yarray[i], larray[i]), ctx->ub_real);
    PetscCall(VecRestoreArray(x, &xarray));
    PetscCall(VecRestoreArray(y, &yarray));
    PetscCall(VecRestoreArray(ctx->lb_vec, &larray));
  } else if (ctx->ub_vec) {
    /* Case 3. Ignore ub_real */
    PetscCall(VecGetArray(x, &xarray));
    PetscCall(VecGetArray(y, &yarray));
    PetscCall(VecGetArray(ctx->ub_vec, &uarray));
    for (i = 0; i < n; i++) xarray[i] = PetscMin(PetscMax(yarray[i], ctx->lb_real), uarray[i]);
    PetscCall(VecRestoreArray(x, &xarray));
    PetscCall(VecRestoreArray(y, &yarray));
    PetscCall(VecRestoreArray(ctx->ub_vec, &uarray));
  } else {
    /* Case 4 */
    PetscCall(VecGetArray(x, &xarray));
    PetscCall(VecGetArray(y, &yarray));
    for (i = 0; i < n; i++) xarray[i] = PetscMin(PetscMax(yarray[i], ctx->lb_real), ctx->ub_real);
    PetscCall(VecRestoreArray(x, &xarray));
    PetscCall(VecRestoreArray(y, &yarray));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   DMTAOBOX - Box constraint DMTao object. This represents a convex set, where
   C = {x: lb \leq x \leq ub}. Lowerbound and upperbound must satisfy lb \leq ub.

  Level: beginner

.seealso: `DMTao`
M*/
PETSC_EXTERN PetscErrorCode DMTaoCreate_Box_Private(DMTao dm)
{
  DMTao_Box *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));

  ctx->lb_real = 0.;
  ctx->ub_real = 0.;
  ctx->lb_vec  = NULL;
  ctx->ub_vec  = NULL;

  dm->ops->applyproximalmap = DMTaoApplyProximalMap_Box;
  dm->ops->computeobjective = NULL; //Ignoring obj func for indicator functions...
  dm->data                  = (void *)ctx;
  dm->ops->setup            = NULL;
  dm->ops->destroy          = DMTaoContextDestroy_Box;
  dm->ops->view             = DMTaoView_Box;
  dm->ops->setfromoptions   = DMTaoSetFromOptions_Box;
  dm->ops->reset            = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
