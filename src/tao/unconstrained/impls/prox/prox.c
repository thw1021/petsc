#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petscmath.h>
#include <../src/tao/unconstrained/impls/prox/prox.h>

static PetscFunctionList TaoProxList = NULL;
static PetscBool         TaoProxPackageInitialized;

static PetscErrorCode TaoSolve_Prox(Tao tao)
{
  TAO_PROX          *proxP = (TAO_PROX *)tao->data;
  TaoRegularizerType reg_type;
  TaoRegularizer     reg;

  PetscFunctionBegin;

  PetscCall(TaoGetRegularizer(tao, &reg));
  PetscCall(TaoRegularizerGetType(reg, &reg_type));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);
    /* Solve */
    PetscUseTypeMethod(tao, applyproximalmap, proxP->stepsize, proxP->y, tao->solution, tao->user_proxP);
    /* TODO is there better converged_reason than user?
     * Also, if Method is iterative using subtao, how to deal with it wrt converged reason and monitor?  */
    tao->reason = TAO_CONVERGED_USER;
    /* Note: Changes to stepsize / VM should be done outside TAOPROX */
    tao->niter++;
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_Prox(Tao tao)
{
  TAO_PROX      *proxP = (TAO_PROX *)tao->data;
  TaoRegularizer reg;
  Vec            y;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!proxP->X_old) PetscCall(VecDuplicate(tao->solution, &proxP->X_old));
  if (!proxP->G_old) PetscCall(VecDuplicate(tao->gradient, &proxP->G_old));
  if (!proxP->workvec1) PetscCall(VecDuplicate(tao->solution, &proxP->workvec1));
  if (!proxP->workvec1) PetscCall(VecDuplicate(tao->solution, &proxP->workvec1));

  PetscCall(TaoGetRegularizer(tao, &reg));
  PetscCheck(reg, PetscObjectComm((PetscObject)reg), PETSC_ERR_ARG_WRONG, "TaoRegularizer has not been set for TAOPROX.");
  PetscCall(TaoRegularizerGetCentralVector(reg, &y));
  proxP->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_Prox(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&proxP->X_old));
    PetscCall(VecDestroy(&proxP->G_old));
    PetscCall(VecDestroy(&proxP->workvec1));
    PetscCall(VecDestroy(&proxP->y));
  }

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoProxMSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoProxMGetType_C", NULL));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_Prox(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Proximal algorithm optimization");
  /* Not supporting options to change prox type, as it doesn't makes sense */
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_Prox(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     TAOPROX -  Proximal algorithm.

   Solves proximal operator. User gives function, f(x), and this algorithm solves

   min_x f(x) + \rho/2 \|x-y \|_2^2.

   \rho is stepsize, which can be adaptive. (should i have user-given strategy for adaptiveness?? probably not...)

   Or, if the user preferrs, one can use variable-metric version, which is

   min_x f(x) + 1/2 \|x-y\|_M^2, where \|x\|_M^2 = x^T M x. VM will be diagonal. (future support for diag+ vv^T, a la becker et al?)

   Should default solve method be like CG? Gradient descent?

   Issue: prox_f(y) = inf_x f(x) \rho/2 \|x-y\|_2^2.

   But TAO is essentially, min_x T(x).

   If we are thinking in non=iterative, but just direct ones, (1-iter), this is easy.

   If we want to solve via other methods, it gets tricky..

   "Built-in f(x)", see Beck ...

   Options Database Keys:

  Notes:
     Prox formulas are:
  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_PROX(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoProxInitializePackage());
  PetscCall(PetscNew(&proxP));
  PetscCall(PetscNew(&proxP->L1));

  tao->ops->setup          = TaoSetUp_Prox;
  tao->ops->solve          = TaoSolve_Prox;
  tao->ops->view           = TaoView_Prox;
  tao->ops->setfromoptions = TaoSetFromOptions_Prox;
  tao->ops->destroy        = TaoDestroy_Prox;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  tao->data                = (void *)proxP;
  proxP->y                 = NULL;
  proxP->eta               = 0.1;
  proxP->stepsize          = 1.;
  proxP->stepsize_old      = 0.;
  proxP->orig_objP         = NULL;
  proxP->orig_objgradP     = NULL;
  proxP->orig_gradP        = NULL;
  proxP->orig_hessP        = NULL;
  proxP->ops->orig_obj     = NULL;
  proxP->ops->orig_objgrad = NULL;
  proxP->ops->orig_grad    = NULL;
  proxP->ops->orig_hess    = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxSetStepSize(Tao tao, PetscReal step)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->stepsize = step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxGetStepSize(Tao tao, PetscReal *step)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *step = proxP->stepsize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxSetVM(Tao tao, Mat vm)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->vm = vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxGetVM(Tao tao, Mat *vm)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *vm = proxP->vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxSetInitialVector(Tao tao, Vec y)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  proxP->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoProxGetInitialVector(Tao tao, Vec *y)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  *y = proxP->y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoProxSetType - Sets the `TaoProxType` for the prox routine.

   Collective
TAO_PROX *proxP  = (TAO_PROX *)tao->data;
   Input Parameters:
+  tao - the `Tao` solver context
-  type - a known method

   Options Database Key:
.  -tao_prox_type <type> - Sets the method; use -help for a list
   of available methods (for instance, "-tao_prox_type prox_l2")

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoCreate()`, `TaoProxGetType()`, `TaoProxType`
@*/
PetscErrorCode TaoProxSetType(Tao tao, TaoProxType type)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;
  PetscErrorCode (*prox_xxx)(Tao, PetscReal, Vec, Vec, void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscFunctionListFind(TaoProxList, type, &prox_xxx));
  PetscCheck(prox_xxx, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested Tao prox type %s", type);
  proxP->type                = type;
  tao->ops->applyproximalmap = prox_xxx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoProxGetType - Gets the current `TaoProxType` being used in the `Tao` object

   Not Collective

   Input Parameter:
.  tao - the `Tao` solver context

   Output Parameter:
.  type - the `TaoProxType`

   Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoProxType`, `TaoProxGetType()`
@*/

PetscErrorCode TaoProxGetType(Tao tao, TaoProxType *type)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = proxP->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoProxInitializePackage - This function initializes everything in the `TaoProx` package.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode TaoProxInitializePackage(void)
{
  PetscFunctionBegin;
  if (TaoProxPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TaoProxPackageInitialized = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFunctionListAdd(&TaoProxList, TAOPROX_L1, TaoApplyProximalMap_L1));
  PetscCall(PetscFunctionListAdd(&TaoProxList, TAOPROX_SIMPLEX, TaoApplyProximalMap_Simplex));
  PetscCall(PetscFunctionListAdd(&TaoProxList, TAOPROX_AFFINE, TaoApplyProximalMap_Affine));
#endif
  PetscCall(PetscRegisterFinalize(TaoProxFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoProxFinalizePackage - This function frees everything from the `TaoProx` package. It is
  called from `PetscFinalize()` automatically.

  Level: developer

.seealso: `PetscFinalize()`
@*/
PetscErrorCode TaoProxFinalizePackage(void)
{
  PetscFunctionBegin;
  TaoProxPackageInitialized = PETSC_FALSE;
  PetscCall(PetscFunctionListDestroy(&TaoProxList));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TaoProxRegister - Adds a method to the Tao package for minimization.

   Not Collective

   Input Parameters:
+  type - name of a new user-defined proximal mapping
-  func -  a callback for proximal mapping solver

   Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoSetType()`, `TaoProxRegisterAll()`, `TaoProxRegisterDestroy()`
@*/
PetscErrorCode TaoProxRegister(TaoProxType type, PetscErrorCode (*func)(Tao, PetscReal, Vec, Vec, void *))
{
  PetscFunctionBegin;
  PetscCall(TaoProxInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoProxList, type, func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO should we support this? wrt to usemethod in taosolve in taoprox... */
PETSC_EXTERN PetscErrorCode TaoApplyProximalMap(Tao tao, PetscReal lambda, Vec y, Vec x)
{
  PetscFunctionBegin;
  /*TODO lock push, etc boilerplates... */
  PetscUseTypeMethod(tao, applyproximalmap, lambda, y, x, tao->user_proxP);
  PetscFunctionReturn(PETSC_SUCCESS);
}
