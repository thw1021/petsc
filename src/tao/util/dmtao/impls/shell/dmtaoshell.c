#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/
#include <../src/tao/util/dmtao/impls/shell/dmtaoshell.h>

static PetscErrorCode DMTaoContextDestroy_Shell(DMTao dm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoSetFromOptions_Shell(DMTao dm, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoSetUp_Shell(DMTao dm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoView_Shell(DMTao dm, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoApplyProximalMap_Shell(DMTao tdm0, DMTao tdm1, PetscReal lambda, Vec y, Vec x, PetscBool flg)
{
  PetscBool is_0_shell;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm0, DMTAOSHELL, &is_0_shell));
  PetscCheck(is_0_shell, PetscObjectComm((PetscObject)tdm0), PETSC_ERR_ARG_WRONGSTATE, "DMTaoApplyProximalMap_Shell requires first DMTao to be of Shell type");

  DMTao_Shell *shell = (DMTao_Shell *)tdm0->data;

  PetscCheck(shell->applyproximalmap, PetscObjectComm((PetscObject)tdm0), PETSC_ERR_ARG_WRONGSTATE, "Must call DMTaoShellSetProximalMap() first");
  PetscCall((*shell->applyproximalmap)(tdm0, tdm1, lambda, y, x, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoShellSetContext - sets the context for a `DMTAOSHELL`

  Logically Collective

  Input Parameters:
+ dm  - the shell DMTao
- ctx - the context

  Level: advanced

  Fortran Notes:
  The context can only be an integer or a `PetscObject`

.seealso: `DMTao`, `DMTAOSHELL`, `DMTaoCreateShell()`, `DMTaoShellGetContext()`
@*/
PetscErrorCode DMTaoShellSetContext(DM dm, void *ctx)
{
  DMTao     tdm;
  PetscBool is_shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, DMTAOSHELL, &is_shell));

  DMTao_Shell *shell = (DMTao_Shell *)tdm->data;
  if (is_shell) shell->data = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoShellGetContext - Returns the user-provided context associated with a `DMTAOSHELL`

  Not Collective

  Input Parameter:
. dm - should have been created with `DMTaoSetType`(dm,`DMTAOSHELL`);

  Output Parameter:
. ctx - the user provided context

  Level: advanced

  Note:
  This routine is intended for use within various shell routines

.seealso: `DMTao`, `DMTAOSHELL`, `DMTaoCreateShell()`, `DMTaoShellSetContext()`
@*/
PetscErrorCode DMTaoShellGetContext(DM dm, void *ctx)
{
  DMTao     tdm;
  PetscBool is_shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(ctx, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, DMTAOSHELL, &is_shell));
  if (!is_shell) *(void **)ctx = NULL;
  else *(void **)ctx = ((DMTao_Shell *)tdm->data)->data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoShellSetProximalMap - Sets routine to apply as proximal map

  Logically Collective

  Input Parameters:
+ dm               - the main `DM` object
- applyproximalmap - the application-provided proximal map routine

  Calling sequence of `applyproximalmap`:
+ dm0    - the `DMTao` primary  context
. dm1    - the `DMTao` regularizer context
. lambda - the scale of regularizer
. y      - the central vector of regularizer
. x      - the solution vector
- is_cj  - bool to denote conjugate or not. TRUE for conjugate, FALSE for regular

  Level: advanced

.seealso: `DMTao`, `DMTAOSHELL`, `DMTaoShellSetContext()`, `DMTaoShellGetContext()`
@*/
PetscErrorCode DMTaoShellSetProximalMap(DM dm, PetscErrorCode (*applyproximalmap)(DMTao dm0, DMTao dm1, PetscReal lambda, Vec y, Vec x, PetscBool is_cj))
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));

  DMTao_Shell *shell = (DMTao_Shell *)tdm->data;

  shell->applyproximalmap = applyproximalmap;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  DMTAOSHELL - a user provided DMTAO

   Level: advanced

.seealso: `dmTaoCreate()`, `DMTao`, `DMTaoSetType()`, `DMTaoType`
M*/
PETSC_EXTERN PetscErrorCode DMTaoCreate_Shell_Private(DMTao dm)
{
  DMTao_Shell *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  dm->ops->applyproximalmap = DMTaoApplyProximalMap_Shell;
  dm->data                  = (void *)ctx;
  dm->ops->setup            = DMTaoSetUp_Shell;
  dm->ops->destroy          = DMTaoContextDestroy_Shell;
  dm->ops->view             = DMTaoView_Shell;
  dm->ops->setfromoptions   = DMTaoSetFromOptions_Shell;
  dm->ops->reset            = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
