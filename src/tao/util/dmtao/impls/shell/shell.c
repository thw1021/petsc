#include <petsc/private/taoimpl.h>
#include <petsc/private/dmimpl.h>
#include <../src/tao/util/dmtao/impls/shell/shell.h>

static PetscErrorCode DMTaoContextDestroy_Shell(DMTao dm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(dm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoSetFromOptions_Shell(DMTao dm, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMTao Shell options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoApplyProximalMap_Shell(DMTao dm0, DMTao dm1, PetscReal lambda, Vec y, Vec x, void *ctx)
{
  PetscBool is_shell;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)dm0, DMTAOSHELL, &is_shell));
  PetscAssert(is_shell, PetscObjectComm((PetscObject)dm0), PETSC_ERR_USER, "DMShell does not have proximal map.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMTaoCreate_Shell_Private(DMTao dm)
{
  DMTao_Shell *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(PetscNew(&ctx));
  dm->data                  = (void *)ctx;
  dm->ops->applyproximalmap = DMTaoApplyProximalMap_Shell;
  dm->ops->destroy          = DMTaoContextDestroy_Shell;
  dm->ops->setfromoptions   = DMTaoSetFromOptions_Shell;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMTaoCreate_Shell(MPI_Comm comm, DM *dm, Mat vm, Vec y, PetscReal size, PetscReal tol)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMGetDMTao(*dm, &tdm));
  PetscCall(DMTaoSetType(*dm, DMTAOSHELL));
  PetscCall(DMTaoSetCentralVector(*dm, y));
  PetscCall(DMTaoSetVM(*dm, vm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
