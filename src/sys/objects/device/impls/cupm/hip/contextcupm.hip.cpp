#include "../contextcupm.hpp" /*I "petscdevice.h" I*/

static const Petsc::cupmContextHip contextHip(PetscDeviceContextCreate_HIP);

PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext dctx)
{
  PetscDeviceContext_(HIP) *dci;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = reinterpret_cast<void*>(dci);
  ierr = PetscMemcpy(dctx->ops,&contextHip.ops,sizeof(contextHip.ops));CHKERRQ(ierr);
  ierr = PetscFree(dctx->type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(PETSCDEVICECONTEXTHIP,&dctx->type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
