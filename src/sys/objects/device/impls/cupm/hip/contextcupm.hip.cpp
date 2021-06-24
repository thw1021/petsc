#include "../contextcupm.hpp" /*I "petscdevice.h" I*/

using namespace Petsc;

static const cupmContextHip contextHip(PetscDeviceContextCreate_HIP);

PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext dctx)
{
  PetscDeviceContext_(HIP) *dci;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = (void *)dci;
  ierr = PetscMemcpy(dctx->ops,&contextHip.ops,sizeof(contextHip.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
