#include "contextcupm.hpp" /*I "petscdevice.h" I*/

using namespace Petsc;

static const cupmContextHip contextHip(PetscDeviceContextCreate_HIPM);

PetscErrorCode PetscDeviceContextCreate_HIPM(PetscDeviceContext dctx)
{
  PetscDeviceContext_(HIP) *dci;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = (void *)dci;
  ierr = PetscMemcpy(dctx->ops,&contextHip.ops,sizeof(contextHip.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
