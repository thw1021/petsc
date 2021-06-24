#include "../contextcupm.hpp" /*I "petscdevice.h" I*/

using namespace Petsc;

static const cupmContextCuda contextCuda(PetscDeviceContextCreate_CUDA);

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_(CUDA) *dci;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = (void *)dci;
  ierr = PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
