#include "../contextcupm.hpp" /*I "petscdevice.h" I*/

static const Petsc::cupmContextCuda contextCuda(PetscDeviceContextCreate_CUDA);

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  PetscDeviceContext_(CUDA) *dci;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = reinterpret_cast<void*>(dci);
  ierr = PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops));CHKERRQ(ierr);
  ierr = PetscFree(dctx->type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(PETSCDEVICECONTEXTCUDA,&dctx->type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
