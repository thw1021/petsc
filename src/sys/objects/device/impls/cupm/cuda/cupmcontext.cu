#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

using namespace Petsc::device::cupm;

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextCreate_CUPM<DeviceType::CUDA>(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Management of CUBLAS and CUSOLVER handles */
PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t *handle)
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMBLASGetHandle<DeviceType::CUDA>(handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t *handle)
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMSolverGetHandle<DeviceType::CUDA>(handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}
