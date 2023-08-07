#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

using namespace Petsc::device::cupm;

PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextCreate_CUPM<DeviceType::HIP>(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Management of HIPBLAS and HIPSOLVER handles

 Unlike CUDA, hipSOLVER is just for dense matrices so there is
 no distinguishing being dense and sparse.  Also, hipSOLVER is
 very immature so we often have to do the mapping between roc and
 cuda manually.
 */

PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t *handle)
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMBLASGetHandle<DeviceType::HIP>(handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscHIPSOLVERDnGetHandle(hipsolverDnHandle_t *handle)
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMSolverGetHandle<DeviceType::HIP>(handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}
