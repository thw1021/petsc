#include "petscdevice.hpp"

using namespace Petsc;

#if PetscDefined(HAVE_CUDA)
static CUPMDevice<CUPMDeviceKind::CUDA> cudaDevice(PetscDeviceContextCreate_CUDA);
#endif
#if PetscDefined(HAVE_HIP)
static CUPMDevice<CUPMDeviceKind::HIP>  hipDevice(PetscDeviceContextCreate_HIP);
#endif

const char *const PetscDeviceKinds[] = {"invalid","cuda","hip","max","PetscDeviceKind","PETSC_DEVICE_",PETSC_NULLPTR};

PetscErrorCode PetscDeviceGetDevice(PetscDeviceKind kind, PetscDevice *device)
{
  PetscDevice    dev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceKind(kind,1);
  PetscValidPointer(device,2);
  ierr = PetscNew(&dev);CHKERRQ(ierr);
  dev->kind = kind;
  switch (kind) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    ierr = cudaDevice.getDefaultDevice(dev);CHKERRQ(ierr);
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    ierr = hipDevice.getDefaultDevice(dev);CHKERRQ(ierr);
    break;
#endif
  default:
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[kind],kind);
    break;
  }
  *device = dev;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceConfigure(PetscDevice device)
{
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  PetscValidDevice(device,1);
  switch (device->kind) {
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    ierr = cudaDevice.configureDevice(device);CHKERRQ(ierr);
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    ierr = hipDevice.configureDevice(device);CHKERRQ(ierr);
    break;
#endif
  default:
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[device->kind],device->kind);
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceDestroy(PetscDevice *device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*device) PetscFunctionReturn(0);
  if ((*device)->ops->destroy) {ierr = (*(*device)->ops->destroy)(*device);CHKERRQ(ierr);}
  ierr = PetscFree(device);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
