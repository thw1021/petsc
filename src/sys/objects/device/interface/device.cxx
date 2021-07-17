#include "petscdevice.hpp"

using namespace Petsc;

#if PetscDefined(HAVE_CUDA)
static CUPMDevice<CUPMDeviceKind::CUDA> cudaDevice(PetscDeviceContextCreate_CUDA);
#endif
#if PetscDefined(HAVE_HIP)
static CUPMDevice<CUPMDeviceKind::HIP>  hipDevice(PetscDeviceContextCreate_HIP);
#endif

PetscErrorCode PetscDeviceGetDevice(PetscDeviceKind kind, PetscDevice *device)
{
  PetscDevice    dev;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(device,2);
  ierr = PetscNew(&dev);CHKERRQ(ierr);
  dev->kind = kind;
  switch (kind) {
  case PETSC_DEVICE_KIND_CUDA:
#if PetscDefined(HAVE_CUDA)
    ierr = cudaDevice.getDefaultDevice(dev);CHKERRQ(ierr);
    break;
#endif
  case PETSC_DEVICE_KIND_HIP:
#if PetscDefined(HAVE_HIP)
    ierr = hipDevice.getDefaultDevice(dev);CHKERRQ(ierr);
    break;
#endif
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown or invalid PetscDeviceKind %d\n",kind);
    break;
  }
  *device = dev;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceConfigure(PetscDevice device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (device->kind) {
  case PETSC_DEVICE_KIND_CUDA:
#if PetscDefined(HAVE_CUDA)
    ierr = cudaDevice.configureDevice(device);CHKERRQ(ierr);
    break;
#endif
  case PETSC_DEVICE_KIND_HIP:
#if PetscDefined(HAVE_HIP)
    ierr = hipDevice.configureDevice(device);CHKERRQ(ierr);
    break;
#endif
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown or invalid PetscDeviceKind %d\n",device->kind);
    break;
  }
  PetscFunctionReturn(0);
}
