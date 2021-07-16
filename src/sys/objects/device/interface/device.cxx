#include "petscdevice.hpp"

static Petsc::CUPMDevice<Petsc::CUPMDeviceKind::CUDA> cudaDevice;
static Petsc::CUPMDevice<Petsc::CUPMDeviceKind::HIP>  hipDevice;

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
    ierr = cudaDevice.getDefaultDevice(dev);CHKERRQ(ierr);
    break;
  case PETSC_DEVICE_KIND_HIP:
    ierr = hipDevice.getDefaultDevice(dev);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscDeviceKind %d\n",kind);
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
    ierr = cudaDevice.configureDevice(device);CHKERRQ(ierr);
    break;
  case PETSC_DEVICE_KIND_HIP:
    ierr = hipDevice.configureDevice(device);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscDeviceKind %d\n",device->kind);
    break;
  }
  PetscFunctionReturn(0);
}
