#include "cupmdevice.hpp"

using namespace Petsc;

#if PetscDefined(HAVE_CUDA)
static CUPMDevice<CUPMDeviceKind::CUDA> cudaDevice(PetscDeviceContextCreate_CUDA);
#endif
#if PetscDefined(HAVE_HIP)
static CUPMDevice<CUPMDeviceKind::HIP>  hipDevice(PetscDeviceContextCreate_HIP);
#endif

const char *const PetscDeviceKinds[] = {"invalid","cuda","hip","max","PetscDeviceKind","PETSC_DEVICE_",PETSC_NULLPTR};

/*@C
  PetscDeviceGetDevice - Get a handle for a particular device kind

  Not Collective, Possibly Synchronous

  Input Parameter:
. kind - The kind of PetscDevice

  Output Parameter:
. device - The PetscDevice

  Notes:
  If this is the first time that a PetscDevice is created, this routine may initialize
  the corresponding backend. If this is the case, this will cause a device
  synchronization.

  Level: beginner

.seealso: PetscDeviceConfigure(), PetscDeviceDestroy()
@*/
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
    ierr = cudaDevice.getDevice(dev);CHKERRQ(ierr);
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    ierr = hipDevice.getDevice(dev);CHKERRQ(ierr);
    break;
#endif
  case PETSC_DEVICE_INVALID:
    break;
  default:
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[kind],kind);
    break;
  }
  *device = dev;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceConfigure - Configure a particular PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice to Configure

  Developer Notes:
  Currently a no-op

  Level: developer

.seealso: PetscDeviceGetDevice(), PetscDeviceDestroy()
@*/
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
  case PETSC_DEVICE_INVALID:
    break;
  default:
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[device->kind],device->kind);
    break;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceDestroy - Free a PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice

  Level: beginner

.seealso: PetscDeviceGetDevice(), PetscDeviceConfigure()
@*/
PetscErrorCode PetscDeviceDestroy(PetscDevice *device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*device) PetscFunctionReturn(0);
  if ((*device)->ops->destroy) {ierr = (*(*device)->ops->destroy)(*device);CHKERRQ(ierr);}
  ierr = PetscFree(device);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 1 default device for every petsc device kind (even invalid) */
static PetscDevice defaultDevice[PETSC_DEVICE_MAX];

/* so you don't have to repeat yourself for every kind */
static PetscErrorCode InitializeDeviceHelper_Internal(PetscDeviceKind kind)
{
  const int      kindIdx = static_cast<int>(kind);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(NULL,"Initializing default %s PetscDevice\n",PetscDeviceKinds[kindIdx]);CHKERRQ(ierr);
  ierr = PetscDeviceGetDevice(kind,defaultDevice+kindIdx);CHKERRQ(ierr);
  ierr = PetscDeviceConfigure(defaultDevice[kindIdx]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* called in PetscFinalize(), don't call this yourself! */
PetscErrorCode PetscDeviceFinalizeDefaultDevices_Internal(void)
{
  const int      maxIdx = static_cast<int>(PETSC_DEVICE_MAX);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (int i = 0; i < maxIdx; ++i) {
    ierr = PetscDeviceDestroy(defaultDevice+i);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* called in PetscInitialize(), don't call this yourself! */
PetscErrorCode PetscDeviceInitializeDefaultDevices_Internal(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscRegisterFinalize(PetscDeviceFinalizeDefaultDevices_Internal);CHKERRQ(ierr);
  defaultDevice[static_cast<int>(PETSC_DEVICE_INVALID)] = PETSC_NULLPTR;
  if (PetscDefined(HAVE_CUDA)) {
    ierr = InitializeDeviceHelper_Internal(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  } else {
    defaultDevice[static_cast<int>(PETSC_DEVICE_CUDA)] = PETSC_NULLPTR;
  }
  if (PetscDefined(HAVE_HIP)) {
    ierr = InitializeDeviceHelper_Internal(PETSC_DEVICE_HIP);CHKERRQ(ierr);
  } else {
    defaultDevice[static_cast<int>(PETSC_DEVICE_HIP)] = PETSC_NULLPTR;
  }
  PetscFunctionReturn(0);
}

/* Get the default PetscDevice for a particular kind, usually one should use
   PetscDeviceDefault_Internal() since that will return the automatically selected
   default kind. */
PetscDevice PetscDeviceDefaultKind_Internal(PetscDeviceKind kind)
{
  return defaultDevice[static_cast<int>(kind)];
}
