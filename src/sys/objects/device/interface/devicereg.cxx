#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

const char *const PetscStreamTypes[] = {"global_blocking","default_blocking","global_nonblocking","max","PetscStreamType","PETSC_STREAM_",PETSC_NULLPTR};

const char *const PetscDeviceContextJoinModes[] = {"destroy","sync","no_sync","PetscDeviceContextJoinMode","PETSC_DEVICE_CONTEXT_JOIN_",PETSC_NULLPTR};

static PetscBool PetscDeviceRegisterAllCalled  = PETSC_FALSE;
static PetscBool PetscDevicePackageInitialized = PETSC_FALSE;

/*@C
  PetscDeviceRegisterAll - Registers all the components in the PetscDevice package.

  Not Collective

  Level: developer

.seealso:  PetscDeviceContextCreate(), PetscDeviceFinalizePackage(), PetscDeviceInitializePackage(), PetscDeviceContextRegister()
@*/
PetscErrorCode PetscDeviceRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDeviceRegisterAllCalled) PetscFunctionReturn(0);
  PetscDeviceRegisterAllCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceFinalizePackage - This function cleans up all components of the PetscDevice package.
  It is called from PetscFinalize().

  Developer Notes:
  This function is automatically registered to be called during PetscFinalize() by PetscDeviceInitializePackage() so
  there should be no need to call it yourself.

  Level: developer

.seealso: PetscFinalize(), PetscDeviceInitializePackage()
@*/
PetscErrorCode PetscDeviceFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscDeviceRegisterAllCalled  = PETSC_FALSE;
  PetscDevicePackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceInitializePackage - This function initializes everything in the PetscDevice package. It is called from
  PetscDLLibraryRegister_petscsys() when using dynamic libraries, and on the first call to PetscDeviceContextCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), PetscDeviceFinalizePackage(), PetscDeviceContextCreate()
@*/
PetscErrorCode PetscDeviceInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PETSC_DEVICE_DEFAULT == PETSC_DEVICE_INVALID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"No suitable PetscDeviceKind found, must configure PETSc with a device backend enabled");
  if (PetscDevicePackageInitialized) PetscFunctionReturn(0);
  PetscDevicePackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscDeviceFinalizePackage);CHKERRQ(ierr);
  ierr = PetscDeviceRegisterAll();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
