#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

const char *const PetscStreamTypes[] = {"global_blocking","default_blocking","global_nonblocking","max","PetscStreamType","PETSC_STREAM_",PETSC_NULLPTR};

const char *const PetscDeviceContextJoinModes[] = {"destroy","sync","no_sync","PetscDeviceContextJoinMode","PETSC_DEVICE_CONTEXT_JOIN_",PETSC_NULLPTR};

static PetscBool PetscDeviceRegisterAllCalled  = PETSC_FALSE;
static PetscBool PetscDevicePackageInitialized = PETSC_FALSE;

/*@C
  PetscDeviceContextSetFromOptions - Configure a PetscDeviceContext from the options database

  Collective on comm, Asynchronous

  Input Parameters:
+ comm   - MPI communicator on which to query the options database
. prefix - prefix to prepend to all options database queries, NULL if not needed
- dctx   - The PetscDeviceContext to configure

  Output Parameter:
. dctx - The PetscDeviceContext

  Options Database:
. -device_context_stream_type - type of stream to create inside the PetscDeviceContext -
  PetscDeviceContextSetStreamType()

  Level: beginner

.seealso: PetscDeviceContextSetStreamType()
@*/
PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm comm, const char prefix[], PetscDeviceContext dctx)
{
  PetscBool      flag;
  PetscInt       stype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (prefix) {PetscValidCharPointer(prefix,2);}
  PetscValidDeviceContext(dctx,3);
  ierr = PetscOptionsBegin(comm,prefix,"PetscDeviceContext Options","Sys");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-device_context_stream_type","PetscDeviceContext PetscStreamType","PetscDeviceContextSetStreamType",PetscStreamTypes,3,PetscStreamTypes[dctx->streamType],&stype,&flag);CHKERRQ(ierr);
  if (flag) {ierr = PetscDeviceContextSetStreamType(dctx,(PetscStreamType)stype);CHKERRQ(ierr);}
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  if (PetscDevicePackageInitialized) PetscFunctionReturn(0);
  PetscDevicePackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscDeviceFinalizePackage);CHKERRQ(ierr);
  ierr = PetscDeviceRegisterAll();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
