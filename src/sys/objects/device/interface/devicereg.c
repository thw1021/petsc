#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

const char *const PetscStreamTypes[] = {"global_blocking","default_blocking","global_nonblocking","MAX_TYPE","PetscStreamType","PETSC_STREAM_",NULL};

const char *const PetscDeviceContextJoinModes[] = {"destroy","sync","no_sync","PetscDeviceContextJoinMode","PETSC_DEVICE_CONTEXT_JOIN_",NULL};

static PetscFunctionList PetscDeviceContextList        = NULL;
static PetscBool         PetscDeviceRegisterAllCalled  = PETSC_FALSE;
static PetscBool         PetscDevicePackageInitialized = PETSC_FALSE;

/*@C
  PetscDeviceContextSetType - Builds a PetscDeviceContext for a particular implementation

  Not Collective, Synchronous on PetscDeviceContext

  Input Parameters:
+ dctx - The PetscDeviceContext object
- type - The PetscDeviceContextType

  Notes:
  See "include/petscdevicetypes.h" for available context types. When converting types this routine should be considered
  synchronous. It is possible that during destruction of the previous type it deallocates device-side memory which is
  synchronous.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextGetType()
@*/
PetscErrorCode PetscDeviceContextSetType(PetscDeviceContext dctx, PetscDeviceContextType type)
{
  PetscErrorCode (*create)(PetscDeviceContext);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!type)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscDeviceContext to NULL type");
  ierr = PetscDeviceContextTypeCompare(dctx->type,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscDeviceContextList,type,&create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscDeviceContextType: %s",type);
  if (dctx->ops->destroy) {ierr = (*dctx->ops->destroy)(dctx);CHKERRQ(ierr);}
  ierr = PetscMemzero(dctx->ops,sizeof(struct _DeviceContextOps));CHKERRQ(ierr);
  ierr = (*create)(dctx);CHKERRQ(ierr);
  ierr = PetscFree(dctx->type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&dctx->type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetType - Gets the typename of a PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameter:
. strm - The PetscDeviceContext object

  Output Parameter:
. type - The PetscDeviceContextType

  Notes:
  See "include/petscdevicetypes.h" for available stream types.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType()
@*/
PetscErrorCode PetscDeviceContextGetType(PetscDeviceContext dctx, PetscDeviceContextType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(type,2);
  *type = dctx->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextRegister - Adds a new PetscDeviceContext implementation

  Not Collective, Asynchronous

  Input Parameters:
+ name     - The name of a new user-defined creation routine
- function - The creation routine itself

  Notes:
  PetscDeviceContextRegister() may be called multiple times to add several user-defined vectors

  Sample usage:
.vb
    PetscDeviceContextRegister("my_pdc_name",MyPetscDeviceContextCreate);
.ve

  Then, your PetscDeviceContext type can be chosen with the procedural interface via
.vb
    PetscDeviceContextCreate(PetscDeviceContext *);
    PetscDeviceContextSetType(PetscDeviceContext,"my_pdc_name");
.ve

  Level: advanced

.seealso: PetscDeviceContextRegisterAll()
@*/
PetscErrorCode PetscDeviceContextRegister(const char sname[], PetscErrorCode (*function)(PetscDeviceContext))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscDeviceContextList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext);
#endif

/*@C
  PetscDeviceRegisterAll - Registers all of the stream components in the PetscDevice package.

  Not Collective

  Level: developer

.seealso:  PetscDeviceContextCreate(), PetscDeviceFinalizePackage()
@*/
PetscErrorCode PetscDeviceRegisterAll(void)
{
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  if (PetscDeviceRegisterAllCalled) PetscFunctionReturn(0);
  PetscDeviceRegisterAllCalled = PETSC_TRUE;
#if PetscDefined(HAVE_CUDA)
  ierr = PetscDeviceContextRegister(PETSCDEVICECONTEXTCUDA,PetscDeviceContextCreate_CUDA);CHKERRQ(ierr);
#endif
#if PetscDefined(HAVE_HIP)
  ierr = PetscDeviceContextRegister(PETSCDEVICECONTEXTHIP,PetscDeviceContextCreate_HIP);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceFinalizePackage - This function cleans up all components of the PetscDevice package.
  It is called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize(), PetscDeviceInitializePackage()
@*/
PetscErrorCode PetscDeviceFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscDeviceContextList);CHKERRQ(ierr);
  PetscDeviceRegisterAllCalled  = PETSC_FALSE;
  PetscDevicePackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceInitializePackage - This function initializes everything in the PetscDevice package. It is called from
  PetscDLLibraryRegister_petscsys() when using dynamic libraries, and on the first call to PetscEventCreate(),
  PetscStreamCreate(), or PetscDeviceContextCreate() when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), PetscDeviceFinalizePackage()
@*/
PetscErrorCode PetscDeviceInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscDevicePackageInitialized) PetscFunctionReturn(0);
  PetscDevicePackageInitialized = PETSC_TRUE;
  ierr = PetscDeviceRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscDeviceFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
