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

.seealso: PetscDeviceContextCreate(), PetscDeviceContextGetType(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextSetType(PetscDeviceContext dctx, PetscDeviceContextType type)
{
  PetscErrorCode (*create)(PetscDeviceContext);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  PetscValidCharPointer(type,2);
  ierr = PetscDeviceContextTypeCompare(dctx->type,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscDeviceContextList,type,&create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscDeviceContextType: %s",type);
  if (dctx->ops->destroy) {ierr = (*dctx->ops->destroy)(dctx);CHKERRQ(ierr);}
  ierr = PetscMemzero(dctx->ops,sizeof(struct _DeviceContextOps));CHKERRQ(ierr);
  ierr = (*create)(dctx);CHKERRQ(ierr);
  /* If we've swapped types we shouldn't be marked "setup" */
  dctx->setup = PETSC_FALSE;
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

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextGetType(PetscDeviceContext dctx, PetscDeviceContextType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidCharPointer(type,2);
  *type = dctx->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetFromOptions - Configure a PetscDeviceContext from the options database

  Collective on comm, Possibly Synchronous

  Input Parameters:
+ comm   - MPI communicator on which to query the options database
. prefix - prefix to prepend to all options database queries, NULL if not needed
- dctx   - The PetscDeviceContext to configure

  Output Parameter:
. dctx - The PetscDeviceContext

  Notes:
  Certain operations -- such as setting the type of the object -- incurs a rebuild of device-side data structures, which
  may be synchronous. Care is therefore taken in this routine not to unnecessarily trigger these changes.

  Options Database:
+ -device_context_type        - type of PetscDeviceContext to create - PetscDeviceContextSetType()
- -device_context_stream_type - type of stream to create inside the PetscDeviceContext -
  PetscDeviceContextSetStreamType()

  Level: beginner

.seealso: PetscDeviceContextSetStreamType(), PetscDeviceContextSetType()
@*/
PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm comm, const char prefix[], PetscDeviceContext dctx)
{
  char                   type[256];
  PetscBool              flag;
  PetscInt               stype;
  PetscDeviceContextType deft;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(prefix,2);
  PetscValidPointer(dctx,3);
  ierr = PetscOptionsBegin(comm,prefix,"PetscDeviceContext Options","Sys");CHKERRQ(ierr);
#if PetscDefined(HAVE_HIP)
  deft = dctx->type ? dctx->type : PETSCDEVICECONTEXTHIP;
#else
  deft = dctx->type ? dctx->type : PETSCDEVICECONTEXTCUDA;
#endif
  ierr = PetscOptionsFList("-device_context_type","PetscDeviceContext implementation type","PetscDeviceContextSetType",PetscDeviceContextList,deft,type,sizeof(type),&flag);CHKERRQ(ierr);
  if (flag) {ierr = PetscDeviceContextSetType(dctx,type);CHKERRQ(ierr);}
  ierr = PetscOptionsEList("-device_context_stream_type","PetscDeviceContext PetscStreamType","PetscDeviceContextSetStreamType",PetscStreamTypes,3,PetscStreamTypes[dctx->streamType],&stype,&flag);CHKERRQ(ierr);
  if (flag) {ierr = PetscDeviceContextSetStreamType(dctx,(PetscStreamType)stype);CHKERRQ(ierr);}
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextRegister - Adds a new PetscDeviceContext implementation

  Not Collective, Asynchronous

  Input Parameters:
+ name     - The name of a new user-defined creation routine
- function - The creation routine itself

  Notes:
  PetscDeviceContextRegister() may be called multiple times to add several user-defined implementations

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
  PetscValidCharPointer(sname,1);
  PetscValidFunction(function,2);
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
  PetscDeviceRegisterAll - Registers all the components in the PetscDevice package.

  Not Collective

  Level: developer

.seealso:  PetscDeviceContextCreate(), PetscDeviceFinalizePackage(), PetscDeviceInitializePackage(), PetscDeviceContextRegister()
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

  Developer Notes:
  This function is automatically registered to be called during PetscFinalize() by PetscDeviceInitializePackage() so
  there should be no need to call it yourself.

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
