#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

static PetscFunctionList PetscEventList                = NULL;
static PetscFunctionList PetscDeviceContextList        = NULL;
static PetscBool         PetscDeviceRegisterAllCalled  = PETSC_FALSE;
static PetscBool         PetscDevicePackageInitialized = PETSC_FALSE;

/*@C
  PetscEventSetType - Builds a PetscEvent for a particular stream implementation

  Not Collective

  Input Parameters:
+ event - The PetscEvent object
- type - The PetscStream type

  Notes:
  See "petsc/include/petscdevice.h" for available stream types

  Level: intermediate

.seealso: PetscEventCreate(), PetscEventGetType()
@*/
PetscErrorCode PetscEventSetType(PetscEvent event, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscEvent);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!type)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscEvent to NULL type");
  ierr = PetscStreamTypeCompare(event->type,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscEventList,type,&create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscEvent type: %s",type);
  if (event->ops->destroy) {ierr = (*event->ops->destroy)(event);CHKERRQ(ierr);}
  ierr = PetscMemzero(event->ops,sizeof(struct _EventOps));CHKERRQ(ierr);
  ierr = (*create)(event);CHKERRQ(ierr);
  ierr = PetscFree(event->type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&event->type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscEventGetType - Gets the typename of a PetscEvent

  Not Collective

  Input Parameter:
. event - The PetscEvent object

  Output Parameter:
. type - The PetscStream type

  Level: intermediate

.seealso: PetscEventCreate(), PetscEventSetType()
@*/
PetscErrorCode PetscEventGetType(PetscEvent event, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  PetscValidPointer(type,2);
  *type = event->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetType - Builds a PetscDeviceContext for a particular stream implementation

  Not Collective

  Input Parameters:
+ dctx - The PetscDeviceContext object
- type - The PetscStreamType

  Notes:
  See "include/petscdevicetypes.h" for available stream types.

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextGetType()
@*/
PetscErrorCode PetscDeviceContextSetType(PetscDeviceContext dctx, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscDeviceContext);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!type)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscDeviceContext to NULL type");
  ierr = PetscStreamTypeCompare(dctx->type,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscDeviceContextList,type,&create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscStreamType: %s",type);
  if (dctx->ops->destroy) {ierr = (*dctx->ops->destroy)(dctx);CHKERRQ(ierr);}
  ierr = PetscMemzero(dctx->ops,sizeof(struct _DeviceContextOps));CHKERRQ(ierr);
  ierr = (*create)(dctx);CHKERRQ(ierr);
  ierr = PetscFree(dctx->type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&dctx->type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetType - Gets the typename of a PetscDeviceContext

  Not Collective

  Input Parameter:
. strm - The PetscDeviceContext object

  Output Parameter:
. type - The PetscStreamType

  Notes:
  See "include/petscdevicetypes.h" for available stream types.

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType()
@*/
PetscErrorCode PetscDeviceContextGetType(PetscDeviceContext dctx, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(type,2);
  *type = dctx->type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventRegister(const char sname[], PetscErrorCode (*function)(PetscEvent))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscEventList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextRegister(const char sname[], PetscErrorCode (*function)(PetscDeviceContext))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscDeviceContextList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscEventCreate_CUDA(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_EXTERN PetscErrorCode PetscEventCreate_HIP(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext);
#endif

/*@C
  PetscDeviceRegisterAll - Registers all of the stream components in the PetscDevice package.

  Not Collective

  Level: advanced

.seealso:  PetscEventCreate(), PetscStreamCreate(), PetscDeviceContextCreate(), PetscDeviceFinalizePackage()
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
  ierr = PetscEventRegister(PETSCSTREAMCUDA,PetscEventCreate_CUDA);CHKERRQ(ierr);
  ierr = PetscDeviceContextRegister(PETSCSTREAMCUDA,PetscDeviceContextCreate_CUDA);CHKERRQ(ierr);
#endif
#if PetscDefined(HAVE_HIP)
  ierr = PetscEventRegister(PETSCSTREAMHIP,PetscEventCreate_HIP);CHKERRQ(ierr);
  ierr = PetscDeviceContextRegister(PETSCSTREAMHIP,PetscDeviceContextCreate_HIP);CHKERRQ(ierr);
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
  ierr = PetscFunctionListDestroy(&PetscEventList);CHKERRQ(ierr);
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
PetscErrorCode PetsDeviceInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscDevicePackageInitialized) PetscFunctionReturn(0);
  PetscDevicePackageInitialized = PETSC_TRUE;
  ierr = PetscDeviceRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscDeviceFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscEventSetFromOptions - Configures a PetscEvent from the options database.

  Collective on comm

  Input Parameters:
+ comm - The communicator on which to query the options database
. prefix - Optional prefix to prepend to all queries using this call
- event - The PetscEvent

  Options Database Keys:
+ -stream_type <type> - cuda, hip, see PetscStreamType for complete list
. -event_create_flag <int> - Flags for special event behavior, such as disabling timing. See PetscEventSetFlags() for
more information
- -event_wait_flag <int> - Flags for special wait-on-event behavior. See PetscEventSetFlags() for more information

  Notes:
  Must be called after creating the PetscEvent, but before the PetscEvent is used. Run with -help to see all available
  options for a particular stream type.

  Level: beginner

.seealso: PetscEventCreate(), PetscEventSetType(), PetscEventSetFlags()
@*/
PetscErrorCode PetscEventSetFromOptions(MPI_Comm comm, const char prefix[], PetscEvent event)
{
  PetscErrorCode  ierr;
  PetscStreamType defaultType;

  PetscFunctionBegin;
  if (event->setfromoptionscalled) PetscFunctionReturn(0);
  event->setfromoptionscalled = PETSC_TRUE;
  if (event->type) {defaultType = event->type;}
  else {
#if PetscDefined(HAVE_CUDA)
    defaultType = PETSCSTREAMCUDA;
#elif PetscDefined(HAVE_HIP)
    defaultType = PETSCSTREAMHIP;
#else
    SETERRQ(comm,PETSC_ERR_SUP,"No suitable default stream type exists");
    defaultType = "invalidType";
#endif
  }
  {
    PetscBool opt;
    char      typeName[256];

    ierr = PetscOptionsBegin(comm,prefix,"PetscEvent Options","Sys");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-event_type","PetscStream type","PetscEventSetType",PetscEventList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
    ierr = PetscEventSetType(event,opt ? typeName : defaultType);CHKERRQ(ierr);
    if (event->ops->setfromoptions) {
      ierr = (*event->ops->setfromoptions)(PetscOptionsObject,event);CHKERRQ(ierr);
    }
    /* Use PetscOptionsRangeInt since the flag variables are unsigned */
    ierr = PetscOptionsRangeInt("-event_create_flag","PetscEvent creation flag","PetscEventSetFlags",(PetscInt)event->eventFlags,(PetscInt*)&event->eventFlags,NULL,0,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = PetscOptionsRangeInt("-event_wait_flag","PetscEvent wait flag","PetscEventSetFlags",(PetscInt)event->waitFlags,(PetscInt*)&event->waitFlags,NULL,0,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  ierr = PetscEventSetUp(event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
