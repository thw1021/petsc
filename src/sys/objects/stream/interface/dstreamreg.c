#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

static PetscFunctionList PetscStreamList                     = NULL;
static PetscFunctionList PetscEventList                      = NULL;
static PetscFunctionList PetscStreamScalarList               = NULL;
static PetscFunctionList PetscStreamGraphList                = NULL;
static PetscBool         PetscStreamRegisterAllCalled        = PETSC_FALSE;
static PetscBool         PetscEventRegisterAllCalled         = PETSC_FALSE;
static PetscBool         PetscStreamScalarRegisterAllCalled  = PETSC_FALSE;
static PetscBool         PetscStreamGraphRegisterAllCalled   = PETSC_FALSE;
static PetscBool         PetscStreamPackageInitialized       = PETSC_FALSE;
static PetscBool         PetscEventPackageInitialized        = PETSC_FALSE;
static PetscBool         PetscStreamScalarPackageInitialized = PETSC_FALSE;
static PetscBool         PetscStreamGraphPackageInitialized  = PETSC_FALSE;

/*@C
  PetscStreamSetType - Builds a PetscStream for a particular stream implementation

  Not Collective

  Input Parameters:
+ strm - The PetscStream object
- type - The PetscStream type

  Notes:
  See "petsc/include/petscdevice.h" for available stream types.

  Level: intermediate

.seealso: PetscStreamCreate(), PetscStreamGetType()
@*/
PetscErrorCode PetscStreamSetType(PetscStream strm, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscStream);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!type)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscStream to NULL type");
  if (PetscUnlikelyDebug(strm->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change type on already setup PetscStream");
  ierr = PetscStreamTypeCompare(strm->type,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscStreamList,type,&create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStream type: %s",type);
  if (strm->ops->destroy) {ierr = (*strm->ops->destroy)(strm);CHKERRQ(ierr);}
  ierr = PetscMemzero(strm->ops,sizeof(struct _StreamOps));CHKERRQ(ierr);
  ierr = (*create)(strm);CHKERRQ(ierr);
  ierr = PetscFree(strm->type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&strm->type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGetType - Gets the typename of a PetscStream

  Not Collective

  Input Parameter:
. strm - The PetscStream object

  Output Parameter:
. type - The PetscStream type

  Level: intermediate

.seealso: PetscStreamCreate(), PetscStreamSetType()
@*/
PetscErrorCode PetscStreamGetType(PetscStream strm, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(type,2);
  *type = strm->type;
  PetscFunctionReturn(0);
}

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
  PetscStreamScalarSetType - Builds a PetscStreamScalar for a particular stream implementation

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
- type - The PetscStream type

  Notes:
  See "petsc/include/petscdevice.h" for available stream types

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamScalarGetType()
@*/
PetscErrorCode PetscStreamScalarSetType(PetscStreamScalar pscal, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscStreamScalar);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!type)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscEvent to NULL type");
  ierr = PetscStreamTypeCompare(pscal->type,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscStreamScalarList,type,&create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStreamScalar type: %s",type);
  if (pscal->ops->destroy) {ierr = (*pscal->ops->destroy)(pscal);CHKERRQ(ierr);}
  ierr = PetscMemzero(pscal->ops,sizeof(struct _ScalOps));CHKERRQ(ierr);
  ierr = (*create)(pscal);CHKERRQ(ierr);
  ierr = PetscFree(pscal->type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&pscal->type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarGetType - Gets the typename of a PetscStreamScalar

  Not Collective

  Input Parameter:
. pscal - The PetscStreamScalar object

  Output Parameter:
. type - The PetscStream type

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType()
@*/
PetscErrorCode PetscStreamScalarGetType(PetscStreamScalar pscal, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(pscal,1);
  PetscValidPointer(type,2);
  *type = pscal->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphSetType - Builds a PetscStreamGraph for a particular stream implementation

  Not Collective

  Input Parameters:
+ sgraph - The PetscStreamGraph object
- type - The PetscStream type

  Notes:
  See "petsc/include/petscdevice.h" for available stream types

  Level: intermediate

.seealso: PetscStreamGraphCreate(), PetscStreamGraphGetType()
@*/
PetscErrorCode PetscStreamGraphSetType(PetscStreamGraph sgraph, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscStreamGraph);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!type)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscStreamGraph to NULL type");
  ierr = PetscStreamTypeCompare(sgraph->type,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscStreamGraphList,type,&create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStreamGraph type: %s",type);
  if (sgraph->ops->destroy) {ierr = (*sgraph->ops->destroy)(sgraph);CHKERRQ(ierr);}
  ierr = PetscMemzero(sgraph->ops,sizeof(struct _GraphOps));CHKERRQ(ierr);
  ierr = (*create)(sgraph);CHKERRQ(ierr);
  ierr = PetscFree(sgraph->type);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&sgraph->type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphGetType - Gets the typename of a PetscStreamGraph

  Not Collective

  Input Parameter:
. sgraph - The PetscStreamGraph object

  Output Parameter:
. type - The PetscStream type

  Level: intermediate

.seealso: PetscStreamGraphCreate(), PetscStreamGraphSetType()
@*/
PetscErrorCode PetscStreamGraphGetType(PetscStreamGraph sgraph, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(sgraph,1);
  PetscValidPointer(type,2);
  *type = sgraph->type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRegister(const char sname[], PetscErrorCode (*function)(PetscStream))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscStreamList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventRegister(const char sname[], PetscErrorCode (*function)(PetscEvent))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscEventList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarRegister(const char sname[], PetscErrorCode (*function)(PetscStreamScalar))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscStreamScalarList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphRegister(const char sname[], PetscErrorCode (*function)(PetscStreamGraph))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscStreamGraphList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscStreamCreate_CUDA(PetscStream);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_EXTERN PetscErrorCode PetscStreamCreate_HIP(PetscStream);
#endif

/*@C
  PetscStreamRegisterAll - Registers all of the stream components in the PetscStream package.

  Not Collective

  Level: advanced

.seealso:  PetscStreamCreate(), PetscStreamSetType(), PetscStreamGetType()
@*/
PetscErrorCode PetscStreamRegisterAll(void)
{
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  if (PetscStreamRegisterAllCalled) PetscFunctionReturn(0);
  PetscStreamRegisterAllCalled = PETSC_TRUE;
#if PetscDefined(HAVE_CUDA)
  ierr = PetscStreamRegister(PETSCSTREAMCUDA,PetscStreamCreate_CUDA);CHKERRQ(ierr);
#endif
#if PetscDefined(HAVE_HIP)
  ierr = PetscStreamRegister(PETSCSTREAMHIP,PetscStreamCreate_HIP);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscEventCreate_CUDA(PetscEvent);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_EXTERN PetscErrorCode PetscEventCreate_HIP(PetscEvent);
#endif

/*@C
  PetscEventRegisterAll - Registers all of the event components in the PetscStream package.

  Not Collective

  Level: advanced

.seealso:  PetscEventCreate(), PetscEventSetType(), PetscEventGetType()
@*/
PetscErrorCode PetscEventRegisterAll(void)
{
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  if (PetscEventRegisterAllCalled) PetscFunctionReturn(0);
  PetscEventRegisterAllCalled = PETSC_TRUE;
#if PetscDefined(HAVE_CUDA)
  ierr = PetscEventRegister(PETSCSTREAMCUDA,PetscEventCreate_CUDA);CHKERRQ(ierr);
#endif
#if PetscDefined(HAVE_HIP)
  ierr = PetscEventRegister(PETSCSTREAMHIP,PetscEventCreate_HIP);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate_CUDA(PetscStreamScalar);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate_HIP(PetscStreamScalar);
#endif

/*@C
  PetscStreamScalarRegisterAll - Registers all of the stream scalar components in the PetscStream package.

  Not Collective

  Level: advanced

.seealso:  PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamScalarGetType()
@*/
PetscErrorCode PetscStreamScalarRegisterAll(void)
{
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
  if (PetscStreamScalarRegisterAllCalled) PetscFunctionReturn(0);
  PetscStreamScalarRegisterAllCalled = PETSC_TRUE;
#if PetscDefined(HAVE_CUDA)
  ierr = PetscStreamScalarRegister(PETSCSTREAMCUDA,PetscStreamScalarCreate_CUDA);CHKERRQ(ierr);
#endif
#if PetscDefined(HAVE_HIP)
  ierr = PetscStreamScalarRegister(PETSCSTREAMHIP,PetscStreamScalarCreate_HIP);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscStreamGraphCreate_CUDA(PetscStreamGraph);
#endif

/*@C
  PetscStreamGraphRegisterAll - Registers all of the stream graph components in the PetscStream package.

  Not Collective

  Level: advanced

.seealso:  PetscStreamGraphCreate(), PetscStreamGraphSetType(), PetscStreamGraphGetType()
@*/
PetscErrorCode PetscStreamGraphRegisterAll(void)
{
#if PetscDefined(HAVE_CUDA)
  PetscErrorCode ierr;
#endif

  if (PetscStreamGraphRegisterAllCalled) PetscFunctionReturn(0);
  PetscStreamGraphRegisterAllCalled = PETSC_TRUE;
#if PetscDefined(HAVE_CUDA)
  ierr = PetscStreamGraphRegister(PETSCSTREAMCUDA,PetscStreamGraphCreate_CUDA);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamFinalizePackage - This function cleans up all components of the PetscStream ppacakge.
  It is called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize(), PetscStreamInitializePackage()
@*/
PetscErrorCode PetscStreamFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscStreamList);CHKERRQ(ierr);
  PetscStreamRegisterAllCalled = PETSC_FALSE;
  PetscStreamPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamInitializePackage - This function initializes everything in the PetscStream package. It is called from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to PetscStreamCreate() when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), PetscStreamFinalizePackage()
@*/
PetscErrorCode PetscStreamInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscStreamPackageInitialized) PetscFunctionReturn(0);
  PetscStreamPackageInitialized = PETSC_TRUE;
  ierr = PetscStreamRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscStreamFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscEventFinalizePackage - This function cleans up all components of the PetscEvent ppacakge.
  It is called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize(), PetscEventInitializePackage()
@*/
PetscErrorCode PetscEventFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscEventList);CHKERRQ(ierr);
  PetscEventRegisterAllCalled = PETSC_FALSE;
  PetscEventPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventInitializePackage - This function initializes everything in the PetscEvent package. It is called from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to PetscEventCreate() when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), PetscEventFinalizePackage()
@*/
PetscErrorCode PetscEventInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscEventPackageInitialized) PetscFunctionReturn(0);
  PetscEventPackageInitialized = PETSC_TRUE;
  ierr = PetscEventRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscEventFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarFinalizePackage - This function cleans up all components of the PetscStreamScalar ppacakge.
  It is called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize(), PetscStreamScalarInitializePackage()
@*/
PetscErrorCode PetscStreamScalarFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscStreamScalarList);CHKERRQ(ierr);
  PetscStreamScalarRegisterAllCalled = PETSC_FALSE;
  PetscStreamScalarPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarInitializePackage - This function initializes everything in the PetscStreamScalar package. It is called from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to PetscStreamScalarCreate() when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), PetscStreamScalarFinalizePackage()
@*/
PetscErrorCode PetscStreamScalarInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscStreamScalarPackageInitialized) PetscFunctionReturn(0);
  PetscStreamScalarPackageInitialized = PETSC_TRUE;
  ierr = PetscStreamScalarRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscStreamScalarFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphFinalizePackage - This function cleans up all components of the PetscStreamGraph ppacakge.
  It is called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize(), PetscStreamGraphInitializePackage()
@*/
PetscErrorCode PetscStreamGraphFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscStreamGraphList);CHKERRQ(ierr);
  PetscStreamGraphRegisterAllCalled = PETSC_FALSE;
  PetscStreamGraphPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphInitializePackage - This function initializes everything in the PetscStreamGraph package. It is called from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to PetscStreamGraphCreate() when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), PetscStreamGraphFinalizePackage()
@*/
PetscErrorCode PetscStreamGraphInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscStreamGraphPackageInitialized) PetscFunctionReturn(0);
  PetscStreamGraphPackageInitialized = PETSC_TRUE;
  ierr = PetscStreamGraphRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscStreamGraphFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamSetFromOptions - Configures a PetscStream from the options database.

  Collective on comm

  Input Parameters:
+ comm - The communicator on which to query the options database
. prefix - Optional prefix to prepend to all queries using this call
- strm - The PetscStream

  Options Database Keys:
+ -stream_type <type> - cuda, hip, see PetscStreamType for complete list
- -stream_mode <mode> - global_blocking, default_blocking, global_nonblocking, see PetscStreamMode for complete list

  Notes:
  Must be called after creating the PetscStream, but before the PetscStream is used. Run with -help to see all available
  options for a particular stream type.

  Level: beginner

.seealso: PetscStreamCreate(), PetscStreamSetMode(), PetscStreamSetType()
@*/
PetscErrorCode PetscStreamSetFromOptions(MPI_Comm comm, const char prefix[], PetscStream strm)
{
  PetscErrorCode  ierr;
  PetscStreamType defaultType;

  PetscFunctionBegin;
  if (strm->setfromoptionscalled) PetscFunctionReturn(0);
  strm->setfromoptionscalled = PETSC_TRUE;
  if (strm->type) {defaultType = strm->type;}
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
    PetscInt  idx;
    char      typeName[256];

    ierr = PetscOptionsBegin(comm,prefix,"PetscStream Options","Sys");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-stream_mode","PetscStream mode","PetscStreamSetMode",PetscStreamModes,PETSC_STREAM_MAX_MODE,PetscStreamModes[strm->mode],&idx,&opt);CHKERRQ(ierr);
    if (opt) {ierr = PetscStreamSetMode(strm,(PetscStreamMode)idx);CHKERRQ(ierr);}
    ierr = PetscOptionsFList("-stream_type","PetscStream type","PetscStreamSetType",PetscStreamList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
    ierr = PetscStreamSetType(strm,opt ? typeName : defaultType);CHKERRQ(ierr);
    if (strm->ops->setfromoptions) {
      ierr = (*strm->ops->setfromoptions)(PetscOptionsObject,strm);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  ierr = PetscStreamSetUp(strm);CHKERRQ(ierr);
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
