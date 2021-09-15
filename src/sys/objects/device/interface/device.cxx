#include "cupmdevice.hpp" /* I "petscdevice.h" */

using namespace Petsc;

#if PetscDefined(HAVE_CUDA)
static CUPMDevice<CUPMDeviceKind::CUDA> CUDADevice(PetscDeviceContextCreate_CUDA);
#endif
#if PetscDefined(HAVE_HIP)
static CUPMDevice<CUPMDeviceKind::HIP>  HIPDevice(PetscDeviceContextCreate_HIP);
#endif

const char *const PetscDeviceKinds[] = {"invalid","cuda","hip","default","max","PetscDeviceKind","PETSC_DEVICE_",PETSC_NULLPTR};

const char *const PetscDeviceInitKinds[] = {"none","lazy","greedy","PetscDeviceInitKind","PETSC_DEVICE_INIT_",PETSC_NULLPTR};
static_assert(sizeof(PetscDeviceInitKinds)/sizeof(*PetscDeviceInitKinds) == 6,"Must change CUPMDevice<T>::initialize number of enum values in -device_enable_cupm to match!");

#define CAT_(a,...) a ## __VA_ARGS__
#define CAT(a,...)  CAT_(a,__VA_ARGS__)

/* Need to do the ugly ## directly here rather than use macro since I guess PetscDefined
 * doesn't have an initial indirection layer? */
#define PETSC_DEVICE_DEFAULT_CASE(comm,kind) SETERRQ1(comm,PETSC_ERR_PLIB,"PETSc was seeminly configured for PetscDeviceKind %s but we've fallen through all cases in a switch",PetscDeviceKinds[kind])

#define PETSC_DEVICE_CASE_IF_PETSC_DEFINED(IMPLS,...)                   \
  CAT(PETSC_DEVICE_CASE_IF_PETSC_DEFINED_,PetscDefined(HAVE_##IMPLS))(IMPLS,__VA_ARGS__)

#define PETSC_DEVICE_CASE_IF_PETSC_DEFINED_0(IMPLS,func,...)
#define PETSC_DEVICE_CASE_IF_PETSC_DEFINED_1(IMPLS,func,...)            \
  case CAT(PETSC_DEVICE_,IMPLS):                                        \
  {                                                                     \
    PetscErrorCode ierr;                                                \
    ierr = CAT(IMPLS,Device).func(__VA_ARGS__);CHKERRQ(ierr);           \
    break;                                                              \
  }

/*@C
  PetscDeviceCreate - Get a new handle for a particular device kind

  Not Collective, Possibly Synchronous

  Input Parameter:
. kind  - The kind of PetscDevice
. devid - The numeric ID# of the device (pass PETSC_DECIDE to assign automatically)

  Output Parameter:
. device - The PetscDevice

  Notes:
  If this is the first time that a PetscDevice is created, this routine may initialize
  the corresponding backend. If this is the case, this will most likely cause some sort of
  device synchronization.

  devid is what you might pass to cudaSetDevice() for example.

  Level: beginner

.seealso: PetscDeviceConfigure(), PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceCreate(PetscDeviceKind kind, PetscInt devid, PetscDevice *device)
{
  static PetscInt PetscDeviceCounter = 0;
  PetscDevice     dev;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidDeviceKind(kind,1);
  PetscValidPointer(device,3);
  ierr = PetscDeviceInitializePackage();CHKERRQ(ierr);
  ierr = PetscNew(&dev);CHKERRQ(ierr);
  dev->id   = PetscDeviceCounter++;
  dev->kind = kind;
  /* if you are adding a device, you also need to add it's initialization in
     PetscDeviceInitializeKind_Internal below */
  switch (kind) {
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,getDevice,dev,devid);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HIP,getDevice,dev,devid);
  default:
    PETSC_DEVICE_DEFAULT_CASE(PETSC_COMM_SELF,kind);
    break;
  }
  *device = dev;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceDestroy - Free a PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice

  Level: beginner

.seealso: PetscDeviceCreate(), PetscDeviceConfigure()
@*/
PetscErrorCode PetscDeviceDestroy(PetscDevice *device)
{
  PetscFunctionBegin;
  if (!*device) PetscFunctionReturn(0);
  if (!--(*device)->refcnt) {
    PetscErrorCode ierr;

    if (PetscUnlikelyDebug((*device)->refcnt < 0)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDevice %D reference count %D < 0",(*device)->id,(*device)->refcnt);
    ierr = PetscFree((*device)->data);CHKERRQ(ierr);
    ierr = PetscFree(*device);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceConfigure - Configure a particular PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice to configure

  Notes:
  The user should not assume that this is a cheap operation

  Level: developer

.seealso: PetscDeviceCreate(), PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceConfigure(PetscDevice device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDevice(device,1);
  if (PetscDefined(USE_DEBUG)) {
    /* if no available configuration is available, this cascades all the way down to default
       and error */
    switch (device->kind) {
    case PETSC_DEVICE_CUDA: if (PetscDefined(HAVE_CUDA)) break;
    case PETSC_DEVICE_HIP:  if (PetscDefined(HAVE_HIP))  break;
    default:
      PETSC_DEVICE_DEFAULT_CASE(PETSC_COMM_SELF,device->kind);
      break;
    }
  }
  ierr = (*device->ops->configure)(device);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceView(PetscDevice device, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDevice(device,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = (*device->ops->view)(device,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static std::array<PetscBool,PETSC_DEVICE_MAX>   initializedDevice;
static std::array<PetscDevice,PETSC_DEVICE_MAX> defaultDevices;
static_assert(initializedDevice.size() == defaultDevices.size(),"");

PetscBool PetscDeviceInitializedFor(PetscDeviceKind kind)
{
  return PetscDeviceConfiguredFor(kind) ? initializedDevice[kind] : PETSC_FALSE;
}

PetscErrorCode PetscDeviceInitializeDefaultDevice_Internal(PetscDeviceKind kind, PetscInt defaultDeviceId)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLikely(defaultDevices[kind])) PetscFunctionReturn(0);
  ierr = PetscDeviceCreate(kind,defaultDeviceId,&defaultDevices[kind]);CHKERRQ(ierr);
  ierr = PetscDeviceConfigure(defaultDevices[kind]);CHKERRQ(ierr);
  /* the default devices are all automatically "referenced" at least once, otherwise the
     reference counting is off for them. We could alternatively increase the reference
     count when they are retrieved but that is a lot more brittle; whats to stop someone
     from doing the following?

     for (int i = 0; i < 10000; ++i) auto device = PetscDeviceDefault_Internal();
   */
  defaultDevices[kind] = PetscDeviceReference(defaultDevices[kind]);
  initializedDevice[kind] = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceAllocateDefaultDeviceFromOptions_Private(MPI_Comm comm, PetscDeviceKind kind, PetscDeviceInitKind defaultInitKind = PETSC_DEVICE_INIT_LAZY, PetscInt defaultDeviceId = PETSC_DECIDE, PetscBool defaultView = PETSC_FALSE)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscDeviceConfiguredFor(kind)) {
    ierr = PetscInfo1(PETSC_NULLPTR,"PetscDeviceKind %s supported, initializing\n",PetscDeviceKinds[kind]);CHKERRQ(ierr);
    /* ugly switch needed to pick the right global variable... could maybe do this as a union? */
    switch (kind) {
      PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,initialize,comm,&defaultInitKind,&defaultDeviceId);
      PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HIP,initialize,comm,&defaultInitKind,&defaultDeviceId);
    default:
      PETSC_DEVICE_DEFAULT_CASE(comm,kind);
      break;
    }
    /* defaultInitKind and defaultDeviceId now represent what the individual TYPES have decided
     * to initialize as */
    if (defaultInitKind == PETSC_DEVICE_INIT_GREEDY) {
      ierr = PetscDeviceInitializeDefaultDevice_Internal(kind,defaultDeviceId);CHKERRQ(ierr);
      if (defaultView) {
        PetscViewer vwr;

        ierr = PetscViewerASCIIGetStdout(comm,&vwr);CHKERRQ(ierr);
        ierr = PetscDeviceView(defaultDevices[kind],vwr);CHKERRQ(ierr);
      }
    }
  } else {
    ierr = PetscInfo1(PETSC_NULLPTR,"PetscDeviceKind %s not supported\n",PetscDeviceKinds[kind]);CHKERRQ(ierr);
    defaultDevices[kind] = PETSC_NULLPTR;
  }
  PetscFunctionReturn(0);
}

/* called from PetscFinalize() do not call yourself! */
static PetscErrorCode PetscDeviceFinalizeDefaultDevices_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (auto &device : defaultDevices) {
    ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
    if (PetscUnlikelyDebug(device)) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_COR,"Device of kind '%s' had reference count %D and was not fully destroyed during PetscFinalize()",device->kind,device->refcnt);
  }
  CHKERRCXX(initializedDevice.fill(PETSC_FALSE));
  PetscFunctionReturn(0);
}

/* called from PetscDeviceContextInitializeRootContext_Internal() do not call yourself! */
PetscErrorCode PetscDeviceInitializeAllDefaultDevicesFromOptions_Internal(MPI_Comm comm, PetscDeviceInitKind defaultInitKind)
{
  PetscBool      defaultView   = PETSC_FALSE;
  PetscInt       defaultDevice = PETSC_DECIDE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscRegisterFinalize(PetscDeviceFinalizeDefaultDevices_Private);CHKERRQ(ierr);
  {
    PetscInt  initIdx = defaultInitKind;
    PetscBool flg;

    ierr = PetscOptionsBegin(comm,PETSC_NULLPTR,"PetscDevice Options","Sys");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-device_enable","How (or whether to) initialize PetscDevices","PetscDeviceInitializeAllDevices_Internal()",PetscDeviceInitKinds,3,PetscDeviceInitKinds[initIdx],&initIdx,PETSC_NULLPTR);CHKERRQ(ierr);
    ierr = PetscOptionsRangeInt("-device_select","Which device to use. Pass -1 to have PETSc decide or (given they exist) [0-NUM_DEVICE) for a specific device","PetscDeviceCreate",defaultDevice,&defaultDevice,PETSC_NULLPTR,PETSC_DECIDE,std::numeric_limits<int>::max());CHKERRQ(ierr);
    ierr = PetscOptionsBool("-device_view","Display device information and assignments (note this implies greedy initialization, but is overridden by disabling devices)",PETSC_NULLPTR,defaultView,&defaultView,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (initIdx == PETSC_DEVICE_INIT_NONE) {
      /* disabled all device initialization if devices are globally disabled */
      if (PetscUnlikelyDebug(defaultDevice != PETSC_DECIDE)) SETERRQ(comm,PETSC_ERR_USER_INPUT,"You have disabled devices but also specified a particular device to use, these options are mutually  exlusive");
      defaultView = PETSC_FALSE;
    } else {
      defaultView = static_cast<PetscBool>(defaultView && flg);
      if (defaultView) initIdx = PETSC_DEVICE_INIT_GREEDY;
    }
    defaultInitKind = static_cast<PetscDeviceInitKind>(initIdx);
  }
  ierr = PetscDeviceAllocateDefaultDeviceFromOptions_Private(comm,PETSC_DEVICE_INVALID);CHKERRQ(ierr);
  ierr = PetscDeviceAllocateDefaultDeviceFromOptions_Private(comm,PETSC_DEVICE_CUDA,defaultInitKind,defaultDevice,defaultView);CHKERRQ(ierr);
  ierr = PetscDeviceAllocateDefaultDeviceFromOptions_Private(comm,PETSC_DEVICE_HIP,defaultInitKind,defaultDevice,defaultView);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Get the default PetscDevice for a particular kind and constructs them if lazily initialized. */
PetscErrorCode PetscDeviceGetDefaultForKind_Internal(PetscDeviceKind kind, PetscDevice *device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceInitialize_Internal(kind);CHKERRQ(ierr);
  *device = defaultDevices[kind];
  PetscFunctionReturn(0);
}
