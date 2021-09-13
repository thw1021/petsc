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
#if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA:
    ierr = CUDADevice.getDevice(dev,devid);CHKERRQ(ierr);
    break;
#endif
#if PetscDefined(HAVE_HIP)
  case PETSC_DEVICE_HIP:
    ierr = HIPDevice.getDevice(dev,devid);CHKERRQ(ierr);
    break;
#endif
  default:
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[kind],kind);
  }
  *device = dev;
  PetscFunctionReturn(0);
#undef PETSCDEVICE_CASE
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
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[device->kind],device->kind);
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

// TODO decide whether to remove or make fully featured
constexpr PetscBool PetscDeviceEnabledFor(PetscDeviceKind kind)
{
  bool enabled = false;

  switch(kind) {
  case PETSC_DEVICE_INVALID: break;
  case PETSC_DEVICE_CUDA:    {enabled = PetscDefined(HAVE_CUDA); break;}
  case PETSC_DEVICE_HIP:     {enabled = PetscDefined(HAVE_HIP);  break;}
  case PETSC_DEVICE_MAX:     break;
  }
  return static_cast<PetscBool>(enabled);
}

static PetscDevice defaultDevices[PETSC_DEVICE_MAX];

/* called from PetscFinalize() do not call yourself! */
static PetscErrorCode PetscDeviceFinalizeDefaultDevices_Private(void)
{
  PetscFunctionBegin;
  for (int i = 0; i < PETSC_DEVICE_MAX; ++i) {
    PetscErrorCode ierr;

    ierr = PetscDeviceDestroy(defaultDevices+i);CHKERRQ(ierr);
    if (PetscUnlikelyDebug(defaultDevices[i])) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_COR,"Device of kind '%s' had reference count %D and was not fully destroyed during PetscFinalize()",PetscDeviceKinds[i],defaultDevices[i]->refcnt);
  }
  PetscFunctionReturn(0);
}

#define CAT_(a,...) a ## __VA_ARGS__
#define CAT(a,...)  CAT_(a,__VA_ARGS__)

/* Need to do the ugly ## directly here rather than use macro since I guess PetscDefined
 * doesn't have an initial indirection layer? */
#define PETSC_DEVICE_CASE_IF_PETSC_DEFINED(IMPLS,...)                      \
  CAT(PETSC_DEVICE_CASE_IF_PETSC_DEFINED_,PetscDefined(HAVE_##IMPLS))(IMPLS,__VA_ARGS__)

#define PETSC_DEVICE_CASE_IF_PETSC_DEFINED_0(IMPLS,...)
#define PETSC_DEVICE_CASE_IF_PETSC_DEFINED_1(IMPLS,...) do {            \
    case CAT(PETSC_DEVICE_,IMPLS):                                      \
    {                                                                   \
      PetscErrorCode ierr;                                              \
      ierr = CAT(IMPLS,Device).initialize(__VA_ARGS__);CHKERRQ(ierr);   \
    }                                                                   \
  } while (0)

static PetscErrorCode PetscDeviceInitializeKind_Private(MPI_Comm comm, PetscDeviceKind kind, PetscDeviceInitKind *defaultInitKind, PetscInt *defaultDeviceId)
{

  PetscFunctionBegin;
  switch (kind) {
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,comm,defaultInitKind,defaultDeviceId);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HIP,comm,defaultInitKind,defaultDeviceId);
  default:
    SETERRQ2(comm,PETSC_ERR_SUP_SYS,"Must have configured PETSc with %s support to use PetscDeviceKind %d",PetscDeviceKinds[kind],kind);
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceConstructDefaultDevice_Internal(PetscDeviceKind kind, PetscInt defaultDeviceId)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(!PetscDeviceEnabledFor(kind))) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trying to construct the default PetscDevice for disabled kind %s",PetscDeviceKinds[kind]);
  if (PetscLikely(defaultDevices[kind])) PetscFunctionReturn(0);
  ierr = PetscDeviceCreate(kind,defaultDeviceId,defaultDevices+kind);CHKERRQ(ierr);
  ierr = PetscDeviceConfigure(defaultDevices[kind]);CHKERRQ(ierr);
  /* the default devices are all automatically "referenced" at least once, otherwise the
     reference counting is off for them. We could alternatively increase the reference
     count when they are retrieved but that is a lot more brittle; whats to stop someone
     from doing the following?

     for (int i = 0; i < 10000; ++i) auto device = PetscDeviceDefault_Internal();
   */
  defaultDevices[kind] = PetscDeviceReference(defaultDevices[kind]);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDeviceSetupDefaultDevice_Private(MPI_Comm comm, PetscDeviceKind kind, PetscDeviceInitKind defaultInitKind = PETSC_DEVICE_INIT_LAZY, PetscInt defaultDevice = PETSC_DECIDE, PetscBool defaultView = PETSC_FALSE)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscDeviceEnabledFor(kind)) {
    /* on the off chance that someone fumbles calling this with INVALID or MAX */
    PetscValidDeviceKind(kind,2);
    ierr = PetscInfo1(PETSC_NULLPTR,"PetscDeviceKind %s supported, initializing\n",PetscDeviceKinds[kind]);CHKERRQ(ierr);
    ierr = PetscDeviceInitializeKind_Private(comm,kind,&defaultInitKind,&defaultDevice);CHKERRQ(ierr);
    /* defaultInitKind and defaultDevice now represent what the individual TYPES have decided
     * to initialize as */
    if (defaultInitKind == PETSC_DEVICE_INIT_GREEDY) {
      ierr = PetscDeviceConstructDefaultDevice_Internal(kind,defaultDevice);CHKERRQ(ierr);
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

/* called from PetscDeviceContextInitializeRootContext_Internal() do not call yourself! */
PetscErrorCode PetscDeviceInitializeAllDefaultDevices_Internal(MPI_Comm comm, PetscDeviceInitKind defaultInitKind)
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
    ierr = PetscOptionsEList("-device_enable","How (or whether to) initialize PetscDevices","PetscDeviceInitializeAllDevices_Internal()",PetscDeviceInitKinds,sizeof(PetscDeviceInitKinds)/sizeof(*PetscDeviceInitKinds),PetscDeviceInitKinds[initIdx],&initIdx,PETSC_NULLPTR);CHKERRQ(ierr);
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
  ierr = PetscDeviceSetupDefaultDevice_Private(comm,PETSC_DEVICE_INVALID);CHKERRQ(ierr);
  ierr = PetscDeviceSetupDefaultDevice_Private(comm,PETSC_DEVICE_CUDA,defaultInitKind,defaultDevice,defaultView);CHKERRQ(ierr);
  ierr = PetscDeviceSetupDefaultDevice_Private(comm,PETSC_DEVICE_HIP,defaultInitKind,defaultDevice,defaultView);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Get the default PetscDevice for a particular kind and constructs them if lazily initialized. */
PetscErrorCode PetscDeviceGetDefaultForKind_Internal(PetscDeviceKind kind, PetscDevice *device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceConstructDefaultDevice_Internal(kind,PETSC_DECIDE);CHKERRQ(ierr);
  *device = defaultDevices[kind];
  PetscFunctionReturn(0);
}
