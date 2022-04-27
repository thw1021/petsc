#include <petsc/private/petscadvancedmacros.h>
#include <petsc/private/deviceimpl.h>  /* I "petscdevice.h" */
#include "hostdevice.hpp"
#include "cupmdevice.hpp"
#include <limits> // for std::numeric_limits

// REVIEW ME: this should probably go somewhere better
#define PETSC_HAVE_HOST 1

using namespace Petsc::Device;

/*
  note to anyone adding more classes, the name must be ALL_CAPS_SHORT_NAME + Device exactly to
  be picked up by the switch-case macros below
*/
static Host::Device                         HOSTDevice{PetscDeviceContextCreate_HOST};
#if PetscDefined(HAVE_CUDA)
static CUPM::Device<CUPM::DeviceType::CUDA> CUDADevice{PetscDeviceContextCreate_CUDA};
#endif
#if PetscDefined(HAVE_HIP)
static CUPM::Device<CUPM::DeviceType::HIP>  HIPDevice{PetscDeviceContextCreate_HIP};
#endif
#if PetscDefined(HAVE_SYCL)
#include "sycldevice.hpp"
static SYCL::Device                         SYCLDevice{PetscDeviceContextCreate_SYCL};
#endif

static_assert(Petsc::util::integral_value(PETSC_DEVICE_HOST) == 0,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_CUDA) == 1,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_HIP)  == 2,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_SYCL) == 3,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_MAX)  == 4,"");
const char *const PetscDeviceTypes[] = {
  "host",
  "cuda",
  "hip",
  "sycl",
  "max",
  "PetscDeviceType",
  "PETSC_DEVICE_",
  nullptr
};

static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_NONE)  == 0,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_LAZY)  == 1,"");
static_assert(Petsc::util::integral_value(PETSC_DEVICE_INIT_EAGER) == 2,"");
const char *const PetscDeviceInitTypes[] = {
  "none",
  "lazy",
  "eager",
  "PetscDeviceInitType",
  "PETSC_DEVICE_INIT_",
  nullptr
};
static_assert(
  sizeof(PetscDeviceInitTypes)/sizeof(*PetscDeviceInitTypes) == 6,
  "Must change CUPMDevice<T>::initialize number of enum values in -device_enable_cupm to match!"
);

#define PETSC_DEVICE_CASE(IMPLS,func,...)                                     \
  case PetscConcat_(PETSC_DEVICE_,IMPLS): {                                   \
    PetscCall(PetscConcat_(IMPLS,Device).func(__VA_ARGS__));                    \
  } break

/*
  Suppose you have:

  CUDADevice.myFunction(arg1,arg2)

  that you would like to conditionally define and call in a switch-case:

  switch(PetscDeviceType) {
  #if PetscDefined(HAVE_CUDA)
  case PETSC_DEVICE_CUDA: {
    PetscCall(CUDADevice.myFunction(arg1,arg2));
  } break;
  #endif
  }

  then calling this macro:

  PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,myFunction,arg1,arg2)

  will expand to the following case statement:

  case PETSC_DEVICE_CUDA: {
    PetscCall(CUDADevice.myFunction(arg1,arg2));
  } break

  if PetscDefined(HAVE_CUDA) evaluates to 1, and expand to nothing otherwise
*/
#define PETSC_DEVICE_CASE_IF_PETSC_DEFINED(IMPLS,func,...) PetscIfPetscDefined(PetscConcat_(HAVE_,IMPLS),PETSC_DEVICE_CASE,PetscExpandToNothing)(IMPLS,func,__VA_ARGS__)

/*@C
  PetscDeviceCreate - Get a new handle for a particular device type

  Not Collective, Possibly Synchronous

  Input Parameters:
+ type  - The type of PetscDevice
- devid - The numeric ID# of the device (pass PETSC_DECIDE to assign automatically)

  Output Parameter:
. device - The PetscDevice

  Notes:
  This routine may initialize PetscDevice. If this is the case, this will most likely cause
  some sort of device synchronization.

  devid is what you might pass to cudaSetDevice() for example.

  Level: beginner

.seealso: PetscDevice, PetscDeviceInitType, PetscDeviceInitialize(),
PetscDeviceInitialized(), PetscDeviceConfigure(), PetscDeviceView(), PetscDeviceDestroy(),
PetscDeviceGetType(), PetscDeviceGetDeviceId()
@*/
PetscErrorCode PetscDeviceCreate(PetscDeviceType type, PetscInt devid, PetscDevice *device)
{
  static PetscInt PetscDeviceCounter = 0;

  PetscFunctionBegin;
  PetscValidDeviceType(type,1);
  PetscValidPointer(device,3);
  PetscCall(PetscDeviceInitializePackage());
  PetscCall(PetscNew(device));
  (*device)->id     = PetscDeviceCounter++;
  (*device)->type   = type;
  (*device)->refcnt = 1;
  /*
    if you are adding a device, you also need to add it's initialization in
    PetscDeviceInitializeTypeFromOptions_Private() below
  */
  switch (type) {
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HOST,getDevice,*device,devid);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,getDevice,*device,devid);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HIP,getDevice,*device,devid);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(SYCL,getDevice,*device,devid);
  default:
    /* in case the above macros expand to nothing this silences any unused variable warnings */
    (void)(devid);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PETSc was seemingly configured for PetscDeviceType %s but we've fallen through all cases in a switch",PetscDeviceTypes[type]);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceDestroy - Free a PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice

  Level: beginner

.seealso: PetscDevice, PetscDeviceCreate(), PetscDeviceConfigure(), PetscDeviceView(),
PetscDeviceGetType(), PetscDeviceGetDeviceId()
@*/
PetscErrorCode PetscDeviceDestroy(PetscDevice *device)
{
  PetscFunctionBegin;
  if (!*device) PetscFunctionReturn(0);
  PetscValidDevice(*device,1);
  PetscCall(PetscDeviceDereference_Internal(*device));
  if ((*device)->refcnt) {
    *device = nullptr;
    PetscFunctionReturn(0);
  }
  PetscCall(PetscFree((*device)->data));
  PetscCall(PetscFree(*device));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceConfigure - Configure a particular PetscDevice

  Not Collective, Asynchronous

  Input Parameter:
. device - The PetscDevice to configure

  Notes:
  The user should not assume that this is a cheap operation

  Level: beginner

.seealso: PetscDevice, PetscDeviceCreate(), PetscDeviceView(), PetscDeviceDestroy(),
PetscDeviceGetType(), PetscDeviceGetDeviceId()
@*/
PetscErrorCode PetscDeviceConfigure(PetscDevice device)
{
  PetscFunctionBegin;
  PetscValidDevice(device,1);
  /*
    if no available configuration is available, this cascades all the way down to default
    and error
  */
  switch (const auto dtype = device->type) {
  case PETSC_DEVICE_HOST: if (PetscDefined(HAVE_HOST)) break; // always true
  case PETSC_DEVICE_CUDA: if (PetscDefined(HAVE_CUDA)) break;
  case PETSC_DEVICE_HIP:  if (PetscDefined(HAVE_HIP))  break;
  case PETSC_DEVICE_SYCL: if (PetscDefined(HAVE_SYCL)) break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc was not configured for PetscDeviceType %s",PetscDeviceTypes[dtype]);
  }
  PetscCall((*device->ops->configure)(device));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceView - View a PetscDevice

  Collective on viewer, Asynchronous

  Input Parameters:
+ device - The PetscDevice to view
- viewer - The PetscViewer to view the device with (NULL for PETSC_VIEWER_STDOUT_WORLD)

  Level: beginner

.seealso: PetscDevice, PetscDeviceCreate(), PetscDeviceConfigure(), PetscDeviceDestroy(),
PetscDeviceGetType(), PetscDeviceGetDeviceId()
@*/
PetscErrorCode PetscDeviceView(PetscDevice device, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidDevice(device,1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall((*device->ops->view)(device,viewer));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceGetType - Get the type of device

  Asynchronous

  Input Parameter:
. device - The PetscDevice

  Output Parameter:
. type - The PetscDeviceType

  Level: beginner

.seealso: PetscDevice, PetscDeviceType, PetscDeviceCreate(), PetscDeviceConfigure(),
PetscDeviceDestroy(), PetscDeviceGetDeviceId()
@*/
PetscErrorCode PetscDeviceGetType(PetscDevice device, PetscDeviceType *type)
{
  PetscFunctionBegin;
  PetscValidDevice(device,1);
  PetscValidPointer(type,2);
  *type = device->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceGetDeviceId - Get the device ID for a PetscDevice

  Asynchronous

  Input Parameter:
. device - The PetscDevice

  Output Parameter:
. id - The id

  Notes:
  The returned ID may have been assigned by the underlying device backend. For example if the
  backend is CUDA then id is exactly the value returned by cudaGetDevice() at the time when
  this device was configured.

  Level: beginner

.seealso: PetscDevice, PetscDeviceCreate(), PetscDeviceGetType()
@*/
PetscErrorCode PetscDeviceGetDeviceId(PetscDevice device, PetscInt *id)
{
  PetscFunctionBegin;
  PetscValidDevice(device,1);
  PetscValidIntPointer(id,2);
  *id = device->deviceId;
  PetscFunctionReturn(0);
}

static std::array<std::pair<PetscDevice,bool>,PETSC_DEVICE_MAX> defaultDevices = {};

/*@C
  PetscDeviceInitialize - Initialize PetscDevice

  Not Collective, Possibly Synchronous

  Input Parameter:
. type - The PetscDeviceType to initialize

  Notes:
  Eagerly initializes the corresponding PetscDeviceType if needed.

  Level: beginner

.seealso: PetscDevice, PetscDeviceInitType, PetscDeviceInitialized(), PetscDeviceCreate(),
PetscDeviceDestroy()
@*/
PetscErrorCode PetscDeviceInitialize(PetscDeviceType type)
{
  PetscFunctionBegin;
  PetscValidDeviceType(type,1);
  PetscCall(PetscDeviceInitializeDefaultDevice_Internal(type,PETSC_DECIDE));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceInitialized - Determines whether PetscDevice is initialized for a particular
  PetscDeviceType

  Not Collective, Asynchronous

  Input Parameter:
. type - The PetscDeviceType to check

  Output Parameter:
. [return value] - PETSC_TRUE if type is initialized, PETSC_FALSE otherwise

  Notes:
  If one has not configured PETSc for a particular PetscDeviceType then this routine will
  return PETSC_FALSE for that PetscDeviceType.

  Level: beginner

.seealso: PetscDevice, PetscDeviceInitType, PetscDeviceInitialize(), PetscDeviceCreate(),
PetscDeviceDestroy()
@*/
PetscBool PetscDeviceInitialized(PetscDeviceType type)
{
  return static_cast<PetscBool>(PetscDeviceConfiguredFor_Internal(type) && defaultDevices[type].second);
}

/* Get the default PetscDevice for a particular type and constructs them if lazily initialized. */
PetscErrorCode PetscDeviceGetDefaultForType_Internal(PetscDeviceType type, PetscDevice *device)
{
  PetscFunctionBegin;
  PetscValidPointer(device,2);
  PetscCall(PetscDeviceInitialize(type));
  *device = defaultDevices[type].first;
  PetscFunctionReturn(0);
}

/*
  Actual intialization function; any functions claiming to initialize PetscDevice or
  PetscDeviceContext will have to run through this one
*/
PetscErrorCode PetscDeviceInitializeDefaultDevice_Internal(PetscDeviceType type, PetscInt defaultDeviceId)
{
  PetscFunctionBegin;
  PetscValidDeviceType(type,1);
  if (PetscUnlikely(!PetscDeviceInitialized(type))) {
    auto& dev  = defaultDevices[type].first;
    auto& init = defaultDevices[type].second;

    PetscAssert(!dev,PETSC_COMM_SELF,PETSC_ERR_MEM,"Trying to overwrite existing default device of type %s",PetscDeviceTypes[type]);
    PetscCall(PetscDeviceCreate(type,defaultDeviceId,&dev));
    PetscCall(PetscDeviceConfigure(dev));
    init = true;
  }
  PetscFunctionReturn(0);
}

#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#else
#define PetscLogInitialize() 0
#endif

static PetscErrorCode PetscDeviceInitializeTypeFromOptions_Private(MPI_Comm comm, PetscDeviceType type, PetscInt defaultDeviceId, PetscBool defaultView, PetscDeviceInitType *defaultInitType)
{
  PetscFunctionBegin;
  if (!PetscDeviceConfiguredFor_Internal(type)) {
    PetscCall(PetscInfo(nullptr,"PetscDeviceType %s not available\n",PetscDeviceTypes[type]));
    defaultDevices[type].first = nullptr;
    PetscFunctionReturn(0);
  }
  PetscCall(PetscInfo(nullptr,"PetscDeviceType %s available, initializing\n",PetscDeviceTypes[type]));
  /* ugly switch needed to pick the right global variable... could maybe do this as a union? */
  switch (type) {
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HOST,initialize,comm,&defaultDeviceId,defaultInitType);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(CUDA,initialize,comm,&defaultDeviceId,defaultInitType);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(HIP,initialize,comm,&defaultDeviceId,defaultInitType);
    PETSC_DEVICE_CASE_IF_PETSC_DEFINED(SYCL,initialize,comm,&defaultDeviceId,defaultInitType);
  default:
    SETERRQ(comm,PETSC_ERR_PLIB,"PETSc was seemingly configured for PetscDeviceType %s but we've fallen through all cases in a switch",PetscDeviceTypes[type]);
  }
  PetscCall(PetscInfo(nullptr,"PetscDevice %s initialized, device id %" PetscInt_FMT ", init type %s\n",PetscDeviceTypes[type],defaultDeviceId,PetscDeviceInitTypes[Petsc::util::integral_value(*defaultInitType)]));
  /*
    defaultInitType and defaultDeviceId now represent what the individual TYPES have decided to
    initialize as
  */
  if (*defaultInitType == PETSC_DEVICE_INIT_EAGER) {
    PetscCall(PetscDeviceInitializeDefaultDevice_Internal(type,defaultDeviceId));
    if (defaultView) {
      PetscViewer vwr;

      PetscCall(PetscLogInitialize());
      PetscCall(PetscViewerASCIIGetStdout(comm,&vwr));
      PetscCall(PetscDeviceView(defaultDevices[type].first,vwr));
    }
  }
  PetscFunctionReturn(0);
}

/* called from PetscFinalize() do not call yourself! */
static PetscErrorCode PetscDeviceFinalize_Private(void)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    const auto PetscDeviceCheckAllDestroyedAfterFinalize = []
    {
      PetscFunctionBegin;
      for (auto&& device : defaultDevices) {
        const auto dev = device.first;

        PetscCheck(!dev,PETSC_COMM_WORLD,PETSC_ERR_COR,"Device of type '%s' had reference count %" PetscInt_FMT " and was not fully destroyed during PetscFinalize()",PetscDeviceTypes[dev->type],dev->refcnt);
      }
      PetscFunctionReturn(0);
    };
    /*
      you might be thinking, why on earth are you registered yet another finalizer in a
      function already called during PetscRegisterFinalizeAll()? If this seems stupid it's
      because it is.

      The crux of the problem is that the initializer (and therefore the ~finalizer~) of
      PetscDeviceContext is guaranteed to run after PetscDevice's. So if the global context had
      a default PetscDevice attached, that PetscDevice will have a reference count >0 and hence
      won't be destroyed yet. So we need to repeat the check that all devices have been
      destroyed again ~after~ the global context is destroyed. In summary:

      1. This finalizer runs and destroys all devices, except it may not because the global
         context may still hold a reference!
      2. The global context finalizer runs and does the final reference count decrement
         required, which actually destroys the held device.
      3. Our newly added finalizer runs and checks that all is well.
    */
    PetscCall(PetscRegisterFinalize(PetscDeviceCheckAllDestroyedAfterFinalize));
  }
  for (auto &&device : defaultDevices) {
    PetscCall(PetscDeviceDestroy(&device.first));
    device.second = false;
  }
  PetscFunctionReturn(0);
}

/*
  Begins the init proceeedings for the entire PetscDevice stack. there are 3 stages of
  initialization types:

  1. defaultInitType - how does PetscDevice as a whole expect to initialize?
  2. subTypeDefaultInitType - how does each PetscDevice implementation expect to initialize?
     e.g. you may want to blanket disable PetscDevice init (and disable say Kokkos init), but
     have all CUDA devices still initialize.

  All told the following happens:

  0. defaultInitType -> LAZY
  1. Check for log_view/log_summary, if yes defaultInitType -> EAGER
  2. PetscDevice initializes each sub type with deviceDefaultInitType.
  2.1 Each enabled PetscDevice sub-type then does the above disable or view check in addition
      to checking for specific device init. if view or specific device init
      subTypeDefaultInitType -> EAGER. disabled once again overrides all.
*/

/* can't put this in a header since its not C-portable and only used here and in dcontext.cxx */
extern PETSC_VISIBILITY_INTERNAL PetscErrorCode PetscDeviceContextQueryOptions_Internal(MPI_Comm,const char[],std::pair<PetscDeviceType,PetscBool>&,std::pair<PetscStreamType,PetscBool>&);

PetscErrorCode PetscDeviceInitializeFromOptions_Internal(MPI_Comm comm)
{
  PetscBool           flg,defaultView = PETSC_FALSE,initializeDeviceContextEagerly = PETSC_FALSE;
  PetscInt            defaultDevice   = PETSC_DECIDE;
  PetscDeviceType     deviceContextInitDevice = PETSC_DEVICE_DEFAULT;
  PetscDeviceInitType defaultInitType;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    int result;

    PetscCallMPI(MPI_Comm_compare(comm,PETSC_COMM_WORLD,&result));
    /* in order to accurately assign ranks to gpus we need to get the MPI_Comm_rank of the
     * global space */
    if (PetscUnlikely(result != MPI_IDENT)) {
      char name[MPI_MAX_OBJECT_NAME] = {};
      int  len; /* unused */

      PetscCallMPI(MPI_Comm_get_name(comm,name,&len));
      SETERRQ(comm,PETSC_ERR_MPI,"Default devices being initialized on MPI_Comm '%s' not PETSC_COMM_WORLD",name);
    }
  }
  comm = PETSC_COMM_WORLD; /* from this point on we assume we're on PETSC_COMM_WORLD */
  PetscCall(PetscRegisterFinalize(PetscDeviceFinalize_Private));
  if (PetscDefined(USE_LOG)) {
    PetscCall(PetscOptionsHasName(nullptr,nullptr,"-log_view_gpu_time",&flg));
    if (flg) {
      PetscCall(PetscLogGpuTime());
    } else {
      PetscCall(PetscOptionsHasName(nullptr,nullptr,"-log_view",&flg));
      if (!flg) PetscCall(PetscOptionsHasName(nullptr,nullptr,"-log_summary",&flg));
    }
  } else flg = PETSC_FALSE;

  {
    PetscInt initIdx = flg ? PETSC_DEVICE_INIT_EAGER : PETSC_DEVICE_INIT_LAZY;

    PetscOptionsBegin(comm,nullptr,"PetscDevice Options","Sys");
    PetscCall(PetscOptionsEList("-device_enable","How (or whether) to initialize PetscDevices","PetscDeviceInitialize()",PetscDeviceInitTypes,3,PetscDeviceInitTypes[initIdx],&initIdx,nullptr));
    PetscCall(PetscOptionsRangeInt("-device_select","Which device to use. Pass " PetscStringize(PETSC_DECIDE) " to have PETSc decide or (given they exist) [0-NUM_DEVICE) for a specific device","PetscDeviceCreate()",defaultDevice,&defaultDevice,nullptr,PETSC_DECIDE,PETSC_DEVICE_MAX_DEVICES));
    PetscCall(PetscOptionsBool("-device_view","Display device information and assignments (forces eager initialization)","PetscDeviceView()",defaultView,&defaultView,&flg));
    PetscOptionsEnd();

    if (initIdx == PETSC_DEVICE_INIT_NONE) {
      /* disabled all device initialization if devices are globally disabled */
      PetscCheck(defaultDevice == PETSC_DECIDE,comm,PETSC_ERR_USER_INPUT,"You have disabled devices but also specified a particular device to use, these options are mutually exlusive");
      defaultView = PETSC_FALSE;
    } else {
      defaultView = static_cast<decltype(defaultView)>(defaultView && flg);
      if (defaultView) initIdx = PETSC_DEVICE_INIT_EAGER;
    }
    defaultInitType = static_cast<decltype(defaultInitType)>(initIdx);
  }

  static_assert((PETSC_DEVICE_HOST < PETSC_DEVICE_CUDA) && (PETSC_DEVICE_MAX < std::numeric_limits<int>::max()),"PETSC_DEVICE_HOST must be the lowest device and be < INT_MAX");
  for (int i = PETSC_DEVICE_HOST; i < PETSC_DEVICE_MAX; ++i) {
    const auto deviceType = PetscDeviceTypeCast(i);
    auto initType         = defaultInitType;

    PetscCall(PetscDeviceInitializeTypeFromOptions_Private(comm,deviceType,defaultDevice,defaultView,&initType));
    if (PetscDeviceConfiguredFor_Internal(deviceType) && (initType == PETSC_DEVICE_INIT_EAGER)) {
      initializeDeviceContextEagerly = PETSC_TRUE;
      deviceContextInitDevice        = deviceType;
      PetscCall(PetscInfo(nullptr,"PetscDevice %s set as default device type due to eager initialization\n",PetscDeviceTypes[deviceType]));
    }
  }

  {
    /*
      query the options db to get the root settings from the user (if any).

      This section is a bit of a hack. We have to reach across to dcontext.cxx to all but call
      PetscDeviceContextSetFromOptions() before we even have one, then set a few static
      variables in that file with the results.
    */
    auto dtype = std::make_pair(deviceContextInitDevice,PETSC_FALSE);
    auto stype = std::make_pair(PETSC_STREAM_GLOBAL_BLOCKING,PETSC_FALSE);

    PetscCall(PetscDeviceContextQueryOptions_Internal(comm,"root_",dtype,stype));
    if (initializeDeviceContextEagerly || dtype.second) {
      PetscCall(PetscDeviceContextSetRootDeviceType_Internal(dtype.first));
    }
    if (stype.second) PetscCall(PetscDeviceContextSetRootStreamType_Internal(stype.first));
  }
  if (initializeDeviceContextEagerly) {
    PetscDeviceContext dctx;

    PetscCall(PetscInfo(nullptr,"Eagerly initializing PetscDeviceContext with %s device\n",PetscDeviceTypes[deviceContextInitDevice]));
    /* instantiates the device context */
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextSetUp(dctx));
  }
  PetscFunctionReturn(0);
}
