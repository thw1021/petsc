#include "../../interface/sycldevice.hpp"
#include <csetjmp> // for MPI sycl device awareness
#include <csignal> // SIGSEGV
#include <vector>
#include <CL/sycl.hpp>

#if PetscDefined(USE_LOG)
  PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#else
  #define PetscLogInitialize() 0
#endif

namespace Petsc {

// definition for static
std::array<SyclDevice::SyclDeviceInternal*,PETSC_DEVICE_MAX_DEVICES> SyclDevice::_devices_array = {};
SyclDevice::SyclDeviceInternal**                                     SyclDevice::_devices       = &SyclDevice::_devices_array[1];
int                                                                  SyclDevice::_defaultDevice = PETSC_SYCL_DEVICE_NONE;
bool                                                                 SyclDevice::_initialized   = false;

static std::jmp_buf MPISyclAwareJumpBuffer;
static bool         MPISyclAwareJumpBufferSet;

// internal "impls" class for SyclDevice. Each instance represents a single sycl device
class SyclDevice::SyclDeviceInternal
{
  const int        _id; // -1 for the host device; 0 and up for gpu devices
  bool             _devInitialized;
  sycl::device     _syclDevice;

public:
  // default constructor
  SyclDeviceInternal(int id) noexcept : _id(id),_devInitialized(false) {
    if (_id == PETSC_SYCL_DEVICE_HOST) {
      _syclDevice = sycl::device(sycl::host_selector());
    } else if (_id >= 0) {
      _syclDevice= sycl::device::get_devices(sycl::info::device_type::gpu)[_id];
    }
  }
  int  id() const {return _id;}
  bool initialized() {return _devInitialized;}

  PetscErrorCode initialize() noexcept
  {
    PetscFunctionBegin;
    if (_devInitialized) PetscFunctionReturn(0);
    if (_syclDevice.is_gpu() && use_gpu_aware_mpi) {
      if (!_isMPISyclAware()) {
        (*PetscErrorPrintf)("PETSc is configured with sycl support, but your MPI is not aware of sycl GPU devices. For better performance, please use a sycl GPU-aware MPI.\n");
        (*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n");
        PETSCABORT(PETSC_COMM_SELF,PETSC_ERR_LIB);
      }
    }
    _devInitialized = true;
    PetscFunctionReturn(0);
  }

  PetscErrorCode view(PetscViewer viewer) const noexcept
  {
    PetscErrorCode ierr;
    MPI_Comm       comm;
    PetscMPIInt    rank;
    PetscBool      iascii;
    sycl::device   dev;

    PetscFunctionBegin;
    if ((!_devInitialized)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being viewed before it was initialized or configured",_id);
    ierr = PetscObjectTypeCompare(reinterpret_cast<PetscObject>(viewer),PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
    ierr = PetscObjectGetComm(reinterpret_cast<PetscObject>(viewer),&comm);CHKERRQ(ierr);
    if ((!iascii)) SETERRQ(comm,PETSC_ERR_SUP,"Only PetscViewer of type PETSCVIEWERASCII is supported");
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] device %d: %s\n",rank,_syclDevice.get_info<sycl::info::device::name>().c_str());CHKERRQ(ierr);
    // flush the assignment information
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"-> Device vendor: %s\n",_syclDevice.get_info<sycl::info::device::vendor>().c_str());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

private:
  // Is the underlying MPI aware of sycl (GPU) devices?
  bool _isMPISyclAware() noexcept
  {
    PetscErrorCode ierr;
    const int      bufSize = 2;
    const int      hbuf[bufSize] = {1,0};
    int            *dbuf = nullptr;
    bool           awareness = false;
    const auto     SyclSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
      if ((signal == SIGSEGV) && MPISyclAwareJumpBufferSet) std::longjmp(MPISyclAwareJumpBuffer,1);
      return PetscSignalHandlerDefault(signal,ptr);
    };

    PetscFunctionBegin;
    auto Q = sycl::queue(_syclDevice);
    dbuf   = sycl::malloc_device<int>(bufSize,Q);
    Q.memcpy(dbuf,hbuf,sizeof(int)*bufSize).wait();
    ierr = PetscPushSignalHandler(SyclSignalHandler,nullptr);CHKERRABORT(PETSC_COMM_SELF,ierr);
    MPISyclAwareJumpBufferSet = true;
    if (setjmp(MPISyclAwareJumpBuffer)) {
      // if a segv was triggered in the MPI_Allreduce below, it is very likely due to MPI not being GPU-aware
      awareness = false;
      PetscStackPop;
    } else if (!MPI_Allreduce(dbuf,dbuf+1,1,MPI_INT,MPI_SUM,PETSC_COMM_SELF)) awareness = true;
    MPISyclAwareJumpBufferSet = false;
    ierr = PetscPopSignalHandler();CHKERRABORT(PETSC_COMM_SELF,ierr);
    sycl::free(dbuf,Q);
    PetscFunctionReturn(awareness);
  }
};

PetscErrorCode SyclDevice::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscDeviceInitType *defaultInitType) noexcept
{
  PetscInt       initType = *defaultInitType,id = *defaultDeviceId;
  PetscBool      view = PETSC_FALSE,flg;
  int            ngpus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (_initialized) PetscFunctionReturn(0);
  _initialized = true;
  ierr = PetscRegisterFinalize(_finalize);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,nullptr,"PetscDevice SYCL Options","Sys");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-device_enable_sycl","How (or whether) to initialize a device","SyclDevice::initialize()",PetscDeviceInitTypes,3,PetscDeviceInitTypes[initType],&initType,nullptr);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-device_select_sycl","Which sycl device to use. Pass -2 for host, PETSC_DECIDE (-1) to have PETSc decide, 0 and up for GPUs","PetscDeviceCreate",id,&id,nullptr,-2,std::numeric_limits<int>::max());CHKERRQ(ierr);
  ierr = PetscOptionsBool("-device_view_sycl","Display device information and assignments (forces eager initialization)",nullptr,view,&view,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // post-process the options and lay the groundwork for initialization if needs be
  std::vector<sycl::device> gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  ngpus = gpu_devices.size();
  if (ngpus == 0 && id >= 0) SETERRQ1(comm,PETSC_ERR_USER_INPUT,"You specified a sycl gpu device with -device_select_sycl %d but there is no GPU", (int)id);

  if (initType == PETSC_DEVICE_INIT_NONE) id = PETSC_SYCL_DEVICE_NONE; /* user wants to disable all sycl devices */
  else {
    ierr = PetscDeviceCheckDeviceCount_Internal(ngpus);CHKERRQ(ierr);
    if (id == PETSC_DECIDE) { /* petsc will choose a GPU device if any, otherwise a CPU device */
      if (ngpus) {
        PetscMPIInt rank;
        ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
        id   = rank % ngpus;
      } else id = PETSC_SYCL_DEVICE_HOST;
    }
    view = static_cast<decltype(view)>(view && flg);
    if (view) initType = PETSC_DEVICE_INIT_EAGER;
  }

  if (id == -2) id = PETSC_SYCL_DEVICE_HOST; // user passed in '-device_select_sycl -2'. We transform it into canonical form

  _defaultDevice = static_cast<int>(id);
  if (initType == PETSC_DEVICE_INIT_EAGER && id == PETSC_SYCL_DEVICE_NONE) SETERRQ(comm,PETSC_ERR_USER_INPUT,"Cannot eagerly initialize sycl devices as you disabled them by -device_enable_sycl none");

  if (initType == PETSC_DEVICE_INIT_EAGER) {
    _devices[_defaultDevice] = new SyclDeviceInternal(_defaultDevice);
    ierr = _devices[_defaultDevice]->initialize();CHKERRQ(ierr);
    if (view) {
      PetscViewer viewer;
      ierr = PetscLogInitialize();CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
      ierr = _devices[_defaultDevice]->view(viewer);CHKERRQ(ierr);
    }
  }

  // record the results of the initialization
  *defaultInitType = static_cast<PetscDeviceInitType>(initType);
  *defaultDeviceId = id;
  PetscFunctionReturn(0);
}

PetscErrorCode SyclDevice::_finalize() noexcept
{
  PetscFunctionBegin;
  if (!_initialized) PetscFunctionReturn(0);
  for (auto&& devPtr : _devices_array) delete devPtr;
  _defaultDevice = PETSC_SYCL_DEVICE_NONE;  // disabled by default
  _initialized   = false;
  PetscFunctionReturn(0);
}

PetscErrorCode SyclDevice::getDevice(PetscDevice device, PetscInt id) const noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (_defaultDevice == PETSC_SYCL_DEVICE_NONE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Trying to retrieve a SYCL PetscDevice when it has been disabled");
  if (id == PETSC_DECIDE) id = _defaultDevice;
  if ((id < PETSC_SYCL_DEVICE_HOST) || (id-PETSC_SYCL_DEVICE_HOST >= PETSC_DEVICE_MAX_DEVICES)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only supports %zu number of devices but trying to get device with id %" PetscInt_FMT,_devices_array.size(),id);
  if (_devices[id]) {
    if (id != _devices[id]->id()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Entry %" PetscInt_FMT " contains device with mismatching id %" PetscInt_FMT,id,_devices[id]->id());
  } else _devices[id] = new SyclDeviceInternal(id);
  ierr = _devices[id]->initialize();CHKERRQ(ierr);
  device->deviceId           = _devices[id]->id(); // technically id = _devices[id]->_id here
  device->ops->createcontext = _create;
  device->ops->configure     = this->configureDevice;
  device->ops->view          = this->viewDevice;
  PetscFunctionReturn(0);
}

PetscErrorCode SyclDevice::configureDevice(PetscDevice device) noexcept
{
  PetscFunctionBegin;
  // Nothing for now
  PetscFunctionReturn(0);
}

PetscErrorCode SyclDevice::viewDevice(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = _devices[device->deviceId]->view(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

}
