#include "../../interface/sycldevice.hpp"
#include <algorithm>
#include <csetjmp> // for gpu mpi awareness
#include <csignal> // SIGSEGV
#include <iterator>
#include <type_traits>
#include <vector>
#include <CL/sycl.hpp>


#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#else
#define PetscLogInitialize() 0
#endif

PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

namespace Petsc
{

// define static variables
bool SyclDevice::_initialized = false;
std::array<std::unique_ptr<SyclDevice::SyclDeviceInternal>,PETSC_DEVICE_MAX_DEVICES> SyclDevice::_devices_array = {};
std::unique_ptr<SyclDevice::SyclDeviceInternal> *SyclDevice::_devices = &SyclDevice::_devices_array[1];
int SyclDevice::_defaultDevice = PETSC_SYCL_DEVICE_NONE;

static std::jmp_buf MPISyclAwareJumpBuffer;
static bool         MPISyclAwareJumpBufferSet;

// internal "impls" class for SyclDevice. Each instance represents a single sycl device
class SyclDevice::SyclDeviceInternal
{
  const int        _id; // -1 for host; 0+ for gpu
  bool             _devInitialized = false;

public:
  // default constructor
  explicit constexpr SyclDeviceInternal(int id) noexcept : _id(id) { }
  int  id() const { return _id; }
  bool initialized() const { return _devInitialized; }

  // factory
  static constexpr std::unique_ptr<SyclDeviceInternal> makeDevice(int id) noexcept
  {
    return std::unique_ptr<SyclDeviceInternal>(new SyclDeviceInternal(id));
  }

  PetscErrorCode initialize() noexcept
  {
    PetscFunctionBegin;
    if (_devInitialized) PetscFunctionReturn(0);
    if (_id >= 0 && use_gpu_aware_mpi) {
      if (!_isMPISyclAware()) {
        (*PetscErrorPrintf)("PETSc is configured with sycl support, but your MPI is not aware of sycl GPU devices. For better performance, please use a sycl GPU-aware MPI.\n");
        (*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n");
        PETSCABORT(PETSC_COMM_SELF,PETSC_ERR_LIB);
      }
    }
    _devInitialized = true;
    PetscFunctionReturn(0);
  }

  PetscErrorCode finalize() noexcept
  {
    PetscFunctionBegin;
    _devInitialized = false;
    PetscFunctionReturn(0);
  }

  PetscErrorCode configure() noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (!_devInitialized) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being configured before it was initialized",_id);
    ierr = PetscInfo1(nullptr,"Configured device %d\n",_id);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode view(PetscViewer viewer) const noexcept
  {
    MPI_Comm       comm;
    PetscMPIInt    rank;
    PetscBool      iascii;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if ((!_devInitialized)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being viewed before it was initialized or configured",_id);
    ierr = PetscObjectTypeCompare(reinterpret_cast<PetscObject>(viewer),PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
    ierr = PetscObjectGetComm(reinterpret_cast<PetscObject>(viewer),&comm);CHKERRQ(ierr);
    if ((!iascii)) SETERRQ(comm,PETSC_ERR_SUP,"Only PetscViewer of type PETSCVIEWERASCII is supported");
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);

    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] device %d: %s\n",rank,_id,Q.get_device().get_info<info::device::name>().data());CHKERRQ(ierr);
    // // flush the assignment information
    // ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    // ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Compute capability: %d.%d\n",_dprop.major,_dprop.minor);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Multiprocessor Count: %d\n",_dprop.multiProcessorCount);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Maximum Grid Dimensions: %d x %d x %d\n",_dprop.maxGridSize[0],_dprop.maxGridSize[1],_dprop.maxGridSize[2]);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Maximum Block Dimensions: %d x %d x %d\n",_dprop.maxThreadsDim[0],_dprop.maxThreadsDim[1],_dprop.maxThreadsDim[2]);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Maximum Threads Per Block: %d\n",_dprop.maxThreadsPerBlock);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Warp Size: %d\n",_dprop.warpSize);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Total Global Memory (bytes): %zu\n",_dprop.totalGlobalMem);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Total Constant Memory (bytes): %zu\n",_dprop.totalConstMem);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Shared Memory Per Block (bytes): %zu\n",_dprop.sharedMemPerBlock);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Multiprocessor Clock Rate (KHz): %d\n",_dprop.clockRate);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Memory Clock Rate (KHz): %d\n",_dprop.memoryClockRate);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Memory Bus Width (bits): %d\n",_dprop.memoryBusWidth);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Peak Memory Bandwidth (GB/s): %f\n",2.0*_dprop.memoryClockRate*(_dprop.memoryBusWidth/8)/1.0e6);CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Can map host memory: %s\n",_dprop.canMapHostMemory ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
    // ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Can execute multiple kernels concurrently: %s\n",_dprop.concurrentKernels ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

private:
  // Is the underlying MPI aware of sycl (GPU) devices?
  static bool _isMPISyclAware() noexcept
  {
    PetscErrorCode ierr;
    constexpr int  bufSize = 2;
    constexpr int  hbuf[bufSize] = {1,0};
    int            *dbuf = nullptr;
    bool           awareness = false;
    const auto     SyclSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
      if ((signal == SIGSEGV) && MPISyclAwareJumpBufferSet) std::longjmp(MPISyclAwareJumpBuffer,1);
      return PetscSignalHandlerDefault(signal,ptr);
    };

    PetscFunctionBegin;
    auto Q = sycl::queue(sycl::gpu_selector());
    dbuf   = sycl::malloc_device<int>(bufSize,Q);
    Q.memcpy(dbuf,hbuf,sizeof(int)*bufSize).wait();
    ierr = PetscPushSignalHandler(SyclSignalHandler,nullptr);CHKERRABORT(PETSC_COMM_SELF,ierr);
    MPISyclAwareJumpBufferSet = true;
    if (setjmp(MPISyclAwareJumpBuffer)) {
      // if a segv was triggered in the MPI_Allreduce below, it is very likely due to MPI not being GPU-aware
      awareness = false;
      // control flow up until this point:
      // 1. SyclDevice::SyclDeviceInternal::__MPISyclAware()
      // 2. MPI_Allreduce
      // 3. SIGSEGV
      // 4. PetscSignalHandler_Private
      // 5. SyclSignalHandler (lambda function)
      // 6. here
      // PetscSignalHandler_Private starts with PetscFunctionBegin and is pushed onto the stack
      // so we must undo this. This would be most naturally done in SyclSignalHandler, however
      // the C/C++ standard dictates:
      //
      // After invoking longjmp(), non-volatile-qualified local objects should not be accessed if
      // their values could have changed since the invocation of setjmp(). Their value in this
      // case is considered indeterminate, and accessing them is undefined behavior.
      //
      // so for safety (since we don't know what PetscStackPop may try to read/declare) we do it
      // outside of the longjmp control flow
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
  _devices = &_devices_array[1];
  ierr = PetscRegisterFinalize(_finalize);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,nullptr,"PetscDevice SYCL Options","Sys");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-device_enable_sycl","How (or whether) to initialize a device","SyclDevice::initialize()",PetscDeviceInitTypes,3,PetscDeviceInitTypes[initType],&initType,nullptr);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-device_select_sycl","Which device to use. Pass PETSC_DECIDE to have PETSc decide or (given they exist) [0-NUM_DEVICE) for a specific device","PetscDeviceCreate",id,&id,nullptr,PETSC_DECIDE,std::numeric_limits<int>::max());CHKERRQ(ierr);
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
      } else id = PETSC_SYCL_HOST_DEVICE;
    }
    view = static_cast<decltype(view)>(view && flg);
    if (view) initType = PETSC_DEVICE_INIT_EAGER;
  }

  _defaultDevice = static_cast<int>(id);
  if (initType == PETSC_DEVICE_INIT_EAGER && id == PETSC_SYCL_DEVICE_NONE) SETERRQ(comm,PETSC_ERR_USER_INPUT,"Cannot eagerly initialize sycl devices as you disabled them by -device_enable_sycl none");

  if (initType == PETSC_DEVICE_INIT_EAGER) {
    _devices[_defaultDevice] = SyclDeviceInternal::makeDevice(_defaultDevice);
    ierr = _devices[_defaultDevice]->initialize();CHKERRQ(ierr);
    ierr = _devices[_defaultDevice]->configure();CHKERRQ(ierr);
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
  for (auto&& device : _devices_array) {
    if (device) {
      const auto ierr = device->finalize();CHKERRQ(ierr);
      device.reset();
    }
  }
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
  if (id+1 >= _devices_array.size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only supports %zu number of devices but trying to get device with id %" PetscInt_FMT,_devices_array.size(),id);
  if (_devices[id]) {
    if (id != _devices[id]->id()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Entry %" PetscInt_FMT " contains device with mismatching id %" PetscInt_FMT,id,_devices[id]->id());
  } else _devices[id] = SyclDeviceInternal::makeDevice(id);
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
  PetscFunctionReturn(0);
}

PetscErrorCode SyclDevice::viewDevice(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

} // namespace Petsc
