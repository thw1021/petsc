#include "../../interface/cupmdevice.hpp"
#include <algorithm>
#include <csetjmp> // for cuda mpi awareness
#include <csignal> // SIGSEGV
#include <iterator>
#include <type_traits>

namespace Petsc
{

namespace Device
{

namespace CUPM
{

// internal "impls" class for CUPMDevice. Each instance represents a single cupm device
template <DeviceType T>
class Device<T>::DeviceInternal
{
  const int        id_;
  bool             devInitialized_ = false;
  cupmDeviceProp_t dprop_; // cudaDeviceProp appears to be an actual struct, i.e. you can't
                           // initialize it with nullptr or NULL (i've tried)

  PETSC_CXX_COMPAT_DECL(bool CUPMAwareMPI_());

public:
  // default constructor
  explicit constexpr DeviceInternal(int dev) noexcept : id_(dev) { }

  // gather all relevant information for a particular device, a cupmDeviceProp_t is
  // usually sufficient here
  PETSC_NODISCARD PetscErrorCode initialize() noexcept;
  PETSC_NODISCARD PetscErrorCode configure() noexcept;
  PETSC_NODISCARD PetscErrorCode view(PetscViewer) const noexcept;

  PETSC_NODISCARD auto id()          const -> decltype(id_)             { return id_;             }
  PETSC_NODISCARD auto initialized() const -> decltype(devInitialized_) { return devInitialized_; }
  PETSC_NODISCARD auto prop()        const -> const decltype(dprop_)&   { return dprop_;          }

  // factory
  PETSC_CXX_COMPAT_DECL(std::unique_ptr<DeviceInternal> makeDevice(int i))
  {
    return std::unique_ptr<DeviceInternal>(new DeviceInternal(i));
  }
};

// the goal here is simply to get the cupm backend to create its context, not to do any type of
// modification of it, or create objects (since these may be affected by subsequent
// configuration changes)
template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::initialize() noexcept
{
  cupmError_t cerr;

  PetscFunctionBegin;
  if (devInitialized_) PetscFunctionReturn(0);
  devInitialized_ = true;
  // need to do this BEFORE device has been set, although if the user
  // has already done this then we just ignore it
  if (cupmSetDeviceFlags(cupmDeviceMapHost) == cupmErrorSetOnActiveProcess) {
    // reset the error if it was cupmErrorSetOnActiveProcess
    const auto PETSC_UNUSED unused = cupmGetLastError();
  } else {CHKERRCUPM(cupmGetLastError());}
  // cuda 5.0+ will create a context when cupmSetDevice is called
  if (cupmSetDevice(id_) != cupmErrorDeviceAlreadyInUse) CHKERRCUPM(cupmGetLastError());
  // forces cuda < 5.0 to initialize a context
  cerr = cupmFree(nullptr);CHKERRCUPM(cerr);
  // where is this variable defined and when is it set? who knows! but it is defined and set
  // at this point. either way, each device must make this check since I guess MPI might not be
  // aware of all of them?
  if (use_gpu_aware_mpi) {
    // For OpenMPI, we could do a compile time check with
    // "defined(PETSC_HAVE_OMPI_MAJOR_VERSION) && defined(MPIX_CUDA_AWARE_SUPPORT) &&
    // MPIX_CUDA_AWARE_SUPPORT" to see if it is CUDA-aware. However, recent versions of IBM
    // Spectrum MPI (e.g., 10.3.1) on Summit meet above conditions, but one has to use jsrun
    // --smpiargs=-gpu to really enable GPU-aware MPI. So we do the check at runtime with a
    // code that works only with GPU-aware MPI.
    if (PetscUnlikely(!CUPMAwareMPI_())) {
      (*PetscErrorPrintf)("PETSc is configured with GPU support, but your MPI is not GPU-aware. For better performance, please use a GPU-aware MPI.\n");
      (*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n");
      (*PetscErrorPrintf)("If you do care, for IBM Spectrum MPI on OLCF Summit, you may need jsrun --smpiargs=-gpu.\n");
      (*PetscErrorPrintf)("For OpenMPI, you need to configure it --with-cuda (https://www.open-mpi.org/faq/?category=buildcuda)\n");
      (*PetscErrorPrintf)("For MVAPICH2-GDR, you need to set MV2_USE_CUDA=1 (http://mvapich.cse.ohio-state.edu/userguide/gdr/)\n");
      (*PetscErrorPrintf)("For Cray-MPICH, you need to set MPICH_RDMA_ENABLED_CUDA=1 (https://www.olcf.ornl.gov/tutorials/gpudirect-mpich-enabled-cuda/)\n");
      PETSCABORT(PETSC_COMM_SELF,PETSC_ERR_LIB);
    }
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::configure() noexcept
{
  cupmError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssert(devInitialized_,PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being configured before it was initialized",id_);
  // why on EARTH nvidia insists on making otherwise informational states into
  // fully-fledged error codes is beyond me. Why couldn't a pointer to bool argument have
  // sufficed?!?!?!
  if (cupmSetDevice(id_) != cupmErrorDeviceAlreadyInUse) CHKERRCUPM(cupmGetLastError());
  // need to update the device properties
  cerr = cupmGetDeviceProperties(&dprop_,id_);CHKERRCUPM(cerr);
  ierr = PetscInfo(nullptr,"Configured device %d\n",id_);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::view(PetscViewer viewer) const noexcept
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssert(devInitialized_,PETSC_COMM_SELF,PETSC_ERR_COR,"Device %d being viewed before it was initialized or configured",id_);
  ierr = PetscObjectTypeCompare(PetscObjectCast(viewer),PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    MPI_Comm    comm;
    PetscMPIInt rank;
    PetscViewer sviewer;

    ierr = PetscObjectGetComm(PetscObjectCast(viewer),&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"[%d] device %d: %s\n",rank,id_,dprop_.name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(sviewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Compute capability: %d.%d\n",dprop_.major,dprop_.minor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Multiprocessor Count: %d\n",dprop_.multiProcessorCount);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Maximum Grid Dimensions: %d x %d x %d\n",dprop_.maxGridSize[0],dprop_.maxGridSize[1],dprop_.maxGridSize[2]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Maximum Block Dimensions: %d x %d x %d\n",dprop_.maxThreadsDim[0],dprop_.maxThreadsDim[1],dprop_.maxThreadsDim[2]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Maximum Threads Per Block: %d\n",dprop_.maxThreadsPerBlock);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Warp Size: %d\n",dprop_.warpSize);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Total Global Memory (bytes): %zu\n",dprop_.totalGlobalMem);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Total Constant Memory (bytes): %zu\n",dprop_.totalConstMem);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Shared Memory Per Block (bytes): %zu\n",dprop_.sharedMemPerBlock);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Multiprocessor Clock Rate (KHz): %d\n",dprop_.clockRate);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Memory Clock Rate (KHz): %d\n",dprop_.memoryClockRate);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Memory Bus Width (bits): %d\n",dprop_.memoryBusWidth);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Peak Memory Bandwidth (GB/s): %f\n",2.0*dprop_.memoryClockRate*(dprop_.memoryBusWidth/8)/1.0e6);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Can map host memory: %s\n",dprop_.canMapHostMemory ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(sviewer,"Can execute multiple kernels concurrently: %s\n",dprop_.concurrentKernels ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static std::jmp_buf cupmMPIAwareJumpBuffer;
static bool         cupmMPIAwareJumpBufferSet;

// godspeed to anyone that attempts to call this function
void SilenceVariableIsNotNeededAndWillNotBeEmittedWarning_ThisFunctionShouldNeverBeCalled()
{
  PETSCABORT(MPI_COMM_NULL,INT_MAX);
  if (cupmMPIAwareJumpBufferSet) (void)cupmMPIAwareJumpBuffer;
}

#define CHKCUPMAWARE(...) if (PetscUnlikely((__VA_ARGS__) != cupmSuccess)) return false

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(bool Device<T>::DeviceInternal::CUPMAwareMPI_())
{
  constexpr int  bufSize = 2;
  constexpr int  hbuf[bufSize] = {1,0};
  int            *dbuf = nullptr;
  constexpr auto bytes = bufSize*sizeof(*dbuf);
  auto           awareness = false;
  cupmError_t    cerr;
  PetscErrorCode ierr;
  const auto     cupmSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
    if ((signal == SIGSEGV) && cupmMPIAwareJumpBufferSet) std::longjmp(cupmMPIAwareJumpBuffer,1);
    return PetscSignalHandlerDefault(signal,ptr);
  };

  PetscFunctionBegin;
  cerr = cupmMalloc(reinterpret_cast<void**>(&dbuf),bytes);CHKCUPMAWARE(cerr);
  cerr = cupmMemcpy(dbuf,hbuf,bytes,cupmMemcpyHostToDevice);CHKCUPMAWARE(cerr);
  ierr = PetscPushSignalHandler(cupmSignalHandler,nullptr);CHKERRABORT(PETSC_COMM_SELF,ierr);
  cupmMPIAwareJumpBufferSet = true;
  if (setjmp(cupmMPIAwareJumpBuffer)) {
    // if a segv was triggered in the MPI_Allreduce below, it is very likely due to MPI not
    // being GPU-aware
    awareness = false;
    // control flow up until this point:
    // 1. CUPMDevice<T>::CUPMDeviceInternal::MPICUPMAware__()
    // 2. MPI_Allreduce
    // 3. SIGSEGV
    // 4. PetscSignalHandler_Private
    // 5. cupmSignalHandler (lambda function)
    // 6. here
    // PetscSignalHandler_Private starts with PetscFunctionBegin and is pushed onto the stack
    // so we must undo this. This would be most naturally done in cupmSignalHandler, however
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
  cupmMPIAwareJumpBufferSet = false;
  ierr = PetscPopSignalHandler();CHKERRABORT(PETSC_COMM_SELF,ierr);
  cerr = cupmFree(dbuf);CHKCUPMAWARE(cerr);
  PetscFunctionReturn(awareness);
}

#undef CHKCUPMAWARE

template <DeviceType T>
PetscErrorCode Device<T>::finalize_() noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(!initialized_)) PetscFunctionReturn(0);
  for (auto&& device : devices_) device.reset();
  defaultDevice_ = PETSC_CUPM_DEVICE_NONE;  // disabled by default
  initialized_   = false;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscDeviceInitType *defaultInitType) noexcept
{
  auto           initType = std::make_pair(*defaultInitType,PETSC_FALSE);
  auto           initId   = std::make_pair(*defaultDeviceId,PETSC_FALSE);
  auto           initView = std::make_pair(PETSC_FALSE,PETSC_FALSE);
  int            ndev;
  cupmError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (initialized_) PetscFunctionReturn(0);
  initialized_ = true;
  ierr = PetscRegisterFinalize(finalize_);CHKERRQ(ierr);
  ierr = base_type::PetscOptionDeviceAll(comm,initType,initId,initView);CHKERRQ(ierr);
  initView.first = static_cast<decltype(initView.first)>(initView.first && initView.second);
  cerr = cupmGetDeviceCount(&ndev);
  // post-process the options and lay the groundwork for initialization if needs be
  if (PetscUnlikely((cerr == cupmErrorStubLibrary) || (cerr == cupmErrorNoDevice))) {
    if (PetscUnlikely((initType.first == PETSC_DEVICE_INIT_EAGER) || initView.first)) {
      const auto name    = cupmGetErrorName(cerr);
      const auto desc    = cupmGetErrorString(cerr);
      const auto backend = cupmName();
      SETERRQ(comm,PETSC_ERR_USER_INPUT,"Cannot eagerly initialize %s, as doing so results in %s error %d (%s) : %s",backend,backend,static_cast<PetscErrorCode>(cerr),name,desc);
    }
    initId.first   = -cerr;
    cerr = cupmGetLastError(); // reset error
    initType.first = PETSC_DEVICE_INIT_NONE;
  } else CHKERRCUPM(cerr);

  if (initType.first == PETSC_DEVICE_INIT_NONE) {
    if ((initId.first > 0) || (initId.first == PETSC_DECIDE)) initId.first = PETSC_CUPM_DEVICE_NONE;
  } else {
    ierr = PetscDeviceCheckDeviceCount_Internal(ndev);CHKERRQ(ierr);
    if (initId.first == PETSC_DECIDE) {
      if (ndev) {
        PetscMPIInt rank;

        ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
        initId.first   = rank % ndev;
      } else initId.first = 0;
    }
    if (initView.first) initType.first = PETSC_DEVICE_INIT_EAGER;
  }

  static_assert(std::is_same<PetscMPIInt,decltype(defaultDevice_)>::value,"");
  // initId.first is PetscInt, _defaultDevice is int
  ierr = PetscMPIIntCast(initId.first,&defaultDevice_);CHKERRQ(ierr);
  if (initType.first == PETSC_DEVICE_INIT_EAGER) {
    devices_[defaultDevice_] = DeviceInternal::makeDevice(defaultDevice_);
    ierr = devices_[defaultDevice_]->initialize();CHKERRQ(ierr);
    ierr = devices_[defaultDevice_]->configure();CHKERRQ(ierr);
    if (initView.first) {
      PetscViewer vwr;

      ierr = PetscLogInitialize();CHKERRQ(ierr);
      ierr = PetscViewerASCIIGetStdout(comm,&vwr);CHKERRQ(ierr);
      ierr = devices_[defaultDevice_]->view(vwr);CHKERRQ(ierr);
    }
  }

  // record the results of the initialization
  *defaultInitType = initType.first;
  *defaultDeviceId = initId.first;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::getDevice(PetscDevice device, PetscInt id) const noexcept
{
  const auto     cerr = static_cast<cupmError_t>(-defaultDevice_);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheck(defaultDevice_ != PETSC_CUPM_DEVICE_NONE,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Trying to retrieve a %s PetscDevice when it has been disabled",cupmName());
  PetscCheck(defaultDevice_ >= 0,PETSC_COMM_SELF,PETSC_ERR_GPU,"Cannot lazily initialize PetscDevice: %s error %d (%s) : %s",cupmName(),static_cast<PetscErrorCode>(cerr),cupmGetErrorName(cerr),cupmGetErrorString(cerr));
  if (id == PETSC_DECIDE) id = defaultDevice_;
  PetscAssert(static_cast<decltype(devices_.size())>(id) < devices_.size(),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only supports %zu number of devices but trying to get device with id %" PetscInt_FMT,devices_.size(),id);
  if (devices_[id]) {
    PetscAssert(id == devices_[id]->id(),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Entry %" PetscInt_FMT " contains device with mismatching id %d",id,devices_[id]->id());
  } else devices_[id] = DeviceInternal::makeDevice(id);
  ierr = devices_[id]->initialize();CHKERRQ(ierr);
  device->deviceId           = devices_[id]->id(); // technically id = _devices[id]->_id here
  device->ops->createcontext = create_;
  device->ops->configure     = this->configureDevice;
  device->ops->view          = this->viewDevice;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::configureDevice(PetscDevice device) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = devices_[device->deviceId]->configure();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <DeviceType T>
PetscErrorCode Device<T>::viewDevice(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // now this __shouldn't__ reconfigure the device, but there is a petscinfo call to indicate
  // it is being reconfigured
  ierr = devices_[device->deviceId]->configure();CHKERRQ(ierr);
  ierr = devices_[device->deviceId]->view(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// explicitly instantiate the classes
#if PetscDefined(HAVE_CUDA)
template class Device<DeviceType::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template class Device<DeviceType::HIP>;
#endif

} // namespace CUPM

} // namespace Device

} // namespace Petsc
