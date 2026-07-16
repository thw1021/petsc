#include <petsc/private/cpp/memory.hpp> // make_unique

#include "cupmdevice.hpp"

#include <algorithm>
#include <csetjmp> // for cuda mpi awareness
#include <csignal> // SIGSEGV
#include <iterator>
#include <type_traits>

#if PetscDefined(HAVE_HWLOC)
#include <hwloc.h>
#endif

namespace Petsc
{

namespace device
{

namespace cupm
{

// internal "impls" class for CUPMDevice. Each instance represents a single cupm device
template <DeviceType T>
class Device<T>::DeviceInternal {
  const int        id_;
  bool             devInitialized_ = false;
  cupmDeviceProp_t dprop_{}; // cudaDeviceProp appears to be an actual struct, i.e. you can't
                             // initialize it with nullptr or NULL (i've tried)

  static PetscErrorCode CUPMAwareMPI_(bool *) noexcept;

public:
  // default constructor
  explicit constexpr DeviceInternal(int dev) noexcept : id_(dev) { }

  // gather all relevant information for a particular device, a cupmDeviceProp_t is
  // usually sufficient here
  PetscErrorCode initialize() noexcept;
  PetscErrorCode configure() noexcept;
  PetscErrorCode view(PetscViewer) const noexcept;
  PetscErrorCode getattribute(PetscDeviceAttribute, void *) const noexcept;
  PetscErrorCode shutdown() noexcept;

  PETSC_NODISCARD auto id() const -> decltype(id_) { return id_; }
  PETSC_NODISCARD auto initialized() const -> decltype(devInitialized_) { return devInitialized_; }
  PETSC_NODISCARD auto prop() const -> const decltype(dprop_) & { return dprop_; }
};

// the goal here is simply to get the cupm backend to create its context, not to do any type of
// modification of it, or create objects (since these may be affected by subsequent
// configuration changes)
template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::initialize() noexcept
{
  PetscFunctionBegin;
  if (initialized()) PetscFunctionReturn(PETSC_SUCCESS);
  devInitialized_ = true;
  // need to do this BEFORE device has been set, although if the user
  // has already done this then we just ignore it
  if (cupmSetDeviceFlags(cupmDeviceMapHost) == cupmErrorSetOnActiveProcess) {
    // reset the error if it was cupmErrorSetOnActiveProcess
    const auto PETSC_UNUSED unused = cupmGetLastError();
  } else PetscCallCUPM(cupmGetLastError());
  // cuda 5.0+ will create a context when cupmSetDevice is called
  if (cupmSetDevice(id()) != cupmErrorDeviceAlreadyInUse) PetscCallCUPM(cupmGetLastError());
  // and in case it doesn't, explicitly call init here
  PetscCallCUPM(cupmInit(0));
#if PetscDefined(HAVE_CUDA)
  // nvmlInit() deprecated in NVML 5.319
  PetscCallNVML(nvmlInit_v2());
#endif
  // where is this variable defined and when is it set? who knows! but it is defined and set
  // at this point. either way, each device must make this check since I guess MPI might not be
  // aware of all of them?
  if (use_gpu_aware_mpi) {
    bool aware;

    // Even the MPI implementation is configured with GPU-aware, it might still need extra settings to enable it.
    // So we do the check at runtime with a code that works only with GPU-aware MPI.
    PetscCall(CUPMAwareMPI_(&aware));
    if (PetscUnlikely(!aware)) {
      PetscCall((*PetscErrorPrintf)("PETSc is configured with GPU support, but your MPI is not GPU-aware. For better performance, please use a GPU-aware MPI.\n"));
      PetscCall((*PetscErrorPrintf)("If you do not care, add option -use_gpu_aware_mpi 0. To not see the message again, add the option to your .petscrc, OR add it to the env var PETSC_OPTIONS.\n"));
      PetscCall((*PetscErrorPrintf)("For Open MPI, you need to configure it with CUDA, ROCm or GPU-aware UCX (https://docs.open-mpi.org/en/main/tuning-apps/accelerators/index.html)\n"));
      PetscCall((*PetscErrorPrintf)("If you already configured it with GPU-aware UCX, you may need 'mpiexec -n <np> --mca pml ucx' or export 'OMPI_MCA_pml=\"ucx\"' to use it.\n"));
      PetscCall((*PetscErrorPrintf)("For MVAPICH2-GDR, you need to set MV2_USE_CUDA=1 (http://mvapich.cse.ohio-state.edu/userguide/gdr/)\n"));
      PetscCall((*PetscErrorPrintf)("For Cray-MPICH, export MPICH_GPU_SUPPORT_ENABLED=1 (see its 'man mpi'); for MPICH, export MPIR_CVAR_ENABLE_GPU=1\n"));
      PETSCABORT(PETSC_COMM_SELF, PETSC_ERR_LIB);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::configure() noexcept
{
  PetscFunctionBegin;
  PetscAssert(initialized(), PETSC_COMM_SELF, PETSC_ERR_COR, "Device %d being configured before it was initialized", id());
  // why on EARTH nvidia insists on making otherwise informational states into
  // fully-fledged error codes is beyond me. Why couldn't a pointer to bool argument have
  // sufficed?!?!?!
  if (cupmSetDevice(id_) != cupmErrorDeviceAlreadyInUse) PetscCallCUPM(cupmGetLastError());
  // need to update the device properties
  PetscCallCUPM(cupmGetDeviceProperties(&dprop_, id_));
  PetscDeviceCUPMRuntimeArch = dprop_.major * 10 + dprop_.minor;
  PetscCall(PetscInfo(nullptr, "Configured device %d\n", id_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::view(PetscViewer viewer) const noexcept
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscAssert(initialized(), PETSC_COMM_SELF, PETSC_ERR_COR, "Device %d being viewed before it was initialized or configured", id());
  // we don't print device-specific info in CI-mode
  if (PetscUnlikely(PetscCIEnabled)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(viewer), PETSCVIEWERASCII, &isascii));
  if (isascii) {
    MPI_Comm    comm;
    PetscMPIInt rank;
    PetscViewer sviewer;

    int clock, memclock;
    PetscCallCUPM(cupmDeviceGetAttribute(&clock, cupmDevAttrClockRate, id_));
    PetscCallCUPM(cupmDeviceGetAttribute(&memclock, cupmDevAttrMemoryClockRate, id_));

    PetscCall(PetscObjectGetComm(PetscObjectCast(viewer), &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "[%d] name: %s\n", rank, dprop_.name));
    PetscCall(PetscViewerASCIIPushTab(sviewer));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Compute capability: %d.%d\n", dprop_.major, dprop_.minor));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Multiprocessor Count: %d\n", dprop_.multiProcessorCount));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Maximum Grid Dimensions: %d x %d x %d\n", dprop_.maxGridSize[0], dprop_.maxGridSize[1], dprop_.maxGridSize[2]));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Maximum Block Dimensions: %d x %d x %d\n", dprop_.maxThreadsDim[0], dprop_.maxThreadsDim[1], dprop_.maxThreadsDim[2]));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Maximum Threads Per Block: %d\n", dprop_.maxThreadsPerBlock));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Warp Size: %d\n", dprop_.warpSize));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Total Global Memory (bytes): %zu\n", dprop_.totalGlobalMem));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Total Constant Memory (bytes): %zu\n", dprop_.totalConstMem));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Shared Memory Per Block (bytes): %zu\n", dprop_.sharedMemPerBlock));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Multiprocessor Clock Rate (kHz): %d\n", clock));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Memory Clock Rate (kHz): %d\n", memclock));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Memory Bus Width (bits): %d\n", dprop_.memoryBusWidth));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Peak Memory Bandwidth (GB/s): %f\n", 2.0 * memclock * (dprop_.memoryBusWidth / 8) / 1.0e6));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Can map host memory: %s\n", dprop_.canMapHostMemory ? "PETSC_TRUE" : "PETSC_FALSE"));
    PetscCall(PetscViewerASCIIPrintf(sviewer, "Can execute multiple kernels concurrently: %s\n", dprop_.concurrentKernels ? "PETSC_TRUE" : "PETSC_FALSE"));
    PetscCall(PetscViewerASCIIPopTab(sviewer));
    PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::getattribute(PetscDeviceAttribute attr, void *value) const noexcept
{
  PetscFunctionBegin;
  PetscAssert(initialized(), PETSC_COMM_SELF, PETSC_ERR_COR, "Device %d was not initialized", id());
  switch (attr) {
  case PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK:
    *static_cast<std::size_t *>(value) = prop().sharedMemPerBlock;
  case PETSC_DEVICE_ATTR_MAX:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::shutdown() noexcept
{
  PetscFunctionBegin;
  if (!initialized()) PetscFunctionReturn(PETSC_SUCCESS);
#if PetscDefined(HAVE_CUDA)
  PetscCallNVML(nvmlShutdown());
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static std::jmp_buf cupmMPIAwareJumpBuffer;
static bool         cupmMPIAwareJumpBufferSet;

// godspeed to anyone that attempts to call this function
void SilenceVariableIsNotNeededAndWillNotBeEmittedWarning_ThisFunctionShouldNeverBeCalled()
{
  PETSCABORT(MPI_COMM_NULL, (PetscErrorCode)INT_MAX);
  if (cupmMPIAwareJumpBufferSet) (void)cupmMPIAwareJumpBuffer;
}

template <DeviceType T>
PetscErrorCode Device<T>::DeviceInternal::CUPMAwareMPI_(bool *awareness) noexcept
{
  constexpr int hbuf[]            = {1, 0};
  int          *dbuf              = nullptr;
  const auto    cupmSignalHandler = [](int signal, void *ptr) -> PetscErrorCode {
    if ((signal == SIGSEGV) && cupmMPIAwareJumpBufferSet) std::longjmp(cupmMPIAwareJumpBuffer, 1);
    return PetscSignalHandlerDefault(signal, ptr);
  };

  PetscFunctionBegin;
  *awareness = false;
  PetscCallCUPM(cupmMalloc(reinterpret_cast<void **>(&dbuf), sizeof(hbuf)));
  PetscCallCUPM(cupmMemcpy(dbuf, hbuf, sizeof(hbuf), cupmMemcpyHostToDevice));
  PetscCallCUPM(cupmDeviceSynchronize());
  PetscCall(PetscPushSignalHandler(cupmSignalHandler, nullptr));
  cupmMPIAwareJumpBufferSet = true;
  if (!setjmp(cupmMPIAwareJumpBuffer) && !MPI_Allreduce(dbuf, dbuf + 1, 1, MPI_INT, MPI_SUM, PETSC_COMM_SELF)) *awareness = true;
  cupmMPIAwareJumpBufferSet = false;
  PetscCall(PetscPopSignalHandler());
  PetscCallCUPM(cupmFree(dbuf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::finalize_() noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(!initialized_)) PetscFunctionReturn(PETSC_SUCCESS);
  for (auto &&device : devices_) {
    if (device) PetscCall(device->shutdown());
    device.reset();
  }
  defaultDevice_ = PETSC_CUPM_DEVICE_NONE; // disabled by default
  initialized_   = false;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PETSC_NODISCARD static PETSC_CONSTEXPR_14 const char *CUPM_VISIBLE_DEVICES() noexcept
{
  switch (T) {
  case DeviceType::CUDA:
    return "CUDA_VISIBLE_DEVICES";
  case DeviceType::HIP:
    return "ROCR_VISIBLE_DEVICES";
  }
  PetscUnreachable();
  return "PETSC_ERROR_PLIB";
}

/*
     The default device ID is
       MPI     -- rank % number_local_devices
       PyTorch -- getenv("LOCAL_RANK")
*/
template <DeviceType T>
PetscErrorCode Device<T>::select_device_petsc_decide_(MPI_Comm comm, PetscInt ndev, std::pair<PetscInt, PetscBool> *initId) noexcept
{
  PetscFunctionBegin;
  if (ndev) {
    /* TORCHELASTIC_RUN_ID is used as a proxy to determine if the current process was launched with torchrun */
    char *pytorch_exists = (char *)getenv("TORCHELASTIC_RUN_ID");
    char *pytorch_rank   = (char *)getenv("LOCAL_RANK");

    if (pytorch_exists && pytorch_rank) {
      char *endptr;

      initId->first = (PetscInt)strtol(pytorch_rank, &endptr, 10);
      PetscCheck(initId->first < ndev, PETSC_COMM_SELF, PETSC_ERR_LIB, "PyTorch environmental variable LOCAL_RANK %s > number devices %d", pytorch_rank, ndev);
    } else {
      PetscMPIInt rank;

      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      initId->first = rank % ndev;
    }
  } else initId->first = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(HAVE_HWLOC)
template <DeviceType T>
PetscErrorCode Device<T>::get_device_placement_in_cpuset_(PetscInt dev_count, hwloc_cpuset_t superset_cpuset, hwloc_cpuset_t process_cpuset, PetscInt *relative_device_index) noexcept
{
  // hwloc_bitmap_weight returns the number of non-zero entires in a cpuset.
  PetscInt cores_in_anc_obj = hwloc_bitmap_weight(superset_cpuset);
  PetscInt ctr              = 0;
  PetscFunctionBegin;
  // Enumerate cpuset in topological order
  for (auto icore = hwloc_bitmap_next(superset_cpuset, -1); icore != -1; icore = hwloc_bitmap_next(superset_cpuset, icore)) {
    // If the first CPU core in this thread's cpuset is found, set relative_device_index and return.
    if (icore == hwloc_bitmap_first(process_cpuset)) {
      *relative_device_index = dev_count * ctr / cores_in_anc_obj;
      break;
    }
    ctr++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::select_device_topology_aware_(PetscInt ndev, std::pair<PetscInt, PetscBool> *initId) noexcept
{
  PetscFunctionBegin;
  if (ndev == 1) {
    initId->first = 0;
  } else {
    // Ensure initId->first is set to a sensble fallback value if any hwloc
    // calls fail.
    initId->first = PETSC_DECIDE;
    std::vector<std::string> device_addrs(ndev, std::string(32, 0));
    // Get PCI Bus addresses for each CUPM device
    for (PetscInt idev = 0; idev < ndev; idev++) {
      PetscCallCUPM(cupmDeviceGetPCIBusId(device_addrs[idev].data(), 32, idev));
    }
    // Initialise hwloc topology object
    hwloc_topology_t topology;
    if (hwloc_topology_init(&topology) == -1) PetscFunctionReturn(PETSC_ERR_LIB);
    // Enables some internal optimisations in hwloc
    if (hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM) == -1) PetscFunctionReturn(PETSC_ERR_LIB);
    // By default IO devices (i.e. GPUs, storage, etc.) are filtered out. GPUs are considered
    // important. This filter makes sure those are included in the detected topology.
    if (hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT) == -1) PetscFunctionReturn(PETSC_ERR_LIB);
    if (hwloc_topology_load(topology) == -1) PetscFunctionReturn(PETSC_ERR_LIB);

    // Get the current thread's CPU binding mask
    hwloc_cpuset_t cpuset_mine = nullptr;
    cpuset_mine                = hwloc_bitmap_alloc();
    if (!cpuset_mine) PetscFunctionReturn(PETSC_ERR_LIB);
    if (hwloc_get_cpubind(topology, cpuset_mine, HWLOC_CPUBIND_THREAD) == -1) PetscFunctionReturn(PETSC_ERR_LIB);
    // The cpuset returned from hwloc_get_cpubind is in OS-order, which is not necessarily
    // the same as the topological order. Create a PU object from the first CPU detected
    // in this cpuset. A PU object is the lowest object in any hwloc topology. It is not
    // allowed to have any child objects.
    hwloc_obj *first_cpu = hwloc_get_pu_obj_by_os_index(topology, hwloc_bitmap_first(cpuset_mine));
    if (!first_cpu) PetscFunctionReturn(PETSC_ERR_LIB);

    std::vector<hwloc_obj_t> hwloc_devs(ndev);
    std::vector<hwloc_obj_t> common_ancestors(ndev);
    PetscInt                 selected_device = -1;
    PetscInt                 max_depth       = -1;
    PetscInt                 max_count       = 1;

    for (PetscInt idev = 0; idev < ndev; idev++) {
      // Use the PCI Bus address of each GPU to find it among the hwloc
      // topology object
      hwloc_devs[idev] = hwloc_get_pcidev_by_busidstring(topology, device_addrs[idev].c_str());
      if (!hwloc_devs[idev]) PetscFunctionReturn(PETSC_ERR_LIB);
      // hwloc does not consider IO devices to have ancestor or child objects, therefore
      // a call to hwloc_get_non_io_ancestor_obj is required for each device to find the
      // nearest non-IO device that has the same locality as the GPU. hwloc_get_common_ancestor
      // object returns the lowest-level object ('Group', CPU socket, Machine, etc) that contains
      // both the GPU and the first CPU core in the current cpuset.
      common_ancestors[idev] = hwloc_get_common_ancestor_obj(topology, first_cpu, hwloc_get_non_io_ancestor_obj(topology, hwloc_devs[idev]));
      // Every object at depth n+1 contains a subset of the PU and IO objects at depth n. Therefore
      // the smallest object containing both device idev and CPU core will have the largest
      // 'depth' value.
      if (common_ancestors[idev]->depth > max_depth) {
        max_depth       = common_ancestors[idev]->depth;
        max_count       = 1;
        selected_device = idev;
        // Prepare for the case where multiple devices are reported at the same depth level.
      } else if (common_ancestors[idev]->depth == max_depth) {
        max_count++;
      }
    }
    initId->first = selected_device;
    if (max_count > 1) {
      // Handle the case where multiple devices are reported at the same depth level. Find
      // all devices at the highest depth value.
      std::vector<PetscInt> devices_at_max_depth(max_count);
      PetscInt              ctr = 0;
      for (PetscInt idev = 0; idev < ndev; idev++) {
        if (common_ancestors[idev]->depth == max_depth) {
          devices_at_max_depth[ctr] = idev;
          ctr++;
        }
      }
      // Determine the cpuset of common ancestor of each device and the first CPU core
      // of this thread's cpuset. Then enumerate the ancestor cpuset in topological
      // order and determine how far through that cpuset the current core is bound.
      // Use that to select a device id out of the devices at max_depth.
      // Note that by the reasoning above, the common ancestor
      // cpuset should be identical for each device, so this loop should always break
      // in the first entry. If the loop does not break, initId->first has already been
      // set to a sensible value.
      for (PetscInt idev = 0; idev < max_count; idev++) {
        hwloc_cpuset_t anc_cpuset = common_ancestors[devices_at_max_depth[idev]]->cpuset;
        if (hwloc_bitmap_isincluded(first_cpu->cpuset, anc_cpuset)) {
          PetscInt relative_device_idx = 0;
          PetscCall(get_device_placement_in_cpuset_(max_count, anc_cpuset, cpuset_mine, &relative_device_idx));
          initId->first = devices_at_max_depth[relative_device_idx];
          break;
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

template <DeviceType T>
PetscErrorCode Device<T>::initialize(MPI_Comm comm, PetscInt *defaultDeviceId, PetscBool *defaultView, PetscDeviceInitType *defaultInitType) noexcept
{
  auto initId   = std::make_pair(*defaultDeviceId, PETSC_FALSE);
  auto initView = std::make_pair(*defaultView, PETSC_FALSE);
  auto initType = std::make_pair(*defaultInitType, PETSC_FALSE);
  int  ndev     = 0;

  PetscFunctionBegin;
  if (initialized_) PetscFunctionReturn(PETSC_SUCCESS);
  initialized_ = true;
  PetscCall(PetscRegisterFinalize(finalize_));
  PetscCall(base_type::PetscOptionDeviceAll(comm, initType, initId, initView));

  if (initType.first == PETSC_DEVICE_INIT_NONE) {
    initId.first = PETSC_CUPM_DEVICE_NONE;
  } else if (const auto cerr = cupmGetDeviceCount(&ndev)) {
    auto PETSC_UNUSED ignored = cupmGetLastError();

    PetscCheck((initType.first != PETSC_DEVICE_INIT_EAGER) && !initView.first, comm, PETSC_ERR_USER_INPUT, "Cannot eagerly initialize %s, as doing so results in %s error %d (%s) : %s", cupmName(), cupmName(), static_cast<PetscErrorCode>(cerr), cupmGetErrorName(cerr), cupmGetErrorString(cerr));
    // we won't be initializing anything anyways
    initType.first = PETSC_DEVICE_INIT_NONE;
    // save the error code for later
    initId.first = -static_cast<decltype(initId.first)>(cerr);
  }

  // check again for init type, since the device count may have changed it
  if (initType.first == PETSC_DEVICE_INIT_NONE) {
    // id < 0 (excluding PETSC_DECIDE and PETSC_DEVICE_TOPOLOGY_AWARE) indicates an error has occurred during setup
#if PetscDefined(HAVE_HWLOC)
    if ((initId.first > 0) || (initId.first == PETSC_DECIDE) || (initId.first == PETSC_DEVICE_TOPOLOGY_AWARE)) initId.first = PETSC_CUPM_DEVICE_NONE;
#else
    if ((initId.first > 0) || (initId.first == PETSC_DECIDE)) initId.first = PETSC_CUPM_DEVICE_NONE;
#endif
    // initType overrides initView
    initView.first = PETSC_FALSE;
  } else {
    PetscCall(PetscDeviceCheckDeviceCount_Internal(ndev));
#if PetscDefined(HAVE_HWLOC)
    if (initId.first == PETSC_DEVICE_TOPOLOGY_AWARE) {
      auto ierr = select_device_topology_aware_(ndev, &initId);
      // If select_device_topology_aware_ fails, it will set initId.first to
      // PETSC_DECIDE in order to fall through to the default algorithm
      if (ierr == PETSC_ERR_LIB) PetscCall(PetscInfo(nullptr, "Topology aware GPU device allocation failed. Falling back to default algorithm\n"));
    }
#endif
    if (initId.first == PETSC_DECIDE) {
      PetscCall(select_device_petsc_decide_(comm, ndev, &initId));
    }
    if (initView.first) initType.first = PETSC_DEVICE_INIT_EAGER;
    PetscCall(PetscInfo(nullptr, "GPU device id selected: %d\n", initId.first));
  }

  static_assert(std::is_same<PetscMPIInt, decltype(defaultDevice_)>::value, "");
  // initId.first is PetscInt, _defaultDevice is int
  PetscCall(PetscMPIIntCast(initId.first, &defaultDevice_));
  // record the results of the initialization
  *defaultDeviceId = initId.first;
  *defaultView     = initView.first;
  *defaultInitType = initType.first;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::init_device_id_(PetscInt *inid) const noexcept
{
  const auto id   = *inid == PETSC_DECIDE ? defaultDevice_ : (int)*inid;
  const auto cerr = static_cast<cupmError_t>(-defaultDevice_);

  PetscFunctionBegin;
  PetscCheck(defaultDevice_ != PETSC_CUPM_DEVICE_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Trying to retrieve a %s PetscDevice when it has been disabled", cupmName());
  PetscCheck(defaultDevice_ >= 0, PETSC_COMM_SELF, PETSC_ERR_GPU, "Cannot lazily initialize PetscDevice: %s error %d (%s) : %s", cupmName(), static_cast<PetscErrorCode>(cerr), cupmGetErrorName(cerr), cupmGetErrorString(cerr));
  PetscAssert(static_cast<decltype(devices_.size())>(id) < devices_.size(), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Only supports %zu number of devices but trying to get device with id %d", devices_.size(), id);

  if (!devices_[id]) devices_[id] = util::make_unique<DeviceInternal>(id);
  PetscAssert(id == devices_[id]->id(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Entry %d contains device with mismatching id %d", id, devices_[id]->id());
  PetscCall(devices_[id]->initialize());
  *inid = id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::configure_device_(PetscDevice device) noexcept
{
  PetscFunctionBegin;
  PetscCall(devices_[device->deviceId]->configure());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::view_device_(PetscDevice device, PetscViewer viewer) noexcept
{
  PetscFunctionBegin;
  // now this __shouldn't__ reconfigure the device, but there is a petscinfo call to indicate
  // it is being reconfigured
  PetscCall(devices_[device->deviceId]->configure());
  PetscCall(devices_[device->deviceId]->view(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
PetscErrorCode Device<T>::get_attribute_(PetscInt id, PetscDeviceAttribute attr, void *value) noexcept
{
  PetscFunctionBegin;
  PetscCall(devices_[id]->getattribute(attr, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// explicitly instantiate the classes
#if PetscDefined(HAVE_CUDA)
template class Device<DeviceType::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template class Device<DeviceType::HIP>;
#endif

} // namespace cupm

} // namespace device

} // namespace Petsc
