#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmsolverinterface.hpp>
#include <petsc/private/logimpl.h>

#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpp/array.hpp>
#include <petsc/private/cpp/type_traits.hpp>

#include "../segmentedmempool.hpp"
#include "cupmallocator.hpp"
#include "cupmstream.hpp"
#include "cupmevent.hpp"

#if PetscDefined(HAVE_CUDA)
  #include <nvToolsExtCudaRt.h>
#else
  #define nvtxNameCudaStreamA(strm, name) ((void)name)
#endif

#if PetscDefined(HAVE_MPIX_STREAM)
  #include <unordered_map>
#endif

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

template <DeviceType T>
class DeviceContext : public RegisterFinalizeable<DeviceContext<T>>, SolverInterface<T> {
public:
  PETSC_CUPMSOLVER_INHERIT_INTERFACE_TYPEDEFS_USING(T);

private:
  using stream_tag = util::type_identity<cupmStream_t>;
  using blas_tag   = util::type_identity<cupmBlasHandle_t>;
  using solver_tag = util::type_identity<cupmSolverHandle_t>;

  using stream_type       = CUPMStream<T>;
  using stream_cache_type = std::unordered_map<int, std::unordered_map<MPI_Comm, MPI_Comm>>;

public:
  // This is the canonical PETSc "impls" struct that normally resides in a standalone impls
  // header, but since we are using the power of templates it must be declared part of
  // this class to have easy access the same typedefs. Technically one can make a
  // templated struct outside the class but it's more code for the same result.
  struct PetscDeviceContext_IMPLS : memory::PoolAllocated<PetscDeviceContext_IMPLS> {
    stream_type stream{};
    cupmEvent_t event{};
    cupmEvent_t begin{}; // timer-only
    cupmEvent_t end{};   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse{};
#endif
    cupmBlasHandle_t   blas{};
    cupmSolverHandle_t solver{};
#if PetscDefined(HAVE_MPIX_STREAM)
    class MPIUX_Stream {
    public:
      PETSC_NODISCARD const MPIX_Stream &get() const noexcept { return stream_; }

      PetscErrorCode init(const cupmStream_t &strm) noexcept
      {
        PetscFunctionBegin;
        if (PetscUnlikely(!init_)) {
          MPI_Info info;

          init_ = true;
          PetscCallMPI(MPI_Info_create(&info));
          PetscCallMPI(MPI_Info_set(info, "type", T == DeviceType::CUDA ? "cudaStream_t" : "hipStream_t"));
          PetscCallMPI(MPIX_Info_set_hex(info, "value", &strm, sizeof(strm)));
          PetscCallMPI(MPIX_Stream_create(info, &stream_));
          PetscCallMPI(MPI_Info_free(&info));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
      }

      PetscErrorCode destroy(stream_cache_type &cache, int stream_id) noexcept
      {
        PetscFunctionBegin;
        if (init_) {
          const auto it = cache.find(stream_id);

          if (it != cache.end()) {
            for (auto &&subit : it->second) PetscCallMPI(MPI_Comm_free(&subit.second));
            PetscCallCXX(cache.erase(it));
          }
          PetscCallMPI(MPIX_Stream_free(&stream_));
          init_ = false;
        }
        PetscFunctionReturn(PETSC_SUCCESS);
      }

    private:
      bool        init_{};
      MPIX_Stream stream_{};
    } mpi_stream{};
#endif

    constexpr PetscDeviceContext_IMPLS() noexcept = default;

    PETSC_NODISCARD const cupmStream_t &get(stream_tag) const noexcept { return this->stream.get_stream(); }

    PETSC_NODISCARD const cupmBlasHandle_t &get(blas_tag) const noexcept { return this->blas; }

    PETSC_NODISCARD const cupmSolverHandle_t &get(solver_tag) const noexcept { return this->solver; }
  };

  static PetscErrorCode register_finalize_(PetscDevice device) noexcept
  {
    uint64_t      threshold = UINT64_MAX;
    cupmMemPool_t mempool;
    PetscInt      id;

    PetscFunctionBegin;
    PetscCall(PetscDeviceGetDeviceId(device, &id));
    PetscCallCUPM(cupmDeviceGetMemPool(&mempool, static_cast<int>(id)));
    PetscCallCUPM(cupmMemPoolSetAttribute(mempool, cupmMemPoolAttrReleaseThreshold, &threshold));
    blashandles_.fill(nullptr);
    solverhandles_.fill(nullptr);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode finalize_() noexcept
  {
    PetscFunctionBegin;
    for (auto &&handle : blashandles_) {
      if (handle) {
        PetscCallCUPMBLAS(cupmBlasDestroy(handle));
        handle = nullptr;
      }
    }

    for (auto &&handle : solverhandles_) {
      if (handle) {
        PetscCallCUPMSOLVER(cupmSolverDestroy(handle));
        handle = nullptr;
      }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  static std::array<cupmBlasHandle_t, PETSC_DEVICE_MAX_DEVICES>   blashandles_;
  static std::array<cupmSolverHandle_t, PETSC_DEVICE_MAX_DEVICES> solverhandles_;
#if PetscDefined(HAVE_MPIX_STREAM)
  static stream_cache_type stream_comm_cache_;
#endif

  PETSC_NODISCARD static constexpr PetscDeviceContext_IMPLS *impls_cast_(PetscDeviceContext ptr) noexcept { return static_cast<PetscDeviceContext_IMPLS *>(ptr->data); }

  PETSC_NODISCARD static constexpr cupmEvent_t event_cast_(PetscEvent event) noexcept { return static_cast<cupmEvent_t>(event->data); }

  PETSC_NODISCARD static PetscLogEvent CUPMBLAS_HANDLE_CREATE() noexcept { return T == DeviceType::CUDA ? CUBLAS_HANDLE_CREATE : HIPBLAS_HANDLE_CREATE; }

  PETSC_NODISCARD static PetscLogEvent CUPMSOLVER_HANDLE_CREATE() noexcept { return T == DeviceType::CUDA ? CUSOLVER_HANDLE_CREATE : HIPSOLVER_HANDLE_CREATE; }

  // this exists purely to satisfy the compiler so the tag-based dispatch works for the other
  // handles
  static PetscErrorCode initialize_handle_(stream_tag, PetscDeviceContext) noexcept { return PETSC_SUCCESS; }

  static PetscErrorCode initialize_handle_(blas_tag, PetscDeviceContext dctx) noexcept
  {
    const auto dci    = impls_cast_(dctx);
    auto      &handle = blashandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    if (!handle) {
      PetscCall(PetscLogEventsPause());
      PetscCall(PetscLogEventBegin(CUPMBLAS_HANDLE_CREATE(), 0, 0, 0, 0));
      for (auto i = 0; i < 3; ++i) {
        const auto cberr = cupmBlasCreate(handle.ptr_to());

        if (PetscLikely(cberr == CUPMBLAS_STATUS_SUCCESS)) break;
        if (PetscUnlikely(cberr != CUPMBLAS_STATUS_ALLOC_FAILED) && (cberr != CUPMBLAS_STATUS_NOT_INITIALIZED)) PetscCallCUPMBLAS(cberr);
        if (i != 2) {
          PetscCall(PetscSleep(3));
          continue;
        }
        PetscCheck(cberr == CUPMBLAS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, "Unable to initialize %s", cupmBlasName());
      }
      PetscCall(PetscLogEventEnd(CUPMBLAS_HANDLE_CREATE(), 0, 0, 0, 0));
      PetscCall(PetscLogEventsResume());
    }
    PetscCallCUPMBLAS(cupmBlasSetStream(handle, dci->stream.get_stream()));
    dci->blas = handle;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode initialize_handle_(solver_tag, PetscDeviceContext dctx) noexcept
  {
    const auto dci    = impls_cast_(dctx);
    auto      &handle = solverhandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    if (!handle) {
      PetscCall(PetscLogEventsPause());
      PetscCall(PetscLogEventBegin(CUPMSOLVER_HANDLE_CREATE(), 0, 0, 0, 0));
      for (auto i = 0; i < 3; ++i) {
        const auto cerr = cupmSolverCreate(&handle);

        if (PetscLikely(cerr == CUPMSOLVER_STATUS_SUCCESS)) break;
        if ((cerr != CUPMSOLVER_STATUS_NOT_INITIALIZED) && (cerr != CUPMSOLVER_STATUS_ALLOC_FAILED)) PetscCallCUPMSOLVER(cerr);
        if (i < 2) {
          PetscCall(PetscSleep(3));
          continue;
        }
        PetscCheck(cerr == CUPMSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_GPU_RESOURCE, "Unable to initialize %s", cupmSolverName());
      }
      PetscCall(PetscLogEventEnd(CUPMSOLVER_HANDLE_CREATE(), 0, 0, 0, 0));
      PetscCall(PetscLogEventsResume());
    }
    PetscCallCUPMSOLVER(cupmSolverSetStream(handle, dci->stream.get_stream()));
    dci->solver = handle;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode check_current_device_(PetscDeviceContext dctxl, PetscDeviceContext dctxr) noexcept
  {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG)) {
      PetscInt      devids[2];
      PetscObjectId ids[2];
      int           i = 0;

      for (auto &&dctx : {dctxl, dctxr}) {
        PetscDevice device;

        PetscCall(PetscDeviceContextGetDevice(dctx, &device));
        PetscCall(PetscDeviceGetDeviceId(device, devids + i));
        PetscCall(PetscDeviceCheckDeviceCount_Internal(devids[i]));
        PetscCall(PetscObjectGetId(PetscObjectCast(dctx), ids + i));
        ++i;
      }
      PetscCheck(devids[0] == devids[1], PETSC_COMM_SELF, PETSC_ERR_GPU, "Device contexts must be on the same device; dctx A (id %" PetscInt64_FMT " device id %" PetscInt_FMT ") dctx B (id %" PetscInt64_FMT " device id %" PetscInt_FMT ")", ids[0], devids[0], ids[1], devids[1]);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode check_current_device_(PetscDeviceContext dctx) noexcept { return check_current_device_(dctx, dctx); }

  template <typename Allocator, typename PoolType = ::Petsc::memory::SegmentedMemoryPool<typename Allocator::value_type, Allocator, 256 * sizeof(PetscScalar)>>
  PETSC_NODISCARD static PoolType &default_pool_() noexcept
  {
    static PoolType pool;
    return pool;
  }

  static PetscErrorCode check_memtype_(PetscMemType mtype, const char mess[]) noexcept
  {
    PetscFunctionBegin;
    PetscCheck(PetscMemTypeHost(mtype) || (mtype == PETSC_MEMTYPE_DEVICE) || (mtype == PETSC_MEMTYPE_CUPM()), PETSC_COMM_SELF, PETSC_ERR_SUP, "%s device context can only handle %s (pinned) host or device memory", cupmName(), mess);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

public:
  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  static PetscErrorCode destroy(PetscDeviceContext) noexcept;
  static PetscErrorCode changeStreamType(PetscDeviceContext, PetscStreamType) noexcept;
  static PetscErrorCode setUp(PetscDeviceContext) noexcept;
  static PetscErrorCode query(PetscDeviceContext, PetscBool *) noexcept;
  static PetscErrorCode waitForContext(PetscDeviceContext, PetscDeviceContext) noexcept;
  static PetscErrorCode synchronize(PetscDeviceContext) noexcept;
  template <typename Handle_t>
  static PetscErrorCode getHandle(PetscDeviceContext, void *) noexcept;
  template <typename Handle_t>
  static PetscErrorCode getHandlePtr(PetscDeviceContext, void **) noexcept;
  static PetscErrorCode beginTimer(PetscDeviceContext) noexcept;
  static PetscErrorCode endTimer(PetscDeviceContext, PetscLogDouble *) noexcept;
  static PetscErrorCode memAlloc(PetscDeviceContext, PetscBool, PetscMemType, std::size_t, std::size_t, void **) noexcept;
  static PetscErrorCode memFree(PetscDeviceContext, PetscPointerAttributes *, void **) noexcept;
  static PetscErrorCode memCopy(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, std::size_t, PetscDeviceCopyMode) noexcept;
  static PetscErrorCode memSet(PetscDeviceContext, PetscMemType, void *, PetscInt, std::size_t) noexcept;
  static PetscErrorCode createEvent(PetscDeviceContext, PetscEvent) noexcept;
  static PetscErrorCode recordEvent(PetscDeviceContext, PetscEvent) noexcept;
  static PetscErrorCode waitForEvent(PetscDeviceContext, PetscEvent) noexcept;

  static PetscErrorCode getStreamComm(PetscDeviceContext, MPI_Comm, MPI_Comm *) noexcept;

  // clang-format off
  static constexpr _DeviceContextOps ops = {
    PetscDesignatedInitializer(destroy, destroy),
    PetscDesignatedInitializer(changestreamtype, changeStreamType),
    PetscDesignatedInitializer(setup, setUp),
    PetscDesignatedInitializer(query, query),
    PetscDesignatedInitializer(waitforcontext, waitForContext),
    PetscDesignatedInitializer(synchronize, synchronize),
    PetscDesignatedInitializer(getblashandle, getHandle<blas_tag>),
    PetscDesignatedInitializer(getsolverhandle, getHandle<solver_tag>),
    PetscDesignatedInitializer(getstreamhandle, getHandlePtr<stream_tag>),
    PetscDesignatedInitializer(begintimer, beginTimer),
    PetscDesignatedInitializer(endtimer, endTimer),
    PetscDesignatedInitializer(memalloc, memAlloc),
    PetscDesignatedInitializer(memfree, memFree),
    PetscDesignatedInitializer(memrealloc, nullptr),
    PetscDesignatedInitializer(memcopy, memCopy),
    PetscDesignatedInitializer(memset, memSet),
    PetscDesignatedInitializer(createevent, createEvent),
    PetscDesignatedInitializer(recordevent, recordEvent),
    PetscDesignatedInitializer(waitforevent, waitForEvent),
    PetscDesignatedInitializer(getstreamcomm, getStreamComm)
  };
  // clang-format on
};

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::destroy(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (const auto dci = impls_cast_(dctx)) {
    auto &stream = dci->stream;

#if PetscDefined(HAVE_MPIX_STREAM)
    PetscCall(dci->mpi_stream.destroy(stream_comm_cache_, stream.get_id()));
#endif
    PetscCall(stream.destroy());
    if (dci->event) PetscCall(cupm_fast_event_pool<T>().deallocate(&dci->event));
    if (dci->begin) PetscCallCUPM(cupmEventDestroy(dci->begin));
    if (dci->end) PetscCallCUPM(cupmEventDestroy(dci->end));
    delete dci;
    dctx->data = nullptr;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::changeStreamType(PetscDeviceContext dctx, PetscStreamType stype) noexcept
{
  const auto dci    = impls_cast_(dctx);
  auto      &stream = dci->stream;
  const auto old_id = stream.get_id();
  bool       did_change;

  PetscFunctionBegin;
  PetscCall(stream.change_type(stype, &did_change));
#if PetscDefined(HAVE_MPIX_STREAM)
  if (did_change) PetscCall(dci->mpi_stream.destroy(stream_comm_cache_, old_id));
#else
  static_cast<void>(did_change);
  static_cast<void>(old_id);
#endif
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::setUp(PetscDeviceContext dctx) noexcept
{
  const auto dci   = impls_cast_(dctx);
  auto      &event = dci->event;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(changeStreamType(dctx, dctx->streamType));
  if (const auto name = PetscObjectCast(dctx)->name) nvtxNameCudaStreamA(dci->stream.get_stream(), name);
  if (!event) PetscCall(cupm_fast_event_pool<T>().allocate(&event));
#if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::query(PetscDeviceContext dctx, PetscBool *idle) noexcept
{
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  switch (auto cerr = cupmStreamQuery(impls_cast_(dctx)->stream.get_stream())) {
  case cupmSuccess:
    *idle = PETSC_TRUE;
    break;
  case cupmErrorNotReady:
    *idle = PETSC_FALSE;
    // reset the error
    cerr = cupmGetLastError();
    static_cast<void>(cerr);
    break;
  default:
    PetscCallCUPM(cerr);
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) noexcept
{
  const auto dcib  = impls_cast_(dctxb);
  const auto event = dcib->event;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctxa, dctxb));
  PetscCallCUPM(cupmEventRecord(event, dcib->stream.get_stream()));
  PetscCallCUPM(cupmStreamWaitEvent(impls_cast_(dctxa)->stream.get_stream(), event, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::synchronize(PetscDeviceContext dctx) noexcept
{
  auto idle = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(query(dctx, &idle));
  if (!idle) PetscCallCUPM(cupmStreamSynchronize(impls_cast_(dctx)->stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
template <typename handle_t>
inline PetscErrorCode DeviceContext<T>::getHandle(PetscDeviceContext dctx, void *handle) noexcept
{
  PetscFunctionBegin;
  PetscCall(initialize_handle_(handle_t{}, dctx));
  *static_cast<typename handle_t::type *>(handle) = impls_cast_(dctx)->get(handle_t{});
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
template <typename handle_t>
inline PetscErrorCode DeviceContext<T>::getHandlePtr(PetscDeviceContext dctx, void **handle) noexcept
{
  using handle_type = typename handle_t::type;

  PetscFunctionBegin;
  PetscCall(initialize_handle_(handle_t{}, dctx));
  *reinterpret_cast<handle_type **>(handle) = const_cast<handle_type *>(std::addressof(impls_cast_(dctx)->get(handle_t{})));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::beginTimer(PetscDeviceContext dctx) noexcept
{
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
#if PetscDefined(USE_DEBUG)
  PetscCheck(!dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
#endif
  if (!dci->begin) {
    PetscAssert(!dci->end, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Don't have a 'begin' event, but somehow have a end event %p", (void *)dci->end);
    PetscCallCUPM(cupmEventCreate(&dci->begin));
    PetscCallCUPM(cupmEventCreate(&dci->end));
  }
  PetscCallCUPM(cupmEventRecord(dci->begin, dci->stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed) noexcept
{
  float      gtime;
  const auto dci = impls_cast_(dctx);
  const auto end = dci->end;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
#if PetscDefined(USE_DEBUG)
  PetscCheck(dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscCallCUPM(cupmEventRecord(end, dci->stream.get_stream()));
  PetscCallCUPM(cupmEventSynchronize(end));
  PetscCallCUPM(cupmEventElapsedTime(&gtime, dci->begin, end));
  *elapsed = static_cast<util::remove_pointer_t<decltype(elapsed)>>(gtime);
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::memAlloc(PetscDeviceContext dctx, PetscBool clear, PetscMemType mtype, std::size_t n, std::size_t alignment, void **dest) noexcept
{
  const auto &stream = impls_cast_(dctx)->stream;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "allocating"));
  if (PetscMemTypeHost(mtype)) {
    PetscCall(default_pool_<HostAllocator<T>>().allocate(dctx, stream.get_id(), n, reinterpret_cast<char **>(dest), alignment));
  } else {
    PetscCall(default_pool_<DeviceAllocator<T>>().allocate(dctx, stream.get_id(), n, reinterpret_cast<char **>(dest), alignment));
  }
  if (clear && n) PetscCallCUPM(cupmMemsetAsync(*dest, 0, n, stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::memFree(PetscDeviceContext dctx, PetscPointerAttributes *attr, void **ptr) noexcept
{
  const auto &stream = impls_cast_(dctx)->stream;
  const auto  mtype  = attr->mtype;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "freeing"));
  if (!*ptr) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscMemTypeHost(mtype)) {
    PetscCall(default_pool_<HostAllocator<T>>().deallocate(dctx, stream.get_id(), reinterpret_cast<char **>(ptr)));

    // if ptr exists still exists the pool didn't own it
    if (*ptr) {
      auto registered = PETSC_FALSE, managed = PETSC_FALSE;

      PetscCall(PetscCUPMGetMemType(*ptr, nullptr, &registered, &managed));
      if (registered) {
        PetscCallCUPM(cupmFreeHost(*ptr));
        attr->id = PETSC_DELETED_MEMORY_ID;
      } else if (managed) {
        PetscCallCUPM(cupmFreeAsync(*ptr, stream.get_stream()));
        attr->id = PETSC_DELETED_MEMORY_ID;
      }
    }
  } else {
    PetscCall(default_pool_<DeviceAllocator<T>>().deallocate(dctx, stream.get_id(), reinterpret_cast<char **>(ptr)));

    // if ptr still exists the pool didn't own it
    if (*ptr) {
      PetscCallCUPM(cupmFreeAsync(*ptr, stream.get_stream()));
      attr->id = PETSC_DELETED_MEMORY_ID;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::memCopy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, PetscDeviceCopyMode mode) noexcept
{
  PetscFunctionBegin;
  // can't use PetscCUPMMemcpyAsync here since we don't know sizeof(*src)...
  if (mode == PETSC_DEVICE_COPY_HTOH) {
    PetscBool idle;

    PetscCall(query(dctx, &idle));
    if (idle) {
      PetscCall(PetscMemcpy(dest, src, n));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCallCUPM(cupmMemcpyAsync(dest, src, n, PetscDeviceCopyModeToCUPMMemcpyKind(mode), impls_cast_(dctx)->stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::memSet(PetscDeviceContext dctx, PetscMemType mtype, void *ptr, PetscInt v, std::size_t n) noexcept
{
  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  PetscCall(check_memtype_(mtype, "zeroing"));
  PetscCallCUPM(cupmMemsetAsync(ptr, static_cast<int>(v), n, impls_cast_(dctx)->stream.get_stream()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::createEvent(PetscDeviceContext, PetscEvent event) noexcept
{
  cupmEvent_t cupm_event;

  PetscFunctionBegin;
  PetscCall(cupm_fast_event_pool<T>().allocate(&cupm_event));
  event->data    = cupm_event;
  event->destroy = [](PetscEvent event) {
    auto cupm_event = event_cast_(event);

    PetscFunctionBegin;
    PetscCall(cupm_fast_event_pool<T>().deallocate(&cupm_event));
    event->data = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  };
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace
{

// ASYNC TODO: remove
#define TIME_EVENTS PetscDefined(USE_DEBUG)
#if PetscHasBuiltin(__builtin_ia32_rdtsc)
  #include <cstddef>
  #define PETSC_USE_IA32 1
#else
  #include <chrono>
  #define PETSC_USE_IA32 0
#endif

struct EventCounter : Petsc::RegisterFinalizeable<EventCounter> {
  std::string name;
  std::size_t cnt = 0;
#if PETSC_USE_IA32
  std::uint64_t t1;
  std::uint64_t duration = 0;
#else
  using clock_t = std::chrono::steady_clock;
  typename clock_t::time_point t1;
  std::chrono::nanoseconds     duration{0};
#endif

  EventCounter(std::string name) : name{std::move(name)} { }

  void tick()
  {
#if TIME_EVENTS
    (void)(this->register_finalize());
    ++cnt;
#endif
  }

  void begin()
  {
    this->tick();
#if TIME_EVENTS
  #if PETSC_USE_IA32
    t1 = __builtin_ia32_rdtsc();
  #else
    t1      = clock_t::now();
  #endif
#endif
    return;
  }

  void end()
  {
#if TIME_EVENTS
  #if PETSC_USE_IA32
    auto t2 = __builtin_ia32_rdtsc();
    duration += t2 - t1;
  #else
    auto t2 = clock_t::now();
    duration += t2 - t1;
  #endif
#endif
    return;
  }

  PetscErrorCode finalize_() noexcept
  {
#if PETSC_USE_IA32
    auto count = (double)(duration / 2.85);
#else
    auto count = (double)(duration.count());
#endif
    auto       avg_count = count / cnt;
    char       duration_name, avg_duration_name;
    const auto scale_count = [](double &count, char &dur_name) {
      if (count > 1000000.) {
        count /= 1000000.;
        dur_name = 'm';
      } else if (count > 1000.) {
        count /= 1000.;
        dur_name = 'u';
      } else {
        dur_name = 'n';
      }
    };

    PetscFunctionBegin;
    scale_count(count, duration_name);
    scale_count(avg_count, avg_duration_name);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: %zu calls, total %g %cs (avg %g %cs)\n", name.c_str(), cnt, count, duration_name, avg_count, avg_duration_name));
    cnt = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

} // namespace

static EventCounter record_counter("Event record");

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::recordEvent(PetscDeviceContext dctx, PetscEvent event) noexcept
{
  PetscFunctionBegin;
  record_counter.begin();
  PetscCallCUPM(cupmEventRecord(event_cast_(event), impls_cast_(dctx)->stream.get_stream()));
  record_counter.end();
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::waitForEvent(PetscDeviceContext dctx, PetscEvent event) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmStreamWaitEvent(impls_cast_(dctx)->stream.get_stream(), event_cast_(event), 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode DeviceContext<T>::getStreamComm(PetscDeviceContext dctx, MPI_Comm comm, MPI_Comm *scomm) noexcept
{
#if PetscDefined(HAVE_MPIX_STREAM)
  const auto  dci     = impls_cast_(dctx);
  const auto &stream  = dci->stream;
  auto       &retcomm = stream_comm_cache_[stream.get_id()].emplace(comm, MPI_COMM_NULL).first->second;

  PetscFunctionBegin;
  if (retcomm == MPI_COMM_NULL) {
    auto &mpi_stream = dci->mpi_stream;

    // the comm did not exist yet
    PetscCall(mpi_stream.init(stream.get_stream()));
    PetscCallMPI(MPIX_Stream_comm_create(comm, mpi_stream.get(), &retcomm));
  }
  *scomm = retcomm;
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  PetscFunctionBegin;
  SETERRQ(comm, PETSC_ERR_SUP, "MPI Implementation does not support MPIX_Stream");
#endif
}

// initialize the static member variables
template <DeviceType T>
std::array<typename DeviceContext<T>::cupmBlasHandle_t, PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::blashandles_ = {};

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmSolverHandle_t, PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::solverhandles_ = {};

#if PetscDefined(HAVE_MPIX_STREAM)
template <DeviceType T>
std::unordered_map<int, std::unordered_map<MPI_Comm, MPI_Comm>> DeviceContext<T>::stream_comm_cache_ = {};
#endif

template <DeviceType T>
constexpr _DeviceContextOps DeviceContext<T>::ops;

} // namespace impl

template <DeviceType T>
inline PetscErrorCode PetscDeviceContextCreate_CUPM(PetscDeviceContext dctx) noexcept
{
  using CUPMContextIMPL    = impl::DeviceContext<T>;
  static auto cupm_context = CUPMContextIMPL{};

  PetscFunctionBegin;
  PetscCall(cupm_context.register_finalize(dctx->device));
  PetscCallCXX(dctx->data = new typename CUPMContextIMPL::PetscDeviceContext_IMPLS{});
  *dctx->ops = cupm_context.ops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode PetscCUPMBLASGetHandle(void *handle) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscAssertPointer(handle, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, impl::DeviceContext<T>::PETSC_DEVICE_CUPM()));
  PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx, handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode PetscCUPMSolverGetHandle(void *handle) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscAssertPointer(handle, 1);
  PetscCall(PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx, impl::DeviceContext<T>::PETSC_DEVICE_CUPM()));
  PetscCall(PetscDeviceContextGetSOLVERHandle_Internal(dctx, handle));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // PETSCDEVICECONTEXTCUDA_HPP
