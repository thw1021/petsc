#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>

#include <array>
#include <vector>

namespace Petsc
{

namespace Device
{

namespace CUPM
{

namespace Impl
{

namespace detail
{

// for tag-based dispatch of handle retrieval
template <typename T> struct HandleTag { using type = T; };

struct PETSC_TEMPLATE_VISIBILITY_INTERNAL MemoryBlock;

struct MemoryBlock
{
  using size_type = std::size_t;

  size_type start;
  size_type size;
  bool      open;

  constexpr MemoryBlock(size_type start_, size_type size_, bool open_ = false) noexcept
    : start(start_), size(size_), open(open_)
  { }
};

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize> class PETSC_TEMPLATE_VISIBILITY_INTERNAL SegmentedMemoryPool;

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
class SegmentedMemoryPool
{
  using BlocksType = std::vector<MemoryBlock>;
  using size_type  = BlocksType::value_type::size_type;

  const AllocType  allocate_;
  const FreeType   destroy_;
  BlocksType       blocks_;
  MemType         *mem_pool_;

public:
  constexpr SegmentedMemoryPool(AllocType&& alloc, FreeType&& destroy) noexcept
    : allocate_(std::forward<AllocType>(alloc)), destroy_(std::forward<FreeType>(destroy)),
      blocks_(), mem_pool_(nullptr)
  { }

  PETSC_NODISCARD PetscErrorCode finalize() noexcept;
  PETSC_NODISCARD PetscErrorCode initialize() noexcept;
  PETSC_NODISCARD PetscErrorCode get(PetscInt,MemType**) noexcept;
  PETSC_NODISCARD PetscErrorCode release(MemType**) noexcept;
  PETSC_NODISCARD bool           owns_pointer(const MemType*) const noexcept;
};

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::finalize() noexcept
{
  PetscFunctionBegin;
  PetscCall(destroy_(mem_pool_));
  mem_pool_ = nullptr;
  PetscCallCXX(blocks_.clear());
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::initialize() noexcept
{
  constexpr auto alloc_size = PoolSize*sizeof(MemType);
  const     auto finalizer  = [](void *ptr) {
    PetscFunctionBegin;
    PetscCall(static_cast<decltype(this)>(ptr)->finalize());
    PetscFunctionReturn(0);
  };
  PetscContainer contain;

  PetscFunctionBegin;
  PetscCall(allocate_(&mem_pool_,alloc_size));
  PetscCall(PetscContainerCreate(PETSC_COMM_SELF,&contain));
  PetscCall(PetscContainerSetPointer(contain,this));
  PetscCall(PetscContainerSetUserDestroy(contain,finalizer));
  PetscCall(PetscObjectRegisterDestroy(reinterpret_cast<PetscObject>(contain)));
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::get(PetscInt size, MemType **ptr) noexcept
{
  auto result = mem_pool_;

  PetscFunctionBegin;
  PetscAssert(size < PoolSize,PETSC_COMM_SELF,PETSC_ERR_MEM,"Cannot allocate pool larger than %zu elements",PoolSize);
  // use host_mem as canary
  if (PetscUnlikely(!mem_pool_)) PetscCall(initialize());

  if (blocks_.empty()) {
    PetscCallCXX(blocks_.emplace_back(0,size));
  } else {
    auto block_alloced = size_type{0};
    // first, search the blocks
    for (auto& block : blocks_) {
      const auto bsize = block.size;

      if (block.open && (bsize <= size)) {
        // ok found open block of suitable size, claim it.
        // could maybe have shared blocks in the future
        result     = mem_pool_+bsize;
        block.open = false;
        break;
      }
      block_alloced += bsize;
    }
    // no open block found, need to make one
    if (result == mem_pool_) {
      // check that the pool has enough room
      PetscCheck(block_alloced+size <= PoolSize,PETSC_COMM_SELF,PETSC_ERR_MEM,"Allocating block of size %zu would exceed maximum pool size %zu",size,PoolSize);
      PetscCallCXX(blocks_.emplace_back(block_alloced,size));
    }
  }
  if (ptr) *ptr = result;
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::release(MemType **ptr) noexcept
{
  const auto offset = *ptr-mem_pool_;

  PetscFunctionBegin;
  if (!this->owns_pointer(*ptr)) PetscFunctionReturn(0); // don't own it, bail

  for (auto block = blocks_.begin(); block != blocks_.end(); ++block) {
    if (block->start == offset) {
      // ok, found ourselves
      if (std::next(block) == blocks_.end()) {
        // last element of the vector, just destroy it
        PetscCallCXX(blocks_.pop_back());
      } else {
        // somewhere inside, so mark the block free again
        block->open = true;
      }
      break;
    }
    PetscAssert(std::next(block) != blocks_.end(),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find block owning offset %zu in pool",offset);
  }
  *ptr = nullptr;
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline bool SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::owns_pointer(const MemType *ptr) const noexcept
{
  return ptr >= mem_pool_ && ptr < std::next(mem_pool_,PoolSize);
}

} // namespace detail

// Forward declare
template <DeviceType> class PETSC_TEMPLATE_VISIBILITY_INTERNAL DeviceContext;

template <DeviceType T>
class DeviceContext : BlasInterface<T>
{
public:
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t,T);

private:
  template <typename H> using HandleTag = typename detail::HandleTag<H>;
  using stream_tag = HandleTag<cupmStream_t>;
  using blas_tag   = HandleTag<cupmBlasHandle_t>;
  using solver_tag = HandleTag<cupmSolverHandle_t>;

public:
  // This is the canonical PETSc "impls" struct that normally resides in a standalone impls
  // header, but since we are using the power of templates it must be declared part of
  // this class to have easy access the same typedefs. Technically one can make a
  // templated struct outside the class but it's more code for the same result.
  struct PetscDeviceContext_IMPLS
  {
    cupmStream_t       stream;
    cupmEvent_t        event;
    cupmEvent_t        begin; // timer-only
    cupmEvent_t        end;   // timer-only
#if PetscDefined(USE_DEBUG)
    PetscBool          timerInUse;
#endif
    cupmBlasHandle_t   blas;
    cupmSolverHandle_t solver;

    PETSC_NODISCARD auto get(stream_tag) const noexcept PETSC_DECLTYPE_AUTO_RETURNS(this->stream);
    PETSC_NODISCARD auto get(blas_tag)   const noexcept PETSC_DECLTYPE_AUTO_RETURNS(this->blas);
    PETSC_NODISCARD auto get(solver_tag) const noexcept PETSC_DECLTYPE_AUTO_RETURNS(this->solver);
  };

private:
  static bool initialized_;
  static std::array<cupmBlasHandle_t,PETSC_DEVICE_MAX_DEVICES>   blashandles_;
  static std::array<cupmSolverHandle_t,PETSC_DEVICE_MAX_DEVICES> solverhandles_;

  PETSC_CXX_COMPAT_DECL(constexpr auto impls_cast_(PetscDeviceContext ptr))
  PETSC_DECLTYPE_AUTO_RETURNS(static_cast<PetscDeviceContext_IMPLS*>(ptr->data));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode check_current_device_(PetscDeviceContext dctxl, PetscDeviceContext dctxr))
  {
    const auto devidl = dctxl->device->deviceId,devidr = dctxr->device->deviceId;

    PetscFunctionBegin;
    PetscCheck(devidl == devidr,PETSC_COMM_SELF,PETSC_ERR_GPU,"Device contexts must be on the same device; dctx A (id %" PetscInt_FMT " device id %" PetscInt_FMT ") dctx B (id %" PetscInt_FMT " device id %" PetscInt_FMT ")",dctxl->id,devidl,dctxr->id,devidr);
    PetscCall(PetscDeviceCheckDeviceCount_Internal(devidl));
    PetscCall(PetscDeviceCheckDeviceCount_Internal(devidr));
    PetscCallCUPM(cupmSetDevice(static_cast<int>(devidl)));
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(auto check_current_device_(PetscDeviceContext dctx))
  PETSC_DECLTYPE_AUTO_RETURNS(check_current_device_(dctx,dctx));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode finalize_())
  {
    PetscFunctionBegin;
    for (auto&& handle : blashandles_) {
      if (handle) PetscCallCUPMBLAS(cupmBlasDestroy(handle));
      handle = nullptr;
    }
    for (auto&& handle : solverhandles_) {
      if (handle) PetscCall(cupmBlasInterface_t::DestroyHandle(handle));
      handle = nullptr;
    }
    initialized_ = false;
    PetscFunctionReturn(0);
  }

  // this exists purely to satisfy the tag interface for the other handles
  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle2_(stream_tag,PetscDeviceContext)) { return 0; }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_handle_(cupmBlasHandle_t &handle))
  {
    PetscFunctionBegin;
    if (PetscLikely(handle)) PetscFunctionReturn(0);
    for (auto i = 0; i < 3; ++i) {
      auto cberr = cupmBlasCreate(&handle);
      if (PetscLikely(cberr == CUPMBLAS_STATUS_SUCCESS)) break;
      if (PetscUnlikely(cberr != CUPMBLAS_STATUS_ALLOC_FAILED) && (cberr != CUPMBLAS_STATUS_NOT_INITIALIZED)) CHKERRCUPMBLAS(cberr);
      if (i != 2) {PetscCall(PetscSleep(3));continue;}
      PetscCheck(cberr == CUPMBLAS_STATUS_SUCCESS,PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize %s",cupmBlasName());
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle2_(blas_tag, PetscDeviceContext dctx))
  {
    const auto   dci    = impls_cast_(dctx);
    auto&        handle = blashandles_[dctx->device->deviceId];
    cupmStream_t stream;

    PetscFunctionBegin;
    PetscCall(create_handle_(handle));
    PetscCallCUPMBLAS(cupmBlasGetStream(handle,&stream));
    if (stream != dci->stream) PetscCallCUPMBLAS(cupmBlasSetStream(handle,dci->stream));
    dci->blas = handle;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_handle_(cupmSolverHandle_t &handle))
  {
    return cupmBlasInterface_t::InitializeHandle(handle);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle2_(solver_tag, PetscDeviceContext dctx))
  {
    const auto dci    = impls_cast_(dctx);
    auto&      handle = solverhandles_[dctx->device->deviceId];

    PetscFunctionBegin;
    PetscCall(create_handle_(handle));
    PetscCall(cupmBlasInterface_t::SetHandleStream(handle,dci->stream));
    dci->solver = handle;
    PetscFunctionReturn(0);
  }

  template <typename TagType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(PetscDeviceContext dctx))
  {
    PetscFunctionBegin;
    PetscCall(check_current_device_(dctx));
    PetscCall(initialize_handle2_(TagType{},dctx));
    PetscFunctionReturn(0);
  }

  template <typename PetscType, typename AllocType, typename FreeType>
  PETSC_CXX_COMPAT_DECL(auto managed_pool_(AllocType&& allocfn, FreeType&& freefn)) PETSC_DECLTYPE_AUTO_RETURNS(detail::SegmentedMemoryPool<PetscType,AllocType,FreeType,100>{std::forward<AllocType>(allocfn),std::forward<FreeType>(freefn)});

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(auto managed_host_pool_())
    -> decltype(managed_pool_<PetscType>([](PetscType**,std::size_t) {return 0;},[](PetscType*) {return 0;}))&
  {
    static auto pool = managed_pool_<PetscType>(
      [](PetscType **ptr, std::size_t n) {
        PetscFunctionBegin;
        PetscCallCUPM(cupmMallocHost(reinterpret_cast<void**>(ptr),n));
        PetscFunctionReturn(0);
      },
      [](PetscType *ptr) {
        PetscFunctionBegin;
	PetscCallCUPM(cupmFreeHost(ptr));
        PetscFunctionReturn(0);
      }
    );
    return pool;
  }

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(auto managed_device_pool_())
    -> decltype(managed_pool_<PetscType>([](PetscType**,std::size_t) {return 0;},[](PetscType*) {return 0;}))&
  {
    static auto pool = managed_pool_<PetscType>(
      [](PetscType **ptr, std::size_t n) {
        PetscFunctionBegin;
        PetscCallCUPM(cupmMalloc(reinterpret_cast<void**>(ptr),n));
        PetscFunctionReturn(0);
      },
      [](PetscType *ptr) {
        PetscFunctionBegin;
	PetscCallCUPM(cupmFree(ptr));
        PetscFunctionReturn(0);
      }
    );
    return pool;
  }

public:
  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize());
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode changeStreamType(PetscDeviceContext,PetscStreamType));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setUp(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode query(PetscDeviceContext,PetscBool*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waitForContext(PetscDeviceContext,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode synchronize(PetscDeviceContext));
  template <typename Handle_t>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getHandle(PetscDeviceContext,void*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode beginTimer(PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode endTimer(PetscDeviceContext,PetscLogDouble*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode arrayCopy(PetscDeviceContext,void*PETSC_RESTRICT,const void*PETSC_RESTRICT,std::size_t,PetscDeviceCopyMode));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createManagedType(PetscDeviceContext,PetscManagedType));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroyManagedType(PetscDeviceContext,PetscManagedType));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getManagedTypeValues(PetscDeviceContext,PetscManagedType,PetscOffloadMask,PetscType**));

  const struct _DeviceContextOps ops = {
    destroy,
    changeStreamType,
    setUp,
    query,
    waitForContext,
    synchronize,
    getHandle<blas_tag>,
    getHandle<solver_tag>,
    getHandle<stream_tag>,
    beginTimer,
    endTimer,
    arrayCopy,
    createManagedType<PetscScalar,PetscManagedScalar>,
    destroyManagedType<PetscScalar,PetscManagedScalar>
  };
};

// not a PetscDeviceContext method, this initializes the CLASS
template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::initialize())
{
  PetscFunctionBegin;
  if (PetscUnlikely(!initialized_)) {
    initialized_ = true;
    PetscCall(PetscRegisterFinalize(finalize_));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::destroy(PetscDeviceContext dctx))
{
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (!dci) PetscFunctionReturn(0);
  if (dci->stream) PetscCallCUPM(cupmStreamDestroy(dci->stream));
  if (dci->event)  PetscCallCUPM(cupmEventDestroy(dci->event));
  if (dci->begin)  PetscCallCUPM(cupmEventDestroy(dci->begin));
  if (dci->end)    PetscCallCUPM(cupmEventDestroy(dci->end));
  PetscCall(PetscFree(dctx->data));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::changeStreamType(PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype))
{
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (dci->stream) {
    PetscCallCUPM(cupmStreamDestroy(dci->stream));
    dci->stream = nullptr;
  }
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::setUp(PetscDeviceContext dctx))
{
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  if (dci->stream) {
    PetscCallCUPM(cupmStreamDestroy(dci->stream));
    dci->stream = nullptr;
  }
  switch (dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    // don't create a stream for global blocking
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
    PetscCallCUPM(cupmStreamCreate(&dci->stream));
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    PetscCallCUPM(cupmStreamCreateWithFlags(&dci->stream,cupmStreamNonBlocking));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %s",PetscStreamTypes[util::integral_value(dctx->streamType)]);
    break;
  }
  if (!dci->event) PetscCallCUPM(cupmEventCreateWithFlags(&dci->event,cupmEventDisableTiming));
#if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::query(PetscDeviceContext dctx, PetscBool *idle))
{
  cupmError_t cerr;

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  cerr = cupmStreamQuery(impls_cast_(dctx)->stream);
  if (cerr == cupmSuccess) *idle = PETSC_TRUE;
  else {
    // somethings gone wrong
    if (PetscUnlikely(cerr != cupmErrorNotReady)) PetscCallCUPM(cerr);
    *idle = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb))
{
  const auto dcib = impls_cast_(dctxb);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctxa,dctxb));
  PetscCallCUPM(cupmEventRecord(dcib->event,dcib->stream));
  PetscCallCUPM(cupmStreamWaitEvent(impls_cast_(dctxa)->stream,dcib->event,0));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::synchronize(PetscDeviceContext dctx))
{
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
  // in case anything was queued on the event
  PetscCallCUPM(cupmStreamWaitEvent(dci->stream,dci->event,0));
  PetscCallCUPM(cupmStreamSynchronize(dci->stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename handle_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getHandle(PetscDeviceContext dctx, void *handle))
{
  PetscFunctionBegin;
  PetscCall(initialize_handle_(handle_t{},dctx));
  *static_cast<typename handle_t::type*>(handle) = impls_cast_(dctx)->get(handle_t{});
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::beginTimer(PetscDeviceContext dctx))
{
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
#if PetscDefined(USE_DEBUG)
  PetscCheck(!dci->timerInUse,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
#endif
  if (!dci->begin) {
    PetscCallCUPM(cupmEventCreate(&dci->begin));
    PetscCallCUPM(cupmEventCreate(&dci->end));
  }
  PetscCallCUPM(cupmEventRecord(dci->begin,dci->stream));
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed))
{
  float      gtime;
  const auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  PetscCall(check_current_device_(dctx));
#if PetscDefined(USE_DEBUG)
  PetscCheck(dci->timerInUse,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
#endif
  PetscCallCUPM(cupmEventRecord(dci->end,dci->stream));
  PetscCallCUPM(cupmEventSynchronize(dci->end));
  PetscCallCUPM(cupmEventElapsedTime(&gtime,dci->begin,dci->end));
  *elapsed = static_cast<util::remove_pointer_t<decltype(elapsed)>>(gtime);
  PetscFunctionReturn(0);
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::arrayCopy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, PetscDeviceCopyMode mode))
{
  PetscFunctionBegin;
  if (PetscUnlikely(mode == PETSC_DEVICE_COPY_HTOH)) {
    PetscCall(PetscMemcpy(dest,src,n));
  } else {
    PetscCallCUPM(cupmMemcpyAsync(dest,src,n,PetscDeviceCopyModeToCUPMMemcpyKind(mode),impls_cast_(dctx)->stream));
  }
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::createManagedType(PetscDeviceContext dctx, PetscManagedType scal))
{
  const auto n = scal->n;

  PetscFunctionBegin;
  if (!scal->host) PetscCall(managed_host_pool_<PetscType>().get(n,&scal->host));
  if (!scal->device) PetscCall(managed_device_pool_<PetscType>().get(n,&scal->device));
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::destroyManagedType(PetscDeviceContext dctx, PetscManagedType scal))
{
  PetscFunctionBegin;
  // try returning them to the pool
  PetscCall(managed_host_pool_<PetscType>().release(&scal->host));
  PetscCall(managed_device_pool_<PetscType>().release(&scal->device));
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getManagedTypeValues(PetscDeviceContext dctx, PetscManagedType scal, PetscOffloadMask mask, PetscType **ptr))
{
  // sets ptr, and scal->mask!
  const auto copyToDestination = [&](PetscType *&dest, const PetscType *src)
  {
    const auto device = dest == scal->device;
    const auto n      = scal->n;

    PetscFunctionBegin;
    // first get the destination buffer
    if (!dest) {
      if (device) PetscCall(managed_device_pool_<PetscType>().get(n,&dest));
      else PetscCall(managed_host_pool_<PetscType>().get(n,&dest));
    }
    // now see if we can copy the other
    if (src) {
      const auto kind = device ? cupmMemcpyHostToDevice : cupmMemcpyDeviceToHost;
      PetscCallCUPM(cupmMemcpyAsync(dest,src,n*sizeof(PetscType),kind,impls_cast_(dctx)->stream));
      // if we copied the other we are up to date on both
      scal->mask = PETSC_OFFLOAD_BOTH;
    } else {
      scal->mask = device ? PETSC_OFFLOAD_GPU : PETSC_OFFLOAD_CPU;
    }
    *ptr = dest;
    PetscFunctionReturn(0);
  };

  const auto smask = scal->mask;

  PetscFunctionBegin;
  PetscAssert(mask != smask,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Implementations should only be called if destination offloadmask (%d) != source offloadmask (%d)",static_cast<int>(mask),static_cast<int>(smask));
  PetscAssert(smask != PETSC_OFFLOAD_BOTH,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Implementations should only be called if source offloadmask (%d) != PETSC_OFFLOAD_BOTH (%d)",static_cast<int>(smask),static_cast<int>(PETSC_OFFLOAD_BOTH));
  switch (mask) {
  case PETSC_OFFLOAD_CPU:
    // must mean scal->mask is PETSC_OFFLOAD_GPU or unallocated
    PetscCall(copyToDestination(scal->host,scal->device));
    break;
  case PETSC_OFFLOAD_GPU:
    // must mean scal->mask is PETSC_OFFLOAD_CPU or unallocated
    PetscCall(copyToDestination(scal->device,scal->host));
    break;
  case PETSC_OFFLOAD_BOTH:
    // ok we have one or the other, but not both
    if (smask == PETSC_OFFLOAD_UNALLOCATED) {
      const auto n = scal->n;

      // we have none of them, let's make sure
      PetscAssert(!scal->host,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Have offloadmask PETSC_OFFLOAD_UNALLOCATED but have host pointer");
      PetscAssert(!scal->device,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Have offloadmask PETSC_OFFLOAD_UNALLOCATED but have device pointer");

      // get both
      PetscCall(managed_device_pool_<PetscType>().get(n,&scal->host));
      PetscCall(managed_device_pool_<PetscType>().get(n,&scal->device));
      scal->mask = PETSC_OFFLOAD_BOTH;
      *ptr       = scal->host;
    } else if (scal->host) {
      PetscCall(copyToDestination(scal->host,scal->device));
    } else {
      // presumably have device, let's check though
      PetscAssert(scal->device,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Have offloadmask (%d) but neither host nor device pointer",static_cast<int>(smask));
      PetscCall(copyToDestination(scal->device,scal->host));
    }
    break;
  default:
    PetscUnreachable();
    break;
  }
  PetscFunctionReturn(0);
}

// initialize the static member variables
template <DeviceType T> bool DeviceContext<T>::initialized_ = false;

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmBlasHandle_t,PETSC_DEVICE_MAX_DEVICES>   DeviceContext<T>::blashandles_ = {};

template <DeviceType T>
std::array<typename DeviceContext<T>::cupmSolverHandle_t,PETSC_DEVICE_MAX_DEVICES> DeviceContext<T>::solverhandles_ = {};

} // namespace Impl

// shorten this one up a bit (and instantiate the templates)
using CUPMContextCuda = Impl::DeviceContext<DeviceType::CUDA>;
using CUPMContextHip  = Impl::DeviceContext<DeviceType::HIP>;

// shorthand for what is an EXTREMELY long name
#define PetscDeviceContext_(IMPLS) Petsc::Device::CUPM::Impl::DeviceContext<Petsc::Device::CUPM::DeviceType::IMPLS>::PetscDeviceContext_IMPLS

} // namespace CUPM

} // namespace Device

} // namespace Petsc

#endif // PETSCDEVICECONTEXTCUDA_HPP
