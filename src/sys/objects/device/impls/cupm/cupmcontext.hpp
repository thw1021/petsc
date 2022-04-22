#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>
#include "../impldevicecontextbase.hpp"
#include "cupmthrustutility.hpp"

#include <array>

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
      if (PetscUnlikely(cberr != CUPMBLAS_STATUS_ALLOC_FAILED) && (cberr != CUPMBLAS_STATUS_NOT_INITIALIZED)) PetscCallCUPMBLAS(cberr);
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

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode host_malloc_wrapper(PetscType **ptr, std::size_t n))
  {
    PetscFunctionBegin;
    PetscCallCUPM(cupmMallocHost(reinterpret_cast<void**>(ptr),n*sizeof(PetscType)));
    PetscFunctionReturn(0);
  }

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode host_free_wrapper(PetscType *ptr))
  {
    PetscFunctionBegin;
    PetscCallCUPM(cupmFreeHost(ptr));
    PetscFunctionReturn(0);
  }

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(auto managed_host_pool_()) -> decltype(Petsc::Device::Impl::make_segmented_memory_pool<PetscType>(host_malloc_wrapper<PetscType>,host_free_wrapper<PetscType>))&
  {
    static auto pool = Petsc::Device::Impl::make_segmented_memory_pool<PetscType>(
      host_malloc_wrapper<PetscType>,host_free_wrapper<PetscType>
    );
    return pool;
  }

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode device_malloc_wrapper(PetscType **ptr, std::size_t n))
  {
    PetscFunctionBegin;
    PetscCallCUPM(cupmMalloc(reinterpret_cast<void**>(ptr),n*sizeof(PetscType)));
    PetscFunctionReturn(0);
  }

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode device_free_wrapper(PetscType *ptr))
  {
    PetscFunctionBegin;
    PetscCallCUPM(cupmFree(ptr));
    PetscFunctionReturn(0);
  }

  template <typename PetscType>
  PETSC_CXX_COMPAT_DECL(auto managed_device_pool_()) -> decltype(Petsc::Device::Impl::make_segmented_memory_pool<PetscType>(device_malloc_wrapper<PetscType>,device_free_wrapper<PetscType>))&
  {
    static auto pool = Petsc::Device::Impl::make_segmented_memory_pool<PetscType>(
      device_malloc_wrapper<PetscType>,device_free_wrapper<PetscType>
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
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroyManagedType(PetscDeviceContext,PetscManagedType));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getManagedTypeValues(PetscDeviceContext,PetscManagedType,PetscMemType,PetscMemoryAccessMode,PetscType**));
  template <typename PetscType, typename PetscManagedType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode applyOperatorType(PetscDeviceContext,PetscManagedType,PetscOperatorType,const PetscType*,PetscManagedType));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode launchHostFunction(PetscDeviceContext,PetscHostFunction,void*));

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
    destroyManagedType<PetscScalar,PetscManagedScalar>,
    getManagedTypeValues<PetscScalar,PetscManagedScalar>,
    applyOperatorType<PetscScalar,PetscManagedScalar>,
    destroyManagedType<PetscReal,PetscManagedReal>,
    getManagedTypeValues<PetscReal,PetscManagedReal>,
    applyOperatorType<PetscReal,PetscManagedReal>,
    destroyManagedType<PetscInt,PetscManagedInt>,
    getManagedTypeValues<PetscInt,PetscManagedInt>,
    applyOperatorType<PetscInt,PetscManagedInt>,
    launchHostFunction
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
  PetscCall(initialize_handle_<handle_t>(dctx));
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
  PetscCallCUPM(cupmMemcpyAsync(dest,src,n,PetscDeviceCopyModeToCUPMMemcpyKind(mode),impls_cast_(dctx)->stream));
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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::getManagedTypeValues(PetscDeviceContext dctx, PetscManagedType scal, PetscMemType mtype, PetscMemoryAccessMode mode, PetscType **ptr))
{
  const auto       n        = scal->n;
  const auto       xfersize = n*sizeof(PetscType);
  const auto       stream   = impls_cast_(dctx)->stream;
  PetscOffloadMask mask;

  PetscFunctionBegin;
  switch (mtype) {
  case PETSC_MEMTYPE_HOST: {
    const auto src  = scal->device;
    auto&      dest = scal->host;

    // read or write, get a pointer if we don't have one yet
    if (!dest) PetscCall(managed_host_pool_<PetscType>().get(n,&dest));
    mask = PETSC_OFFLOAD_CPU;
    // if we want any kind of read (read or read_write) and we have valid SRC, we need to copy
    // it now
    if (mode != PETSC_MEMORY_ACCESS_WRITE && src) {
      PetscCallCUPM(cupmMemcpyAsync(dest,src,xfersize,cupmMemcpyDeviceToHost,stream));
      // if read-only then update the offloadmask
      if (mode == PETSC_MEMORY_ACCESS_READ) mask = PETSC_OFFLOAD_BOTH;
    }
    *ptr = dest;
  } break;
  case PETSC_MEMTYPE_DEVICE: {
    const auto src  = scal->host;
    auto&      dest = scal->device;

    if (!dest) PetscCall(managed_device_pool_<PetscType>().get(n,&dest));
    mask = PETSC_OFFLOAD_GPU;
    if (mode != PETSC_MEMORY_ACCESS_WRITE && src) {
      PetscCallCUPM(cupmMemcpyAsync(dest,src,xfersize,cupmMemcpyHostToDevice,stream));
      if (mode == PETSC_MEMORY_ACCESS_READ) mask = PETSC_OFFLOAD_BOTH;
    }
    *ptr = dest;
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscMemType must be either PETSC_MEMTYPE_HOST (%d) or PETSC_MEMTYPE_DEVICE (%d) not %d",static_cast<int>(PETSC_MEMTYPE_HOST),static_cast<int>(PETSC_MEMTYPE_DEVICE),static_cast<int>(mtype));
    break;
  }
  scal->mask = mask;
  PetscFunctionReturn(0);
}

template <DeviceType T>
template <typename PetscType, typename PetscManagedType>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::applyOperatorType(PetscDeviceContext dctx, PetscManagedType scal, PetscOperatorType otype, const PetscType *rhs, PetscManagedType ret))
{
  auto         stream = impls_cast_(dctx)->stream;
  const auto   src_access = ret ? PETSC_MEMORY_ACCESS_READ : PETSC_MEMORY_ACCESS_READ_WRITE;
  PetscScalar *ptr,*retptr;
  PetscInt     n;

  PetscFunctionBegin;
  PetscCall(getManagedTypeValues(dctx,scal,PETSC_MEMTYPE_DEVICE,src_access,&ptr,&n));
  if (ret) {
    PetscCall(getManagedTypeValues(dctx,ret,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_WRITE,&retptr,nullptr));
  } else {
    // in place
    retptr = ptr;
  }

  switch (otype) {
  case PETSC_OPERATOR_PLUS:
    PetscCall(ThrustApplyPointwiseUnary<T>(shift_operator<PetscType,thrust::plus<PetscType>>{*rhs},stream,n,ptr,retptr));
    break;
  case PETSC_OPERATOR_MINUS:
    PetscCall(ThrustApplyPointwiseUnary<T>(shift_operator<PetscType,thrust::minus<PetscType>>{*rhs},stream,n,ptr,retptr));
    break;
  case PETSC_OPERATOR_MULTIPLY:
    PetscCall(ThrustApplyPointwiseUnary<T>(shift_operator<PetscType,thrust::multiplies<PetscType>>{*rhs},stream,n,ptr,retptr));
    break;
  case PETSC_OPERATOR_DIVIDE:
    PetscCall(ThrustApplyPointwiseUnary<T>(shift_operator<PetscType,thrust::divides<PetscType>>{*rhs},stream,n,ptr,retptr));
    break;
  case PETSC_OPERATOR_EQUAL:
    PetscCall(ThrustSet(stream,n,retptr,rhs));
    break;
  }
  PetscFunctionReturn(0);
}

struct HostFunctionContext
{
  const PetscDeviceContext dctx;
  const PetscHostFunction  fn;
  void *const              ctx;
};

static inline CUPM_CALLBACK_FN void hostFuncStaging(void *ctx) noexcept
{
  const auto hfc = static_cast<HostFunctionContext*>(ctx);

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF,(*hfc->fn)(hfc->dctx,hfc->ctx));
  try {
    delete hfc;
  } catch (const std::exception &e) {
    SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_MEM,"%s",e.what());
  }
  PetscFunctionReturnVoid();
}

template <DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceContext<T>::launchHostFunction(PetscDeviceContext dctx, PetscHostFunction func, void *ctx))
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmLaunchHostFunc(impls_cast_(dctx)->stream,hostFuncStaging,new HostFunctionContext{dctx,func,ctx}));
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
