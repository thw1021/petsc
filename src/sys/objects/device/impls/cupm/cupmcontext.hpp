#ifndef PETSCDEVICECONTEXTCUPM_HPP
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>

#if !defined(PETSC_HAVE_CXX_DIALECT_CXX11)
#  error PetscDeviceContext backends for CUDA and HIP requires C++11
#endif

#include <array>

namespace Petsc
{

namespace Impl
{

namespace detail
{

// for tag-based dispatch of handle retrieval
template <typename T> struct HandleTag { using type = T; };

} // namespace detail

// Forward declare
template <CUPMDeviceType T> class PETSC_VISIBILITY_INTERNAL CUPMContext;

template <CUPMDeviceType T>
class CUPMContext : CUPMBlasInterface<T>
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

    PETSC_NODISCARD auto get(stream_tag) const -> decltype(this->stream) { return this->stream; }
    PETSC_NODISCARD auto get(blas_tag)   const -> decltype(this->blas)   { return this->blas;   }
    PETSC_NODISCARD auto get(solver_tag) const -> decltype(this->solver) { return this->solver; }
  };

private:
  static bool initialized_;
  static std::array<cupmBlasHandle_t,PETSC_DEVICE_MAX_DEVICES>   blashandles_;
  static std::array<cupmSolverHandle_t,PETSC_DEVICE_MAX_DEVICES> solverhandles_;

  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceContext_IMPLS* impls_cast_(PetscDeviceContext ptr))
  {
    return static_cast<PetscDeviceContext_IMPLS*>(ptr->data);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_handle_(cupmBlasHandle_t &handle))
  {
    PetscFunctionBegin;
    if (handle) PetscFunctionReturn(0);
    for (auto i = 0; i < 3; ++i) {
      auto cberr = cupmBlasCreate(&handle);
      if (PetscLikely(cberr == CUPMBLAS_STATUS_SUCCESS)) break;
      if (PetscUnlikely(cberr != CUPMBLAS_STATUS_ALLOC_FAILED) && (cberr != CUPMBLAS_STATUS_NOT_INITIALIZED)) CHKERRCUPMBLAS(cberr);
      if (i != 2) {
        auto ierr = PetscSleep(3);CHKERRQ(ierr);
        continue;
      }
      if (PetscUnlikely(cberr != CUPMBLAS_STATUS_SUCCESS)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"Unable to initialize %s",cupmBlasName());
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_handle_stream_(cupmBlasHandle_t &handle, cupmStream_t &stream))
  {
    cupmStream_t    cupmStream;
    cupmBlasError_t cberr;

    PetscFunctionBegin;
    cberr = cupmBlasGetStream(handle,&cupmStream);CHKERRCUPMBLAS(cberr);
    if (cupmStream != stream) {cberr = cupmBlasSetStream(handle,stream);CHKERRCUPMBLAS(cberr);}
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode finalize_())
  {
    PetscFunctionBegin;
    for (auto&& handle : blashandles_) {
      if (handle) {
        auto cberr = cupmBlasDestroy(handle);CHKERRCUPMBLAS(cberr);
        handle     = nullptr;
      }
    }
    for (auto&& handle : solverhandles_) {
      if (handle) {
        auto ierr = cupmBlasInterface_t::DestroyHandle(handle);CHKERRQ(ierr);
        handle    = nullptr;
      }
    }
    initialized_ = false;
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode initialize_(PetscInt id, PetscDeviceContext_IMPLS *dci))
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceCheckDeviceCount_Internal(id);CHKERRQ(ierr);
    if (!initialized_) {
      initialized_ = true;
      ierr = PetscRegisterFinalize(finalize_);CHKERRQ(ierr);
    }
    // use the blashandle as a canary
    if (!blashandles_[id]) {
      ierr = initialize_handle_(blashandles_[id]);CHKERRQ(ierr);
      ierr = cupmBlasInterface_t::InitializeHandle(solverhandles_[id]);CHKERRQ(ierr);
    }
    ierr = set_handle_stream_(blashandles_[id],dci->stream);CHKERRQ(ierr);
    ierr = cupmBlasInterface_t::SetHandleStream(solverhandles_[id],dci->stream);CHKERRQ(ierr);
    dci->blas   = blashandles_[id];
    dci->solver = solverhandles_[id];
    PetscFunctionReturn(0);
  }

public:
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
  };

  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
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
};

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::destroy(PetscDeviceContext dctx))
{
  cupmError_t    cerr;
  PetscErrorCode ierr;
  auto           dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (dci->stream) {cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);}
  if (dci->event)  {
    cerr = cupmEventDestroy(dci->event);CHKERRCUPM(cerr);
    cerr = cupmEventDestroy(dci->begin);CHKERRCUPM(cerr);
    cerr = cupmEventDestroy(dci->end);CHKERRCUPM(cerr);
  }
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::changeStreamType(PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype))
{
  auto dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (dci->stream) {
    auto cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);
    dci->stream = nullptr;
  }
  // set these to null so they aren't usable until setup is called again
  dci->blas   = nullptr;
  dci->solver = nullptr;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::setUp(PetscDeviceContext dctx))
{
  PetscErrorCode ierr;
  cupmError_t    cerr;
  auto           dci = impls_cast_(dctx);

  PetscFunctionBegin;
  if (dci->stream) {cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);}
  switch (dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    // don't create a stream for global blocking
    dci->stream = nullptr;
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
    cerr = cupmStreamCreate(&dci->stream);CHKERRCUPM(cerr);
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    cerr = cupmStreamCreateWithFlags(&dci->stream,cupmStreamNonBlocking);CHKERRCUPM(cerr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %s",PetscStreamTypes[util::integral_value(dctx->streamType)]);
    break;
  }
  if (!dci->event) {cerr = cupmEventCreate(&dci->event);CHKERRCUPM(cerr);}
#if PetscDefined(USE_DEBUG)
  dci->timerInUse = PETSC_FALSE;
#endif
  ierr = initialize_(dctx->device->deviceId,dci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::query(PetscDeviceContext dctx, PetscBool *idle))
{
  cupmError_t cerr;

  PetscFunctionBegin;
  cerr = cupmStreamQuery(impls_cast_(dctx)->stream);
  if (cerr == cupmSuccess) *idle = PETSC_TRUE;
  else {
    // somethings gone wrong
    if (PetscUnlikely(cerr != cupmErrorNotReady)) CHKERRCUPM(cerr);
    *idle = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb))
{
  cupmError_t cerr;
  auto        dcib = impls_cast_(dctxb);

  PetscFunctionBegin;
  cerr = cupmEventRecord(dcib->event,dcib->stream);CHKERRCUPM(cerr);
  cerr = cupmStreamWaitEvent(impls_cast_(dctxa)->stream,dcib->event,0);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::synchronize(PetscDeviceContext dctx))
{
  cupmError_t cerr;
  auto        dci = impls_cast_(dctx);

  PetscFunctionBegin;
  // in case anything was queued on the event
  cerr = cupmStreamWaitEvent(dci->stream,dci->event,0);CHKERRCUPM(cerr);
  cerr = cupmStreamSynchronize(dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <typename handle_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::getHandle(PetscDeviceContext dctx, void *handle))
{
  PetscFunctionBegin;
  *static_cast<typename handle_t::type*>(handle) = impls_cast_(dctx)->get(handle_t());
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::beginTimer(PetscDeviceContext dctx))
{
  auto        dci = impls_cast_(dctx);
  cupmError_t cerr;

  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  if (PetscUnlikely(dci->timerInUse)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Forgot to call PetscLogGpuTimeEnd()?");
  dci->timerInUse = PETSC_TRUE;
#endif
  if (!dci->begin) {
    cerr = cupmEventCreate(&dci->begin);CHKERRCUPM(cerr);
    cerr = cupmEventCreate(&dci->end);CHKERRCUPM(cerr);
  }
  cerr = cupmEventRecord(dci->begin,dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode CUPMContext<T>::endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed))
{
  cupmError_t cerr;
  float       gtime;
  auto        dci = impls_cast_(dctx);

  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  if (PetscUnlikely(!dci->timerInUse)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Forgot to call PetscLogGpuTimeBegin()?");
  dci->timerInUse = PETSC_FALSE;
#endif
  cerr = cupmEventRecord(dci->end,dci->stream);CHKERRCUPM(cerr);
  cerr = cupmEventSynchronize(dci->end);CHKERRCUPM(cerr);
  cerr = cupmEventElapsedTime(&gtime,dci->begin,dci->end);CHKERRCUPM(cerr);
  *elapsed = static_cast<util::remove_pointer_t<decltype(elapsed)>>(gtime);
  PetscFunctionReturn(0);
}

// initialize the static member variables
template <CUPMDeviceType T> bool CUPMContext<T>::initialized_ = false;

template <CUPMDeviceType T>
std::array<typename CUPMContext<T>::cupmBlasHandle_t,PETSC_DEVICE_MAX_DEVICES>   CUPMContext<T>::blashandles_ = {};

template <CUPMDeviceType T>
std::array<typename CUPMContext<T>::cupmSolverHandle_t,PETSC_DEVICE_MAX_DEVICES> CUPMContext<T>::solverhandles_ = {};

// shorten this one up a bit (and instantiate the templates)
using CUPMContextCuda = CUPMContext<CUPMDeviceType::CUDA>;
using CUPMContextHip  = CUPMContext<CUPMDeviceType::HIP>;

} // namespace Impl

} // namespace Petsc

// shorthand for what is an EXTREMELY long name
#define PetscDeviceContext_(IMPLS)                                      \
  Petsc::Impl::CUPMContext<Petsc::CUPMDeviceType::IMPLS>::PetscDeviceContext_IMPLS

#endif // PETSCDEVICECONTEXTCUDA_HPP
