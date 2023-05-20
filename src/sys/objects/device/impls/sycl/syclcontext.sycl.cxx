#include "sycldevice.hpp"
#include <CL/sycl.hpp>
#include <Kokkos_Core.hpp>

namespace Petsc
{

namespace device
{

namespace sycl
{

namespace impl
{

class DeviceContext {
public:
  struct PetscDeviceContext_SYCL {
    ::sycl::event event{};
    ::sycl::event begin{}; // timer-only
    ::sycl::event end{};   // timer-only
    Kokkos::Timer timer{}; // use cpu time since sycl events are return value of queue submission and we have no infrastructure to store them
    double        timeBegin{};
#if PetscDefined(USE_DEBUG)
    PetscBool timerInUse{};
#endif
    ::sycl::queue queue{};
  };

private:
  PETSC_NODISCARD static PetscDeviceContext_SYCL *impls_cast_(PetscDeviceContext dctx) noexcept { return static_cast<PetscDeviceContext_SYCL *>(dctx->data); }

public:
  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  static PetscErrorCode destroy(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    delete impls_cast_(dctx);
    dctx->data = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  static PetscErrorCode setUp(PetscDeviceContext dctx) noexcept
  {
    const auto dci = impls_cast_(dctx);

    PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
    dci->timerInUse = PETSC_FALSE;
#endif
    // petsc/sycl currently only uses Kokkos's default execution space (and its queue),
    // so in some sense, we have only one petsc device context.
    PetscCall(PetscKokkosInitializeCheck());
    dci->queue = Kokkos::DefaultExecutionSpace().sycl_queue();
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  static PetscErrorCode query(PetscDeviceContext, PetscBool *idle) noexcept
  {
    PetscFunctionBegin;
    // available in future, https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_queue_empty.asciidoc
    // *idle = static_cast<PetscDeviceContext_SYCL*>(dctx->data)->queue.empty() ? PETSC_TRUE : PETSC_FALSE;
    *idle = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  static PetscErrorCode synchronize(PetscDeviceContext dctx) noexcept
  {
    PetscBool  idle = PETSC_TRUE;
    const auto dci  = impls_cast_(dctx);

    PetscFunctionBegin;
    PetscCall(query(dctx, &idle));
    if (!idle) PetscCallCXX(dci->queue.wait());
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  static PetscErrorCode getStreamHandle(PetscDeviceContext dctx, void **handle) noexcept
  {
    PetscFunctionBegin;
    *reinterpret_cast<::sycl::queue **>(handle) = &(impls_cast_(dctx)->queue);
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  static PetscErrorCode beginTimer(PetscDeviceContext dctx) noexcept
  {
    const auto dci = impls_cast_(dctx);

    PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
    PetscCheck(!dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeEnd()?");
    dci->timerInUse = PETSC_TRUE;
#endif
    PetscCallCXX(dci->timeBegin = dci->timer.seconds());
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  static PetscErrorCode endTimer(PetscDeviceContext dctx, PetscLogDouble *elapsed) noexcept
  {
    const auto dci = impls_cast_(dctx);

    PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
    PetscCheck(dci->timerInUse, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forgot to call PetscLogGpuTimeBegin()?");
    dci->timerInUse = PETSC_FALSE;
#endif
    PetscCall(synchronize(dctx));
    PetscCallCXX(*elapsed = dci->timer.seconds() - dci->timeBegin);
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  // clang-format off
  static constexpr _DeviceContextOps ops = {
    PetscDesignatedInitializer(destroy, destroy),
    PetscDesignatedInitializer(changestreamtype, nullptr),
    PetscDesignatedInitializer(setup, setUp),
    PetscDesignatedInitializer(query, query),
    PetscDesignatedInitializer(waitforcontext, nullptr),
    PetscDesignatedInitializer(synchronize, synchronize),
    PetscDesignatedInitializer(getblashandle, nullptr),
    PetscDesignatedInitializer(getsolverhandle, nullptr),
    PetscDesignatedInitializer(getstreamhandle, nullptr),
    PetscDesignatedInitializer(begintimer, nullptr),
    PetscDesignatedInitializer(endtimer, nullptr),
    PetscDesignatedInitializer(memalloc, nullptr),
    PetscDesignatedInitializer(memfree, nullptr),
    PetscDesignatedInitializer(memcopy, nullptr),
    PetscDesignatedInitializer(memset, nullptr),
    PetscDesignatedInitializer(createevent, nullptr),
    PetscDesignatedInitializer(recordevent, nullptr),
    PetscDesignatedInitializer(waitforevent, nullptr)
  };
  // clang-format on
};

constexpr _DeviceContextOps DeviceContext::ops;

} // namespace impl

} // namespace sycl

} // namespace device

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext dctx)
{
  using namespace Petsc::device::sycl::impl;

  static constexpr DeviceContext syclctx;

  PetscFunctionBegin;
  PetscCallCXX(dctx->data = new DeviceContext::PetscDeviceContext_SYCL{});
  *(dctx->ops) = syclctx.ops;
  PetscFunctionReturn(PETSC_SUCCESS);
}
