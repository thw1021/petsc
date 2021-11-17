#include "../../interface/sycldevice.hpp"
#include <CL/sycl.hpp>

namespace Petsc {

class SyclContext
{
public:
  struct PetscDeviceContext_IMPLS {
    sycl::event        event;
    sycl::event        begin; // timer-only
    sycl::event        end;   // timer-only
  #if PetscDefined(USE_DEBUG)
    PetscBool          timerInUse;
  #endif
  };

private:
  static bool _initialized;

  PETSC_NODISCARD static PetscErrorCode _finalize() noexcept
  {
    PetscFunctionBegin;
    _initialized = false;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode _initialize(PetscInt id, SyclContext *dci) noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDeviceCheckDeviceCount_Internal(id);CHKERRQ(ierr);
    if (!_initialized) {
      _initialized = true;
      ierr = PetscRegisterFinalize(_finalize);CHKERRQ(ierr);
    }
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
    getBlasHandle,
    getSolverHandle,
    beginTimer,
    endTimer
  };

  // default constructor
  SyclContext() noexcept = default;

  // All of these functions MUST be static in order to be callable from C, otherwise they
  // get the implicit 'this' pointer tacked on
  PETSC_NODISCARD static PetscErrorCode destroy(PetscDeviceContext dctx) noexcept {
    PetscFunctionBegin;
    delete static_cast<PetscDeviceContext_IMPLS*>(dctx->data);
    PetscFunctionReturn(0);
  };
  PETSC_NODISCARD static PetscErrorCode changeStreamType(PetscDeviceContext,PetscStreamType) noexcept {return 0;};
  PETSC_NODISCARD static PetscErrorCode setUp(PetscDeviceContext) noexcept {return 0;};
  PETSC_NODISCARD static PetscErrorCode query(PetscDeviceContext,PetscBool*) noexcept {return 0;};
  PETSC_NODISCARD static PetscErrorCode waitForContext(PetscDeviceContext,PetscDeviceContext) noexcept {return 0;};
  PETSC_NODISCARD static PetscErrorCode synchronize(PetscDeviceContext) noexcept {return 0;};
  PETSC_NODISCARD static PetscErrorCode getBlasHandle(PetscDeviceContext,void*) noexcept {return 0;};
  PETSC_NODISCARD static PetscErrorCode getSolverHandle(PetscDeviceContext,void*) noexcept {return 0;};
  PETSC_NODISCARD static PetscErrorCode beginTimer(PetscDeviceContext) noexcept {return 0;};
  PETSC_NODISCARD static PetscErrorCode endTimer(PetscDeviceContext,PetscLogDouble*) noexcept {return 0;};
};

} // namespace Petsc

PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext dctx)
{
  PetscErrorCode                      ierr;
  static const Petsc::SyclContext     syclctx;

  PetscFunctionBegin;
  dctx->data = new Petsc::SyclContext::PetscDeviceContext_IMPLS();
  ierr = PetscMemcpy(dctx->ops,&syclctx.ops,sizeof(syclctx.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}