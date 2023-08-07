#include <petsc/private/cpp/register_finalize.hpp>

#include <petscmanagedmemory.hpp>

namespace Petsc
{

namespace
{

template <typename MM>
class StaticScalar : public RegisterFinalizeable<StaticScalar<MM>> {
public:
  using managed_type = MM;
  using value_type   = typename managed_type::value_type;

  explicit StaticScalar(value_type val) noexcept : val_{std::move(val)} { }

  const managed_type &get() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, this->register_finalize());
    PetscFunctionReturn(v_);
  }

  PetscErrorCode register_finalize_() noexcept
  {
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(v_.Reserve(dctx, 1));
    PetscCallCXX(v_.front(dctx) = val_);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode finalize_() noexcept
  {
    // ASYNC TODO:
    // fix this with the register finalize rewrite. The problem is that (when it is not
    // finalized backwards) the finalizer for the memory pool runs before this one, so the
    // internal pointers are free'd out from under these objects by the time this finalizer
    // runs.
    // PetscDeviceContext dctx;

    PetscFunctionBegin;
    // PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    // PetscCall(v_.Destroy(dctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  value_type   val_{};
  managed_type v_;
};

} // namespace

const ManagedReal &MANAGED_REAL_ONE() noexcept
{
  static StaticScalar<ManagedReal> scal{1.0};

  return scal.get();
}

const ManagedReal &MANAGED_REAL_ZERO() noexcept
{
  static StaticScalar<ManagedReal> scal{0.0};

  return scal.get();
}

const ManagedReal &MANAGED_REAL_MINUS_ONE() noexcept
{
  static StaticScalar<ManagedReal> scal{-1.0};

  return scal.get();
}

const ManagedScalar &MANAGED_SCAL_ONE() noexcept
{
#if PetscDefined(USE_COMPLEX)
  static StaticScalar<ManagedScalar> scal{1.0};

  return scal.get();
#else
  return MANAGED_REAL_ONE();
#endif
}

const ManagedScalar &MANAGED_SCAL_ZERO() noexcept
{
#if PetscDefined(USE_COMPLEX)
  static StaticScalar<ManagedScalar> scal{0.0};

  return scal.get();
#else
  return MANAGED_REAL_ZERO();
#endif
}

const ManagedScalar &MANAGED_SCAL_MINUS_ONE() noexcept
{
#if PetscDefined(USE_COMPLEX)
  static StaticScalar<ManagedScalar> scal{-1.0};

  return scal.get();
#else
  return MANAGED_REAL_MINUS_ONE();
#endif
}

} // namespace Petsc
