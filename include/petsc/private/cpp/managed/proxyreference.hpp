#ifndef PETSC_CPP_MANAGED_PROXYREFERENCE_HPP
#define PETSC_CPP_MANAGED_PROXYREFERENCE_HPP

#include <petscdevice.h>

#include <petsc/private/cpp/memory.hpp> // std::addressof

namespace Petsc
{

template <typename>
class ManagedMemory;

// ==========================================================================================
// ProxyReference
// ==========================================================================================

template <typename T>
class ProxyReference {
public:
  using value_type   = T;
  using managed_type = ManagedMemory<value_type>;
  using size_type    = typename managed_type::size_type;

  ProxyReference() noexcept = delete;

  explicit ProxyReference(managed_type *, PetscDeviceContext, size_type) noexcept;

  ProxyReference  &operator=(const value_type &) const  & = delete;
  ProxyReference &&operator=(const value_type &) && noexcept;

  operator value_type() const & noexcept = delete;
  operator value_type() const && noexcept;

private:
  managed_type      *man_{};
  PetscDeviceContext dctx_{};
  size_type          idx_{};
};

// ==========================================================================================
// ProxyReference -- Public API
// ==========================================================================================

template <typename T>
inline ProxyReference<T>::ProxyReference(managed_type *man, PetscDeviceContext dctx, size_type idx) noexcept : man_{man}, dctx_{dctx}, idx_{idx}
{
  PetscFunctionBegin;
  PetscAssertAbort(man_->size() > idx_, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "index %zu >= size %zu", idx_, man_->size());
  PetscFunctionReturnVoid();
}

template <typename T>
inline ProxyReference<T> &&ProxyReference<T>::operator=(const value_type &val) && noexcept
{
  auto        mtype = PETSC_MEMTYPE_HOST;
  value_type *ptr;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, man_->GetArrayAndMemType(dctx_, PETSC_MEMORY_ACCESS_WRITE, &ptr, &mtype));
  if (man_->is_nosync_available(PETSC_MEMTYPE_HOST) && PetscMemTypeHost(mtype)) {
    ptr[idx_] = val;
  } else {
    constexpr auto src_attr = PetscPointerAttributes{PETSC_MEMTYPE_HOST, PETSC_STACK_MEMORY_ID, sizeof(val), alignof(decltype(val))};

    PetscCallAbort(PETSC_COMM_SELF, PetscDeviceMemcpy(dctx_, ptr + idx_, std::addressof(val), 1 * sizeof(*ptr), nullptr, &src_attr));
  }
  PetscFunctionReturn(std::move(*this));
}

template <typename T>
inline ProxyReference<T>::operator value_type() const && noexcept
{
  const value_type *ptr;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, man_->GetArrayRead(dctx_, PETSC_MEMTYPE_HOST, PETSC_TRUE, &ptr));
  PetscFunctionReturn(ptr[idx_]);
}

} // namespace Petsc

#endif // PETSC_CPP_MANAGED_PROXYREFERENCE_HPP
