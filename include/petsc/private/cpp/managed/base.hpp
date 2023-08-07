#ifndef PETSC_CPP_MANAGED_BASE_HPP
#define PETSC_CPP_MANAGED_BASE_HPP

#include <petscdevicetypes.h>

#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/tuple.hpp>

#include <cstddef> // std::size_t

namespace Petsc
{

// ==========================================================================================
// ManagedMemoryFacadeBase
// ==========================================================================================

template <typename... Storages>
class ManagedMemoryFacadeBase {
public:
  using storages_type   = std::tuple<Storages...>;
  using size_type       = util::common_type_t<typename Storages::size_type...>;
  using value_type      = util::common_type_t<typename Storages::value_type...>;
  using pointer         = value_type *;
  using const_pointer   = const value_type *;
  using reference       = value_type &;
  using const_reference = const value_type &;

  template <typename... T>
  explicit ManagedMemoryFacadeBase(size_type, T &&...) noexcept;
  ManagedMemoryFacadeBase(ManagedMemoryFacadeBase &&) noexcept;
  ManagedMemoryFacadeBase &operator=(ManagedMemoryFacadeBase &&) noexcept;

  PETSC_NODISCARD bool      empty() const noexcept;
  PETSC_NODISCARD size_type capacity() const noexcept;

  PETSC_NODISCARD auto host() noexcept -> util::tuple_element_t<0, storages_type> &;
  PETSC_NODISCARD auto host() const noexcept -> const util::tuple_element_t<0, storages_type> &;

  void        swap(ManagedMemoryFacadeBase &) noexcept;
  friend void swap(ManagedMemoryFacadeBase &lhs, ManagedMemoryFacadeBase &rhs) noexcept { lhs.swap(rhs); }

protected:
  size_type             size_{};
  mutable storages_type storage_{};

  PETSC_NODISCARD storages_type       &storage() noexcept;
  PETSC_NODISCARD const storages_type &storage() const noexcept;

  template <std::size_t Idx>
  PETSC_NODISCARD util::tuple_element_t<Idx, storages_type> &storage() noexcept;
  template <std::size_t Idx>
  PETSC_NODISCARD const util::tuple_element_t<Idx, storages_type> &storage() const noexcept;

  PETSC_NODISCARD size_type size_impl_() const noexcept;
};

// ==========================================================================================
// ManagedMemoryFacadeBase -- Protected API
// ==========================================================================================

template <typename... S>
inline typename ManagedMemoryFacadeBase<S...>::storages_type &ManagedMemoryFacadeBase<S...>::storage() noexcept
{
  return storage_;
}

template <typename... S>
inline const typename ManagedMemoryFacadeBase<S...>::storages_type &ManagedMemoryFacadeBase<S...>::storage() const noexcept
{
  return storage_;
}

template <typename... S>
template <std::size_t Idx>
inline util::tuple_element_t<Idx, typename ManagedMemoryFacadeBase<S...>::storages_type> &ManagedMemoryFacadeBase<S...>::storage() noexcept
{
  return std::get<Idx>(storage());
}

template <typename... S>
template <std::size_t Idx>
inline const util::tuple_element_t<Idx, typename ManagedMemoryFacadeBase<S...>::storages_type> &ManagedMemoryFacadeBase<S...>::storage() const noexcept
{
  return std::get<Idx>(storage());
}

template <typename... S>
inline typename ManagedMemoryFacadeBase<S...>::size_type ManagedMemoryFacadeBase<S...>::size_impl_() const noexcept
{
  return size_;
}

// ==========================================================================================
// ManagedMemoryFacadeBase -- Public API
// ==========================================================================================

template <typename... S>
template <typename... T>
inline ManagedMemoryFacadeBase<S...>::ManagedMemoryFacadeBase(size_type size, T &&...storages) noexcept : size_{std::move(size)}, storage_{std::forward<T>(storages)...}
{
}

template <typename... S>
inline ManagedMemoryFacadeBase<S...>::ManagedMemoryFacadeBase(ManagedMemoryFacadeBase &&other) noexcept : size_{util::exchange(other.size_, 0)}, storage_{std::move(other.storage_)}
{
}

template <typename... S>
inline ManagedMemoryFacadeBase<S...> &ManagedMemoryFacadeBase<S...>::operator=(ManagedMemoryFacadeBase &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    size_    = util::exchange(other.size_, 0);
    storage_ = std::move(other.storage_);
  }
  PetscFunctionReturn(*this);
}

template <typename... S>
inline bool ManagedMemoryFacadeBase<S...>::empty() const noexcept
{
  return size_ == 0;
}

namespace detail
{

template <typename T>
struct MaxCap {
  template <typename U>
  void operator()(U &&s) noexcept
  {
    this->value = std::max(this->value, s.capacity());
  }

  T value{};
};

} // namespace detail

template <typename... S>
inline typename ManagedMemoryFacadeBase<S...>::size_type ManagedMemoryFacadeBase<S...>::capacity() const noexcept
{
  return util::tuple_for_each(this->storage(), detail::MaxCap<size_type>{}).value;
}

template <typename... S>
inline auto ManagedMemoryFacadeBase<S...>::host() noexcept -> util::tuple_element_t<0, storages_type> &
{
  return this->storage<0>();
}

template <typename... S>
inline auto ManagedMemoryFacadeBase<S...>::host() const noexcept -> const util::tuple_element_t<0, storages_type> &
{
  return this->storage<0>();
}

template <typename... S>
inline void ManagedMemoryFacadeBase<S...>::swap(ManagedMemoryFacadeBase &other) noexcept
{
  using std::swap;

  swap(size_, other.size_);
  swap(storage_, other.storage_);
}

// ==========================================================================================
// ManagedMemoryFacade
// ==========================================================================================

template <typename... Storages>
class ManagedMemoryFacade;

// ==========================================================================================
// ManagedMemoryFacade -- Dual storage specialization
// ==========================================================================================

template <typename Storage1, typename Storage2>
class ManagedMemoryFacade<Storage1, Storage2> : public ManagedMemoryFacadeBase<Storage1, Storage2> {
  using base_type = ManagedMemoryFacadeBase<Storage1, Storage2>;

public:
  template <typename S1, typename S2>
  ManagedMemoryFacade(typename base_type::size_type, PetscOffloadMask, S1 &&, S2 &&) noexcept;
  ManagedMemoryFacade(ManagedMemoryFacade &&) noexcept;
  ManagedMemoryFacade &operator=(ManagedMemoryFacade &&) noexcept;

  PETSC_NODISCARD PetscOffloadMask offload_mask() const noexcept;

  void        swap(ManagedMemoryFacade &) noexcept;
  friend void swap(ManagedMemoryFacade &lhs, ManagedMemoryFacade &rhs) noexcept { lhs.swap(rhs); }

protected:
  PETSC_NODISCARD bool pure() const noexcept;

  PetscErrorCode SetPurity_(bool) noexcept;
  PetscErrorCode SetOffloadMask_(PetscOffloadMask) noexcept;

  PETSC_NODISCARD Storage2       &device() noexcept;
  PETSC_NODISCARD const Storage2 &device() const noexcept;

private:
  mutable PetscOffloadMask mask_{PETSC_OFFLOAD_UNALLOCATED};
  mutable bool             pure_{true};
};

// ==========================================================================================
// ManagedMemoryFacade -- Dual storage specialization -- Public API
// ==========================================================================================

template <typename S1, typename S2>
template <typename ST1, typename ST2>
inline ManagedMemoryFacade<S1, S2>::ManagedMemoryFacade(typename base_type::size_type size, PetscOffloadMask mask, ST1 &&hstorage, ST2 &&dstorage) noexcept :
  base_type{std::move(size), std::forward<ST1>(hstorage), std::forward<ST2>(dstorage)}, mask_{mask}
{
}

template <typename S1, typename S2>
inline ManagedMemoryFacade<S1, S2>::ManagedMemoryFacade(ManagedMemoryFacade &&other) noexcept
  // clang-format off
    : base_type{std::move(other)},
      mask_{util::exchange(other.mask_, PETSC_OFFLOAD_UNALLOCATED)},
      pure_{util::exchange(other.pure_, true)}
// clang-format on
{
}

template <typename S1, typename S2>
inline ManagedMemoryFacade<S1, S2> &ManagedMemoryFacade<S1, S2>::operator=(ManagedMemoryFacade &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    base_type::operator=(std::move(other));
    mask_ = util::exchange(other.mask_, PETSC_OFFLOAD_UNALLOCATED);
    pure_ = util::exchange(other.pure_, true);
  }
  PetscFunctionReturn(*this);
}

template <typename S1, typename S2>
inline PetscOffloadMask ManagedMemoryFacade<S1, S2>::offload_mask() const noexcept
{
  return mask_;
}

template <typename S1, typename S2>
inline void ManagedMemoryFacade<S1, S2>::swap(ManagedMemoryFacade &other) noexcept
{
  using std::swap;

  swap(static_cast<base_type &>(*this), static_cast<base_type &>(other));
  swap(mask_, other.mask_);
  swap(pure_, other.pure_);
}

// ==========================================================================================
// ManagedMemoryFacade -- Dual storage specialization -- Protected API
// ==========================================================================================

template <typename S1, typename S2>
inline bool ManagedMemoryFacade<S1, S2>::pure() const noexcept
{
  return pure_;
}

template <typename S1, typename S2>
inline PetscErrorCode ManagedMemoryFacade<S1, S2>::SetPurity_(bool purity) noexcept
{
  PetscFunctionBegin;
  pure_ = purity;
  //if (!purity && parent_) parent_->pure_ = purity;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename S1, typename S2>
inline PetscErrorCode ManagedMemoryFacade<S1, S2>::SetOffloadMask_(PetscOffloadMask mask) noexcept
{
  PetscFunctionBegin;
  if (mask_ != mask) {
    // should not update the parent if the mask did not change!
    //if (parent_) parent_->mask_ = mask;
  }
  mask_ = mask;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename S1, typename S2>
inline S2 &ManagedMemoryFacade<S1, S2>::device() noexcept
{
  return this->template storage<1>();
}

template <typename S1, typename S2>
inline const S2 &ManagedMemoryFacade<S1, S2>::device() const noexcept
{
  return this->template storage<1>();
}

// ==========================================================================================
// ManagedMemoryFacade -- Single storage specialization
// ==========================================================================================

template <typename Storage>
class ManagedMemoryFacade<Storage> : public ManagedMemoryFacadeBase<Storage> {
  using base_type = ManagedMemoryFacadeBase<Storage>;

public:
  template <typename S>
  ManagedMemoryFacade(typename base_type::size_type, PetscOffloadMask, S &&) noexcept;

  static constexpr PetscErrorCode SetPurity_(bool) noexcept;
  static constexpr PetscErrorCode SetOffloadMask_(PetscOffloadMask) noexcept;

  PETSC_NODISCARD PetscOffloadMask offload_mask() const noexcept;

  void        swap(ManagedMemoryFacade &) noexcept;
  friend void swap(ManagedMemoryFacade &lhs, ManagedMemoryFacade &rhs) noexcept { lhs.swap(rhs); }

protected:
  static constexpr bool pure() noexcept;

  struct DummyStorage;
  static DummyStorage device() noexcept;
};

// ==========================================================================================
// ManagedMemoryFacade -- Single storage specialization -- Public API
// ==========================================================================================

template <typename S>
template <typename S1>
inline ManagedMemoryFacade<S>::ManagedMemoryFacade(typename base_type::size_type size, PetscOffloadMask, S1 &&hstorage) noexcept : base_type{std::move(size), std::forward<S1>(hstorage)}
{
}

template <typename S>
inline constexpr PetscErrorCode ManagedMemoryFacade<S>::SetPurity_(bool) noexcept
{
  return PETSC_SUCCESS;
}

template <typename S>
inline constexpr PetscErrorCode ManagedMemoryFacade<S>::SetOffloadMask_(PetscOffloadMask) noexcept
{
  return PETSC_SUCCESS;
}

template <typename S>
inline PetscOffloadMask ManagedMemoryFacade<S>::offload_mask() const noexcept
{
  auto &&storage = this->host();

  if (storage.data()) {
    const auto mtype = storage.mem_type();

    if (PetscMemTypeHost(mtype)) return PETSC_OFFLOAD_CPU;
    if (PetscMemTypeDevice(mtype)) return PETSC_OFFLOAD_GPU;
    // maybe there will be PETSC_MEMTYPE_BOTH at some point?
  }
  return PETSC_OFFLOAD_UNALLOCATED;
}

template <typename S>
inline void ManagedMemoryFacade<S>::swap(ManagedMemoryFacade &other) noexcept
{
  using std::swap;

  swap(static_cast<base_type &>(*this), static_cast<base_type &>(other));
}

// ==========================================================================================
// ManagedMemoryFacade -- Single storage specialization -- Protected API
// ==========================================================================================

template <typename S>
inline constexpr bool ManagedMemoryFacade<S>::pure() noexcept
{
  return true;
}

template <typename S>
struct ManagedMemoryFacade<S>::DummyStorage {
  static constexpr void        *data() noexcept { return nullptr; }
  static constexpr PetscMemType mem_type() noexcept { return PETSC_MEMTYPE_HOST; }
};

template <typename S>
inline typename ManagedMemoryFacade<S>::DummyStorage ManagedMemoryFacade<S>::device() noexcept
{
  SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_SUP, "Trying to retrieve device storage when no such storage exists. Likely you have not configured PETSc with GPU support!");
  return {};
}

} // namespace Petsc

#endif // PETSC_CPP_MANAGED_BASE_HPP
