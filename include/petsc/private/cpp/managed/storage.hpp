#ifndef PETSC_CPP_MANAGED_STORAGE_HPP
#define PETSC_CPP_MANAGED_STORAGE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>

#include <iterator>  // std::distance()
#include <algorithm> // std::copy_n()

namespace Petsc
{

namespace memory
{

// ==========================================================================================
// ManagedStorage
// ==========================================================================================

template <typename T>
class ManagedStorage {
public:
  using value_type     = T;
  using size_type      = std::size_t;
  using const_iterator = const value_type *;
  using iterator       = value_type *;

  ManagedStorage() noexcept = default;
  ~ManagedStorage() noexcept;

  constexpr explicit ManagedStorage(PetscMemType) noexcept;
  ManagedStorage(PetscDeviceContext, PetscMemType, size_type) noexcept;

  template <typename Iterator>
  ManagedStorage(PetscCopyMode, PetscDeviceContext, Iterator, Iterator, const PetscPointerAttributes &) noexcept;

  // ASYNC TODO
  ManagedStorage(const ManagedStorage &) noexcept            = delete;
  ManagedStorage &operator=(const ManagedStorage &) noexcept = delete;

  ManagedStorage(ManagedStorage &&) noexcept;
  ManagedStorage &operator=(ManagedStorage &&) noexcept;

  PETSC_NODISCARD value_type       *data() noexcept { return ptr_; }
  PETSC_NODISCARD const value_type *cdata() const noexcept { return ptr_; }
  PETSC_NODISCARD const value_type *data() const noexcept { return this->cdata(); }
  PETSC_NODISCARD bool              empty() const noexcept { return this->capacity() == 0; }
  PETSC_NODISCARD size_type         capacity() const noexcept { return attr_.size / sizeof(value_type); }
  PETSC_NODISCARD PetscMemType      mem_type() const noexcept { return attr_.mtype; }

  PETSC_NODISCARD iterator begin() noexcept { return iterator{this->data()}; }
  PETSC_NODISCARD iterator end(size_type size) noexcept { return this->begin() + static_cast<typename std::iterator_traits<iterator>::difference_type>(size); }

  PETSC_NODISCARD const_iterator cbegin() const noexcept { return const_iterator{this->data()}; }
  PETSC_NODISCARD const_iterator cend(size_type size) const noexcept { return this->cbegin() + static_cast<typename std::iterator_traits<const_iterator>::difference_type>(size); }

  PETSC_NODISCARD const_iterator begin() const noexcept { return this->cbegin(); }
  PETSC_NODISCARD const_iterator end(size_type size) const noexcept { return this->cend(size); }

  PetscErrorCode mark_begin(PetscDeviceContext, PetscMemoryAccessMode) const noexcept;
  PetscErrorCode mark_end(PetscDeviceContext, PetscMemoryAccessMode) const noexcept;
  PetscErrorCode reserve(PetscDeviceContext, size_type) noexcept;
  PetscErrorCode destroy(PetscDeviceContext = nullptr) noexcept;
  template <typename Iterator>
  PetscErrorCode assign(PetscDeviceContext, Iterator, Iterator, const PetscPointerAttributes &, bool = false) noexcept;
  PetscErrorCode assign(PetscDeviceContext, const ManagedStorage &) noexcept;
  PetscErrorCode copy_from(PetscDeviceContext, const ManagedStorage &) noexcept;

  void        swap(ManagedStorage &) noexcept;
  friend void swap(ManagedStorage &lhs, ManagedStorage &rhs) noexcept { lhs.swap(rhs); }

private:
  value_type                    *ptr_{nullptr};
  mutable PetscPointerAttributes attr_{};
  bool                           own_ptr_{true};

  PetscErrorCode ensure_ptr_attr_() const noexcept;
};

// ==========================================================================================
// ManagedStorage - Private API
// ==========================================================================================

template <typename T>
inline PetscErrorCode ManagedStorage<T>::ensure_ptr_attr_() const noexcept
{
  PetscFunctionBegin;
  if ((attr_.id == PETSC_DELETED_MEMORY_ID) || (attr_.id == PETSC_UNKNOWN_MEMORY_ID)) {
    PetscBool found;

    PetscCheck(this->data(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Do not have a pointer to get attributes for!");
    PetscCall(PetscDeviceGetPointerAttributes(this->data(), &attr_, &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not find attributes for pointer %p", (void *)(this->data()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// ManagedStorage - Public API
// ==========================================================================================

// memtype constructor
template <typename T>
constexpr inline ManagedStorage<T>::ManagedStorage(PetscMemType mtype) noexcept : attr_{mtype}
{
}

// size constructor
template <typename T>
inline ManagedStorage<T>::ManagedStorage(PetscDeviceContext dctx, PetscMemType mtype, size_type n) noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PetscObjectComm(dctx), PetscDeviceMalloc(dctx, mtype, n, &ptr_));
  PetscCallAbort(PetscObjectComm(dctx), PetscDeviceGetPointerAttributes(this->data(), &attr_, nullptr));
  PetscFunctionReturnVoid();
}

template <typename T>
template <typename Iterator>
inline ManagedStorage<T>::ManagedStorage(PetscCopyMode mode, PetscDeviceContext dctx, Iterator begin, Iterator end, const PetscPointerAttributes &attr) noexcept : attr_{attr}, own_ptr_{mode == PETSC_OWN_POINTER || mode == PETSC_COPY_VALUES}
{
  PetscFunctionBegin;
  switch (mode) {
  case PETSC_OWN_POINTER:
  case PETSC_USE_POINTER:
    ptr_ = &*begin;
    if ((attr_.id == PETSC_UNKNOWN_MEMORY_ID) || (attr_.id == PETSC_DELETED_MEMORY_ID)) {
      auto found = PETSC_FALSE;

      if (this->data()) PetscCallAbort(PETSC_COMM_SELF, PetscDeviceGetPointerAttributes(this->data(), &attr_, &found));
      if (!found) {
        if (this->data()) {
          // if we did not find the pointer then we can only assume this to be host stack memory.
          PetscCheckAbort(PetscMemTypeHost(attr_.mtype), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Don't know ID of unknown pointer to %s", PetscMemTypeToString(attr_.mtype));
          attr_.id = PETSC_STACK_MEMORY_ID;
        } else {
          attr_.id = PETSC_DELETED_MEMORY_ID;
        }
        attr_.size  = std::max(std::distance(begin, end) * sizeof(value_type), attr_.size);
        attr_.align = std::max(alignof(value_type), attr_.align);
      }
    }
    break;
  case PETSC_COPY_VALUES:
    PetscCallAbort(PetscObjectComm(dctx), this->assign(dctx, std::move(begin), std::move(end), attr));
    break;
  }
  PetscFunctionReturnVoid();
}

template <typename T>
inline ManagedStorage<T>::~ManagedStorage() noexcept
{
  PetscFunctionBegin;
  if (ptr_ && own_ptr_) {
    PetscBool init;

    // ASYNC TODO: why??????? This is a *memory leak*!
    PetscCallAbort(PETSC_COMM_SELF, PetscInitialized(&init));
    if (PetscLikely(init)) {
      PetscDeviceContext dctx;

      PetscCallAbort(PETSC_COMM_SELF, PetscDeviceContextGetCurrentContext(&dctx));
      PetscCallAbort(PetscObjectComm(dctx), PetscDeviceFree(dctx, ptr_));
    }
  }
  PetscFunctionReturnVoid();
}

template <typename T>
inline ManagedStorage<T>::ManagedStorage(ManagedStorage &&other) noexcept
  // clang-format off
  : ptr_{util::exchange(other.ptr_, nullptr)},
    attr_{util::exchange(other.attr_, PetscPointerAttributes{other.mem_type()})},
    own_ptr_{util::exchange(other.own_ptr_, true)}
// clang-format on
{
}

template <typename T>
inline ManagedStorage<T> &ManagedStorage<T>::operator=(ManagedStorage &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    // delete our pointer (if we have one)
    PetscAssertAbort(other.mem_type() == this->mem_type(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Memtypes don't match mine %s != theirs %s", PetscMemTypeToString(this->mem_type()), PetscMemTypeToString(other.mem_type()));
    PetscCallAbort(PETSC_COMM_SELF, this->destroy());
    ptr_     = util::exchange(other.ptr_, nullptr);
    attr_    = util::exchange(other.attr_, PetscPointerAttributes{attr_.mtype});
    own_ptr_ = util::exchange(other.own_ptr_, true);
  }
  PetscFunctionReturn(*this);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::mark_begin(PetscDeviceContext dctx, PetscMemoryAccessMode mode) const noexcept
{
  PetscFunctionBegin;
  PetscCall(ensure_ptr_attr_());
  PetscCall(PetscDeviceContextMarkIntentFromIDBegin(dctx, attr_.id, mode, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::mark_end(PetscDeviceContext dctx, PetscMemoryAccessMode mode) const noexcept
{
  PetscFunctionBegin;
  PetscCall(ensure_ptr_attr_());
  PetscCall(PetscDeviceContextMarkIntentFromIDEnd(dctx, attr_.id, mode, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::reserve(PetscDeviceContext dctx, size_type n) noexcept
{
  PetscFunctionBegin;
  if ((n == 0) || (this->capacity() >= n)) PetscFunctionReturn(PETSC_SUCCESS);
  // It is assumed that the user provided enough capacity in the case where we don't own the
  // pointer. We cannot resize it. We have no way of knowing if the pointer is dynamically
  // allocated or pointer to a stack variable. Furthermore, we simply do. not. own. it.
  PetscCheck(own_ptr_, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot increase capacity (from %zu to %zu) for a pointer that is not owned!", this->capacity(), n);
  if (this->data()) {
    PetscCall(PetscDeviceRealloc(dctx, n, &ptr_));
  } else {
    PetscCall(PetscDeviceMalloc(dctx, this->mem_type(), n, &ptr_));
  }
  PetscCall(PetscDeviceGetPointerAttributes(this->data(), &attr_, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::destroy(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (ptr_ && own_ptr_) PetscCall(PetscDeviceFree(dctx, ptr_));
  ptr_     = nullptr;
  attr_    = PetscPointerAttributes{attr_.mtype};
  own_ptr_ = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
template <typename Iterator>
inline PetscErrorCode ManagedStorage<T>::assign(PetscDeviceContext dctx, Iterator begin, Iterator end, const PetscPointerAttributes &attr, bool fast_assign) noexcept
{
  const auto n = std::distance(begin, end);

  PetscFunctionBegin;
  static_assert(sizeof(value_type) == sizeof(typename std::iterator_traits<Iterator>::value_type), "");
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Begin < end");
  PetscCall(this->reserve(dctx, n));
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  if (fast_assign || (PetscMemTypeHost(attr.mtype) && (attr.id == PETSC_STACK_MEMORY_ID))) {
    PetscAssert(PetscMemTypeHost(attr.mtype) && PetscMemTypeHost(this->mem_type()), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot fast assign from %s to %s", PetscMemTypeToString(attr.mtype), PetscMemTypeToString(this->mem_type()));
    PetscCallCXX(std::copy_n(std::move(begin), n, this->begin()));
  } else {
    constexpr auto iter_compat = std::is_base_of<typename std::iterator_traits<Iterator>::iterator_category, std::random_access_iterator_tag>::value;

    PetscAssert(iter_compat, PETSC_COMM_SELF, PETSC_ERR_USER, "Only random access iterators are supported for device copies!");
    PetscCall(ensure_ptr_attr_());
    PetscCall(PetscDeviceMemcpy(dctx, this->data(), std::addressof(*begin), n * sizeof(value_type), &attr_, &attr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::assign(PetscDeviceContext dctx, const ManagedStorage &other) noexcept
{
  PetscFunctionBegin;
  PetscAssert(other.mem_type() == this->mem_type(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Memtypes don't match mine %s != theirs %s", PetscMemTypeToString(this->mem_type()), PetscMemTypeToString(other.mem_type()));
  PetscCall(this->copy_from(dctx, other));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedStorage<T>::copy_from(PetscDeviceContext dctx, const ManagedStorage &other) noexcept
{
  PetscFunctionBegin;
  if (&other == this) PetscFunctionReturn(PETSC_SUCCESS);
  // ASYNC TODO this should really be calling some kind of iterator version of "copy()",
  // currently assign() and copy() do pretty much the same things.
  PetscCall(other.ensure_ptr_attr_());
  PetscCall(this->assign(dctx, other.cbegin(), other.cend(other.capacity()), other.attr_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline void ManagedStorage<T>::swap(ManagedStorage &other) noexcept
{
  using std::swap;

  swap(ptr_, other.ptr_);
  swap(attr_, other.attr_);
  swap(own_ptr_, other.own_ptr_);
}

} // namespace memory

} // namespace Petsc

#endif // PETSC_CPP_MANAGED_STORAGE_HPP
