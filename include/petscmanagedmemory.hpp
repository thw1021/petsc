#ifndef PETSCMANAGEDMEMORY_HPP
#define PETSCMANAGEDMEMORY_HPP

#include <petscdevicetypes.h>

#include <petsc/private/cpp/managed/base.hpp>
#include <petsc/private/cpp/managed/storage.hpp>
#include <petsc/private/cpp/managed/proxyreference.hpp>

#include <petsc/private/cpp/expr/base.hpp>
#include <petsc/private/cpp/expr/expression.hpp>
#include <petsc/private/cpp/expr/executable.hpp>

#include <petscmanagedmemory_fwd.hpp>

namespace Petsc
{

// ==========================================================================================
// ManagedType
// ==========================================================================================

template <typename T>
class ManagedMemory : public ManagedMemoryFacade<memory::ManagedStorage<T>, memory::ManagedStorage<T>>, public expr::ExpressionBase<ManagedMemory<T>> {
  using base_type            = ManagedMemoryFacade<memory::ManagedStorage<T>, memory::ManagedStorage<T>>;
  using expression_base_type = expr::ExpressionBase<ManagedMemory<T>>;
  friend expression_base_type;

  ManagedMemory *mut_this_() const noexcept { return const_cast<ManagedMemory *>(this); }

public:
  using value_type = typename base_type::value_type;
  using size_type  = typename base_type::size_type;

  template <typename>
  friend class ManagedType;

  // ==================================================================================== //
  // constructors
  // ==================================================================================== //

  explicit ManagedMemory() noexcept;

  // the final constructor
  template <typename... Storage>
  ManagedMemory(size_type, PetscOffloadMask, Storage &&...) noexcept;
  ManagedMemory(PetscDeviceContext, value_type *, const PetscPointerAttributes *, value_type *, const PetscPointerAttributes *, size_type, PetscCopyMode, PetscCopyMode, PetscOffloadMask) noexcept;
  ManagedMemory(PetscDeviceContext, value_type *, value_type *, size_type, PetscCopyMode, PetscCopyMode, PetscOffloadMask) noexcept;

  ManagedMemory(PetscDeviceContext, const value_type &) noexcept;
  explicit ManagedMemory(PetscDeviceContext, size_type = 1) noexcept;
  explicit ManagedMemory(const value_type &) noexcept;

  template <typename F, typename... E>
  explicit ManagedMemory(const expr::Expression<F, E...> &) noexcept;

  template <typename E>
  explicit ManagedMemory(const expr::ExecutableExpression<E> &) noexcept;

  ManagedMemory(const ManagedMemory &) noexcept = default;
  ManagedMemory(ManagedMemory &&) noexcept      = default;

  // ==================================================================================== //
  // operators
  // ==================================================================================== //

  template <typename F, typename... E>
  ManagedMemory &operator=(const expr::Expression<F, E...> &) noexcept;

  template <typename E>
  ManagedMemory &operator=(const expr::ExecutableExpression<E> &) noexcept;
  ManagedMemory &operator=(const value_type &) noexcept;

  PETSC_NODISCARD bool operator==(const value_type &) const noexcept;

  // ==================================================================================== //
  // Accessors
  // ==================================================================================== //

  PETSC_NODISCARD bool is_nosync_available(PetscMemType mtype) const noexcept { return PetscMemTypeHost(mtype) ? (this->pure() && this->host_data()) : false; }

  PETSC_NODISCARD ProxyReference<T> at(PetscDeviceContext dctx, size_type idx) noexcept { return ProxyReference<T>{this, dctx, idx}; }
  PETSC_NODISCARD value_type        cat(PetscDeviceContext dctx, size_type idx) const noexcept { return ProxyReference<T>{mut_this_(), dctx, idx}; }
  PETSC_NODISCARD value_type        at(PetscDeviceContext dctx, size_type idx) const noexcept { return this->cat(dctx, idx); }

  // ==========================================================================================
  // Accessors - front
  // ==========================================================================================

  PETSC_NODISCARD auto front(PetscDeviceContext dctx = nullptr) noexcept { return this->at(dctx, 0); }
  PETSC_NODISCARD auto cfront(PetscDeviceContext dctx = nullptr) const noexcept { return this->cat(dctx, 0); }
  PETSC_NODISCARD auto front(PetscDeviceContext dctx = nullptr) const noexcept { return this->cfront(dctx); }

  // ==========================================================================================
  // Accessors - back
  // ==========================================================================================

  PETSC_NODISCARD auto back(PetscDeviceContext dctx = nullptr) noexcept { return this->at(dctx, this->size() - 1); }
  PETSC_NODISCARD auto cback(PetscDeviceContext dctx = nullptr) const noexcept { return this->cat(dctx, this->size() - 1); }
  PETSC_NODISCARD auto back(PetscDeviceContext dctx = nullptr) const noexcept { return this->cback(dctx); }

  // ==================================================================================== //
  // Getters
  // ==================================================================================== //

  PetscErrorCode GetArray(PetscDeviceContext, PetscMemType, PetscMemoryAccessMode, PetscBool, value_type **) noexcept;
  PetscErrorCode GetArrayRead(PetscDeviceContext, PetscMemType, PetscBool, const value_type **) const noexcept;

  PetscErrorCode RestoreArray(PetscDeviceContext, PetscMemType, PetscMemoryAccessMode, PetscBool, value_type **) noexcept;
  PetscErrorCode RestoreArrayRead(PetscDeviceContext, PetscMemType, PetscBool, const value_type **) const noexcept;

  PetscErrorCode GetArrayAndMemType(PetscDeviceContext, PetscMemoryAccessMode, value_type **, PetscMemType * = nullptr) noexcept;
  PetscErrorCode GetArrayAndMemTypeRead(PetscDeviceContext, const value_type **, PetscMemType * = nullptr) const noexcept;

  PetscErrorCode Destroy(PetscDeviceContext) noexcept;
  PetscErrorCode Clear() noexcept;
  PetscErrorCode Reserve(PetscDeviceContext, size_type) noexcept;

  PetscErrorCode            EqualTo(const value_type &, PetscBool *, PetscBool *) const noexcept;
  PETSC_NODISCARD PetscBool KnownAndEqual(const value_type &) const noexcept;

  // ==========================================================================================
  // specific getters
  // ==========================================================================================

  PETSC_NODISCARD auto host_data() noexcept { return this->host().data(); }
  PETSC_NODISCARD auto host_cdata() const noexcept { return this->host().cdata(); }
  PETSC_NODISCARD auto host_data() const noexcept { return this->host_cdata(); }
  PETSC_NODISCARD auto device_data() noexcept { return this->device().data(); }
  PETSC_NODISCARD auto device_cdata() const noexcept { return this->device().cdata(); }
  PETSC_NODISCARD auto device_data() const noexcept { return this->device_cdata(); }

  PETSC_NODISCARD auto data() noexcept { return this->host_data(); }
  PETSC_NODISCARD auto cdata() const noexcept { return this->host_data(); }
  PETSC_NODISCARD auto data() const noexcept { return this->cdata(); }

private:
  PETSC_NODISCARD static memory::ManagedStorage<T> ConstructStorage_(PetscCopyMode, PetscDeviceContext, value_type *, value_type *, const PetscPointerAttributes &) noexcept;
  // ASYNC TODO redesign this?
  PETSC_NODISCARD static memory::ManagedStorage<T> ConstructStorage_(PetscCopyMode, PetscDeviceContext, PetscMemType, value_type *, size_type, const PetscPointerAttributes *) noexcept;

  PetscErrorCode Sync_(PetscDeviceContext, PetscMemType) noexcept;
};

} // namespace Petsc

#include <petsc/private/cpp/managed/managedimpl.inl>

#endif // PETSCMANAGEDMEMORY_HPP
