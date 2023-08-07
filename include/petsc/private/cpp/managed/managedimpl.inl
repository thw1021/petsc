#ifndef PETSC_CPP_MANAGED_MANAGED_INL
#define PETSC_CPP_MANAGED_MANAGED_INL

#include <petsc/private/deviceimpl.h>

#include <petsc/private/cpp/memory.hpp> // std::addressof

#include <algorithm> // std::find
#include <iterator>  // std::next, std::distance

namespace Petsc
{

// ==========================================================================================
// ManagedType -- Private API
// ==========================================================================================

template <typename T>
inline memory::ManagedStorage<T> ManagedMemory<T>::ConstructStorage_(PetscCopyMode mode, PetscDeviceContext dctx, value_type *ptr_begin, value_type *ptr_end, const PetscPointerAttributes &attr) noexcept
{
  return {mode, dctx, ptr_begin, ptr_end, attr};
}

template <typename T>
inline memory::ManagedStorage<T> ManagedMemory<T>::ConstructStorage_(PetscCopyMode mode, PetscDeviceContext dctx, PetscMemType mtype, value_type *ptr, size_type n, const PetscPointerAttributes *attr) noexcept
{
  // clang-format off
  return ConstructStorage_(
    mode, dctx, ptr,
    ptr ? std::next(ptr, n) : ptr,
    attr ? *attr : PetscPointerAttributes{
      mtype, PETSC_UNKNOWN_MEMORY_ID,
      ptr ? n * sizeof(value_type) : 0,
      alignof(value_type)
    }
  );
  // clang-format on
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::Sync_(PetscDeviceContext dctx, PetscMemType mtype) noexcept
{
  PetscFunctionBegin;
  if (this->pure()) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscMemTypeHost(mtype)) PetscCall(this->SetPurity_(true));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// ManagedType -- Public API -- Constructors
// ==========================================================================================

template <typename T>
inline ManagedMemory<T>::ManagedMemory() noexcept : ManagedMemory{PetscDeviceContext{nullptr}}
{
}

template <typename T>
template <typename... Storage>
inline ManagedMemory<T>::ManagedMemory(size_type n, PetscOffloadMask mask, Storage &&...storages) noexcept : base_type{n, mask, std::forward<Storage>(storages)...}
{
}

template <typename T>
inline ManagedMemory<T>::ManagedMemory(PetscDeviceContext dctx, value_type *host_ptr, const PetscPointerAttributes *host_attr, value_type *device_ptr, const PetscPointerAttributes *device_attr, size_type n, PetscCopyMode h_cmode, PetscCopyMode d_cmode, PetscOffloadMask mask) noexcept
  // clang-format off
  : ManagedMemory{
      n,
      [=] {
        if (host_ptr && device_ptr) {
          // this is the only instance in which we believe whatever the user has fed us
          return mask;
        } else if (host_ptr) {
          // clearly no device_ptr, so must be on cpu
          return PETSC_OFFLOAD_CPU;
        } else if (device_ptr) {
          // clearly no host_ptr, so must be on gpu
          return PETSC_OFFLOAD_GPU;
        }
        // user gave us nothing, we are nowhere
        return PETSC_OFFLOAD_UNALLOCATED;
      }(),
      // ASYNC TODO make this constructor "smart" and only construct a single storage buffer in
      // host-only mode
      ConstructStorage_(h_cmode, dctx, PETSC_MEMTYPE_HOST, host_ptr, n, host_attr),
      ConstructStorage_(d_cmode, dctx, PETSC_MEMTYPE_DEVICE, device_ptr, n, device_attr)
    }
// clang-format on
{
  #if PetscDefined(USE_DEBUG)
  PetscFunctionBegin;
  if (host_ptr && device_ptr) PetscCheckAbort(mask != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Set both host and device pointer but offloadmask was %s", PetscOffloadMaskToString(mask));
  PetscFunctionReturnVoid();
  #endif
}

template <typename T>
inline ManagedMemory<T>::ManagedMemory(PetscDeviceContext dctx, value_type *host_ptr, value_type *device_ptr, size_type n, PetscCopyMode h_cmode, PetscCopyMode d_cmode, PetscOffloadMask mask) noexcept :
  ManagedMemory{dctx, host_ptr, nullptr, device_ptr, nullptr, n, h_cmode, d_cmode, mask}
{
}

template <typename T>
inline ManagedMemory<T>::ManagedMemory(PetscDeviceContext dctx, size_type n) noexcept : ManagedMemory{dctx, nullptr, nullptr, n, PETSC_OWN_POINTER, PETSC_OWN_POINTER, PETSC_OFFLOAD_UNALLOCATED}
{
}

template <typename T>
inline ManagedMemory<T>::ManagedMemory(PetscDeviceContext dctx, const value_type &value) noexcept :
  // clang-format off
  ManagedMemory{
    1,
    PETSC_OFFLOAD_CPU,
    ConstructStorage_(
      PETSC_COPY_VALUES, dctx,
      const_cast<value_type *>(std::addressof(value)),
      const_cast<value_type *>(std::addressof(value)) + 1,
      {PETSC_MEMTYPE_HOST, PETSC_STACK_MEMORY_ID, sizeof(value_type), alignof(value_type)}
    ),
    ConstructStorage_(PETSC_OWN_POINTER, dctx, nullptr, nullptr, PetscPointerAttributes{PETSC_MEMTYPE_DEVICE})
  }
// clang-format on
{
}

template <typename T>
inline ManagedMemory<T>::ManagedMemory(const value_type &value) noexcept : ManagedMemory{nullptr, value}
{
}

template <typename T>
template <typename F, typename... E>
inline ManagedMemory<T>::ManagedMemory(const expr::Expression<F, E...> &expr) noexcept : ManagedMemory{Eval(expr)}
{
}

template <typename T>
template <typename E>
inline ManagedMemory<T>::ManagedMemory(const expr::ExecutableExpression<E> &expr) noexcept : ManagedMemory{expr.dctx(), expr.size()}
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, expr.Execute(*this));
  PetscFunctionReturnVoid();
}

// ==========================================================================================
// ManagedMemory -- Public API -- Operators
// ==========================================================================================

template <typename T>
template <typename F, typename... E>
inline ManagedMemory<T> &ManagedMemory<T>::operator=(const expr::Expression<F, E...> &expr) noexcept
{
  return *this = Eval(expr);
}

template <typename T>
template <typename E>
inline ManagedMemory<T> &ManagedMemory<T>::operator=(const expr::ExecutableExpression<E> &expr) noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, this->Reserve(expr.dctx(), expr.size()));
  PetscCallAbort(PETSC_COMM_SELF, expr.Execute(*this));
  PetscFunctionReturn(*this);
}

template <typename T>
inline ManagedMemory<T> &ManagedMemory<T>::operator=(const value_type &value) noexcept
{
  PetscFunctionBegin;
  // ASYNC TODO
  this->front() = value;
  PetscFunctionReturn(*this);
}

template <typename T>
inline bool ManagedMemory<T>::operator==(const value_type &value) const noexcept
{
  const value_type *dummy;
  bool              equal;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, this->GetArrayRead(nullptr, PETSC_MEMTYPE_HOST, PETSC_TRUE, &dummy));
  equal = KnownAndEqual(value);
  PetscFunctionReturn(equal);
}

// ==========================================================================================
// ManagedMemory -- Public API -- Functions
// ==========================================================================================

template <typename T>
inline PetscErrorCode ManagedMemory<T>::Destroy(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(this->host().destroy(dctx));
  if (PetscDefined(HAVE_DEVICE)) PetscCall(this->device().destroy(dctx));
  PetscCall(this->Clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::Clear() noexcept
{
  PetscFunctionBegin;
  this->size_ = 0;
  PetscCall(this->SetPurity_(true));
  PetscCall(this->SetOffloadMask_(PETSC_OFFLOAD_UNALLOCATED));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::GetArray(PetscDeviceContext dctx, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, value_type **ptr) noexcept
{
  const auto get_from_storage = [&, this](auto &dest, const auto &src, PetscOffloadMask mask, value_type **ptr) {
    const auto unallocated = PetscOffloadUnallocated(this->offload_mask());
    auto       mark_write  = dest.capacity() == 0;

    PetscFunctionBegin;
    PetscAssert(&dest != &src, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Getting from equal storages!");
    PetscAssert(mask != PETSC_OFFLOAD_BOTH, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot have %s!", PetscOffloadMaskToString(mask));
    if (PetscUnlikely(unallocated)) {
      PetscAssert(!dest.data(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unallocated but dest has data: %p", static_cast<const void *>(dest.data()));
      PetscAssert(!src.data(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unallocated but src has data: %p", static_cast<const void *>(src.data()));
    }

    PetscCall(dest.reserve(dctx, this->size()));
    if (unallocated) {
      PetscCall(this->SetOffloadMask_(mask));
    } else if (this->offload_mask() != mask) {
      // if we want any kind of read (read or read_write) and we have valid SRC, we need to copy
      // it now
      if (PetscMemoryAccessRead(mode) && !src.empty() && !PetscOffloadBoth(this->offload_mask())) {
        PetscCall(this->SetOffloadMask_(PETSC_OFFLOAD_BOTH));
        PetscCall(dest.copy_from(dctx, src));
        mark_write = true;
      }
      // if we have any kind of write then mask is set to the specific requested version (which
      // must not be OFFLOAD_BOTH)
      if (PetscMemoryAccessWrite(mode)) PetscCall(this->SetOffloadMask_(mask));
    }

    // if the mode is write-only and we either allocated the memory or copied from src then we
    // have already properly marked the destination, so we can skip this call
    if (!mark_write || (mode != PETSC_MEMORY_ACCESS_WRITE)) PetscCall(dest.mark_begin(dctx, mode));
    *ptr = dest.data();
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssertPointer(ptr, 5);
  PetscCheck(!(PetscOffloadUnallocated(this->offload_mask()) && PetscMemoryAccessRead(mode)), PetscObjectComm(dctx), PETSC_ERR_ARG_WRONG, "Trying to read (using %s) from a managed type (id %d) that has not been written to (has offload mask %s)", PetscMemoryAccessModeToString(mode), -1,
             PetscOffloadMaskToString(this->offload_mask()));
  *ptr = nullptr;
  if (this->empty()) PetscFunctionReturn(PETSC_SUCCESS);

  // retrieve the pointer
  switch (mtype) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(get_from_storage(this->host(), this->device(), PETSC_OFFLOAD_CPU, ptr));
    break;
  case PETSC_MEMTYPE_DEVICE:
    PetscCall(get_from_storage(this->device(), this->host(), PETSC_OFFLOAD_GPU, ptr));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscMemType must be either %s or %s not %d", PetscMemTypeToString(PETSC_MEMTYPE_HOST), PetscMemTypeToString(PETSC_MEMTYPE_DEVICE), static_cast<int>(mtype));
    break;
  }

  // if user intends to write to device in any capacity then we are impure
  if (PetscMemTypeDevice(mtype) && PetscMemoryAccessWrite(mode)) PetscCall(this->SetPurity_(false));
  PetscAssert(*ptr, PetscObjectComm(dctx), PETSC_ERR_PLIB, "ManagedType (id %d) Returned null pointer for mtype %s", -1, PetscMemTypeToString(mtype));
  if (sync) PetscCall(this->Sync_(dctx, mtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::GetArrayRead(PetscDeviceContext dctx, PetscMemType mtype, PetscBool sync, const value_type **ptr) const noexcept
{
  return mut_this_()->GetArray(dctx, mtype, PETSC_MEMORY_ACCESS_READ, sync, const_cast<value_type **>(ptr));
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::RestoreArray(PetscDeviceContext dctx, PetscMemType mtype, PetscMemoryAccessMode mode, PetscBool sync, value_type **ptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssertPointer(ptr, 4);
  PetscCheck(!PetscOffloadUnallocated(this->offload_mask()), PetscObjectComm(dctx), PETSC_ERR_ARG_WRONG, "Trying to restore an array to an unallocated managed type (offload mask %s)", PetscOffloadMaskToString(this->offload_mask()));
  *ptr = nullptr;
  switch (mtype) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(this->host().mark_end(dctx, mode));
    break;
  case PETSC_MEMTYPE_DEVICE:
    PetscCall(this->device().mark_end(dctx, mode));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscMemType must be either %s or %s not %d", PetscMemTypeToString(PETSC_MEMTYPE_HOST), PetscMemTypeToString(PETSC_MEMTYPE_DEVICE), static_cast<int>(mtype));
    break;
  }
  if (sync) PetscCall(this->Sync_(dctx, PetscOffloadMaskToMemType(this->offload_mask())));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::RestoreArrayRead(PetscDeviceContext dctx, PetscMemType mtype, PetscBool sync, const value_type **ptr) const noexcept
{
  return mut_this_()->RestoreArray(dctx, mtype, PETSC_MEMORY_ACCESS_READ, sync, const_cast<value_type **>(ptr));
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::GetArrayAndMemType(PetscDeviceContext dctx, PetscMemoryAccessMode mode, value_type **ptr, PetscMemType *mtype) noexcept
{
  PetscMemType retmtype;

  PetscFunctionBegin;
  PetscAssertPointer(ptr, 3);
  if (mtype) PetscAssertPointer(mtype, 4);
  switch (const auto mask = this->offload_mask()) {
  case PETSC_OFFLOAD_CPU:
    retmtype = PETSC_MEMTYPE_HOST;
    break;
    // if both prefer GPU
  case PETSC_OFFLOAD_BOTH:
  case PETSC_OFFLOAD_GPU:
    retmtype = PETSC_MEMTYPE_DEVICE;
    break;
  case PETSC_OFFLOAD_UNALLOCATED: {
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    retmtype = dtype == PETSC_DEVICE_HOST ? PETSC_MEMTYPE_HOST : PETSC_MEMTYPE_DEVICE;
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support yet for offloadmask %d", static_cast<int>(mask));
  }
  PetscCall(this->GetArray(dctx, retmtype, mode, PETSC_FALSE, ptr));
  PetscAssert(*ptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, PetscStringize(PetscManagedType) " returned a null pointer for memtype %s as values", PetscMemTypeToString(retmtype));
  if (mtype) *mtype = retmtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::GetArrayAndMemTypeRead(PetscDeviceContext dctx, const value_type **ptr, PetscMemType *mtype) const noexcept
{
  return mut_this_()->GetArrayAndMemType(dctx, PETSC_MEMORY_ACCESS_READ, const_cast<value_type **>(ptr), mtype);
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::Reserve(PetscDeviceContext dctx, size_type n) noexcept
{
  PetscFunctionBegin;
  // ASYNC TODO reserve only what is possible to reserve!
  PetscCall(this->host().reserve(dctx, n));
  if (PetscDefined(HAVE_DEVICE)) PetscCall(this->device().reserve(dctx, n));
  if (PetscOffloadUnallocated(this->offload_mask())) PetscCall(this->SetOffloadMask_(PETSC_OFFLOAD_CPU));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode ManagedMemory<T>::EqualTo(const value_type &value, PetscBool *equal, PetscBool *known) const noexcept
{
  PetscFunctionBegin;
  PetscAssertPointer(equal, 2);
  PetscAssertPointer(known, 3);
  // REVIEW ME: it if is unknown we can technically just forgo the equal check since it is
  // worthless anyways
  if (this->is_nosync_available(PETSC_MEMTYPE_HOST)) {
    auto hend = this->host().cend(this->size());

    *known = PETSC_TRUE;
    *equal = std::find(this->host().cbegin(), hend, value) == hend ? PETSC_FALSE : PETSC_TRUE;
  } else {
    *known = *equal = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscBool ManagedMemory<T>::KnownAndEqual(const value_type &value) const noexcept
{
  PetscBool known, equal;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, this->EqualTo(value, &known, &equal));
  equal = static_cast<PetscBool>(known && equal);
  PetscFunctionReturn(equal);
}

} // namespace Petsc

#endif // PETSC_CPP_MANAGED_MANAGED_INL
