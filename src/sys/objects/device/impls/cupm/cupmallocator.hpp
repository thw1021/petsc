#ifndef CUPMALLOCATOR_HPP
#define CUPMALLOCATOR_HPP

#include <petsc/private/cupminterface.hpp>
#include <petsc/private/cpp/object_pool.hpp>

#include "../segmentedmempool.hpp"

namespace Petsc
{

namespace device
{

namespace cupm
{

// ==========================================================================================
// CUPM Host Allocator
// ==========================================================================================

template <DeviceType T, typename PetscType = char>
class HostAllocator;

// Allocator class to allocate pinned host memory for use with device
template <DeviceType T, typename PetscType>
class HostAllocator final : public memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>, impl::Interface<T> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);
  using base_type  = memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>;
  using size_type  = typename base_type::size_type;
  using value_type = typename base_type::value_type;

  static PetscErrorCode allocate(PetscDeviceContext, size_type, value_type **) noexcept;
  static PetscErrorCode deallocate(PetscDeviceContext, value_type **) noexcept;
};

template <DeviceType T, typename P>
inline PetscErrorCode HostAllocator<T, P>::allocate(PetscDeviceContext, size_type n, value_type **ptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscCUPMMallocHost(ptr, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T, typename P>
inline PetscErrorCode HostAllocator<T, P>::deallocate(PetscDeviceContext, value_type **ptr) noexcept
{
  PetscFunctionBegin;
  PetscCallCUPM(cupmFreeHost(*ptr));
  *ptr = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// CUPM Device Allocator
// ==========================================================================================

template <DeviceType T, typename PetscType = char>
class DeviceAllocator;

template <DeviceType T, typename PetscType>
class DeviceAllocator final : public memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>, impl::Interface<T> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);
  using base_type  = memory::impl::SegmentedMemoryPoolAllocatorBase<PetscType>;
  using size_type  = typename base_type::size_type;
  using value_type = typename base_type::value_type;

  static PetscErrorCode allocate(PetscDeviceContext, size_type, value_type **) noexcept;
  static PetscErrorCode deallocate(PetscDeviceContext, value_type **) noexcept;
};

template <DeviceType T, typename P>
inline PetscErrorCode DeviceAllocator<T, P>::allocate(PetscDeviceContext dctx, size_type n, value_type **ptr) noexcept
{
  cupmStream_t *stream;

  PetscFunctionBegin;
  // ASYNC TODO: maybe no need to use the accessors here
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, (void **)&stream));
  PetscCall(PetscCUPMMallocAsync(ptr, n, *stream));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T, typename P>
inline PetscErrorCode DeviceAllocator<T, P>::deallocate(PetscDeviceContext dctx, value_type **ptr) noexcept
{
  cupmStream_t *stream;

  PetscFunctionBegin;
  // ASYNC TODO: maybe no need to use the accessors here
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx, (void **)&stream));
  PetscCallCUPM(cupmFreeAsync(*ptr, *stream));
  *ptr = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // CUPMALLOCATOR_HPP
