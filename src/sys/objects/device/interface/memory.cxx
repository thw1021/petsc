#include <petsc/private/deviceimpl.h> /*I <petscdevice.h> I*/

#include <petsc/private/cpp/type_traits.hpp> // integral_value
#include <petsc/private/cpp/unordered_map.hpp>

#include <algorithm> // std::find_if
#include <cstring>   // std::memset

const char *const PetscDeviceCopyModes[] = {"host_to_host", "device_to_host", "host_to_device", "device_to_device", "auto", "PetscDeviceCopyMode", "PETSC_DEVICE_COPY_", nullptr};
static_assert(Petsc::util::to_underlying(PETSC_DEVICE_COPY_HTOH) == 0, "");
static_assert(Petsc::util::to_underlying(PETSC_DEVICE_COPY_DTOH) == 1, "");
static_assert(Petsc::util::to_underlying(PETSC_DEVICE_COPY_HTOD) == 2, "");
static_assert(Petsc::util::to_underlying(PETSC_DEVICE_COPY_DTOD) == 3, "");
static_assert(Petsc::util::to_underlying(PETSC_DEVICE_COPY_AUTO) == 4, "");

// GCC implementation for std::hash<T*>. LLVM's libc++ is almost 2x slower because they do all
// kinds of complicated murmur hashing, so we make sure to enforce GCC's version.
struct PointerHash {
  template <typename T>
  PETSC_NODISCARD std::size_t operator()(const T *ptr) const noexcept
  {
    return reinterpret_cast<std::size_t>(ptr);
  }
};

// ==========================================================================================
// MemoryMap
//
// Since the pointers allocated via PetscDeviceAllocate_Private() may be device pointers we
// cannot just store metadata within the pointer itself (as we can't dereference them). So
// instead we need to keep an extra map to keep track of them.
//
// See PetscPointerAttributes for more information on the metadata.
// ==========================================================================================

class MemoryMap {
public:
  using map_type = Petsc::UnorderedMap<void *, PetscPointerAttributes, PointerHash>;

  map_type map{};

  PETSC_NODISCARD map_type::const_iterator csearch_for(const void *, bool = false) const noexcept;
  PETSC_NODISCARD map_type::iterator       search_for(const void *, bool = false) noexcept;

  PetscErrorCode register_mem(const void *, PetscMemType, std::size_t, std::size_t, bool, PetscObjectId *) noexcept;
  PetscErrorCode get_pointer_attributes(const void *, PetscPointerAttributes *, PetscBool *) const noexcept;

private:
  template <typename T>
  static auto search_for_impls(T &&, const void *, bool) noexcept -> decltype(std::declval<T>().end());
};

// ==========================================================================================
// MemoryMap - Private API
// ==========================================================================================

template <typename T>
auto MemoryMap::search_for_impls(T &&map, const void *ptr, bool must_find) noexcept -> decltype(std::declval<T>().end())
{
  const auto end_it = map.end();
  auto       it     = map.find(const_cast<typename map_type::key_type>(ptr));

  PetscFunctionBegin;
  if (it != end_it) {
    // ptr was found, and points to an entire block
    PetscFunctionReturn(it);
  }
  // wasn't found, but maybe its part of a block. have to search every block for it
  // clang-format off
  it = std::find_if(map.begin(), end_it, [=](typename map_type::const_iterator::reference map_it) {
    return map_it.second.contains(map_it.first, ptr);
  });
  // clang-format on
  if (must_find) PetscCheckAbort(it != end_it, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Pointer %p was not registered with the memory tracker, call PetscDeviceRegisterMemory() on it", ptr);
  PetscFunctionReturn(it);
}

// ==========================================================================================
// MemoryMap - Public API
// ==========================================================================================

// A helper utility, since register is called from PetscDeviceRegisterMemory() and
// PetscDevicAllocate(). The latter also needs the generated id, so instead of making it search
// the map again we just return it here
PetscErrorCode MemoryMap::register_mem(const void *ptr, PetscMemType mtype, std::size_t size, std::size_t align, bool deep_search, PetscObjectId *id) noexcept
{
  const auto vptr = const_cast<typename map_type::key_type>(ptr);
  const auto it   = deep_search ? this->csearch_for(ptr) : this->map.find(vptr);

  PetscFunctionBegin;
  if (it == this->map.cend()) {
    auto attr = PetscPointerAttributes{mtype, PetscObjectNewId_Internal(), size, align};

    *id = attr.id;
    PetscCallCXX(this->map[vptr] = std::move(attr));
  } else {
    const auto &old = it->second;

    if (PetscDefined(USE_DEBUG)) {
      const auto attr2 = PetscPointerAttributes{mtype, old.id, old.size, align};

      PetscCheck(attr2 == old, PETSC_COMM_SELF, PETSC_ERR_LIB, "Pointer %p appears to have been previously allocated (memtype %s, id %" PetscInt64_FMT ", size %zu, align %zu), which does not match new values: (mtype %s, id %" PetscInt64_FMT ", size %zu, align %zu)", it->first,
                 PetscMemTypeToString(old.mtype), old.id, old.size, old.align, PetscMemTypeToString(attr2.mtype), attr2.id, attr2.size, attr2.align);
    }
    *id = old.id;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MemoryMap::get_pointer_attributes(const void *ptr, PetscPointerAttributes *attr, PetscBool *found) const noexcept
{
  auto &&it = this->csearch_for(ptr);

  PetscFunctionBegin;
  if (it == this->map.end()) {
    if (found) *found = PETSC_FALSE;
  } else {
    *attr = it->second;
    if (found) *found = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryMap::csearch_for - retrieve an iterator to the key-value pair for a pointer in the map

  Input Parameters:
+ ptr       - pointer to search for
- must_find - true if an error is raised if the pointer is not found (default: false)

  Notes:
  Accounts for sub-regions, i.e. if ptr is contained within another pointers region, it returns
  the iterator to the super-pointers key-value pair.

  If ptr is not found and must_find is false returns map.end(), otherwise raises an error
*/
MemoryMap::map_type::const_iterator MemoryMap::csearch_for(const void *ptr, bool must_find) const noexcept
{
  return this->search_for_impls(this->map, ptr, must_find);
}

MemoryMap::map_type::iterator MemoryMap::search_for(const void *ptr, bool must_find) noexcept
{
  return this->search_for_impls(this->map, ptr, must_find);
}

namespace
{

MemoryMap memory_map;

// ==========================================================================================
// Utility functions
// ==========================================================================================

PetscErrorCode PetscDeviceCheckCapable_Private(PetscDeviceContext dctx, bool cond, const char descr[])
{
  PetscFunctionBegin;
  PetscCheck(cond, PETSC_COMM_SELF, PETSC_ERR_SUP, "Device context (id: %" PetscInt64_FMT ", name: %s, type: %s) can only handle %s host memory", PetscObjectCast(dctx)->id, PetscObjectCast(dctx)->name, dctx->device ? PetscDeviceTypes[dctx->device->type] : "unknown", descr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace

/*@C
  PetscDeviceRegisterMemory - Register a pointer for use with device-aware memory system

  Not Collective

  Input Parameters:
+ ptr   - The pointer to register
- attr  - The `PetscPointerAttributes` which describe the pointer

  Notes:
  If constructing the `PetscPointerAttributes` by hand, `attr->id` should be set to
  `PETSC_UNKNOWN_MEMORY_ID`.

  It's OK to re-register the same `ptr` repeatedly (subsequent registrations do nothing)
  however the given `attr->mtype` and `attr->size` must match the original registration.

  `attr->size` may be 0 (in which case this routine does nothing).

  Level: intermediate

.seealso: `PetscDeviceMalloc()`, `PetscDeviceArrayCopy()`, `PetscDeviceFree()`,
`PetscDeviceArrayZero()`
@*/
PetscErrorCode PetscDeviceRegisterMemory(const void *ptr, PetscPointerAttributes *attr)
{
  PetscFunctionBegin;
  PetscAssertPointer(attr, 2);
  if (PetscMemTypeHost(attr->mtype)) PetscAssertPointer(ptr, 1);
  if (PetscUnlikely(!attr->size)) PetscFunctionReturn(PETSC_SUCCESS); // there is no point registering empty range
  PetscCall(memory_map.register_mem(ptr, attr->mtype, attr->size, attr->align, true, &attr->id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceGetPointerAttributes(const void *ptr, PetscPointerAttributes *attr, PetscBool *found)
{
  PetscFunctionBegin;
  // cannot PetscAssertPointer(ptr), it may be a device pointer!
  PetscAssertPointer(attr, 2);
  if (found) PetscAssertPointer(found, 3);
  PetscCall(memory_map.get_pointer_attributes(ptr, attr, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDeviceAllocate_Private - Allocate device-aware memory

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx      - The `PetscDeviceContext` used to allocate the memory
. clear     - Whether or not the memory should be zeroed
. mtype     - The type of memory to allocate
. n         - The amount (in bytes) to allocate
- alignment - The alignment requirement (in bytes) of the allocated pointer

  Output Parameter:
. ptr - The pointer to store the result in

  Notes:
  The user should prefer `PetscDeviceMalloc()` over this routine as it automatically computes
  the size of the allocation and alignment based on the size of the datatype.

  If the user is unsure about `alignment` -- or unable to compute it -- passing
  `PETSC_MEMALIGN` will always work, though the user should beware that this may be quite
  wasteful for very small allocations.

  Memory allocated with this function must be freed with `PetscDeviceFree()` (or
  `PetscDeviceDeallocate_Private()`).

  If `n` is zero, then `ptr` is set to `PETSC_NULLPTR`.

  This routine falls back to using `PetscMalloc1()` or `PetscCalloc1()` (depending on the value
  of `clear`) if PETSc was not configured with device support. The user should note that
  `mtype` and `alignment` are ignored in this case, as these routines allocate only host memory
  aligned to `PETSC_MEMALIGN`.

  Note result stored `ptr` is immediately valid and the user may freely inspect or manipulate
  its value on function return, i.e.\:

.vb
  PetscInt *ptr;

  PetscDeviceAllocate_Private(dctx, PETSC_FALSE, PETSC_MEMTYPE_DEVICE, 20, alignof(PetscInt), (void**)&ptr);

  PetscInt *sub_ptr = ptr + 10; // OK, no need to synchronize

  ptr[0] = 10; // ERROR, directly accessing contents of ptr is undefined until synchronization
.ve

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| -\- dctx -->
                         \- ptr ->
.ve

  Level: intermediate

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscDeviceFree()`, `PetscDeviceDeallocate_Private()`,
`PetscDeviceArrayCopy()`, `PetscDeviceArrayZero()`, `PetscMemType`
*/
PetscErrorCode PetscDeviceAllocate_Private(PetscDeviceContext dctx, PetscBool clear, PetscMemType mtype, std::size_t n, std::size_t alignment, void **PETSC_RESTRICT ptr)
{
  PetscObjectId id;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    const auto is_power_of_2 = [](std::size_t num) { return (num & (num - 1)) == 0; };

    PetscCheck(alignment != 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested alignment %zu cannot be 0", alignment);
    PetscCheck(is_power_of_2(alignment), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested alignment %zu must be a power of 2", alignment);
  }
  PetscAssertPointer(ptr, 6);
  *ptr = nullptr;
  if (PetscUnlikely(!n)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

  // get our pointer here
  if (dctx->ops->memalloc) {
    PetscUseTypeMethod(dctx, memalloc, clear, mtype, n, alignment, ptr);
  } else {
    PetscCall(PetscDeviceCheckCapable_Private(dctx, PetscMemTypeHost(mtype), "allocating"));
    PetscCall(PetscMallocA(1, clear, __LINE__, PETSC_FUNCTION_NAME, __FILE__, n, ptr));
  }
  PetscCall(memory_map.register_mem(*ptr, mtype, n, alignment, false, &id));
  // Note this is a "write" so that the next dctx to try and read from the pointer has to wait
  // for the allocation to be ready
  PetscCall(PetscDeviceContextMarkIntentFromIDEnd(dctx, id, PETSC_MEMORY_ACCESS_WRITE, "memory allocation"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDeviceDeallocate_Private - Free device-aware memory

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to free the memory
- ptr   - The pointer to free

  Level: intermediate

  Notes:
  `ptr` must have been allocated using any of `PetscDeviceMalloc()`, `PetscDeviceCalloc()` or
  `PetscDeviceAllocate_Private()`, or registered with the system via `PetscDeviceRegisterMemory()`.

  The user should prefer `PetscDeviceFree()` over this routine as it automatically sets `ptr`
  to `PETSC_NULLPTR` on successful deallocation.

  `ptr` may be `NULL`.

  This routine falls back to using `PetscFree()` if PETSc was not configured with device
  support. The user should note that `PetscFree()` frees only host memory.

  DAG representation:
.vb
  time ->

  -> dctx -/- |= CALL =| - dctx ->
  -> ptr -/
.ve

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceAllocate_Private()`
*/
PetscErrorCode PetscDeviceDeallocate_Private(PetscDeviceContext dctx, void *PETSC_RESTRICT ptr)
{
  PetscFunctionBegin;
  if (ptr) {
    auto &map = memory_map.map;
    auto  it  = map.find(const_cast<MemoryMap::map_type::key_type>(ptr));

    if (PetscUnlikelyDebug(it == map.end())) {
      // OK this is a bad pointer, now determine why
      it = memory_map.search_for(ptr);

      // if it is map.cend() then no allocation owns it, meaning it was not allocated by us!
      PetscCheck(it != map.cend(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Pointer %p was not allocated via PetscDeviceAllocate_Private()", ptr);
      // if we are here then we did allocate it but the user has tried to do something along
      // the lines of:
      //
      // allocate(&ptr, size);
      // deallocate(ptr + 5);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempting to deallocate pointer %p which is a suballocation of %p (memtype %s, id %" PetscInt64_FMT ", size %zu bytes)", ptr, it->first, PetscMemTypeToString(it->second.mtype), it->second.id,
              it->second.size);
    }
    auto     &&attr = it->second;
    const auto id   = attr.id;

    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    // Note this is a "write" operation since deallocating the memory is destructive
    PetscCall(PetscDeviceContextMarkIntentFromIDBegin(dctx, id, PETSC_MEMORY_ACCESS_WRITE, "memory deallocation"));
    // do free
    if (dctx->ops->memfree) {
      PetscUseTypeMethod(dctx, memfree, &attr, (void **)&ptr);
    } else {
      PetscCall(PetscDeviceCheckCapable_Private(dctx, PetscMemTypeHost(attr.mtype), "freeing"));
    }
    // if ptr still exists, then the device context could not handle it
    if (ptr) PetscCall(PetscFree(ptr));
    PetscCall(PetscDeviceContextClearIntentFromID(id));
    PetscCallCXX(map.erase(it));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceReallocate_Private(PetscDeviceContext dctx, std::size_t reqsize, void **PETSC_RESTRICT ptr)
{
  PetscFunctionBegin;
  PetscAssertPointer(ptr, 3);
  if (reqsize == 0) {
    PetscCall(PetscDeviceFree(dctx, *ptr));
  } else {
    auto      &attr = memory_map.search_for(*ptr, true)->second;
    const auto size = attr.size;

    // either already big enough or shrinking, either way, do nothing to it
    if (reqsize <= size) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
    if (dctx->ops->memrealloc) {
      PetscUseTypeMethod(dctx, memrealloc, reqsize, &attr, ptr);
    } else {
      void *tmp = nullptr;

      PetscCall(PetscDeviceAllocate_Private(dctx, PETSC_FALSE, attr.mtype, reqsize, attr.align, &tmp));
      PetscCall(PetscDeviceMemcpy(dctx, tmp, *ptr, size, nullptr, &attr));
      PetscCall(PetscDeviceFree(dctx, *ptr));
      *ptr = tmp;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PETSC_DEVICE_MEMCPY_HEADER(n, dest, src, dctx) \
  do { \
    if (!(n)) PetscFunctionReturn(PETSC_SUCCESS); \
    PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer"); \
    PetscCheck(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer"); \
    if ((dest) == (src)) PetscFunctionReturn(PETSC_SUCCESS); \
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&(dctx))); \
  } while (0)

PetscErrorCode PetscDeviceMemcpyRaw(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, PetscDeviceCopyMode cmode, std::size_t n)
{
  PetscFunctionBegin;
  PETSC_DEVICE_MEMCPY_HEADER(n, dest, src, dctx);
  if (dctx->ops->memcopy) {
    PetscUseTypeMethod(dctx, memcopy, dest, src, n, cmode);
    if (cmode == PETSC_DEVICE_COPY_HTOD) {
      PetscCall(PetscLogCpuToGpu(n));
    } else if (cmode == PETSC_DEVICE_COPY_DTOH) {
      PetscCall(PetscLogGpuToCpu(n));
    }
  } else {
    // REVIEW ME: we might potentially need to sync here if the memory is device-allocated
    // (pinned) but being copied by a host dctx
    PetscCall(PetscDeviceCheckCapable_Private(dctx, cmode == PETSC_DEVICE_COPY_HTOH, "copying"));
    PetscCall(PetscMemcpy(dest, src, n));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscClangLinter pragma disable: -fdoc-section-header-unknown
/*@C
  PetscDeviceMemcpy - Copy memory in a device-aware manner

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx - The `PetscDeviceContext` used to copy the memory
. dest - The pointer to copy to
. src  - The pointer to copy from
. n    - The amount (in bytes) to copy
. dest_ptr_attr - The pointer attributes of `dest` if known, `NULL` if not
- src_ptr_attr  - The pointer attributes of `src` if known, `NULL` if not

  Level: intermediate

  Notes:
  If `dest` or `src` were allocated by `PetscDeviceMalloc()` the user may freely pass `NULL`
  for the corresponding pointer attribute argument. In this case the pointer attributes will be
  looked up by PETSc.

  If, however, either `dest` or `src` was not allocated by `PetscDeviceMalloc()` and the
  corresponding pointer attribute argument is `NULL`, then an error is raised.

  This routine may be used performing dependency-aware memory copies on aritrary pointer types,
  provided the user fully describes the pointer in the corresponding pointer attribute
  argument.

  `src` and `dest` cannot overlap.

  If both `src` and `dest` are on the host this routine is fully synchronous.

  The user should prefer `PetscDeviceArrayCopy()` over this routine as it automatically
  computes the number of bytes to copy from the size of the pointer types. The user should note
  however that it does not allow specifying the pointer attributes.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ->
  -> dest --------------------->
  -> src ---------------------->
.ve

.N ASYNC_API

.seealso: `PetscDeviceArrayCopy()`, `PetscDeviceMalloc()`, `PetscDeviceCalloc()`,
`PetscDeviceFree()`
@*/
PetscErrorCode PetscDeviceMemcpy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, std::size_t n, const PetscPointerAttributes *dest_ptr_attr, const PetscPointerAttributes *src_ptr_attr)
{
  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(dest, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy to a NULL pointer");
  PetscCheck(src, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to copy from a NULL pointer");
  if (dest == src) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  {
    const auto &dest_attr = dest_ptr_attr ? *dest_ptr_attr : memory_map.csearch_for(dest, true)->second;
    const auto &src_attr  = src_ptr_attr ? *src_ptr_attr : memory_map.csearch_for(src, true)->second;
    const auto  mode      = PetscMemTypeToDeviceCopyMode(dest_attr.mtype, src_attr.mtype);
    const auto  sid       = src_attr.id;
    const auto  did       = dest_attr.id;

    PetscCall(PetscDeviceContextMarkIntentFromIDBegin(dctx, did, PETSC_MEMORY_ACCESS_WRITE, "memory copy (dest)"));
    PetscCall(PetscDeviceContextMarkIntentFromIDBegin(dctx, sid, PETSC_MEMORY_ACCESS_READ, "memory copy (src)"));
    // perform the copy
    PetscCall(PetscDeviceMemcpyRaw(dctx, dest, src, mode, n));
    PetscCall(PetscDeviceContextMarkIntentFromIDEnd(dctx, did, PETSC_MEMORY_ACCESS_WRITE, "memory copy (dest)"));
    PetscCall(PetscDeviceContextMarkIntentFromIDEnd(dctx, sid, PETSC_MEMORY_ACCESS_READ, "memory copy (src)"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PETSC_DEVICE_MEMSET_HEADER(n, ptr, dctx) \
  do { \
    if (PetscUnlikely(!(n))) PetscFunctionReturn(PETSC_SUCCESS); \
    PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Trying to memset a NULL pointer"); \
    PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&(dctx))); \
  } while (0)

PetscErrorCode PetscDeviceMemsetRaw(PetscDeviceContext dctx, void *ptr, PetscInt v, std::size_t n, PetscMemType mtype)
{
  PetscFunctionBegin;
  PETSC_DEVICE_MEMSET_HEADER(n, ptr, dctx);
  if (dctx->ops->memset) {
    PetscUseTypeMethod(dctx, memset, mtype, ptr, v, n);
  } else {
    // REVIEW ME: we might potentially need to sync here if the memory is device-allocated
    // (pinned) but being memset by a host dctx
    PetscCall(PetscDeviceCheckCapable_Private(dctx, PetscMemTypeHost(mtype), "memsetting"));
    std::memset(ptr, static_cast<int>(v), n);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PetscClangLinter pragma disable: -fdoc-section-header-unknown
/*@C
  PetscDeviceMemset - Memset device-aware memory

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx - The `PetscDeviceContext` used to memset the memory
. ptr  - The pointer to the memory
. v    - The value to set
- n    - The amount (in bytes) to set

  Level: intermediate

  Notes:
  `ptr` must have been allocated by `PetscDeviceMalloc()` or `PetscDeviceCalloc()`.

  The user should prefer `PetscDeviceArrayZero()` over this routine as it automatically
  computes the number of bytes to copy from the size of the pointer types, though they should
  note that it only zeros memory.

  This routine is analogous to `memset()`. That is, this routine copies the value
  `static_cast<unsigned char>(v)` into each of the first count characters of the object pointed
  to by `dest`.

  If `dest` is on device, this routine is asynchronous.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ->
  -> dest --------------------->
.ve

.N ASYNC_API

.seealso: `PetscDeviceArrayZero()`, `PetscDeviceMalloc()`, `PetscDeviceCalloc()`,
`PetscDeviceFree()`
@*/
PetscErrorCode PetscDeviceMemset(PetscDeviceContext dctx, void *ptr, PetscInt v, std::size_t n)
{
  PetscFunctionBegin;
  PETSC_DEVICE_MEMSET_HEADER(n, ptr, dctx);
  {
    auto     &&attr  = memory_map.csearch_for(ptr, true)->second;
    const auto id    = attr.id;
    const auto mtype = attr.mtype;

    PetscCall(PetscDeviceContextMarkIntentFromIDBegin(dctx, id, PETSC_MEMORY_ACCESS_WRITE, "memory set"));
    PetscCall(PetscDeviceMemsetRaw(dctx, ptr, v, n, mtype));
    PetscCall(PetscDeviceContextMarkIntentFromIDEnd(dctx, id, PETSC_MEMORY_ACCESS_WRITE, "memory set"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
