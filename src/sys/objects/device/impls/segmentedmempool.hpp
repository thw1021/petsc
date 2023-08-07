#ifndef PETSC_SEGMENTEDMEMPOOL_HPP
#define PETSC_SEGMENTEDMEMPOOL_HPP

#include <petsc/private/deviceimpl.h>

#include <petsc/private/cpp/macros.hpp>
#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpp/memory.hpp>

#include <limits>
#include <deque>
#include <vector>

#define PETSC_DEFAULT_STREAM_ID -1

namespace Petsc
{

namespace memory
{

namespace impl
{

// ==========================================================================================
// MemoryChunk
//
// Represents a checked-out region of a MemoryBlock. Tracks the offset into the owning
// MemoryBlock and its size/capacity
// ==========================================================================================

class MemoryChunk {
public:
  using size_type = std::size_t;

  constexpr MemoryChunk(size_type, size_type) noexcept;

  MemoryChunk(MemoryChunk &&) noexcept;
  MemoryChunk &operator=(MemoryChunk &&) noexcept;

  MemoryChunk(const MemoryChunk &) noexcept            = delete;
  MemoryChunk &operator=(const MemoryChunk &) noexcept = delete;

  PETSC_NODISCARD size_type size() const noexcept { return size_; }
  PETSC_NODISCARD size_type start() const noexcept { return start_; }
  // REVIEW ME:
  // make this an actual field, normally each chunk shrinks_to_fit() on begin claimed, but in
  // theory only the last chunk needs to do this
  PETSC_NODISCARD size_type capacity() const noexcept { return size(); }
  PETSC_NODISCARD size_type total_offset() const noexcept { return start() + size(); }

  PetscErrorCode release(PetscDeviceContext, int) noexcept;
  PetscErrorCode claim(PetscDeviceContext, int, size_type, bool) noexcept;
  PetscErrorCode resize(size_type) noexcept;

  PETSC_NODISCARD bool can_claim(int, size_type, bool) const noexcept;
  PETSC_NODISCARD bool contains(size_type) const noexcept;

  void        swap(MemoryChunk &) noexcept;
  friend void swap(MemoryChunk &lhs, MemoryChunk &rhs) noexcept { lhs.swap(rhs); }

private:
  struct PetscEventDeleter {
    void operator()(PetscEvent event) const noexcept
    {
      PetscFunctionBegin;
      PetscCallAbort(PETSC_COMM_SELF, PetscEventDestroy_Internal(&event));
      PetscFunctionReturnVoid();
    }
  };

  using event_type = std::unique_ptr<_n_PetscEvent, PetscEventDeleter>;

  event_type event_{nullptr};
  size_type  size_{0};  // size of the chunk
  size_type  start_{0}; // offset from the start of the owning block
  // id of the last stream to use the chunk, populated on release
  int  stream_id_{PETSC_DEFAULT_STREAM_ID};
  bool open_{true}; // is this chunk open?

  PetscErrorCode       ensure_event_(PetscDeviceContext) noexcept;
  PetscErrorCode       record_event_(PetscDeviceContext) noexcept;
  PetscErrorCode       wait_event_(PetscDeviceContext) noexcept;
  PETSC_NODISCARD bool stream_compat_(int) const noexcept;
};

// ==========================================================================================
// MemoryChunk - Private API
// ==========================================================================================

inline PetscErrorCode MemoryChunk::ensure_event_(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (!event_) {
    PetscEvent tmp;

    PetscCall(PetscDeviceContextCreateEvent_Internal(dctx, &tmp));
    PetscCallCXX(event_.reset(tmp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode MemoryChunk::record_event_(PetscDeviceContext dctx) noexcept
{
  PetscEvent tmp;

  PetscFunctionBegin;
  PetscCall(ensure_event_(dctx));
  tmp = event_.release();
  PetscCall(PetscDeviceContextRecordEvent_Internal(dctx, &tmp));
  PetscCallCXX(event_.reset(tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode MemoryChunk::wait_event_(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(ensure_event_(dctx));
  PetscCall(PetscDeviceContextWaitForEvent_Internal(dctx, event_.get()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// asks and answers the question: can this stream claim this chunk without serializing?
inline bool MemoryChunk::stream_compat_(int sid) const noexcept
{
  return (stream_id_ == PETSC_DEFAULT_STREAM_ID) || (stream_id_ == sid);
}

// ==========================================================================================
// MemoryChunk - Public API
// ==========================================================================================

inline constexpr MemoryChunk::MemoryChunk(size_type start, size_type size) noexcept : size_{size}, start_{start} { }

inline MemoryChunk::MemoryChunk(MemoryChunk &&other) noexcept :
  // clang-format off
  event_{std::move(other.event_)},
  size_{util::exchange(other.size_, 0)},
  start_{util::exchange(other.start_, 0)},
  stream_id_{util::exchange(other.stream_id_, PETSC_DEFAULT_STREAM_ID)},
  open_{util::exchange(other.open_, false)}
// clang-format on
{
}

inline MemoryChunk &MemoryChunk::operator=(MemoryChunk &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    if (event_ == other.event_) {
      if (event_) PetscAssertAbort(event_->refcnt >= 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reference count for shared event < 2: %" PetscInt64_FMT, event_->refcnt);
    } else {
      event_ = std::move(other.event_);
    }
    size_      = util::exchange(other.size_, 0);
    start_     = util::exchange(other.start_, 0);
    stream_id_ = util::exchange(other.stream_id_, PETSC_DEFAULT_STREAM_ID);
    open_      = util::exchange(other.open_, false);
  }
  PetscFunctionReturn(*this);
}

/*
  MemoryChunk::release - release a chunk on a stream

  Input Parameters:
+ dctx - the device context to release the chunk with
- sid  - the id of the device context stream

  Notes:
  Inserts a release operation on dctx and records the state of stream at the time this routine
  was called.

  Future allocation requests which attempt to claim the chunk on the same stream may re-acquire
  the chunk without serialization.

  If another stream attempts to claim the chunk they must wait for the recorded event before
  claiming the chunk.
*/
inline PetscErrorCode MemoryChunk::release(PetscDeviceContext dctx, int sid) noexcept
{
  PetscFunctionBegin;
  open_      = true;
  stream_id_ = sid;
  PetscCall(record_event_(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryChunk::claim - attempt to claim a particular chunk

  Input Parameters:
+ dctx      - the device context on which to attempt to claim
. sid       - the id of the device context stream
. req_size  - the requested size (in elements) to attempt to claim
- serialize - whether the claimant allows serialization

  Notes:
  The claimant must be able to claim the chunk, i.e. can_claim() must return true. This routine
  will error in the case that this is not possible.
*/
inline PetscErrorCode MemoryChunk::claim(PetscDeviceContext dctx, int sid, size_type req_size, bool serialize) noexcept
{
  PetscFunctionBegin;
  PetscAssert(can_claim(sid, req_size, serialize), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Attempting to claim a memory chunk that you are not allowed to claim!");
  open_ = false;
  PetscCall(resize(req_size));
  if (serialize && !stream_compat_(sid)) PetscCall(wait_event_(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryChunk::resize - grow a chunk to new size

  Input Parameter:
. newsize - the new size Requested

  Notes:
  newsize cannot be larger than capacity
*/
inline PetscErrorCode MemoryChunk::resize(size_type newsize) noexcept
{
  PetscFunctionBegin;
  PetscAssert(newsize <= capacity(), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "New size %zu larger than capacity %zu", newsize, capacity());
  size_ = newsize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryChunk::can_claim - test whether a particular chunk can be claimed

  Input Parameters:
+ stream_id - the stream id of the stream attempting to claim
. req_size  - the requested size (in elements) to attempt to claim
- serialize - whether the claimant allows serialization

  Notes:
  Returns true if the chunk is claimable given the configuration, false otherwise.
*/
inline bool MemoryChunk::can_claim(int stream_id, size_type req_size, bool serialize) const noexcept
{
  if (open_ && (req_size <= capacity())) {
    // could claim if we serialized
    if (serialize) return true;
    // fully compatible
    if (stream_compat_(stream_id)) return true;
    // incompatible stream and did not want to serialize
  }
  return false;
}

/*
  MemoryChunk::contains - query whether a memory chunk contains a particular offset

  Input Parameters:
. offset - The offset from the MemoryBlock start

  Notes:
  Returns true if the chunk contains the offset, false otherwise
*/
inline bool MemoryChunk::contains(size_type offset) const noexcept
{
  return (offset >= start()) && (offset < total_offset());
}

inline void MemoryChunk::swap(MemoryChunk &other) noexcept
{
  using std::swap;

  swap(event_, other.event_);
  swap(size_, other.size_);
  swap(start_, other.start_);
  swap(stream_id_, other.stream_id_);
  swap(open_, other.open_);
}

// ==========================================================================================
// MemoryBlock
//
// A "memory block" manager, which owns the pointer to a particular memory range. Retrieving
// and restoring a block is thread-safe (so may be used by multiple device streams).
// ==========================================================================================

template <typename T, typename AllocatorType>
class MemoryBlock {
public:
  using value_type      = T;
  using allocator_type  = AllocatorType;
  using chunk_type      = MemoryChunk;
  using size_type       = typename chunk_type::size_type;
  using chunk_list_type = std::vector<chunk_type>;

  MemoryBlock(PetscDeviceContext, allocator_type *, size_type) noexcept;

  ~MemoryBlock() noexcept(std::is_nothrow_destructible<chunk_list_type>::value);

  MemoryBlock(MemoryBlock &&) noexcept;
  MemoryBlock &operator=(MemoryBlock &&) noexcept;

  // memory blocks are not copyable
  MemoryBlock(const MemoryBlock &)            = delete;
  MemoryBlock &operator=(const MemoryBlock &) = delete;

  /* --- actual functions --- */
  PetscErrorCode       try_allocate_chunk(PetscDeviceContext, int, size_type, T **, bool *) noexcept;
  PetscErrorCode       try_deallocate_chunk(PetscDeviceContext, int, T **, bool *) noexcept;
  PetscErrorCode       try_find_chunk(const T *, chunk_type **) noexcept;
  PetscErrorCode       destroy(PetscDeviceContext) noexcept;
  PETSC_NODISCARD bool owns_pointer(const T *) const noexcept;

  PETSC_NODISCARD size_type size() const noexcept { return size_; }
  PETSC_NODISCARD size_type bytes() const noexcept { return sizeof(value_type) * size(); }
  PETSC_NODISCARD size_type num_chunks() const noexcept { return chunks_.size(); }

  void        swap(MemoryBlock &) noexcept;
  friend void swap(MemoryBlock &lhs, MemoryBlock &rhs) noexcept { lhs.swap(rhs); }

private:
  value_type     *mem_{};
  allocator_type *allocator_{};
  size_type       size_{};
  chunk_list_type chunks_{};
};

// ==========================================================================================
// MemoryBlock - Public API
// ==========================================================================================

// default constructor, allocates memory immediately
template <typename T, typename A>
inline MemoryBlock<T, A>::MemoryBlock(PetscDeviceContext dctx, allocator_type *alloc, size_type s) noexcept : allocator_{alloc}, size_{s}
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, alloc->allocate(dctx, s, &mem_));
  PetscAssertAbort(mem_, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate memory block of size %zu", s);
  PetscFunctionReturnVoid();
}

template <typename T, typename A>
inline MemoryBlock<T, A>::~MemoryBlock() noexcept(std::is_nothrow_destructible<chunk_list_type>::value)
{
  PetscFunctionBegin;
  PetscCheckAbort(!mem_, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not destroy block %p before destructor!", (void *)mem_);
  PetscFunctionReturnVoid();
}

template <typename T, typename A>
inline MemoryBlock<T, A>::MemoryBlock(MemoryBlock &&other) noexcept :
  // clang-format off
  mem_{util::exchange(other.mem_, nullptr)},
  allocator_{other.allocator_},
  size_{util::exchange(other.size_, 0)},
  chunks_{std::move(other.chunks_)}
// clang-format on
{
}

template <typename T, typename A>
inline MemoryBlock<T, A> &MemoryBlock<T, A>::operator=(MemoryBlock &&other) noexcept
{
  if (this != &other) this->swap(other);
  return *this;
}

template <typename T, typename A>
inline void MemoryBlock<T, A>::swap(MemoryBlock &other) noexcept
{
  using std::swap;

  swap(mem_, other.mem_);
  swap(allocator_, other.allocator_);
  swap(size_, other.size_);
  swap(chunks_, other.chunks_);
}

/*
  MemoryBock::owns_pointer - returns true if this block owns a pointer, false otherwise
*/
template <typename T, typename A>
inline bool MemoryBlock<T, A>::owns_pointer(const T *ptr) const noexcept
{
  // each pool is linear in memory, so it suffices to check the bounds
  return (ptr >= mem_) && (ptr < std::next(mem_, size()));
}

/*
  MemoryBlock::try_allocate_chunk - try to get a chunk from this MemoryBlock

  Input Parameters:
+ dctx     - the device context to get the chunk with
. sid      - the stream id of the device context stream
- req_size - the requested size of the allocation (in elements)

  Output Parameters:
+ ptr      - non-null on success, null otherwise
- success  - true if chunk was gotten, false otherwise

  Notes:
  If the current memory could not satisfy the memory request, ptr is unchanged
*/
template <typename T, typename A>
inline PetscErrorCode MemoryBlock<T, A>::try_allocate_chunk(PetscDeviceContext dctx, int sid, size_type req_size, T **ptr, bool *success) noexcept
{
  PetscFunctionBegin;
  if (req_size <= size()) {
    std::size_t idx              = 0;
    std::size_t steal_idx        = chunks_.size();
    bool        pruned           = false;
    const auto  try_create_chunk = [this, req_size, dctx, sid, ptr, success] {
      const auto block_alloced = chunks_.empty() ? 0 : chunks_.back().total_offset();

      PetscFunctionBegin;
      if ((block_alloced + req_size) <= size()) {
        PetscCallCXX(chunks_.emplace_back(block_alloced, req_size));
        PetscCall(chunks_.back().claim(dctx, sid, req_size, false));
        *ptr     = mem_ + block_alloced;
        *success = true;
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    };
    const auto try_claim_chunk = [this, req_size, dctx, sid, ptr, success](chunk_type &chunk, bool serialize) {
      PetscFunctionBegin;
      if (chunk.can_claim(sid, req_size, serialize)) {
        PetscCall(chunk.claim(dctx, sid, req_size, serialize));
        *ptr     = mem_ + chunk.start();
        *success = true;
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    // search previously distributed chunks, but only claim one if it is on the same stream as
    // us
    for (idx = 0; idx < chunks_.size(); ++idx) {
      auto &chunk = chunks_[idx];

      PetscCall(try_claim_chunk(chunk, false));
      if (*success) goto done;
      if (chunk.can_claim(sid, req_size, true)) steal_idx = idx;
    }

    // if we are here we couldn't reuse one of our own chunks so check first if the pool has
    // room for a new one
    PetscCall(try_create_chunk());
    if (*success) goto done;

    // try pruning dead chunks off the back
    while (chunks_.back().can_claim(sid, 0, false)) {
      pruned = true;
      chunks_.pop_back();
      if (chunks_.empty()) break;
    }

    // if previously unsuccessful see if enough space has opened up due to pruning. note that
    // if the chunk list was emptied from the pruning this call must succeed in allocating a
    // chunk, otherwise something is wrong
    if (pruned) {
      PetscCall(try_create_chunk());
      if (*success) goto done;
    }

    // last resort, iterate over the chunks (starting with our steal idx) and see if we can
    // steal one by waiting on the current owner to finish using it. If we never found another
    // chunk to steal then this loop is a no-op
    for (idx = steal_idx; idx < chunks_.size(); ++idx) {
      PetscCall(try_claim_chunk(chunks_[idx], true));
      if (*success) break;
    }
  }
done:
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryBlock::try_deallocate_chunk - try to restore a chunk to this MemoryBlock

  Input Parameters:
+ ptr     - ptr to restore
- stream  - stream to restore the pointer on

  Output Parameter:
. success - true if chunk was restored, false otherwise

  Notes:
  ptr is set to nullptr on successful restore, and is unchanged otherwise. If the ptr is owned
  by this MemoryBlock then it is restored on stream. The same stream may receive ptr again
  without synchronization, but other streams may not do so until either serializing or the
  stream is idle again.
*/
template <typename T, typename A>
inline PetscErrorCode MemoryBlock<T, A>::try_deallocate_chunk(PetscDeviceContext dctx, int sid, T **ptr, bool *success) noexcept
{
  chunk_type *chunk = nullptr;

  PetscFunctionBegin;
  PetscCall(try_find_chunk(*ptr, &chunk));
  if (chunk) {
    PetscCall(chunk->release(dctx, sid));
    *ptr     = nullptr;
    *success = true;
  } else {
    *success = false;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// clear the memory block, called from destructors and move assignment/construction
template <typename T, typename A>
inline PetscErrorCode MemoryBlock<T, A>::destroy(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(mem_)) PetscCall(allocator_->deallocate(dctx, &mem_));
  size_ = 0;
  PetscCallCXX(chunks_.clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MemoryBlock::try_find_chunk - try to find the chunk which owns ptr

  Input Parameter:
. ptr - the pointer to look for

  Output Parameter:
. ret_chunk - pointer to the owning chunk or nullptr if not found
*/
template <typename T, typename A>
inline PetscErrorCode MemoryBlock<T, A>::try_find_chunk(const T *ptr, chunk_type **ret_chunk) noexcept
{
  PetscFunctionBegin;
  *ret_chunk = nullptr;
  if (owns_pointer(ptr)) {
    const auto offset = static_cast<size_type>(ptr - mem_);

    for (auto &chunk : chunks_) {
      if (chunk.contains(offset)) {
        *ret_chunk = &chunk;
        break;
      }
    }

    PetscAssert(*ret_chunk, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to find %zu in block, even though it is within block range [%zu, %zu)", reinterpret_cast<uintptr_t>(ptr), reinterpret_cast<uintptr_t>(mem_), reinterpret_cast<uintptr_t>(std::next(mem_, size())));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
struct SegmentedMemoryPoolAllocatorBase {
  using value_type = T;
  using size_type  = std::size_t;

  static PetscErrorCode allocate(PetscDeviceContext, size_type, value_type **) noexcept;
  static PetscErrorCode deallocate(PetscDeviceContext, value_type **) noexcept;
};

template <typename T>
inline PetscErrorCode SegmentedMemoryPoolAllocatorBase<T>::allocate(PetscDeviceContext, size_type n, value_type **ptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscMalloc1(n, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
inline PetscErrorCode SegmentedMemoryPoolAllocatorBase<T>::deallocate(PetscDeviceContext, value_type **ptr) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

// ==========================================================================================
// SegmentedMemoryPool
//
// Stream-aware async memory allocator. Holds a list of memory "blocks" which each control an
// allocated buffer. This buffer is further split into memory "chunks" which control
// consecutive, non-overlapping regions of the block. Chunks may be in 1 of 2 states:
//
// 1. Open:
//    The chunk is free to be claimed by the next suitable allocation request. If the
//    allocation request is made on the same stream as the chunk was deallocated on, no
//    serialization needs to occur. If not, the allocating stream must wait for the
//    event. Claiming the chunk "closes" the chunk.
//
// 2. Closed:
//    The chunk has been claimed by an allocation request. It cannot be opened again until it
//    is deallocated; doing so "opens" the chunk.
//
// Note that there does not need to be a chunk for every region, chunks are created to satisfy
// an allocation request.
//
// Thus there is usually a region of "unallocated" memory at the end of the buffer, which may
// be claimed by a newly created chunk if existing chunks cannot satisfy the allocation
// request. This region exists _only_ at the end, as there are no gaps between chunks.
//
//
// |-----------------------------------------------------------------------------------------
// | SegmentedMemoryPool
// |
// | ||-------------||
// | ||             ||    -------------------------------------------------------------------
// | ||             ||    | AAAAAAAAAAAAAABBBBBBBCCCCCCCCCCCCCCCCCCCCDDDDDDDDDDDDDXXXXXXXX...
// | ||             ||    | |             |      |                   |            |
// | ||             ||    | x-----x-------x-----xx---------x---------x------x-----x
// | || MemoryBlock || -> | ------|-------------|----------|----------------|--------
// | ||             ||    | | MemoryChunk | MemoryChunk | MemoryChunk | MemoryChunk |
// | ||             ||    | ---------------------------------------------------------
// | ||             ||    -------------------------------------------------------------------
// | ||-------------||
// | ||             ||
// | ||     ...     ||
// | ||             ||
// ==========================================================================================

template <typename MemType, typename AllocType = impl::SegmentedMemoryPoolAllocatorBase<MemType>, std::size_t DefaultChunkSize = 256>
class SegmentedMemoryPool;

// The actual memory pool class. It is in essence just a wrapper for a list of MemoryBlocks.
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
class SegmentedMemoryPool : public RegisterFinalizeable<SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>> {
public:
  using value_type     = MemType;
  using allocator_type = AllocType;
  using block_type     = impl::MemoryBlock<value_type, allocator_type>;
  using pool_type      = std::deque<block_type>;
  using size_type      = typename block_type::size_type;

  explicit SegmentedMemoryPool(AllocType = AllocType{}, std::size_t = DefaultChunkSize) noexcept(std::is_nothrow_default_constructible<pool_type>::value);

  PetscErrorCode allocate(PetscDeviceContext, int, PetscInt, value_type **, size_type = std::alignment_of<MemType>::value) noexcept;
  PetscErrorCode deallocate(PetscDeviceContext, int, value_type **) noexcept;
  PetscErrorCode reallocate(PetscDeviceContext, int, PetscInt, value_type **) noexcept;

private:
  pool_type      pool_{};
  allocator_type allocator_{};
  size_type      chunk_size_{};

  PetscErrorCode make_block_(PetscDeviceContext, size_type) noexcept;

  friend class RegisterFinalizeable<SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>>;
  PetscErrorCode register_finalize_(PetscDeviceContext) noexcept;
  PetscErrorCode finalize_() noexcept;

  PetscErrorCode allocate_(PetscDeviceContext, int, size_type, value_type **) noexcept;
};

// ==========================================================================================
// SegmentedMemoryPool - Private API
// ==========================================================================================

template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::make_block_(PetscDeviceContext dctx, size_type size) noexcept
{
  const auto block_size = std::max(size, chunk_size_);

  PetscFunctionBegin;
  PetscCallCXX(pool_.emplace_back(dctx, &allocator_, block_size));
  PetscCall(PetscInfo(nullptr, "Allocated new block of size %zu, total %zu blocks\n", block_size, pool_.size()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::register_finalize_(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(make_block_(dctx, chunk_size_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::finalize_() noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetNullContext_Internal(&dctx));
  for (auto &block : pool_) PetscCall(block.destroy(dctx));
  PetscCallCXX(pool_.clear());
  chunk_size_ = DefaultChunkSize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::allocate_(PetscDeviceContext dctx, int sid, size_type size, value_type **ptr) noexcept
{
  auto found = false;

  PetscFunctionBegin;
  PetscCall(this->register_finalize(dctx));
  for (auto &block : pool_) {
    PetscCall(block.try_allocate_chunk(dctx, sid, size, ptr, &found));
    if (PetscLikely(found)) PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscInfo(dctx, "Could not find an open block in the pool (%zu blocks) (requested size %zu), allocating new block\n", pool_.size(), size));
  // if we are here we couldn't find an open block in the pool, so make a new block
  PetscCall(make_block_(dctx, size));
  // and assign it
  PetscCall(pool_.back().try_allocate_chunk(dctx, sid, size, ptr, &found));
  PetscAssert(found, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to get a suitable memory chunk (of size %zu) from newly allocated memory block (size %zu)", size, pool_.back().size());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// SegmentedMemoryPool - Public API
// ==========================================================================================

template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::SegmentedMemoryPool(AllocType alloc, std::size_t size) noexcept(std::is_nothrow_default_constructible<pool_type>::value) : allocator_{std::move(alloc)}, chunk_size_{size}
{
}

/*
  SegmentedMemoryPool::allocate - get an allocation from the memory pool

  Input Parameters:
+ dctx      - the device context to get an allocation with
. sid       - the id of the device context stream
. req_size  - size (in elements) to get
- alignment - the desired alignment (in bytes) of the allocation

  Output Parameter:
. ptr - the pointer holding the allocation

  Notes:
  req_size cannot be negative. If req_size if zero, ptr is set to nullptr.

  alignment must be a non-zero power of 2.
*/
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::allocate(PetscDeviceContext dctx, int sid, PetscInt req_size, value_type **ptr, size_type alignment) noexcept
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscAssert(req_size >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested memory amount (%" PetscInt_FMT ") must be >= 0", req_size);
  PetscAssert(alignment > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested memory alignment (%zu) must be > 0", alignment);
  PetscAssertPointer(ptr, 4);
  if (req_size) {
    const auto  size         = static_cast<size_type>(req_size);
    auto        aligned_size = alignment == alignof(char) ? size : size + alignment;
    value_type *ret_ptr      = nullptr;
    void       *vptr         = nullptr;

    PetscCall(allocate_(dctx, sid, aligned_size, &ret_ptr));
    vptr = ret_ptr;
    std::align(alignment, size, vptr, aligned_size);
    *ptr = static_cast<value_type *>(vptr);
  } else {
    *ptr = nullptr;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SegmentedMemoryPool::deallocate - release a pointer back to the memory pool

  Input Parameters:
+ dctx - the device context to deallocate the memory on
. sid  - the id of the device context stream
- ptr  - the pointer to release

  Notes:
  If ptr is not owned by the pool it is unchanged.
*/
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::deallocate(PetscDeviceContext dctx, int sid, value_type **ptr) noexcept
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscAssertPointer(ptr, 3);
  // nobody owns a nullptr, and if they do then they have bigger problems
  if (!*ptr) PetscFunctionReturn(PETSC_SUCCESS);
  for (auto &block : pool_) {
    auto found = false;

    PetscCall(block.try_deallocate_chunk(dctx, sid, ptr, &found));
    if (PetscLikely(found)) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SegmentedMemoryPool::reallocate - Resize an allocated buffer

  Input Parameters:
+ dctx         - the device context to reallocate with
. sid          - the id of the device context stream
. new_req_size - the new buffer size
- ptr          - pointer to the buffer

  Output Parameter:
. ptr - pointer to the new region

  Notes:
  ptr must have been allocated by the pool.

  It's OK to shrink the buffer, even down to 0 (in which case it is just deallocated).
*/
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::reallocate(PetscDeviceContext dctx, int sid, PetscInt new_req_size, value_type **ptr) noexcept
{
  using chunk_type = typename block_type::chunk_type;

  const auto  new_size = static_cast<size_type>(new_req_size);
  const auto  old_ptr  = *ptr;
  chunk_type *chunk    = nullptr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscAssert(new_req_size >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested memory amount (%" PetscInt_FMT ") must be >= 0", new_req_size);
  PetscAssertPointer(ptr, 4);

  // if reallocating to zero, just free
  if (PetscUnlikely(new_size == 0)) {
    PetscCall(deallocate(dctx, sid, ptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // search the blocks for the owning chunk
  for (auto &block : pool_) {
    PetscCall(block.try_find_chunk(old_ptr, &chunk));
    if (chunk) break; // found
  }
  PetscAssert(chunk, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Memory pool does not own %p, so cannot reallocate it", (void *)*ptr);

  if (chunk->capacity() < new_size) {
    // chunk does not have enough room, need to grab a fresh chunk and copy to it
    *ptr = nullptr;
    PetscCall(chunk->release(dctx, sid));
    // ASYNC TODO: handle alignment! The alignment of the reallocated pointer !=
    // alignof(value_type)!
    PetscCall(allocate(dctx, sid, new_size, ptr));
    PetscUseTypeMethod(dctx, memcopy, *ptr, old_ptr, new_size, PETSC_DEVICE_COPY_AUTO);
  } else {
    // chunk had enough room we can simply grow (or shrink) to fit the new size
    PetscCall(chunk->resize(new_size));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace memory

} // namespace Petsc

#endif // PETSC_SEGMENTEDMEMPOOL_HPP
