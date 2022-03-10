#ifndef PETSC_IMPLDEVICECONTEXTBASE_HPP
#define PETSC_IMPLDEVICECONTEXTBASE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include <vector>
#include <deque>

namespace Petsc
{

namespace Device
{

namespace Impl
{

struct MemoryChunk
{
  using size_type = std::size_t;

  const size_type start;
  const size_type size;
  bool            open;

  constexpr MemoryChunk(size_type start_, size_type size_, bool open_ = false) noexcept
    : start(start_), size(size_), open(open_)
  { }

  constexpr MemoryChunk(size_type size_) noexcept : MemoryChunk(0,size_) { }
};

template <typename T, typename AllocType, typename FreeType>
class PETSC_TEMPLATE_VISIBILITY_INTERNAL MemoryBlock;

template <typename T, typename AllocType, typename FreeType>
class MemoryBlock
{
public:
  using value_type = T;
  using ChunksType = std::vector<MemoryChunk>;
  using size_type  = ChunksType::value_type::size_type;

  MemoryBlock(AllocType&& allocate, FreeType&& destroy, size_type s)
    noexcept(noexcept(std::is_nothrow_default_constructible<ChunksType>::value))
    : destructor_(std::forward<FreeType>(destroy)), size_(s), chunks_()
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF,allocate(&mem_,s));
    if (PetscUnlikely(!mem_)) SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_MEM,"Failed to allocate memory block of size %zu",s);
    PetscFunctionReturnVoid();
  }

  ~MemoryBlock() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF,destructor_(mem_));
    PetscFunctionReturnVoid();
  }

  size_type size()       const noexcept { return size_;          }
  size_type num_chunks() const noexcept { return chunks_.size(); }

  PETSC_NODISCARD PetscErrorCode get_chunk(size_type,T**) noexcept;
  PETSC_NODISCARD PetscErrorCode reclaim_chunk(T**)       noexcept;
  PETSC_NODISCARD bool           owns_pointer(T*)   const noexcept;

private:
  value_type      *mem_ = nullptr;
  const FreeType   destructor_;
  const size_type  size_;
  ChunksType       chunks_;
};

template <typename T, typename A, typename F>
inline PetscErrorCode MemoryBlock<T,A,F>::get_chunk(size_type s, T **ptr) noexcept
{
  auto &result = *ptr;

  PetscFunctionBegin;
  if (s > size_) PetscFunctionReturn(0);
  if (chunks_.empty()) {
    PetscCallCXX(chunks_.emplace_back(s));
    result = mem_;
  } else {
    auto block_alloced = size_type{0};

    for (auto& block : chunks_) {
      const auto& bsize = block.size;

      if (block.open && (s <= bsize)) {
        // ok found open block of suitable size, claim it. could maybe have shared blocks
        // in the future
        result     = mem_+block.start;
        block.open = false;
        PetscFunctionReturn(0);
      }
      block_alloced += bsize;
    }
    // if we are here can't steal a block so check if the pool has room for a new one,
    // otherwise bail
    if (block_alloced+s <= size_) {
      PetscCallCXX(chunks_.emplace_back(block_alloced,s));
      result = mem_+block_alloced;
    }
  }
  PetscFunctionReturn(0);
}

template <typename T, typename A, typename F>
inline PetscErrorCode MemoryBlock<T,A,F>::reclaim_chunk(T **ptr) noexcept
{
  const auto offset = *ptr-mem_;

  PetscFunctionBegin;
  if (!this->owns_pointer(*ptr)) PetscFunctionReturn(0); // don't own it, bail

  for (auto block = chunks_.begin(); block != chunks_.end(); ++block) {
    if (block->start == static_cast<size_type>(offset)) {
      // ok, found ourselves
      if (std::next(block) == chunks_.end()) {
        // last element of the vector, just destroy it
        PetscCallCXX(chunks_.pop_back());
      } else {
        // somewhere inside, so mark the block free again
        block->open = true;
      }
      break;
    }
    PetscAssert(std::next(block) != chunks_.end(),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find block owning offset %zu in pool",offset);
  }
  *ptr = nullptr;
  PetscFunctionReturn(0);
}

template <typename T, typename A, typename F>
inline bool MemoryBlock<T,A,F>::owns_pointer(T *ptr) const noexcept
{
  // each pool is linear in memory, so it suffices to check the bounds
  return (ptr >= mem_) && (ptr < std::next(mem_,size_));
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t ChunkSize>
class SegmentedMemoryPool
{
  // list of chunks within the pool
  using PoolType   = std::deque<MemoryBlock<MemType,AllocType,FreeType>>;
  using size_type  = typename PoolType::value_type::size_type;

  const AllocType allocate_;
  const FreeType  destroy_;
  PoolType        pool_;
  bool            init_ = false;

  PETSC_NODISCARD PetscErrorCode make_block_(size_type size = ChunkSize) noexcept
  {
    PetscFunctionBegin;
    PetscCallCXX(pool_.emplace_back(allocate_,destroy_,size));
    PetscFunctionReturn(0);
  }

public:
  constexpr SegmentedMemoryPool(AllocType&& alloc, FreeType&& destroy) noexcept
    : allocate_(std::forward<AllocType>(alloc)), destroy_(std::forward<FreeType>(destroy)),
      pool_()
  { }

  PETSC_NODISCARD PetscErrorCode finalize()              noexcept;
  PETSC_NODISCARD PetscErrorCode initialize()            noexcept;
  PETSC_NODISCARD PetscErrorCode get(PetscInt,MemType**) noexcept;
  PETSC_NODISCARD PetscErrorCode release(MemType**)      noexcept;
};

template <typename MemType, typename AllocType, typename FreeType, std::size_t ChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,ChunkSize>::finalize() noexcept
{
  PetscFunctionBegin;
  PetscCallCXX(pool_.clear());
  PetscCallCXX(pool_.shrink_to_fit());
  init_ = false;
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t ChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,ChunkSize>::initialize() noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(!init_)) {
    init_ = true;
    PetscCall(make_block_());
    PetscCall(PetscCxxObjectRegisterFinalize(this));
  }
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t ChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,ChunkSize>::get(PetscInt sizein, MemType **ptr) noexcept
{
  const auto size = static_cast<size_type>(sizein);

  PetscFunctionBegin;
  PetscAssert(sizein > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot retrieve negative (%" PetscInt_FMT ") memory from the pool",sizein);
  PetscCall(initialize());
  *ptr = nullptr;
  for (auto& block : pool_) {
    PetscCall(block.get_chunk(size,ptr));
    if (*ptr) PetscFunctionReturn(0);
  }

  PetscCall(PetscInfo(nullptr,"Could not find an open block in the pool (requested size %" PetscInt_FMT "), allocating new block\n",sizein));
  // if we are here we couldn't find an open block in the pool, so make a new block
  PetscCall(make_block_(std::max(size,ChunkSize)));
  // and assign it
  PetscCall(pool_.back().get_chunk(size,ptr));
  PetscAssert(*ptr,PETSC_COMM_SELF,PETSC_ERR_MEM,"Failed to get a suitable memory chunk (of size %zu) from newly allocated memory block (size %zu)",size,pool_.back().size());
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t ChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,ChunkSize>::release(MemType **ptr) noexcept
{
  PetscFunctionBegin;
  // nobody owns a nullptr, and if they do then they have bigger problems
  if (!*ptr) PetscFunctionReturn(0);
  for (auto& block : pool_) {
    PetscCall(block.reclaim_chunk(ptr));
    if (!*ptr) break;
  }
  // try to prune the pool in case of large allocations
  if (pool_.size() > 1 && pool_.back().num_chunks() == 0) {
    PetscCall(PetscInfo(nullptr,"Freeing empty block of size %zu from pool\n",pool_.back().size()));
    PetscCallCXX(pool_.pop_back());
  }
  PetscFunctionReturn(0);
}

template <typename MemType, std::size_t ChunkSize = 200, typename AllocType, typename FreeType>
static inline auto make_segmented_memory_pool(AllocType&& alloc, FreeType&& freefn)
PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(SegmentedMemoryPool<MemType,AllocType,FreeType,ChunkSize>{
  std::forward<AllocType>(alloc),std::forward<FreeType>(freefn)
});

} // namespace Impl

} // namespace Device

} // namespace Petsc

#endif // PETSC_IMPLDEVICECONTEXTBASE_HPP
