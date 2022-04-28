#ifndef PETSC_IMPLDEVICECONTEXTBASE_HPP
#define PETSC_IMPLDEVICECONTEXTBASE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include <vector>

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

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
class SegmentedMemoryPool
{
  using ChunksType = std::vector<MemoryChunk>;
  using size_type  = ChunksType::value_type::size_type;

  const AllocType  allocate_;
  const FreeType   destroy_;
  ChunksType       blocks_;
  MemType         *mem_pool_;

public:
  constexpr SegmentedMemoryPool(AllocType&& alloc, FreeType&& destroy) noexcept
    : allocate_(std::forward<AllocType>(alloc)), destroy_(std::forward<FreeType>(destroy)),
      blocks_(), mem_pool_(nullptr)
  { }

  PETSC_NODISCARD PetscErrorCode finalize() noexcept;
  PETSC_NODISCARD PetscErrorCode initialize() noexcept;
  PETSC_NODISCARD PetscErrorCode get(PetscInt,MemType**) noexcept;
  PETSC_NODISCARD PetscErrorCode release(MemType**) noexcept;
  PETSC_NODISCARD bool           owns_pointer(const MemType*) const noexcept;
};

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::finalize() noexcept
{
  PetscFunctionBegin;
  PetscCall(destroy_(mem_pool_));
  mem_pool_ = nullptr;
  PetscCallCXX(blocks_.clear());
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::initialize() noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(!mem_pool_)) {
    PetscCall(allocate_(&mem_pool_,PoolSize));
    PetscCall(PetscCxxObjectRegisterFinalize(this));
  }
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::get(PetscInt size, MemType **ptr) noexcept
{
  PetscFunctionBegin;
  PetscAssert(size >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot retrieve negative (%" PetscInt_FMT ") memory from the pool",size);
  PetscCall(initialize());
  if (static_cast<decltype(PoolSize)>(size) >= (PoolSize/2)) {
    // any allocation requestion more than half of the pool probably shouldn't go in the pool
    // in the first place
  }
  PetscAssert(,PETSC_COMM_SELF,PETSC_ERR_MEM,"Cannot allocate pool larger than %zu elements",PoolSize);
  {
    auto result = mem_pool_;

    if (blocks_.empty()) {
      PetscCallCXX(blocks_.emplace_back(size));
    } else {
      auto block_alloced = size_type{0};
      // first, search the blocks
      for (auto& block : blocks_) {
        const auto bsize = block.size;

        if (block.open && (bsize <= static_cast<size_type>(size))) {
          // ok found open block of suitable size, claim it.
          // could maybe have shared blocks in the future
          result     = mem_pool_+bsize;
          block.open = false;
          break;
        }
        block_alloced += bsize;
      }
      // no open block found, need to make one
      if (result == mem_pool_) {
        // check that the pool has enough room
        PetscCheck(static_cast<decltype(PoolSize)>(block_alloced+size) <= PoolSize,PETSC_COMM_SELF,PETSC_ERR_MEM,"Allocating block of size %" PetscInt_FMT " would exceed maximum pool capacity %zu (current capacity %zu)",size,PoolSize,block_alloced);
        PetscCallCXX(blocks_.emplace_back(block_alloced,size));
      }
    }
    *ptr = result;
  }
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline PetscErrorCode SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::release(MemType **ptr) noexcept
{
  const auto offset = *ptr-mem_pool_;

  PetscFunctionBegin;
  if (!this->owns_pointer(*ptr)) PetscFunctionReturn(0); // don't own it, bail

  for (auto block = blocks_.begin(); block != blocks_.end(); ++block) {
    if (block->start == static_cast<size_type>(offset)) {
      // ok, found ourselves
      if (std::next(block) == blocks_.end()) {
        // last element of the vector, just destroy it
        PetscCallCXX(blocks_.pop_back());
      } else {
        // somewhere inside, so mark the block free again
        block->open = true;
      }
      break;
    }
    PetscAssert(std::next(block) != blocks_.end(),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find block owning offset %zu in pool",offset);
  }
  *ptr = nullptr;
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, typename FreeType, std::size_t PoolSize>
inline bool SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>::owns_pointer(const MemType *ptr) const noexcept
{
  // if we have no mem_pool_ then we don't own any pointers
  return mem_pool_ && ((ptr >= mem_pool_) && (ptr < std::next(mem_pool_,PoolSize)));
}

template <typename MemType, std::size_t PoolSize = 200, typename AllocType, typename FreeType>
static inline auto make_segmented_memory_pool(AllocType&& alloc, FreeType&& freefn)
PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(SegmentedMemoryPool<MemType,AllocType,FreeType,PoolSize>{
  std::forward<AllocType>(alloc),std::forward<FreeType>(freefn)
});

} // namespace Impl

} // namespace Device

} // namespace Petsc

#endif // PETSC_IMPLDEVICECONTEXTBASE_HPP
