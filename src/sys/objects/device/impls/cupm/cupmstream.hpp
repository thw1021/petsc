#ifndef PETSC_CUPMSTREAM_HPP
#define PETSC_CUPMSTREAM_HPP

#include <petsc/private/cupminterface.hpp>

#include "../segmentedmempool.hpp"

namespace Petsc
{

namespace device
{

namespace cupm
{

// A bare wrapper around a cupmStream_t. The reason it exists is because we need to uniquely
// identify separate cupm streams. This is so that the memory pool can accelerate allocation
// calls as it can just pass back a pointer to memory that was used on the same
// stream. Otherwise it must either serialize with another stream or allocate a new chunk.
// Address of the objects does not suffice since cupmStreams are very likely internally reused.

template <DeviceType T>
class CUPMStream : impl::Interface<T> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  using stream_type = cupmStream_t;
  using flag_type   = unsigned int;

  CUPMStream() noexcept = default;

  PetscErrorCode destroy() noexcept;
  PetscErrorCode create(flag_type) noexcept;
  PetscErrorCode change_type(PetscStreamType, bool *) noexcept;

  PETSC_NODISCARD const stream_type &get_stream() const noexcept;
  PETSC_NODISCARD int                get_id() const noexcept;

private:
  stream_type stream_{};
  int         id_ = new_id_();

  PETSC_NODISCARD static int new_id_() noexcept;
};

// ==========================================================================================
// CUPMStream -- Private API
// ==========================================================================================

template <DeviceType T>
inline int CUPMStream<T>::new_id_() noexcept
{
  static int id = 0;
  return id++;
}

// ==========================================================================================
// CUPMStream -- Public API
// ==========================================================================================

template <DeviceType T>
inline PetscErrorCode CUPMStream<T>::destroy() noexcept
{
  PetscFunctionBegin;
  if (stream_) {
    PetscCallCUPM(cupmStreamDestroy(stream_));
    stream_ = cupmStream_t{};
    id_     = PETSC_DEFAULT_STREAM_ID;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode CUPMStream<T>::create(flag_type flags) noexcept
{
  PetscFunctionBegin;
  if (stream_) {
    if (PetscDefined(USE_DEBUG)) {
      flag_type current_flags;

      PetscCallCUPM(cupmStreamGetFlags(stream_, &current_flags));
      PetscCheck(flags == current_flags, PETSC_COMM_SELF, PETSC_ERR_GPU, "Current flags %u != requested flags %u for stream %d", current_flags, flags, id_);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCallCUPM(cupmStreamCreateWithFlags(&stream_, flags));
  id_ = new_id_();
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline PetscErrorCode CUPMStream<T>::change_type(PetscStreamType newtype, bool *did_change) noexcept
{
  PetscFunctionBegin;
  *did_change = true;
  if (newtype == PETSC_STREAM_GLOBAL_BLOCKING) {
    if (!get_stream()) *did_change = false;
    PetscCall(destroy());
  } else {
    const flag_type preferred = newtype == PETSC_STREAM_DEFAULT_BLOCKING ? cupmStreamDefault : cupmStreamNonBlocking;

    if (stream_) {
      flag_type flag;

      PetscCallCUPM(cupmStreamGetFlags(stream_, &flag));
      if (flag == preferred) {
        *did_change = false;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      PetscCall(destroy());
    }
    PetscCall(create(preferred));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T>
inline const typename CUPMStream<T>::stream_type &CUPMStream<T>::get_stream() const noexcept
{
  return stream_;
}

template <DeviceType T>
inline int CUPMStream<T>::get_id() const noexcept
{
  return id_;
}

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // PETSC_CUPMSTREAM_HPP
