#ifndef PETSC_CUPMEVENT_HPP
#define PETSC_CUPMEVENT_HPP

#include <petsc/private/cupminterface.hpp>

#include <petsc/private/cpp/register_finalize.hpp> // RegisterFinalizeable
#include <petsc/private/cpp/utility.hpp>           // util::exchange

#include <stack>

namespace Petsc
{

namespace device
{

namespace cupm
{

// A pool for allocating cupmEvent_t's. While events are generally very cheap to create and
// destroy, they are not free. Using the pool vs on-demand creation and destruction yields a ~20%
// speedup.
template <DeviceType T, unsigned long flags>
class CUPMEventPool : impl::Interface<T>, public RegisterFinalizeable<CUPMEventPool<T, flags>> {
public:
  PETSC_CUPM_INHERIT_INTERFACE_TYPEDEFS_USING(T);

  PetscErrorCode allocate(cupmEvent_t *) noexcept;
  PetscErrorCode deallocate(cupmEvent_t *) noexcept;

  PetscErrorCode finalize_() noexcept;

private:
  std::stack<cupmEvent_t> pool_;
};

template <DeviceType T, unsigned long flags>
inline PetscErrorCode CUPMEventPool<T, flags>::finalize_() noexcept
{
  PetscFunctionBegin;
  while (!pool_.empty()) {
    PetscCallCUPM(cupmEventDestroy(std::move(pool_.top())));
    PetscCallCXX(pool_.pop());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T, unsigned long flags>
inline PetscErrorCode CUPMEventPool<T, flags>::allocate(cupmEvent_t *event) noexcept
{
  PetscFunctionBegin;
  PetscAssertPointer(event, 1);
  if (pool_.empty()) {
    PetscCall(this->register_finalize());
    PetscCallCUPM(cupmEventCreateWithFlags(event, flags));
  } else {
    PetscCallCXX(*event = std::move(pool_.top()));
    PetscCallCXX(pool_.pop());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T, unsigned long flags>
inline PetscErrorCode CUPMEventPool<T, flags>::deallocate(cupmEvent_t *in_event) noexcept
{
  PetscFunctionBegin;
  PetscAssertPointer(in_event, 1);
  if (auto event = util::exchange(*in_event, cupmEvent_t{})) {
    if (this->registered()) {
      PetscCallCXX(pool_.push(std::move(event)));
    } else {
      PetscCallCUPM(cupmEventDestroy(event));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <DeviceType T, unsigned long flags>
CUPMEventPool<T, flags> &cupm_event_pool() noexcept
{
  static CUPMEventPool<T, flags> pool;
  return pool;
}

// pool of events with timing disabled
template <DeviceType T>
inline auto cupm_fast_event_pool() noexcept -> decltype(cupm_event_pool<T, impl::Interface<T>::cupmEventDisableTiming>()) &
{
  return cupm_event_pool<T, impl::Interface<T>::cupmEventDisableTiming>();
}

// pool of events with timing enabled
template <DeviceType T>
inline auto cupm_timer_event_pool() noexcept -> decltype(cupm_event_pool<T, impl::Interface<T>::cupmEventDefault>()) &
{
  return cupm_event_pool<T, impl::Interface<T>::cupmEventDefault>();
}

} // namespace cupm

} // namespace device

} // namespace Petsc

#endif // PETSC_CUPMEVENT_HPP
