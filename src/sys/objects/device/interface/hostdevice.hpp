#ifndef HOSTDEVICE_HPP
#define HOSTDEVICE_HPP

#include "../impls/impldevicebase.hpp" /*I "petscdevice.h" I*/

namespace Petsc
{

namespace Device
{

namespace Host
{

struct Device : ::Petsc::Device::Impl::DeviceBase<Device>
{
private:
  PETSC_DEVICE_IMPL_BASE_CLASS_HEADER(base_type,Device);

  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceType GetPetscDeviceType_()) { return PETSC_DEVICE_HOST; }

public:
  PETSC_NODISCARD static PetscErrorCode initialize(MPI_Comm,PetscInt*,PetscDeviceInitType*) noexcept;

  PETSC_NODISCARD PetscErrorCode getDevice(PetscDevice,PetscInt) const noexcept;

  PETSC_NODISCARD static PetscErrorCode configureDevice(PetscDevice) noexcept;

  PETSC_NODISCARD static PetscErrorCode viewDevice(PetscDevice,PetscViewer) noexcept;
};

} // namespace Host

} // namespace Device

} // namespace Petsc

#endif // HOSTDEVICE_HPP
