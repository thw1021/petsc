#ifndef PETSCSYCLDEVICE_HPP
#define PETSCSYCLDEVICE_HPP

#include <petsc/private/deviceimpl.h> /* I "petscdevice.h" */
#include <petsc/private/cupminterface.hpp>
#include <petscviewer.h>
#include <array>
#include <memory>
#include <limits>

namespace Petsc
{
#define PETSC_SYCL_DEVICE_NONE -3
#define PETSC_SYCL_HOST_DEVICE -2

class SyclDevice {
public:

  using createContextFunction_t = PetscErrorCode (*)(PetscDeviceContext);

  // default constructor
  explicit SyclDevice(createContextFunction_t func) noexcept : _create(func) { }

  PETSC_NODISCARD static PetscErrorCode initialize(MPI_Comm,PetscInt*,PetscDeviceInitType*) noexcept;

  PETSC_NODISCARD PetscErrorCode getDevice(PetscDevice,PetscInt) const noexcept;

  PETSC_NODISCARD static PetscErrorCode configureDevice(PetscDevice) noexcept;

  PETSC_NODISCARD static PetscErrorCode viewDevice(PetscDevice,PetscViewer) noexcept;

private:
  // opaque class representing a single device
  class SyclDeviceInternal;

  const createContextFunction_t _create;

  // all known devices
  static std::array<std::unique_ptr<SyclDeviceInternal>,PETSC_DEVICE_MAX_DEVICES> _devices_array;
  static std::unique_ptr<SyclDeviceInternal> *_devices;

  // this ranks default device, if < 0  then devices are specifically disabled
  static int _defaultDevice;

  // have we tried looking for devices
  static bool _initialized;

  // clean-up
  PETSC_NODISCARD static PetscErrorCode _finalize() noexcept;
};


} // namespace Petsc
#endif /* PETSCSYCLDEVICE_HPP */
