#ifndef PETSCDEVICE_HPP
#define PETSCDEVICE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmtraits.hpp>
#include <vector>

namespace Petsc {

template <CUPMDeviceKind T>
class CUPMDevice : CUPMTypeTraits<T>
{
public:
  typedef PetscErrorCode (*createContextFunc_t)(PetscDeviceContext);

  using cupmType_t = CUPMTypeTraits<T>;
  PETSC_CUPM_INHERIT_TRAITS_TYPEDEFS_USING(cupmType_t);

  //! Default constructor
  explicit CUPMDevice(createContextFunc_t func) : _create(func) {}

  //! Copy constructor
  CUPMDevice(const CUPMDevice &other) PETSC_NOEXCEPT = default;

  //! Move constructor
  CUPMDevice(CUPMDevice &&other) PETSC_NOEXCEPT = default;

  //! Destructor
  ~CUPMDevice() PETSC_NOEXCEPT = default;

  //! Copy assignment operator
  CUPMDevice& operator=(const CUPMDevice &other) = default;

  //! Move assignment operator
  CUPMDevice& operator=(CUPMDevice &&other) PETSC_NOEXCEPT = default;

  PETSC_NODISCARD PetscErrorCode getDefaultDevice(PetscDevice&) PETSC_NOEXCEPT;

  PETSC_NODISCARD PetscErrorCode configureDevice(PetscDevice&) PETSC_NOEXCEPT;

private:
  // Opaque class representing a single device
  class  PetscDeviceInternal;

  // all known devices
  static std::vector<PetscDeviceInternal> _devices;

  createContextFunc_t _create;

  // have we tried looking for devices?
  static PetscBool _initialized;

  // look for devices
  static PETSC_NODISCARD PetscErrorCode __initialize() PETSC_NOEXCEPT;


};

template <CUPMDeviceKind T_>
PetscBool CUPMDevice<T_>::_initialized = PETSC_FALSE;

} // namespace Petsc

#endif
