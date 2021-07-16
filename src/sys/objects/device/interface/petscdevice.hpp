#ifndef PETSCDEVICE_HPP
#define PETSCDEVICE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmtraits.hpp>

namespace Petsc {

template <CUPMDeviceKind T_>
class CUPMDevice : CUPMTypeTraits<T_>
{
public:
  //! Default constructor
  CUPMDevice() = default;

  //! Copy constructor
  CUPMDevice(const CUPMDevice &other) = default;

  //! Move constructor
  CUPMDevice(CUPMDevice &&other) noexcept = default;

  //! Destructor
  ~CUPMDevice() noexcept = default;

  //! Copy assignment operator
  CUPMDevice& operator=(const CUPMDevice &other) = default;

  //! Move assignment operator
  CUPMDevice& operator=(CUPMDevice &&other) noexcept = default;

  PETSC_NODISCARD PetscErrorCode getDefaultDevice(PetscDevice&);

  PETSC_NODISCARD PetscErrorCode configureDevice(PetscDevice&);

protected:
  // implemented by impls

private:
  static std::vector<PetscDevice> devices;    // configured devices
  static PetscInt                 numDevices; // total number of detected devices
};

template <>
class CUPMDevice<CUPMDeviceKind::CUDA> : CUPMTypeTraits<CUPMDeviceKind::CUDA>
{
public:
  using cupmTraits_t = CUPMTypeTraits<CUPMDeviceKind::CUDA>;

  PETSC_NODISCARD PetscErrorCode getDefaultDevice(PetscDevice &dev)
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;

    PetscFunctionReturn(0);
  }
};


}  // namespace Petsc

#endif
