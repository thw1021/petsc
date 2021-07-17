#include "../../interface/petscdevice.hpp"

namespace Petsc {

// Internal "impls" class for CUPMDevice. Each instance represents a single cupm device
template <CUPMDeviceKind T>
class CUPMDevice<T>::PetscDeviceInternal
{
private:
  const int        _id;
  cupmDeviceProp_t _dprop;

protected:
  //! Default constructor
  // protected since this class should never be instantiated outside CUPMDevice
  explicit PetscDeviceInternal(int dev) PETSC_NOEXCEPT : _id(dev) {}

  PETSC_NODISCARD PetscErrorCode __initialize() PETSC_NOEXCEPT
  {
    cupmError_t cerr;

    PetscFunctionBegin;
    cerr = cupmGetDeviceProperties(&_dprop,_id);CHKERRCUPM(cerr);
    PetscFunctionReturn(0);
  }
};

template <CUPMDeviceKind T>
PetscErrorCode CUPMDevice<T>::__initialize() PETSC_NOEXCEPT
{
  int         ndev;
  cupmError_t cerr;

  PetscFunctionBegin;
  if (_initialized) PetscFunctionReturn(0);
  cerr = cupmGetDeviceCount(&ndev);CHKERRCUPM(cerr);
  _devices.reserve(ndev);
  for (int i = 0; i < ndev; ++i) {
    PetscDeviceInternal pdi = new PetscDeviceInternal(i);
    PetscErrorCode      ierr;

    ierr = pdi->__initialize();CHKERRQ(ierr);
    _devices.emplace_back(pdi);
  }
  _initialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
PetscErrorCode CUPMDevice<T>::getDefaultDevice(PetscDevice &device) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = this->__initialize();CHKERRQ(ierr);
  // default device is always the first device for now?
  device->deviceId = _devices[0]->_id;
  device->ops->createcontext = this->_create;
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
PetscErrorCode CUPMDevice<T>::configureDevice(PetscDevice &device) PETSC_NOEXCEPT
{
  PetscFunctionBegin;
  // does nothing for now
  PetscFunctionReturn(0);
}

} // namespace Petsc
