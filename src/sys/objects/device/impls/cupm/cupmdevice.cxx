#include "../../interface/cupmdevice.hpp"

namespace Petsc {

// internal "impls" class for CUPMDevice. Each instance represents a single cupm device
template <CUPMDeviceKind T>
class CUPMDevice<T>::PetscDeviceInternal
{
private:
  const int        _id;
  cupmDeviceProp_t _dprop;

public:
  // default constructor
  explicit PetscDeviceInternal(int dev) PETSC_NOEXCEPT : _id{dev} {}

  PETSC_NODISCARD PetscErrorCode initialize() PETSC_NOEXCEPT
  {
    cupmError_t cerr;

    PetscFunctionBegin;
    cerr = cupmGetDeviceProperties(&this->_dprop,this->_id);CHKERRCUPM(cerr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD int id() const PETSC_NOEXCEPT { return this->_id;}
  PETSC_NODISCARD const cupmDeviceProp_t& deviceProp() const PETSC_NOEXCEPT { return this->_dprop;}
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
    PetscErrorCode ierr;

    try {
      _devices.emplace_back(std::unique_ptr<PetscDeviceInternal>(new PetscDeviceInternal{i}));
      ierr = _devices[i]->initialize();CHKERRQ(ierr);
    } catch(const std::exception &ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"std::vector error %s",ex.what());
    }
  }
  _initialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
PetscErrorCode CUPMDevice<T>::getDevice(PetscDevice &device) PETSC_NOEXCEPT
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = this->__initialize();CHKERRQ(ierr);
  // default device is always the first device for now?
  device->deviceId = this->_devices[0]->id();
  device->ops->createcontext = this->_create;
  PetscFunctionReturn(0);
}

template <CUPMDeviceKind T>
PetscErrorCode CUPMDevice<T>::configureDevice(PetscDevice &device) PETSC_NOEXCEPT
{
  cupmError_t cerr;

  PetscFunctionBegin;
  cerr = cupmSetDevice(device->deviceId);
  // why on EARTH nvidia insists on making otherwise informational states into
  // fully-fledged error codes is beyond me. Why couldn't a pointer to bool argument have
  // sufficed?!?!?!
  if (cerr != cupmErrorDeviceAlreadyInUse) CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

// explicitly instantiate the classes
#if PetscDefined(HAVE_CUDA)
template class CUPMDevice<CUPMDeviceKind::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template class CUPMDevice<CUPMDeviceKind::HIP>;
#endif

} // namespace Petsc
