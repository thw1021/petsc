#include <petsc/private/cupminterface.hpp> /* I "petscdevice.h" */

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc {

#if PetscDefined(HAVE_CUDA)
PETSC_CONSTEXPR const decltype(cudaSuccess)                 CUPMInterface<CUPMDeviceKind::CUDA>::cupmSuccess;
PETSC_CONSTEXPR const decltype(cudaErrorNotReady)           CUPMInterface<CUPMDeviceKind::CUDA>::cupmErrorNotReady;
PETSC_CONSTEXPR const decltype(cudaStreamNonBlocking)       CUPMInterface<CUPMDeviceKind::CUDA>::cupmStreamNonBlocking;
PETSC_CONSTEXPR const decltype(cudaErrorDeviceAlreadyInUse) CUPMInterface<CUPMDeviceKind::CUDA>::cupmErrorDeviceAlreadyInUse;
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
PETSC_CONSTEXPR const decltype(hipSuccess)            CUPMInterface<CUPMDeviceKind::HIP>::cupmSuccess;
PETSC_CONSTEXPR const decltype(hipErrorNotReady)      CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorNotReady;
PETSC_CONSTEXPR const decltype(hipStreamNonBlocking)  CUPMInterface<CUPMDeviceKind::HIP>::cupmStreamNonBlocking;
PETSC_CONSTEXPR const decltype(hipSuccess)            CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorDeviceAlreadyInUse;
#endif // PetscDefined(HAVE_HIP)

} // namespace Petsc
