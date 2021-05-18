#include <petsc/private/cupminterface.hpp> /* I "petscdevice.h" */

/* This file serves simply to store the definitions of all the static variables etc. */

namespace Petsc {

#if PetscDefined(HAVE_CUDA)
const decltype(cudaSuccess)           CUPMInterface<CUPMDeviceKind::CUDA>::cupmSuccess;
const decltype(cudaErrorNotReady)     CUPMInterface<CUPMDeviceKind::CUDA>::cupmErrorNotReady;
const decltype(cudaStreamNonBlocking) CUPMInterface<CUPMDeviceKind::CUDA>::cupmStreamNonBlocking;
#endif

#if PetscDefined(HAVE_HIP)
const decltype(hipSuccess)           CUPMInterface<CUPMDeviceKind::HIP>::cupmSuccess;
const decltype(hipErrorNotReady)     CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorNotReady;
const decltype(hipStreamNonBlocking) CUPMInterface<CUPMDeviceKind::HIP>::cupmStreamNonBlocking;
#endif

} /* namespace Petsc */
