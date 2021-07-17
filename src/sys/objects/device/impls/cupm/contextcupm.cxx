#include "contextcupm.hpp"

// This file serves simply to store the definitions of all the static variables etc.
namespace Petsc {

#if PetscDefined(HAVE_CUDA)
const decltype(cudaSuccess)           CUPMTypeTraits<CUPMDeviceKind::CUDA>::cupmSuccess;
const decltype(cudaErrorNotReady)     CUPMTypeTraits<CUPMDeviceKind::CUDA>::cupmErrorNotReady;
const decltype(cudaStreamNonBlocking) CUPMTypeTraits<CUPMDeviceKind::CUDA>::cupmStreamNonBlocking;
#endif

#if PetscDefined(HAVE_HIP)
const decltype(hipSuccess)           CUPMTypeTraits<CUPMDeviceKind::HIP>::cupmSuccess;
const decltype(hipErrorNotReady)     CUPMTypeTraits<CUPMDeviceKind::HIP>::cupmErrorNotReady;
const decltype(hipStreamNonBlocking) CUPMTypeTraits<CUPMDeviceKind::HIP>::cupmStreamNonBlocking;
#endif

} // namespace Petsc
