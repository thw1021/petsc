#include "contextcupm.hpp"

// This file serves simply to store the definitions of all the static variables etc.
namespace Petsc {

#if PetscDefined(HAVE_CUDA)
const decltype(cudaSuccess)           cupmTypeTraits<PetscDeviceContextBackend::CUDA>::cupmSuccess;
const decltype(cudaErrorNotReady)     cupmTypeTraits<PetscDeviceContextBackend::CUDA>::cupmErrorNotReady;
const decltype(cudaStreamNonBlocking) cupmTypeTraits<PetscDeviceContextBackend::CUDA>::cupmStreamNonBlocking;
#endif

#if PetscDefined(HAVE_HIP)
const decltype(hipSuccess)           cupmTypeTraits<PetscDeviceContextBackend::HIP>::cupmSuccess;
const decltype(hipErrorNotReady)     cupmTypeTraits<PetscDeviceContextBackend::HIP>::cupmErrorNotReady;
const decltype(hipStreamNonBlocking) cupmTypeTraits<PetscDeviceContextBackend::HIP>::cupmStreamNonBlocking;
#endif

} // namespace Petsc
