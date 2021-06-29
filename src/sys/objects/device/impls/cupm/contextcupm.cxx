#include "contextcupm.hpp"

// This file serves simply to store the definitions of all the static variables etc.
PETSC_INTERN const char *const PetscDeviceContextBackends[] = {"cuda","hip","PetscDeviceContextBackend","PetscDeviceContextBackend::",NULL};

namespace Petsc {

#if PetscDefined(HAVE_CUDA)
const decltype(cudaStreamNonBlocking) cupmTypeTraits<PetscDeviceContextBackend::CUDA>::cupmStreamNonBlocking;
#endif

#if PetscDefined(HAVE_HIP)
const decltype(hipStreamNonBlocking) cupmTypeTraits<PetscDeviceContextBackend::HIP>::cupmStreamNonBlocking;
#endif

} // namespace Petsc
