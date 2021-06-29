#include "contextcupm.hpp"

// This file serves simply to store the definitions of all the static variables etc.
#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
PETSC_INTERN const char *const PetscDeviceContextBackends[NUM_BACKENDS] = {"cuda","hip","PetscDeviceContextBackend","PetscDeviceContextBackend::",nullptr};
#else
PETSC_INTERN const char *const PetscDeviceContextBackends[NUM_BACKENDS] = {"cuda","hip","PetscDeviceContextBackend","PetscDeviceContextBackend::",NULL};
#endif

namespace Petsc {

#if PetscDefined(HAVE_CUDA)
const decltype(cudaStreamNonBlocking) cupmTypeTraits<PetscDeviceContextBackend::CUDA>::cupmStreamNonBlocking;
#endif

#if PetscDefined(HAVE_HIP)
const decltype(hipStreamNonBlocking) cupmTypeTraits<PetscDeviceContextBackend::HIP>::cupmStreamNonBlocking;
#endif

} // namespace Petsc
