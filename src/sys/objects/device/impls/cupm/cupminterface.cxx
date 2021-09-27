#include <petsc/private/cupminterface.hpp> /* I "petscdevice.h" */

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc {
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE(PREFIX,prefix,stem)   \
  const decltype(prefix##stem) CUPMInterface<CUPMDeviceKind::##PREFIX>::cupm##stem

#if PetscDefined(HAVE_CUDA)
PETSC_CUPM_DEFINE_STATIC_VARIABLE(CUDA,cuda,Success);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(CUDA,cuda,ErrorNotReady);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(CUDA,cuda,StreamNonBlocking);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(CUDA,cuda,ErrorDeviceAlreadyInUse);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(CUDA,cuda,ErrorSetOnActiveProcess);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(CUDA,cuda,DeviceMapHost);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(CUDA,cuda,MemcpyHostToDevice);
#endif // PetscDefined(HAVE_CUDA)

#if PetscDefined(HAVE_HIP)
PETSC_CUPM_DEFINE_STATIC_VARIABLE(HIP,hip,Success);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(HIP,hip,ErrorNotReady);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(HIP,hip,StreamNonBlocking);
// not conforming, see declaration in cupminterface.hpp
const decltype(hipSuccess) CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorDeviceAlreadyInUse;
PETSC_CUPM_DEFINE_STATIC_VARIABLE(HIP,hip,ErrorSetOnActiveProcess);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(HIP,hip,DeviceMapHost);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(HIP,hip,MemcpyHostToDevice);
#endif // PetscDefined(HAVE_HIP)

} // namespace Petsc
