#include <petsc/private/cupminterface.hpp>

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc
{
// do all of this with macros to enforce that both CUDA and HIP implementations both have
// things defined. If you for example implement something on the HIP side but forget to
// implement it on the CUDA side you'll get an error.

#if PetscDefined(HAVE_CUDA)
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_CUDA(stem)            \
  const decltype(cuda##stem) CUPMInterface<CUPMDeviceKind::CUDA>::cupm##stem
#else
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_CUDA(stem)
#endif

#if PetscDefined(HAVE_HIP)
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_HIP(stem)             \
  const decltype(hip##stem) CUPMInterface<CUPMDeviceKind::HIP>::cupm##stem
#else
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_HIP(stem)
#endif

#define PETSC_CUPM_DEFINE_STATIC_VARIABLE(stem)                         \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_CUDA(stem);                 \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_HIP(stem)

PETSC_CUPM_DEFINE_STATIC_VARIABLE(Success);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(ErrorNotReady);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(StreamNonBlocking);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_CUDA(ErrorDeviceAlreadyInUse);
#if PetscDefined(HAVE_HIP)
// not conforming, see declaration in cupminterface.hpp
const decltype(hipSuccess) CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorDeviceAlreadyInUse;
#endif
PETSC_CUPM_DEFINE_STATIC_VARIABLE(ErrorSetOnActiveProcess);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(DeviceMapHost);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(MemcpyHostToDevice);

} // namespace Petsc
