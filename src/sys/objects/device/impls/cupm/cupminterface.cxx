#include <petsc/private/cupminterface.hpp>

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc
{
// do all of this with macros to enforce that both CUDA and HIP implementations both have
// things defined. If you for example implement something on the HIP side but forget to
// implement it on the CUDA side you'll get an error.

// need these for the indirection when building the if_0 and if_1 variants of the macro
#define CAT_(x,...) x ## __VA_ARGS__
#define CAT(x,...) CAT_(x,__VA_ARGS__)

#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_0(PREFIX,prefix,stem)
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_1(PREFIX,prefix,stem) \
  const decltype(prefix ## stem) CUPMInterface<CUPMDeviceType::PREFIX>::cupm ## stem

#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE(PREFIX,prefix,stem)   \
  CAT(PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE_,PetscDefined(HAVE_ ## PREFIX))(PREFIX,prefix,stem)


#define PETSC_CUPM_DEFINE_STATIC_VARIABLE(stem)                         \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE(CUDA,cuda,stem);            \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE(HIP,hip,stem)

PETSC_CUPM_DEFINE_STATIC_VARIABLE(Success);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(ErrorNotReady);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(StreamNonBlocking);
PETSC_CUPM_DEFINE_STATIC_VARIABLE_IF_HAVE(CUDA,cuda,ErrorDeviceAlreadyInUse);
#if PetscDefined(HAVE_HIP)
// not conforming, see declaration in cupminterface.hpp
const decltype(hipSuccess) CUPMInterface<CUPMDeviceKind::HIP>::cupmErrorDeviceAlreadyInUse;
#endif
PETSC_CUPM_DEFINE_STATIC_VARIABLE(ErrorSetOnActiveProcess);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(DeviceMapHost);
PETSC_CUPM_DEFINE_STATIC_VARIABLE(MemcpyHostToDevice);

} // namespace Petsc
