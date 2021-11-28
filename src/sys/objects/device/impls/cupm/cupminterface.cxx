#include <petsc/private/cupminterface.hpp>

// This file serves simply to store the definitions of all the static variables that we
// DON'T have access to. Ones defined in PETSc-defined enum classes don't seem to have to
// need this declaration...

namespace Petsc
{

namespace Impl
{

#define PETSC_CUPM_STATIC_VARIABLE_DEFN(theirs,DEVICE,ours) \
  const decltype(theirs) CUPMInterface<CUPMDeviceType::DEVICE>::ours;

// in case either one or the other don't agree on a name, you can specify all three here:
//
// PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(cudaSuccess, hipAllGood, cupmSuccess) ->
// const decltype(cudaSuccess) CUPMInterface<CUPMDeviceType::CUDA>::cupmSuccess;
// const decltype(hipAllGood)  CUPMInterface<CUPMDeviceType::HIP>::cupmSuccess;
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(cuoriginal,hiporiginal,ours) \
  PETSC_IF_PETSC_DEFINED(HAVE_CUDA,PETSC_CUPM_STATIC_VARIABLE_DEFN,PETSC_EXPAND_TO_NOTHING)(cuoriginal,CUDA,ours) \
  PETSC_IF_PETSC_DEFINED(HAVE_HIP,PETSC_CUPM_STATIC_VARIABLE_DEFN,PETSC_EXPAND_TO_NOTHING)(hiporiginal,HIP,ours)

// if both cuda and hip agree on the same naming scheme i.e. cudaSuccess and hipSuccess:
//
// PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(Success) ->
// const decltype(cudaSuccess) CUPMInterface<CUPMDeviceType::CUDA>::cupmSuccess;
// const decltype(hipSuccess)  CUPMInterface<CUPMDeviceType::HIP>::cupmSuccess;
#define PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(suffix)         \
  PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(PETSC_CONCAT(cuda,suffix),PETSC_CONCAT(hip,suffix),PETSC_CONCAT(cupm,suffix))

// error codes
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(Success)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(ErrorNotReady)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(ErrorSetOnActiveProcess)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(ErrorNoDevice)

// hip not conforming, see declaration in cupminterface.hpp
PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(cudaErrorDeviceAlreadyInUse,hipErrorContextAlreadyInUse,cupmErrorDeviceAlreadyInUse)

// hip not conforming, and cuda faffs around with versions see declaration in cupminterface.hpp
#if PetscDefined(HAVE_CUDA)
#  if PETSC_PKG_CUDA_VERSION_GE(11,1,0)
#    define PetscCudaErrorStubLibrary ErrorStubLibrary
#  endif
#endif

#ifndef PetscCudaErrorStubLibrary
#define PetscCudaErrorStubLibrary ErrorInsufficientDriver
#endif

PETSC_CUPM_DEFINE_STATIC_VARIABLE_EXACT(PETSC_CONCAT(cuda,PetscCudaErrorStubLibrary),hipErrorInsufficientDriver,cupmErrorStubLibrary)

// enums
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(StreamNonBlocking)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(DeviceMapHost)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(MemcpyHostToDevice)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(MemcpyDeviceToHost)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(MemcpyDeviceToDevice)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(MemcpyHostToHost)
PETSC_CUPM_DEFINE_STATIC_VARIABLE_MATCHING_SCHEME(MemcpyDefault)

namespace detail
{

// instantiate the base class
template struct CUPMInterfaceBase<CUPMDeviceType::CUDA>;
template struct CUPMInterfaceBase<CUPMDeviceType::HIP>;

} // namespace detail

// instantiate the interfaces
#if PetscDefined(HAVE_CUDA)
template struct CUPMInterface<CUPMDeviceType::CUDA>;
#endif
#if PetscDefined(HAVE_HIP)
template struct CUPMInterface<CUPMDeviceType::HIP>;
#endif

} // namespace Impl

} // namespace Petsc
