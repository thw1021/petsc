#ifndef PETSCDEVICE_CUPM_H
#define PETSCDEVICE_CUPM_H

#include <petscmacros.h>
#include <petscdevicetypes.h>
#include <petscdevice_cuda.h>
#include <petscdevice_hip.h>

#if PetscDefined(USING_HCC) && PetscDefined(USING_NVCC)
#  error using both nvcc and hipcc at the same time?
#endif

#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
#  define PETSC_HAVE_CUPM 1
#endif

#if PetscDefined(HAVE_CUPM) && (PetscDefined(USING_NVCC) || PetscDefined(USING_HCC))
#  define PETSC_HOST_DECL      __host__
#  define PETSC_DEVICE_DECL    __device__
#  define PETSC_KERNEL_DECL    __global__
#  define PETSC_SHAREDMEM_DECL __shared__
#  define PETSC_FORCEINLINE    __forceinline__
#  define PETSC_CONSTMEM_DECL  __constant__
#else
#  define PETSC_HOST_DECL
#  define PETSC_DEVICE_DECL
#  define PETSC_KERNEL_DECL
#  define PETSC_SHAREDMEM_DECL
#  define PETSC_FORCEINLINE    inline
#  define PETSC_CONSTMEM_DECL
#endif

#define PETSC_HOSTDEVICE_DECL        PETSC_HOST_DECL PETSC_DEVICE_DECL
#define PETSC_HOSTDEVICE_INLINE_DECL PETSC_HOSTDEVICE_DECL PETSC_FORCEINLINE
#endif // PETSCDEVICE_CUPM_H
