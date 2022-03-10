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
#  define PETSC_HOST_DECL       __host__
#  define PETSC_DEVICE_DECL     __device__
#  define PETSC_KERNEL_DECL     __global__
#  define PETSC_SHAREDMEM_DECL  __shared__
#  define PETSC_FORCEINLINE     __forceinline__
#else
#  define PETSC_HOST_DECL
#  define PETSC_DEVICE_DECL
#  define PETSC_KERNEL_DECL
#  define PETSC_SHAREDMEM_DECL
#  define PETSC_FORCEINLINE     inline
#endif

#define PETSC_HOSTDEVICE_DECL        PETSC_HOST_DECL PETSC_DEVICE_DECL
#define PETSC_HOSTDEVICE_INLINE_DECL PETSC_HOSTDEVICE_DECL PETSC_FORCEINLINE

#if PetscDefined(USING_NVCC)
#  define CUPM_CALLBACK_FN CUDART_CB
#else
#  define CUPM_CALLBACK_FN
#endif

static inline PetscErrorCode PetscGetMemType(const void *ptr, PetscMemType *type)
{
  PetscFunctionBegin;
  *type = PETSC_MEMTYPE_HOST;
  if (!ptr) PetscFunctionReturn(0);
#if PetscDefined(HAVE_CUDA)
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {
    cudaError_t                  cerr;
    struct cudaPointerAttributes attr;
    enum cudaMemoryType          mtype;
    cerr = cudaPointerGetAttributes(&attr,ptr); /* Do not check error since before CUDA 11.0, passing a host pointer returns cudaErrorInvalidValue */
    if (cerr) cerr = cudaGetLastError(); /* If there was an error, return it and then reset it */
    #if (CUDART_VERSION < 10000)
      mtype = attr.memoryType;
    #else
      mtype = attr.type;
    #endif
    if (cerr == cudaSuccess && mtype == cudaMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
    PetscFunctionReturn(0);
  }
#endif

#if PetscDefined(HAVE_HIP)
  if (PetscDeviceInitialized(PETSC_DEVICE_HIP)) {
    hipError_t                   cerr;
    struct hipPointerAttribute_t attr;
    enum hipMemoryType           mtype;
    cerr = hipPointerGetAttributes(&attr,ptr);
    if (cerr) cerr = hipGetLastError();
    mtype = attr.memoryType;
    if (cerr == hipSuccess && mtype == hipMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
  }
#endif
  PetscFunctionReturn(0);
}
#endif // PETSCDEVICE_CUPM_H
