#if !defined(STREAMIMPL_H)
#define STREAMIMPL_H

#include <petscstream.h>
#if defined(PETSC_HAVE_CUDA)
#include <cuda.h>
#endif
#if defined(PETSC_HAVE_HIP)
#include <hip/hip_runtime.h>
#endif

struct _p_PetscStream {
#if defined(PETSC_HAVE_CUDA)
  cudaStream_t cstream;
#endif
#if defined(PETSC_HAVE_HIP)
  hipStream_t hstream;
#endif
};

#endif
