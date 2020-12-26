#if !defined(STREAMIMPL_H)
#define STREAMIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscstream.h>

struct _p_PetscStream {
#if defined(PETSC_HAVE_CUDA)
  cudaStream_t     cstream;
#endif
#if defined(PETSC_HAVE_HIP)
  hipStream_t      hstream;
#endif
  PetscStreamMode  mode;
};

#endif
