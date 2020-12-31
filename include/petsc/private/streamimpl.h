#if !defined(STREAMIMPL_H)
#define STREAMIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscstream.h>

struct _n_PetscStream {
  PetscBool    setup;
#if defined(PETSC_HAVE_CUDA)
  cudaStream_t cstream;
#if defined(PETSC_USE_DEBUG)
  PetscBool    gotCUDA;
#endif
#endif
#if defined(PETSC_HAVE_HIP)
  hipStream_t  hstream;
#if defined(PETSC_USE_DEBUG)
  PetscBool    gotHIP;
#endif
#endif
  PetscStreamMode mode;
};

#endif
