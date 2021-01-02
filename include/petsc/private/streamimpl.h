#if !defined(STREAMIMPL_H)
#define STREAMIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscstream.h>

struct _n_PetscStream {
#if defined(PETSC_HAVE_CUDA)
  cudaStream_t cstream;
  cudaEvent_t  cevent;
#if defined(PETSC_USE_DEBUG)
  PetscBool    gotCUDA;
#endif /* PETSC_USE_DEBUG */
#endif /* PETSC_HAVE_CUDA */
#if defined(PETSC_HAVE_HIP)
  hipStream_t  hstream;
  hipEvent_t   hevent;
#if defined(PETSC_USE_DEBUG)
  PetscBool    gotHIP;
#endif /* PETSC_USE_DEBUG */
#endif /* PETSC_HAVE_HIP */
  PetscStreamMode mode;
};

#endif
