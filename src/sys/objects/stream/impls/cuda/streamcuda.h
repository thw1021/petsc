#if !defined(__STREAMCUDA_H)
#define __STREAMCUDA_H

#include <petsc/private/deviceimpl.h>

#if PetscDefined(HAVE_CUDA)
typedef struct {
  cudaStream_t cstream;
} PetscStream_CUDA;

typedef struct {
  cudaEvent_t  cevent;
} PetscEvent_CUDA;

PETSC_INTERN PetscErrorCode PetscStreamScalarAccumOpDispatch_Internal(PetscStreamScalar,PetscInt,PetscStreamScalar[],PetscStreamComputeOp,PetscStreamComputeOp,PetscStream);
#endif
#endif
