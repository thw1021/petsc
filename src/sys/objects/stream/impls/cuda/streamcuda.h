#if !defined(PETSCSTREAMCUDA_H)
#define PETSCSTREAMCUDA_H

#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

#if PetscDefined(HAVE_CUDA)
typedef struct {
  cudaStream_t cstream;
} PetscStream_CUDA;

typedef struct {
  cudaEvent_t  cevent;
} PetscEvent_CUDA;

typedef struct {
  cudaGraph_t     cgraph;
  cudaGraphExec_t cexec;
} PetscStreamGraph_CUDA;

/* Silence undefined identifier errors for the op structs */
PETSC_EXTERN PetscErrorCode PetscStreamCreate_CUDA(PetscStream);
PETSC_EXTERN PetscErrorCode PetscEventCreate_CUDA(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate_CUDA(PetscStreamScalar);
PETSC_EXTERN PetscErrorCode PetscStreamGraphCreate_CUDA(PetscStreamGraph);
#endif /* HAVE_CUDA */
#endif /* PETSCSTREAMCUDA_H */
