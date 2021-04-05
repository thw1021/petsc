#if !defined(PETSCSTREAMHIP_H)
#define PETSCSTREAMHIP_H

#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

#if PetscDefined(HAVE_HIP)
typedef struct {
  hipStream_t hstream;
} PetscStream_HIP;

typedef struct {
  hipEvent_t hevent;
} PetscEvent_HIP;

PETSC_EXTERN PetscErrorCode PetscStreamCreate_HIP(PetscStream);
PETSC_EXTERN PetscErrorCode PetscEventCreate_HIP(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate_HIP(PetscStreamScalar);
#endif /* HAVE_HIP */
#endif /* PETSCSTREAMHIP_H */
