#if !defined(__STREAMHIP_H)
#define __STREAMHIP_H

#include <petsc/private/deviceimpl.h>

#if PetscDefined(HAVE_HIP)
typedef struct {
  hipStream_t hstream;
} PetscStream_HIP;

typedef struct {
  hipEvent_t hevent;
} PetscEvent_HIP;
#endif
#endif
