#if !defined(PETSCDEVICECONTEXTHIP_HIP_HPP)
#define PETSCDEVICECONTEXTHIP_HIP_HPP

#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

#if PetscDefined(HAVE_HIP)
typedef struct {
  hipStream_t  stream;
  hipEvent_t   event;
} PetscDeviceContext_HIP;

/* Silence undefined identifier errors for the op structs */
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext);
#endif /* HAVE_HIP */

#endif /* PETSCDEVICECONTEXTHIP_HIP_HPP */
