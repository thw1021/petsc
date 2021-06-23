#if !defined(PETSCDEVICECONTEXTCUPM_HPP)
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

/* Silence undefined identifier errors for the op structs */
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_CUDAM(PetscDeviceContext);
#endif /* PETSCDEVICECONTEXTCUDA_HPP */
