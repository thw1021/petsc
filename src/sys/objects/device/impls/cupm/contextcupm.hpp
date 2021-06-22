#if !defined(PETSCDEVICECONTEXTCUPM_HPP)
#define PETSCDEVICECONTEXTCUPM_HPP

#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
typedef struct {
  cudaStream_t       stream;
  cudaEvent_t        event;
  cublasHandle_t     blas;
  cusolverDnHandle_t solver;
} PetscDeviceContext_CUDA;

/* Silence undefined identifier errors for the op structs */
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate_CUPM(PetscDeviceContext);
#endif /* HAVE_CUDA */
#endif /* PETSCDEVICECONTEXTCUDA_HPP */
