#include <petscdevice.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

typedef struct {
  PetscReal dt; /* Per thread \delta t */
  PetscInt elements; /* Dimension of the problem */
  PetscInt i; /* Index of the initializing thread */
  PetscBool converged;
} TSDevice_Euler;

__device__ void TSDeviceStep_Euler(){
    return;
}
