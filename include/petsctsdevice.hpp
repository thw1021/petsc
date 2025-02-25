#pragma once

typedef enum {
  TSDEVICE_EULER = 0,
  TSDEVICE_RK4
} TSDeviceType;

template<class func>
PetscErrorCode TSSolve_Device(TS);

/* Define structs for various types for allocation on device memory in TSSetUp_Device */
typedef struct {
  PetscReal dt; /* Per thread \delta t */
  PetscReal time;
  PetscInt elements; /* Dimension of the problem */
  PetscInt i; /* Index of the initializing thread */
  PetscBool converged;
  PetscInt maxSteps;
  PetscInt step;
} TSDevice_Euler;
