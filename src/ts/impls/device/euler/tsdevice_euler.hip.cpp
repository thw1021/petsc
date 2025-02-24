typedef struct {
  PetscReal dt; /* Per thread \delta t */
  PetscInt dim; /* Problem dimension to compute on blocks per thread */
  PetscInt i; /* Index of the initializing thread */
  PetscBool converged;
} TSDevice_Euler;
