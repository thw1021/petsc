#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  TaoTermMapping f_term, g_term; /* smooth term f and proximal term g of the objective sum */
  Vec            f_param, g_param;
  Vec            workvec, workvec2, dualvec;
  Vec            x_old;    /* base point of the forward step: x_k, or the extrapolated point y_k with acceleration */
  Vec            grad_old; /* gradient of f at the base point */
  Vec            x_accel;  /* previous iterate x_k with acceleration */

  PetscReal xi; /* factor by which the step grows before each backtracking line search */
  PetscReal t_fista, t_fista_old, fista_beta;
  PetscReal lip;          /* Lipschitz constant of the gradient of f */
  PetscReal initial_step; /* user-supplied initial step size; 0 means 1/lip */
  PetscReal step_old;     /* previous step size, used by the adaptive rule */

  PetscBool use_accel;
  PetscBool use_adapt;
} TAO_FB;
