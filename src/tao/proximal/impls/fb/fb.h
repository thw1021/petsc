#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  DM reg; /*dm1 term for DMTaoApplyProximalMap */
  DM smoothterm;
  DM proxterm;

  Vec workvec, workvec2, dualvec, x_old, grad_old;

  PetscReal xi; /* backtracking constant */
  PetscReal t_fista, t_fista_old, fista_beta;
  PetscReal lip; /* L_f */

  PetscReal step_old;
  PetscReal f_scale;
  PetscReal prox_scale;

  PetscBool use_accel;
  PetscBool use_adapt;

  PetscBool lip_set;
} TAO_FB;
