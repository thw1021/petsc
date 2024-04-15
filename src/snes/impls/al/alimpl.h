/*
   Private context for a Newton line search method for solving
   systems of nonlinear equations
 */

#pragma once
#include <petsc/private/snesimpl.h>
#include <petscsnes.h>

typedef struct {
  PetscInt  max_steps;
  PetscReal step_size;
  PetscReal min_step_size;
  PetscReal psisq;
  PetscReal delta_s;
  PetscReal lambda_update;
  PetscReal lambda;
} SNES_NEWTONAL;
