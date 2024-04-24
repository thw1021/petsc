/*
   Private context for a Newton line search method for solving
   systems of nonlinear equations
 */

#pragma once
#include <petsc/private/snesimpl.h>
#include <petscsnes.h>

typedef struct {
  PetscInt                   max_steps;
  PetscReal                  step_size;
  PetscReal                  min_step_size;
  PetscReal                  psisq;
  PetscReal                  lambda_update;
  PetscReal                  lambda;
  PetscReal                  lambda_max;
  PetscReal                  lambda_min;
  PetscBool                  scale_rhs;
  SNESNewtonALCorrectionType correction_type;

  Vec vec_rhs_orig;
} SNES_NEWTONAL;

PETSC_INTERN const char NewtonALExactCitation[];
PETSC_INTERN PetscBool  NewtonALExactCitationSet;
PETSC_INTERN const char NewtonALNormalCitation[];
PETSC_INTERN PetscBool  NewtonALNormalCitationSet;
