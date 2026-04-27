#pragma once

#include <petsc/private/taoimpl.h> /*I "petsctaoterm.h" I*/

typedef struct _n_TaoTermHessianShell TaoTermHessianShell;

struct _n_TaoTermHessianShell {
  TaoTerm          term;
  Vec              x;
  Vec              params;
  PetscObjectState x_state;
  PetscObjectState params_state;
  PetscBool        x_state_change_warning;
  PetscBool        params_state_change_warning;
};

PETSC_INTERN PetscErrorCode TaoTermHessianShellCheck(TaoTermHessianShell *, PetscBool, PetscBool);
