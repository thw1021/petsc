#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  PetscReal lb_real, ub_real;

  Vec lb_vec, ub_vec;
} DMTao_Box;
