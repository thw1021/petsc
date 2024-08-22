#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  void *data;
  PetscErrorCode (*applyproximalmap)(DMTao, DMTao, PetscReal, Vec, Vec, PetscBool);
} DMTao_Shell;
