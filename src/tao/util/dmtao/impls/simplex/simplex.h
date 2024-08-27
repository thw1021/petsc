#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  PetscReal  size;
  PetscReal  tol;
  VecScatter vscat;
  Vec        yseq;
} DMTao_Simplex;
