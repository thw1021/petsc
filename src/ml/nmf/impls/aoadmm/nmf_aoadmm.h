#pragma once
#include <petsc/private/nmfimpl.h>

typedef struct {
  Vec workvec;
  Mat workmat;
} PetscNMF_AOADMM;
