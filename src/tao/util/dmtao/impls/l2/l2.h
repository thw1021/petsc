#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  Vec workvec2; /* Currently, only L2 needs second workvec, so its in its own context */
} DMTao_L2;
