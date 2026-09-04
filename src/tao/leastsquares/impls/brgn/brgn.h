#pragma once

#include <petsc/private/taoimpl.h>
#include <petsctaoterm.h>

typedef enum {
  TAOBRGN_PRESET_L2PROX,
  TAOBRGN_PRESET_L2PURE,
  TAOBRGN_PRESET_L1DICT
} TaoBRGNPreset;

typedef struct {
  Tao           subsolver, parent;
  Mat           H;
  Vec           damping, hessian_x, r_work;
  PetscReal     preset_weight;
  PetscReal     preset_l1_epsilon;
  PetscReal     lm_lambda;
  TaoBRGNPreset preset;
  PetscBool     preset_set;
  PetscBool     use_lm;
  PetscBool     matrix_free;
} TAO_BRGN;
