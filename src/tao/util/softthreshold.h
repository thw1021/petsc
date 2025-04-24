#pragma once

#include <petsctao.h>

PETSC_INTERN PetscErrorCode TaoIsSoftThreshold(Tao, PetscBool *);
PETSC_INTERN PetscErrorCode TaoSolve_SoftThreshold(Tao);
