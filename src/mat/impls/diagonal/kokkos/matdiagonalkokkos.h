#pragma once

#include <petscmat.h>

PETSC_INTERN PetscErrorCode MatDiagonalADot_Seq_Kokkos_Private(Mat, Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode MatDiagonalANormSq_Seq_Kokkos_Private(Mat, Vec, PetscReal *);
