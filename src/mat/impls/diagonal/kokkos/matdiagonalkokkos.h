#pragma once

#include <petscmat.h>

PETSC_INTERN PetscErrorCode MatADot_Diagonal_Seq_Kokkos_Private(Mat, Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode MatANormSq_Diagonal_Seq_Kokkos_Private(Mat, Vec, PetscReal *);
