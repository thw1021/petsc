#pragma once

#include <petscmat.h>

PETSC_INTERN PetscErrorCode VecTypeGetRootTypeForMatDense_Private(MPI_Comm, VecType, VecType *, PetscBool *, PetscBool *);
PETSC_INTERN PetscErrorCode MatCreate_DenseFromVecType_Private(Mat, PetscBool, PetscBool, PetscInt, PetscScalar *);
