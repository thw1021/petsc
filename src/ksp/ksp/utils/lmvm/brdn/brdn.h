#pragma once

#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Limited-memory Broyden's method for approximating the inverse of
  a Jacobian.
*/

typedef enum {
  BROYDEN_BASIS_Y_MINUS_BKS = 0,
  BROYDEN_BASIS_S_MINUS_HKY = 1,
  BROYDEN_BASIS_COUNT
} BroydenBasisType;

typedef enum {
  BROYDEN_PRODUCTS_STHKY           = 0,
  BROYDEN_PRODUCTS_YTBKS           = 1,
  BROYDEN_PRODUCTS_STH0Y_MINUS_STS = 2,
  BROYDEN_PRODUCTS_YTB0S_MINUS_YTY = 3,
  BROYDEN_PRODUCTS_COUNT
} BroydenProductsType;

typedef struct {
  LMBasis    basis[BROYDEN_BASIS_COUNT];
  LMProducts products[BROYDEN_PRODUCTS_COUNT];
  Vec        YtFprev;
} Mat_Brdn;

PETSC_INTERN PetscErrorCode BroydenKernel_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernel_Dense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_Dense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernel_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernel_Dense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_Dense(Mat, MatLMVMMode, Vec, Vec);
