#pragma once

#include <../src/ksp/ksp/utils/lmvm/lmvm.h>
#include <../src/ksp/ksp/utils/lmvm/rescale/symbrdnrescale.h>

/*
  Limited-memory Symmetric Broyden method for approximating both
  the forward product and inverse application of a Jacobian.
*/

typedef enum {
  SYMBROYDEN_BASIS_BKS = 0,
  SYMBROYDEN_BASIS_HKY = 1,
  SYMBROYDEN_BASIS_COUNT
} SymBroydenBasisType;

typedef enum {
  SYMBROYDEN_PRODUCTS_PHI   = 0,
  SYMBROYDEN_PRODUCTS_PSI   = 1,
  SYMBROYDEN_PRODUCTS_STBKS = 2,
  SYMBROYDEN_PRODUCTS_YTHKY = 3,
  SYMBROYDEN_PRODUCTS_M00   = 4,
  SYMBROYDEN_PRODUCTS_N00   = 5,
  SYMBROYDEN_PRODUCTS_M01   = 6,
  SYMBROYDEN_PRODUCTS_N01   = 7,
  SYMBROYDEN_PRODUCTS_M11   = 8,
  SYMBROYDEN_PRODUCTS_N11   = 9,
  SYMBROYDEN_PRODUCTS_COUNT
} SymBroydenProductsType;

typedef struct {
  PetscReal                  phi_scalar, psi_scalar;
  MatLMVMSymBroydenScaleType scale_type;
  PetscInt                   watchdog, max_seq_rejects; /* tracker to reset after a certain # of consecutive rejects */
  SymBroydenRescale          rescale;                   /* context for diagonal or scalar rescaling */
  LMBasis                    basis[SYMBROYDEN_BASIS_COUNT];
  LMProducts                 products[SYMBROYDEN_PRODUCTS_COUNT];
  Vec                        StFprev, YtH0Fprev;
} Mat_SymBrdn;

PETSC_INTERN PetscErrorCode SymBroydenKernel_Recursive(Mat, MatLMVMMode, Vec, Vec, PetscBool);
PETSC_INTERN PetscErrorCode SymBroydenKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec, PetscBool);

PETSC_INTERN PetscErrorCode DFPKernel_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode DFPKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode DFPKernel_Dense(Mat, MatLMVMMode, Vec, Vec);

PETSC_INTERN PetscErrorCode BFGSKernel_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BFGSKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec);

PETSC_INTERN PetscErrorCode SymBroydenCompactDenseKernelUseB0S(Mat, MatLMVMMode, Vec, PetscBool *);
