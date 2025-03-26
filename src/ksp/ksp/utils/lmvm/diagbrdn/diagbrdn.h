#pragma once

#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  "Full" memory implementation of only the diagonal terms in a symmetric Broyden approximation.
*/

typedef struct _SymBroydenRescale *SymBroydenRescale;

struct _SymBroydenRescale {
  PetscInt                   k;
  Vec                        invDnew, BFGS, DFP, U, V, W; /* work vectors for diagonal scaling */
  PetscReal                 *yty, *sts, *yts;             /* scalar arrays for recycling dot products */
  PetscReal                  theta, rho, alpha, beta;     /* convex combination factors for the scalar or diagonal scaling */
  PetscReal                  delta, delta_min, delta_max, sigma, tol;
  PetscInt                   sigma_hist; /* length of update history to be used for scaling */
  PetscBool                  allocated;
  PetscBool                  forward;
  MatLMVMSymBroydenScaleType scale_type;
};

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetDiagonalMode(SymBroydenRescale, PetscBool);
PETSC_INTERN PetscErrorCode SymBroydenRescaleGetType(SymBroydenRescale, MatLMVMSymBroydenScaleType *);
PETSC_INTERN PetscErrorCode SymBroydenRescaleSetType(Mat, SymBroydenRescale, MatLMVMSymBroydenScaleType);
PETSC_INTERN PetscErrorCode SymBroydenRescaleSetDelta(SymBroydenRescale, PetscReal);
PETSC_INTERN PetscErrorCode SymBroydenRescaleInitializeJ0(Mat, SymBroydenRescale);
PETSC_INTERN PetscErrorCode SymBroydenRescaleUpdateJ0(Mat, SymBroydenRescale);
PETSC_INTERN PetscErrorCode SymBroydenRescaleUpdate(Mat, SymBroydenRescale);
PETSC_INTERN PetscErrorCode SymBroydenRescaleCopy(SymBroydenRescale, SymBroydenRescale, PetscInt);
PETSC_INTERN PetscErrorCode SymBroydenRescaleView(SymBroydenRescale, PetscViewer);
PETSC_INTERN PetscErrorCode SymBroydenRescaleSetFromOptions(Mat, SymBroydenRescale, PetscOptionItems PetscOptionsObject);
PETSC_INTERN PetscErrorCode SymBroydenRescaleReset(Mat, SymBroydenRescale, PetscBool);
PETSC_INTERN PetscErrorCode SymBroydenRescaleDestroy(SymBroydenRescale *);
PETSC_INTERN PetscErrorCode SymBroydenRescaleAllocate(Mat, SymBroydenRescale);
PETSC_INTERN PetscErrorCode SymBroydenRescaleCreate(SymBroydenRescale *);
