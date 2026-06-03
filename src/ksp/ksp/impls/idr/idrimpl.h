/*
   Private data structure for the IDR(s) method.
*/
#pragma once

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscInt     s;     /* shadow space dimension; default 4 */
  Vec         *GG;    /* s direction vectors G[0..s-1] */
  Vec         *UU;    /* s update vectors    U[0..s-1] */
  Vec         *PP;    /* s shadow vectors    P[0..s-1] (fixed, random orthonormal) */
  Vec          r;     /* current residual */
  Vec          v;     /* work: K^{-1} current direction */
  Vec          t;     /* work: A v (mat-vec result) */
  PetscScalar *M;     /* s*s matrix M[j,k] = <G[k],P[j]>, column-major */
  PetscScalar *f;     /* length s: P^T r */
  PetscScalar *c;     /* length s: solution of M c = f */
  PetscReal    omega; /* current relaxation parameter */
} KSP_IDR;

PETSC_INTERN PetscErrorCode KSPSetUp_IDR(KSP);
PETSC_INTERN PetscErrorCode KSPSolve_IDR(KSP);
PETSC_INTERN PetscErrorCode KSPReset_IDR(KSP);
PETSC_INTERN PetscErrorCode KSPDestroy_IDR(KSP);
PETSC_INTERN PetscErrorCode KSPView_IDR(KSP, PetscViewer);
PETSC_INTERN PetscErrorCode KSPSetFromOptions_IDR(KSP, PetscOptionItems);
