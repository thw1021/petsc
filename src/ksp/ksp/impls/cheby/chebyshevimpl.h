/*
    Private data structure for Chebyshev Iteration
*/

#ifndef PETSC_CHEBYSHEVIMPL_H
#define PETSC_CHEBYSHEVIMPL_H

#include <petsc/private/kspimpl.h>

typedef enum {
  CHEBYSHEV_FIRST,
  CHEBYSHEV_FOURTH,
  CHEBYSHEV_OPT_FOURTH
} ChebyshevKind;
static const char *const ChebyshevKinds[] = {"FIRST", "FOURTH", "OPT_FOURTH", "ChebyshevKindTypes", "CHEBYSHEV_", 0};

typedef struct {
  PetscReal emin, emax;                   /* store user provided estimates of extreme eigenvalues or computed with kspest and transformed with tform[] */
  PetscReal emin_computed, emax_computed; /* eigenvalues as computed by kspest, if computed */
  PetscReal emin_provided, emax_provided; /* provided by PCGAMG; discarded unless preconditioned by Jacobi */
  PetscReal *betas;                        /* store beta coefficients for 4th-kind Chebyshev smoother */

  KSP       kspest;                       /* KSP used to estimate eigenvalues */
  PetscReal tform[4];                     /* transform from Krylov estimates to Chebyshev bounds */
  PetscInt  eststeps;                     /* number of kspest steps in KSP used to estimate eigenvalues */
  PetscBool usenoisy;                     /* use noisy right hand side vector to estimate eigenvalues */
  ChebyshevKind chebykind;                    /* Chebyshev polynomial kind (must be 1 or 4)*/
  /* For tracking when to update the eigenvalue estimates */
  PetscObjectId    amatid, pmatid;
  PetscObjectState amatstate, pmatstate;
} KSP_Chebyshev;

/* given the polynomial order, return tabulated beta coefficients for use in opt. 4th-kind Chebyshev smoother */
PETSC_INTERN PetscErrorCode KSPChebyshevGetBetas_Private(KSP ksp);

#endif // PETSC_CHEBYSHEVIMPL_H
