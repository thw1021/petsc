#pragma once

#include "petscda.h"
#include <petsc/private/petscimpl.h>

/* PetscDA object cookie */
PETSC_EXTERN PetscClassId PETSCDA_CLASSID;

/* Operator table for PetscDA implementations */
typedef struct _PetscDAOps *PetscDAOps;
struct _PetscDAOps {
  PetscErrorCode (*analysis)(PetscDA, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
  PetscErrorCode (*applymodel)(PetscDA, PetscErrorCode (*)(Vec, Vec, void *), void *);
  PetscErrorCode (*computemean)(PetscDA, Vec);
  PetscErrorCode (*computeanomalies)(PetscDA, Vec, Mat *);
  PetscErrorCode (*destroy)(PetscDA);
  PetscErrorCode (*view)(PetscDA, PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscDA, PetscOptionItems *);
};

/*
  Internal PetscDA structure following PETSc object conventions
*/
struct _p_PetscDA {
  PETSCHEADER(struct _PetscDAOps);

  /* Core PetscDA data */
  PetscInt  ensemble_size; /* Number of ensemble members (m) */
  PetscInt  state_size;    /* State vector dimension (n) */
  PetscInt  obs_size;      /* Observation vector dimension (p) */
  PetscInt  ndof;          /* Number of degrees of freedom per vertex */
  Mat       ensemble;      /* Ensemble matrix (n x m) */
  Vec       obs_error_var; /* Observation error variance (diagonal of R), length p */
  Mat       U;             /* Orthogonal transformation matrix (m x m) */
  PetscReal inflation;     /* Inflation factor */

  /* Algorithm state */
  PetscBool assembled; /* Is the PetscDA object assembled/ready */

  /* T-matrix factorization data (shared across implementations) */
  PetscDASqrtType sqrt_type;       /* Square root factorization type */
  Mat             V;               /* Eigen vectors (LAPACK column-major storage) */
  Mat             L_cholesky;      /* Lower triangular Cholesky factor */
  Vec             sqrt_eigen_vals; /* Square root of eigen values */
  Mat             I_StS;           /* T = I + S^T * S matrix */

  /* Implementation-specific data */
  void *data; /* For implementation-specific storage */
};

/* Internal utility functions shared across PetscDA implementations */
PETSC_INTERN PetscErrorCode PetscDACholeskySqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode PetscDAymmetricEigenSqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode VecSetRandomGaussian_Private(Vec, PetscRandom, PetscReal, PetscReal);
