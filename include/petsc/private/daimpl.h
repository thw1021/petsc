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
  PetscErrorCode (*computeanomalies)(PetscDA, Mat *);
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
  PetscInt ensemble_size; /* Number of ensemble members (m) */
  PetscInt state_size;    /* State vector dimension (n) */
  PetscInt obs_size;      /* Observation vector dimension (p) */
  PetscInt ndof;          /* Number of degrees of freedom per vertex */
  Mat      ensemble;      /* Ensemble matrix (n x m) */
  Vec      obs_error_var; /* Observation error variance (diagonal of R), length p */
  Mat      U;             /* Orthogonal transformation matrix (m x m) */

  /* Algorithm state */
  PetscBool assembled; /* Is the PetscDA object assembled/ready */

  /* Implementation-specific data */
  void *data; /* For implementation-specific storage */
};

/* Internal utility functions shared across PetscDA implementations */
PETSC_INTERN PetscErrorCode PetscDACholeskySqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode PetscDAymmetricEigenSqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode VecSetRandomGaussian_Private(Vec, PetscRandom, PetscReal, PetscReal);
