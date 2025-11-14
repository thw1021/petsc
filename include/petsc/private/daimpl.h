#pragma once

#include "petscda.h"
#include <petsc/private/petscimpl.h>

/* PetscDataAssimilator object cookie */
PETSC_EXTERN PetscClassId PETSCDATAASSIMILATOR_CLASSID;

/* Operator table for PetscDataAssimilator implementations */
typedef struct _PetscDataAssimilatorOps *PetscDataAssimilatorOps;
struct _PetscDataAssimilatorOps {
  PetscErrorCode (*analysis)(PetscDataAssimilator, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
  PetscErrorCode (*applymodel)(PetscDataAssimilator, PetscErrorCode (*)(Vec, Vec, void *), void *);
  PetscErrorCode (*computemean)(PetscDataAssimilator, Vec);
  PetscErrorCode (*computeanomalies)(PetscDataAssimilator, Mat *);
  PetscErrorCode (*destroy)(PetscDataAssimilator);
  PetscErrorCode (*view)(PetscDataAssimilator, PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscDataAssimilator, PetscOptionItems *);
};

/*
  Internal PetscDataAssimilator structure following PETSc object conventions
*/
struct _p_PetscDataAssimilator {
  PETSCHEADER(struct _PetscDataAssimilatorOps);

  /* Core PetscDataAssimilator data */
  PetscInt ensemble_size; /* Number of ensemble members (m) */
  PetscInt state_size;    /* State vector dimension (n) */
  PetscInt obs_size;      /* Observation vector dimension (p) */
  Mat      ensemble;      /* Ensemble matrix (n x m) */
  Vec      obs_error_var; /* Observation error variance (diagonal of R), length p */
  Mat      U;             /* Orthogonal transformation matrix (m x m) */

  /* Algorithm state */
  PetscBool assembled; /* Is the PetscDataAssimilator object assembled/ready */

  /* Implementation-specific data */
  void *data; /* For implementation-specific storage */
};

/* Internal utility functions shared across PetscDataAssimilator implementations */
PETSC_INTERN PetscErrorCode PetscDataAssimilatorCholeskySqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode PetscDataAssimilatorSymmetricEigenSqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode VecSetRandomGaussian_Private(Vec, PetscRandom, PetscReal, PetscReal);
