#pragma once

#include "petscda.h"
#include <petsc/private/petscimpl.h>

/* DA object cookie */
PETSC_EXTERN PetscClassId DA_CLASSID;

/* Operator table for DA implementations */
typedef struct _DAOps *DAOps;
struct _DAOps {
  PetscErrorCode (*analysis)(DA, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
  PetscErrorCode (*applymodel)(DA, PetscErrorCode (*)(Vec, Vec, void *), void *);
  PetscErrorCode (*computemean)(DA, Vec);
  PetscErrorCode (*computeanomalies)(DA, Mat *);
  PetscErrorCode (*destroy)(DA);
  PetscErrorCode (*view)(DA, PetscViewer);
  PetscErrorCode (*setfromoptions)(DA, PetscOptionItems *);
};

/*
  Internal DA structure following PETSc object conventions
*/
struct _p_DA {
  PETSCHEADER(struct _DAOps);

  /* Core DA data */
  PetscInt ensemble_size; /* Number of ensemble members (m) */
  PetscInt state_size;    /* State vector dimension (n) */
  PetscInt obs_size;      /* Observation vector dimension (p) */
  Mat      ensemble;      /* Ensemble matrix (n x m) */
  Vec      obs_error_var; /* Observation error variance (diagonal of R), length p */
  Mat      U;             /* Orthogonal transformation matrix (m x m) */

  /* Algorithm state */
  PetscBool assembled; /* Is the DA object assembled/ready */

  /* Implementation-specific data */
  void *data; /* For implementation-specific storage */
};

/* Internal utility functions shared across DA implementations */
PETSC_INTERN PetscErrorCode DACholeskySqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode DASymmetricEigenSqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode VecSetRandomGaussian_Private(Vec, PetscRandom, PetscReal, PetscReal);
