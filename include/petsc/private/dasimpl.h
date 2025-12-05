#pragma once

#include "petscdas.h"
#include <petsc/private/petscimpl.h>

/* PetscDAS object cookie */
PETSC_EXTERN PetscClassId PETSCDAS_CLASSID;

/* Operator table for PetscDAS implementations */
typedef struct _PetscDASOps *PetscDASOps;
struct _PetscDASOps {
  PetscErrorCode (*analysis)(PetscDAS, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
  PetscErrorCode (*applymodel)(PetscDAS, PetscErrorCode (*)(Vec, Vec, void *), void *);
  PetscErrorCode (*computemean)(PetscDAS, Vec);
  PetscErrorCode (*computeanomalies)(PetscDAS, Mat *);
  PetscErrorCode (*destroy)(PetscDAS);
  PetscErrorCode (*view)(PetscDAS, PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscDAS, PetscOptionItems *);
};

/*
  Internal PetscDAS structure following PETSc object conventions
*/
struct _p_PetscDAS {
  PETSCHEADER(struct _PetscDASOps);

  /* Core PetscDAS data */
  PetscInt ensemble_size; /* Number of ensemble members (m) */
  PetscInt state_size;    /* State vector dimension (n) */
  PetscInt obs_size;      /* Observation vector dimension (p) */
  Mat      ensemble;      /* Ensemble matrix (n x m) */
  Vec      obs_error_var; /* Observation error variance (diagonal of R), length p */
  Mat      U;             /* Orthogonal transformation matrix (m x m) */

  /* Algorithm state */
  PetscBool assembled; /* Is the PetscDAS object assembled/ready */

  /* Implementation-specific data */
  void *data; /* For implementation-specific storage */
};

/* Internal utility functions shared across PetscDAS implementations */
PETSC_INTERN PetscErrorCode PetscDASCholeskySqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode PetscDASSymmetricEigenSqrt_Private(Mat, Mat *);
PETSC_INTERN PetscErrorCode VecSetRandomGaussian_Private(Vec, PetscRandom, PetscReal, PetscReal);
