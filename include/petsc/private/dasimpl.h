#pragma once

#include <petscdas.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      PetscDASRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscDASRegisterAll(void);

typedef struct _PetscDASOps *PetscDASOps;

struct _PetscDASOps {
  PetscErrorCode (*setup)(PetscDAS);
  PetscErrorCode (*setfromoptions)(PetscDAS, PetscOptionItems);
  PetscErrorCode (*assimilate)(PetscDAS, Vec);
  PetscErrorCode (*forecast)(PetscDAS, Mat);
  PetscErrorCode (*destroy)(PetscDAS);
  PetscErrorCode (*reset)(PetscDAS);
  PetscErrorCode (*view)(PetscDAS, PetscViewer);
};

/* Define the PetscDAS data structure */
struct _p_PetscDAS {
  PETSCHEADER(struct _PetscDASOps);

  PetscBool setupcalled; /* True if setup has been called */
  PetscBool assimilated; /* True if assimilate has been called */
  void     *data;        /* Implementation-specific data */

  Vec     *ensemble;      /* Array of m state vectors */
  PetscInt ensemble_size; /* m (ensemble size) */
  PetscInt state_size;    /* n (state dimension) */

  Vec observation;     /* Current observation y_k */
  Mat obs_operator;    /* Observation operator H_k (sparse AIJ) */
  Vec obs_error_cov_diag; /* Observation error covariance diagonal R_k */

  PetscRandom random; /* For Gaussian sampling */
};

/* ETKF-specific structure */
typedef struct {
  Mat S;          /* S = R^(-1/2) * (Z - y_mean*1^T) / sqrt(m-1) */
  Mat I_plus_StS; /* I + S^T S */
  Mat V_T;        /* Eigenvectors of I + S^T S */
  Vec D_T;        /* Eigenvalues of I + S^T S */
  Vec w;          /* Weight vector */
  Vec delta;      /* Innovation delta */

  Vec ensemble_mean; /* Current ensemble mean */
  Mat X;             /* Normalized perturbations */
  Mat Z;             /* H(E) - observation space ensemble */
  
  Vec R_inv_sqrt;    /* R^(-1/2) diagonal (for diagonal obs error covariance) */

  PetscReal inflation;        /* Covariance inflation factor */
  PetscBool use_localization; /* Use localization? */
  PetscReal loc_radius;       /* Localization radius */
} PetscDAS_ETKF;

PETSC_EXTERN PetscLogEvent PetscDAS_SetUp;
PETSC_EXTERN PetscLogEvent PetscDAS_Assimilate;
PETSC_EXTERN PetscLogEvent PetscDAS_Forecast;
