#if !defined(REGRESSORIMPL_H)
  #define REGRESSORIMPL_H

  #include <petscregressor.h>
  #include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      PetscRegressorRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscRegressorRegisterAll(void);

typedef struct _PetscRegressorOps *PetscRegressorOps;

struct _PetscRegressorOps {
  PetscErrorCode (*setup)(PetscRegressor);
  PetscErrorCode (*setfromoptions)(PetscOptionItems *, PetscRegressor); /* sets options from database */
  PetscErrorCode (*settraining)(PetscRegressor, Mat, Vec);              /* set the training data matrix and targets */
  PetscErrorCode (*fit)(PetscRegressor);                                /* compute the transformation to be applied */
  PetscErrorCode (*predict)(PetscRegressor, Mat, Vec);                  /* predict using fitted model */
  PetscErrorCode (*destroy)(PetscRegressor);
  PetscErrorCode (*reset)(PetscRegressor);
  PetscErrorCode (*view)(PetscRegressor, PetscViewer);
};

/* Define the PetscRegressor data structure. */
struct _p_PetscRegressor {
  PETSCHEADER(struct _PetscRegressorOps);

  PetscBool setupcalled; /* True if setup has been called */
  void     *data;        /* Implementation-specific data */
  Mat       training;    /* Matrix holding the training data set */
  Vec       target;      /* Targets for training data (response variables or labels) */
  Tao       tao;         /* Tao optimizer used by many regressor implementations */
  PetscReal regularizer_weight;
  PetscBool regularizer_weight_is_set; /* Indicates that the value in 'regularizer_weight' has been explicitly set by the user */
};

PETSC_EXTERN PetscLogEvent PetscRegressor_SetUp;
PETSC_EXTERN PetscLogEvent PetscRegressor_Fit;
PETSC_EXTERN PetscLogEvent PetscRegressor_Predict;

#endif
