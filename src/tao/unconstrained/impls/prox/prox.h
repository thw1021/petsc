/*
    Context for proximal (unconstrained minimization)
 */

#pragma once

#include <petsc/private/taoimpl.h>
#include <petsc/private/hashmap.h>

typedef struct _TaoProxOps *TaoProxOps;

struct _TaoProxOps {
  PetscErrorCode (*orig_obj)(Tao, Vec, PetscReal *, void *);
  PetscErrorCode (*orig_objgrad)(Tao, Vec, PetscReal *, Vec, void *);
  PetscErrorCode (*orig_grad)(Tao, Vec, Vec, void *);
  PetscErrorCode (*orig_hess)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*destroy)(Tao);
};

typedef struct {
  PetscReal size;
} TAO_PROX_SIMPLEX;

typedef struct {
  PetscReal lb;
  PetscReal ub;
} TAO_PROX_L1;

typedef struct {
  PETSCHEADER(struct _TaoProxOps);
  Tao subsolver;

  void *data;

  Mat vm; /* Variable Metric matrix */
  Mat H_orig, H_pre_orig;

  Vec G_old, X_old, workvec1;
  Vec y; /* Input y vector for prox(y) */

  PetscReal eta;                    /*  Restart tolerance */
  PetscReal stepsize, stepsize_old; /*  Step size */

  PetscInt step_type;

  TaoProxType type;

  void *orig_objP;
  void *orig_objgradP;
  void *orig_gradP;
  void *orig_hessP;

  void *ctxP;
} TAO_PROX;
