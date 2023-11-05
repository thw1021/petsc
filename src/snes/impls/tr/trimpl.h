/*
   Context for a Newton trust region method for solving a system
   of nonlinear equations
 */

#pragma once
#include <petsc/private/snesimpl.h>

typedef struct {
  PetscReal delta;  /* trust region parameter */
  PetscReal delta0; /* initial radius for trust region */
  PetscReal deltaM; /* maximum radius for trust region */
  PetscReal kmdc;   /* sufficient decrease parameter */

  PetscReal lammax;
  PetscReal lammin;
  PetscReal lamup;
  PetscReal lamdown;

  /*
    Given rho = (fk - fkp1) / (m(0) - m(pk))

    The radius is modified as:
      rho < eta2 -> delta *= t1
      rho > eta3 -> delta *= t2
      delta = min(delta,deltaM)

    The step is accepted if rho > eta1
  */
  PetscReal eta1;
  PetscReal eta2;
  PetscReal eta3;
  PetscReal t1;
  PetscReal t2;

  /* Use Quasi-Newton model */
  PetscBool qn;
  Mat       qnB;

  /* The type of norm for the trust region */
  NormType norm;

  SNESNewtonTRFallbackType fallback; /* enum to distinguish fallback in case Newton step is outside of the trust region */

  SNESNewtonTRScalingType scaling; /* enum to distinguish trust region scaling */
  PetscErrorCode (*scaling_update)(SNES, Vec, void *);
  PetscErrorCode (*scaling_apply)(SNES, Vec, Mat, Mat, Mat *, Mat *, void *);
  PetscErrorCode (*scaling_destroy)(void *);
  void *scaling_ctx;

  PetscErrorCode (*precheck)(SNES, Vec, Vec, PetscBool *, void *);
  void *precheckctx;
  PetscErrorCode (*postcheck)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *);
  void *postcheckctx;
} SNES_NEWTONTR;

typedef struct {
  Vec Gacc;
  Vec W;
} SNES_NEWTONTR_DEFAULT_SCALING_CTX;
