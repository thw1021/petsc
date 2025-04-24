#pragma once
#include <petsc/private/taoimpl.h>

typedef enum {
  ADMM_INITIALIZE_SCATTER,
  ADMM_INITIALIZE_Z_AX,
  ADMM_INITIALIZE_X_BZ,
} ADMMInitializeType;

typedef struct {
  PetscErrorCode (*destroy)(Tao);
  PetscErrorCode (*updatexsubproblem)(Tao);
  PetscErrorCode (*updatezsubproblem)(Tao);
  PetscErrorCode (*dualresidual)(Tao);

  void *data;

  // the different ways a Tao with one solution vector can be transformed into an ADMM problem with two solution vectors, x and z
  ADMMInitializeType initialize_type;
  VecScatter         x_scatter;
  VecScatter         z_scatter;

  /* the primal residual is   r_k       = A x_k     + B z_k + c
     the halfstep residual is r_{k+1/2} = A x_{k+1} + B z_k + c */
  Mat A;
  Vec Ax;
  Mat B;
  Vec Bz;
  Vec c;
  Vec r;
  Vec r_halfstep;

  // the algorithm needs primal variables x and z and multipiliers y: x and z are stored as the solution vectors in the subsolvers
  Vec y;

  /* the dual residuals are
     d_x \in \partial f(x) + A' y
     d_z \in \partial g(z) + B' y */
  Vec d_x;
  Vec d_z;

  Vec y_old;
  Vec Ax_old;
  Vec Bz_old;
  Vec x_old;
  Vec z_old;

  // data for the f(x) subproblem
  PetscInt   f_num_terms;
  PetscInt  *f_terms;
  PetscBool3 f_mapped;
  TaoTerm    f;
  Tao        x_subsolver;
  PetscBool  x_inexact;

  // data for the g(z) subproblem
  PetscInt   g_num_terms;
  PetscInt  *g_terms;
  PetscBool3 g_mapped;
  TaoTerm    g;
  Tao        z_subsolver;
  PetscBool  z_inexact;

  // linearized ADMM
  PetscBool linearized;
  PetscReal linearized_mu_x;
  PetscReal linearized_mu_z;

  // relaxation
  PetscReal relaxation_gamma;

  // metric scaling and adaptivity
  TaoADMMUpdateType mu_update;
  PetscReal         mu;
  PetscReal         mu_min;

  // ARADMM parameters
  PetscInt  adaptivity_period;
  PetscReal correlation_epsilon;

  // convergence metrics
  PetscReal c_norm;
  PetscReal Ax_norm;
  PetscReal Bz_norm;
  PetscReal Aty_norm;
  PetscReal Bty_norm;

  PetscBool setfromoptionscalled;

  PetscViewer       debug_viewer;
  PetscViewerFormat debug_viewer_format;
} Tao_ADMM;

PETSC_INTERN PetscErrorCode TaoADMMVecDuplicateAndCopy(Vec, Vec *);
PETSC_INTERN PetscErrorCode TaoADMMSetUp_Basic(Tao);
PETSC_INTERN PetscErrorCode TaoADMMSetUp_ARADMM(Tao);
PETSC_INTERN PetscErrorCode TaoADMMSetUp_Linearized(Tao);
PETSC_INTERN PetscErrorCode TaoADMMSetUp_Linearized_ARADMM(Tao);
PETSC_INTERN PetscErrorCode TaoADMMConfigureSubTao_Default(Tao, Tao);
