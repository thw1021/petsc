#include "admm.h"

typedef struct _n_TaoADMM_Basic {
  Vec y_old;
  PetscReal A_scale;
  PetscReal B_scale;
} TaoADMM_Basic;

PETSC_INTERN PetscErrorCode TaoADMMSetUp_Basic_Internal(Tao, TaoADMM_Basic *);
PETSC_INTERN PetscErrorCode TaoADMMDestroy_Basic_Internal(Tao, TaoADMM_Basic *);
PETSC_INTERN PetscErrorCode TaoADMMUpdateXSubproblem_Basic(Tao);
PETSC_INTERN PetscErrorCode TaoADMMUpdateZSubproblem_Basic(Tao);
