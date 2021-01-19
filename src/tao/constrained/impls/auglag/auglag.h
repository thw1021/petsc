#if !defined(TAOAUGLAG_H)
#define TAOAUGLAG_H
#include <petsc/private/taoimpl.h>

typedef struct {
  Tao              subsolver, parent;                     /* subsolver for aug-lag subproblem */
  PetscErrorCode   (*sub_obj)(Tao);                       /* subsolver objective function */
  TaoAugLagType    type;                                  /* subsolver objective type */

  IS               *Pis, *Yis;                            /* index sets to separate primal and dual vector spaces */
  VecScatter       *Pscatter, *Yscatter;                  /* scatter objects to write into combined vector spaces */

  Mat              Ae, Ai;                                /* aliased constraint Jacobians (do not destroy!) */
  Vec              Px, LgradX, Ce, Ci, G;                 /* aliased vectors (do not destroy!) */
  Vec              Ps, LgradS, Yi, Ye;                    /* sub-vectors for primal variables */
  Vec              *Parr, P, PL, PU, *Yarr, Y, C;         /* arrays and vectors for combined vector spaces */
  Vec              Xwork, Cework, Ciwork, Cizero;         /* work vectors */

  PetscReal        Lval, fval, gnorm, cnorm, cenorm, cinorm, cnorm_old;   /* scalar variables */
  PetscReal        mu0, mu, mu_fac, mu_pow_good, mu_pow_bad;              /* penalty parameters */
  PetscReal        ytol0, ytol, gtol0, gtol;                              /* convergence parameters */
  PetscReal        mu_max, ye_min, yi_min, ye_max, yi_max;                /* parameter safeguards */

  PetscBool        info;
} TAO_AugLag;

PETSC_INTERN PetscErrorCode TaoAugLagGetType_Private(Tao, TaoAugLagType*);
PETSC_INTERN PetscErrorCode TaoAugLagSetType_Private(Tao, TaoAugLagType);
PETSC_INTERN PetscErrorCode TaoAugLagGetSubsolver_Private(Tao, Tao*);
PETSC_INTERN PetscErrorCode TaoAugLagSetSubsolver_Private(Tao, Tao);
PETSC_INTERN PetscErrorCode TaoAugLagGetMultipliers_Private(Tao, Vec*);
PETSC_INTERN PetscErrorCode TaoAugLagSetMultipliers_Private(Tao, Vec);
PETSC_INTERN PetscErrorCode TaoAugLagGetPrimalIS_Private(Tao, IS*, IS*);
PETSC_INTERN PetscErrorCode TaoAugLagGetDualIS_Private(Tao, IS*, IS*);
PETSC_INTERN PetscErrorCode TaoAugLagSubsolverObjectiveAndGradient_Private(Tao, Vec, PetscReal*, Vec, void*);

#endif
