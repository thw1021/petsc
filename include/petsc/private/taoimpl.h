#pragma once

#include <petsctao.h>
#include <petsctaolinesearch.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      TaoRegisterAllCalled;
PETSC_EXTERN PetscErrorCode TaoRegisterAll(void);

typedef struct _TaoOps *TaoOps;

struct _TaoOps {
  /* Methods set by application */
  PetscErrorCode (*computeobjective)(Tao, Vec, PetscReal *, void *);
  PetscErrorCode (*computeobjectiveandgradient)(Tao, Vec, PetscReal *, Vec, void *);
  PetscErrorCode (*computegradient)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computehessian)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computeresidual)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computeresidualjacobian)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computeconstraints)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computeinequalityconstraints)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computeequalityconstraints)(Tao, Vec, Vec, void *);
  PetscErrorCode (*computejacobian)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computejacobianstate)(Tao, Vec, Mat, Mat, Mat, void *);
  PetscErrorCode (*computejacobiandesign)(Tao, Vec, Mat, void *);
  PetscErrorCode (*computejacobianinequality)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computejacobianequality)(Tao, Vec, Mat, Mat, void *);
  PetscErrorCode (*computebounds)(Tao, Vec, Vec, void *);
  PetscErrorCode (*update)(Tao, PetscInt, void *);
  PetscErrorCode (*convergencetest)(Tao, void *);
  PetscErrorCode (*convergencedestroy)(void *);

  /* Methods set by solver */
  PetscErrorCode (*computedual)(Tao, Vec, Vec);
  PetscErrorCode (*setup)(Tao);
  PetscErrorCode (*solve)(Tao);
  PetscErrorCode (*view)(Tao, PetscViewer);
  PetscErrorCode (*setfromoptions)(Tao, PetscOptionItems);
  PetscErrorCode (*destroy)(Tao);
};

#define MAXTAOMONITORS 10

struct _p_Tao {
  PETSCHEADER(struct _TaoOps);
  void *ctx; /* user provided context */
  void *user_objP;
  void *user_objgradP;
  void *user_gradP;
  void *user_hessP;
  void *user_lsresP;
  void *user_lsjacP;
  void *user_conP;
  void *user_con_equalityP;
  void *user_con_inequalityP;
  void *user_jacP;
  void *user_jac_equalityP;
  void *user_jac_inequalityP;
  void *user_jac_stateP;
  void *user_jac_designP;
  void *user_boundsP;
  void *user_update;

  PetscErrorCode (*monitor[MAXTAOMONITORS])(Tao, void *);
  PetscCtxDestroyFn *monitordestroy[MAXTAOMONITORS];
  void              *monitorcontext[MAXTAOMONITORS];
  PetscInt           numbermonitors;
  void              *cnvP;
  TaoConvergedReason reason;
  PetscBool          errorifnotconverged;

  PetscBool objectivedomainerror; /* set with TaoSetObjectiveDomainError() */
  PetscBool gradientdomainerror;  /* set with TaoSetGradientDomainError() */
  PetscBool hessiandomainerror;   /* set with TaoSetHessianDomainError() */

  PetscBool setupcalled;
  void     *data;

  Vec        solution;
  Vec        gradient;
  Vec        stepdirection;
  Vec        XL;
  Vec        XU;
  Vec        IL;
  Vec        IU;
  Vec        DI;
  Vec        DE;
  Mat        hessian;
  Mat        hessian_pre;
  Mat        gradient_norm;
  Vec        gradient_norm_tmp;
  Vec        ls_res;
  Mat        ls_jac;
  Mat        ls_jac_pre;
  Vec        res_weights_v;
  PetscInt   res_weights_n;
  PetscInt  *res_weights_rows;
  PetscInt  *res_weights_cols;
  PetscReal *res_weights_w;
  Vec        constraints;
  Vec        constraints_equality;
  Vec        constraints_inequality;
  Mat        jacobian;
  Mat        jacobian_pre;
  Mat        jacobian_inequality;
  Mat        jacobian_inequality_pre;
  Mat        jacobian_equality;
  Mat        jacobian_equality_pre;
  Mat        jacobian_state;
  Mat        jacobian_state_inv;
  Mat        jacobian_design;
  Mat        jacobian_state_pre;
  Mat        jacobian_design_pre;
  IS         state_is;
  IS         design_is;
  PetscReal  step;
  PetscReal  residual;
  PetscReal  gnorm0;
  PetscReal  cnorm;
  PetscReal  cnorm0;
  PetscReal  fc;

  PetscInt max_constraints;
  PetscInt nfuncs;
  PetscInt ngrads;
  PetscInt nfuncgrads;
  PetscInt nhess;
  PetscInt niter;
  PetscInt ntotalits;
  PetscInt nconstraints;
  PetscInt niconstraints;
  PetscInt neconstraints;
  PetscInt njac;
  PetscInt njac_equality;
  PetscInt njac_inequality;
  PetscInt njac_state;
  PetscInt njac_design;

  PetscInt ksp_its;     /* KSP iterations for this solver iteration */
  PetscInt ksp_tot_its; /* Total (cumulative) KSP iterations */

  TaoLineSearch linesearch;
  PetscBool     lsflag; /* goes up when line search fails */
  KSP           ksp;
  PetscReal     trust; /* Current trust region */

  /* EW type forcing term */
  PetscBool ksp_ewconv;
  SNES      snes_ewdummy;

  PetscObjectParameterDeclare(PetscReal, gatol);
  PetscObjectParameterDeclare(PetscReal, grtol);
  PetscObjectParameterDeclare(PetscReal, gttol);
  PetscObjectParameterDeclare(PetscReal, catol);
  PetscObjectParameterDeclare(PetscReal, crtol);
  PetscObjectParameterDeclare(PetscReal, steptol);
  PetscObjectParameterDeclare(PetscReal, fmin);
  PetscObjectParameterDeclare(PetscInt, max_it);
  PetscObjectParameterDeclare(PetscInt, max_funcs);
  PetscObjectParameterDeclare(PetscReal, trust0); /* initial trust region radius */

  PetscBool printreason;
  PetscBool viewsolution;
  PetscBool viewgradient;
  PetscBool viewconstraints;
  PetscBool viewhessian;
  PetscBool viewjacobian;
  PetscBool bounded;
  PetscBool constrained;
  PetscBool eq_constrained;
  PetscBool ineq_constrained;
  PetscBool ineq_doublesided;
  PetscBool header_printed;
  PetscBool recycle;

  TaoSubsetType subset_type;
  PetscInt      hist_max;   /* Number of iteration histories to keep */
  PetscReal    *hist_obj;   /* obj value at each iteration */
  PetscReal    *hist_resid; /* residual at each iteration */
  PetscReal    *hist_cnorm; /* constraint norm at each iteration */
  PetscInt     *hist_lits;  /* number of ksp its at each TAO iteration */
  PetscInt      hist_len;
  PetscBool     hist_reset;
  PetscBool     hist_malloc;
};

PETSC_EXTERN PetscLogEvent TAO_Solve;
PETSC_EXTERN PetscLogEvent TAO_ObjectiveEval;
PETSC_EXTERN PetscLogEvent TAO_GradientEval;
PETSC_EXTERN PetscLogEvent TAO_ObjGradEval;
PETSC_EXTERN PetscLogEvent TAO_HessianEval;
PETSC_EXTERN PetscLogEvent TAO_ConstraintsEval;
PETSC_EXTERN PetscLogEvent TAO_JacobianEval;

static inline PetscErrorCode TaoLogConvergenceHistory(Tao tao, PetscReal obj, PetscReal resid, PetscReal cnorm, PetscInt totits)
{
  PetscFunctionBegin;
  if (tao->hist_max > tao->hist_len) {
    if (tao->hist_obj) tao->hist_obj[tao->hist_len] = obj;
    if (tao->hist_resid) tao->hist_resid[tao->hist_len] = resid;
    if (tao->hist_cnorm) tao->hist_cnorm[tao->hist_len] = cnorm;
    if (tao->hist_lits) {
      PetscInt sits = totits;
      PetscCheck(tao->hist_len >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "History length cannot be negative");
      for (PetscInt i = 0; i < tao->hist_len; i++) sits -= tao->hist_lits[i];
      tao->hist_lits[tao->hist_len] = sits;
    }
    tao->hist_len++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoCheckInitialObjectiveAndGradient - checks if the initial objective function or gradient norm computed within `TaoSolve()` is infinity or NaN

  Collective

  Input Parameters:
+ tao   - the `Tao` solver object
. f     - the objective function value
- gnorm - the norm of the gradient of the objective function

  Options Database Key:
. -tao_error_if_not_converged - causes `TaoSolve()` to generate an error immediately when it determines the solver cannot converge

  Level: developer

  Notes:
  If `TaoSetErrorIfNotConverged()` has been set and infinity or NaN is returned as the objective function value or the norm of the gradient
  then this errors immediately ending the program.

  If infinity or Nan is returned as the objective function value or gradient norm and `TaoSetObjectiveDomainError()` has been set then the `TaoConvergedReason` of
  `TAO_DIVERGED_OBJECTIVE_DOMAIN` is set, otherwise `TAO_DIVERGED_OBJECTIVE_NAN` is set.

.seealso: [](ch_tao), `Tao`, `TaoCheckObjective()`, `TaoCreate()`, `TaoSetErrorIfNotConverged()`, `TaoGetErrorIfNotConverged()`, `TaoSetObjectiveDomainError()`,
          `TaoCheckObjectiveAndGradient()`
@*/
#define TaoCheckInitialObjectiveAndGradient(tao, f, gnorm) \
  PetscCheck(!tao->errorifnotconverged || (!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm)), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Initial objective function generated Inf or NaN"); \
  if (PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm)) { \
    if (tao->objectivedomainerror) { \
      tao->reason               = TAO_DIVERGED_OBJECTIVE_DOMAIN; \
      tao->objectivedomainerror = PETSC_FALSE; \
    } else tao->reason = TAO_DIVERGED_NAN; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

/*@C
  TaoCheckObjective - checks if an objective function computed within `TaoSolve()` is infinity or NaN

  Collective

  Input Parameters:
+ tao   - the `Tao` solver object
- f     - the objective function value

  Options Database Key:
. -tao_error_if_not_converged - causes `TaoSolve()` to generate an error immediately when it determines the solver cannot converge

  Level: developer

  Notes:
  If `TaoSetErrorIfNotConverged()` has been set and infinity or NaN is returned as the objective function value this errors immediately ending the program.

  If infinity or Nan is returned as the objective function value and `TaoSetObjectiveDomainError()` has been set then the `TaoConvergedReason` of
  `TAO_DIVERGED_OBJECTIVE_DOMAIN` is set, otherwise `TAO_DIVERGED_OBJECTIVE_NAN` is set.

.seealso: [](ch_tao), `Tao`, `TaoCheckInitialObjectiveAndGradient()`, `TaoCreate()`, `TaoSetErrorIfNotConverged()`, `TaoGetErrorIfNotConverged()`,
          `TaoSetObjectiveDomainError()`, `TaoCheckObjectiveAndGradient()`
@*/
#define TaoCheckObjective(tao, f) \
  PetscCheck(!tao->errorifnotconverged || !PetscIsInfOrNanReal(f), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Objective function generated Inf or NaN"); \
  if (PetscIsInfOrNanReal(f)) { \
    if (tao->objectivedomainerror) { \
      tao->reason               = TAO_DIVERGED_OBJECTIVE_DOMAIN; \
      tao->objectivedomainerror = PETSC_FALSE; \
    } else tao->reason = TAO_DIVERGED_NAN; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

/*@C
  TaoCheckObjectiveAndGradient - checks if an objective function or the norm of the gradient computed within `TaoSolve()` is infinity or NaN

  Collective

  Input Parameters:
+ tao   - the `Tao` solver object
. f     - the objective function value
- gnrom - the norm of the gradient

  Options Database Key:
. -tao_error_if_not_converged - causes `TaoSolve()` to generate an error immediately when it determines the solver cannot converge

  Level: developer

  Notes:
  If `TaoSetErrorIfNotConverged()` has been set and infinity or NaN is returned as the objective function value or its gradient this errors immediately ending the program.

  If infinity or Nan is returned as the objective function value or the gradient and `TaoSetObjectiveDomainError()` has been set then the `TaoConvergedReason` of
  `TAO_DIVERGED_OBJECTIVE_DOMAIN` is set, otherwise `TAO_DIVERGED_OBJECTIVE_NAN` is set.

.seealso: [](ch_tao), `Tao`, `TaoCheckInitialObjectiveAndGradient()`, `TaoCreate()`, `TaoSetErrorIfNotConverged()`, `TaoGetErrorIfNotConverged()`,
          `TaoCheckObjective()`, `TaoSetObjectiveDomainError()`
@*/
#define TaoCheckObjectiveAndGradient(tao, f, gnorm) \
  PetscCheck(!tao->errorifnotconverged || (!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm)), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Objective function generated Inf or NaN"); \
  if (PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm)) { \
    if (tao->objectivedomainerror) { \
      tao->reason               = TAO_DIVERGED_OBJECTIVE_DOMAIN; \
      tao->objectivedomainerror = PETSC_FALSE; \
    } else tao->reason = TAO_DIVERGED_NAN; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }
