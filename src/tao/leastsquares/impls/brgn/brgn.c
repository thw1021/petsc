#include <../src/tao/leastsquares/impls/brgn/brgn.h> /*I "petsctao.h" I*/

const char *const TaoBRGNRegularizationTypes[] = {"user", "l2prox", "l2pure", "l1dict", "lm", "TaoBRGNRegularizationType", "TAOBRGN_REGULARIZATION_", NULL};

#define BRGN_GN_PREFIX  "brgn_gauss_newton_"
#define BRGN_REG_PREFIX "regularizer_"

/* ----- USER regularizer shim: wraps the legacy Tao-style callbacks as a TAOTERMSHELL ----- */

typedef struct {
  Tao parent; /* weak; the BRGN parent Tao that the user's callback expects */
  PetscErrorCode (*objgrad)(Tao, Vec, PetscReal *, Vec, PetscCtx);
  PetscCtx objgrad_ctx;
  PetscErrorCode (*hessian)(Tao, Vec, Mat, PetscCtx);
  PetscCtx hessian_ctx;
  Mat      hessian_template; /* user-provided Mat (from TaoBRGNSetRegularizerHessianRoutine) */
} TaoBRGNUserShim;

static PetscErrorCode TaoBRGNUserShimDestroy(PetscCtxRt ctx_arg)
{
  TaoBRGNUserShim **slot = (TaoBRGNUserShim **)ctx_arg;

  PetscFunctionBegin;
  if (*slot) PetscCall(MatDestroy(&(*slot)->hessian_template));
  PetscCall(PetscFree(*slot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNUserShimCreateHessianMatrices(TaoTerm term, Mat *H, Mat *Hpre)
{
  TaoBRGNUserShim *sh;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, &sh));
  PetscCheck(sh->hessian_template, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoBRGN USER regularizer: no Hessian template; call TaoBRGNSetRegularizerHessianRoutine() first");
  if (H) PetscCall(MatDuplicate(sh->hessian_template, MAT_DO_NOT_COPY_VALUES, H));
  if (Hpre) {
    if (H && term->Hpre_is_H) {
      PetscCall(PetscObjectReference((PetscObject)*H));
      *Hpre = *H;
    } else PetscCall(MatDuplicate(sh->hessian_template, MAT_DO_NOT_COPY_VALUES, Hpre));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNUserShimObjAndGrad(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoBRGNUserShim *sh;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, &sh));
  PetscCheck(sh->objgrad, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoBRGN USER regularizer: no objective+gradient callback set");
  PetscCall((*sh->objgrad)(sh->parent, x, value, g, sh->objgrad_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNUserShimHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoBRGNUserShim *sh;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, &sh));
  PetscCheck(sh->hessian, PetscObjectComm((PetscObject)term), PETSC_ERR_USER, "TaoBRGN USER regularizer: no Hessian callback set");
  if (H) PetscCall((*sh->hessian)(sh->parent, x, H, sh->hessian_ctx));
  if (Hpre && Hpre != H) PetscCall((*sh->hessian)(sh->parent, x, Hpre, sh->hessian_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNCreateUserShimTerm(Tao tao, TaoTerm *term)
{
  TAO_BRGN        *gn = (TAO_BRGN *)tao->data;
  TaoBRGNUserShim *sh;
  TaoTerm          _term;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sh));
  sh->parent      = tao;
  sh->objgrad     = gn->user_objgrad;
  sh->objgrad_ctx = gn->user_objgrad_ctx;
  sh->hessian     = gn->user_hessian;
  sh->hessian_ctx = gn->user_hessian_ctx;
  if (gn->user_hessian_mat) {
    PetscCall(PetscObjectReference((PetscObject)gn->user_hessian_mat));
    sh->hessian_template = gn->user_hessian_mat;
  }
  PetscCall(TaoTermCreateShell(PetscObjectComm((PetscObject)tao), (PetscCtx)sh, TaoBRGNUserShimDestroy, &_term));
  PetscCall(TaoTermSetParametersMode(_term, TAOTERM_PARAMETERS_NONE));
  if (tao->solution) PetscCall(TaoTermSetSolutionTemplate(_term, tao->solution));
  if (sh->objgrad) PetscCall(TaoTermShellSetObjectiveAndGradient(_term, TaoBRGNUserShimObjAndGrad));
  if (sh->hessian) PetscCall(TaoTermShellSetHessian(_term, TaoBRGNUserShimHessian));
  if (sh->hessian_template) PetscCall(TaoTermShellSetCreateHessianMatrices(_term, TaoBRGNUserShimCreateHessianMatrices));
  *term = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ----- Build the regularizer subterm for a given reg_type ----- */

static PetscErrorCode TaoBRGNCreateDefaultRegularizer(Tao tao, TaoTerm gn_term, TaoTerm *reg_term, Mat *map)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;
  PetscInt  n, N;

  PetscFunctionBegin;
  *reg_term = NULL;
  *map      = NULL;
  PetscCheck(tao->solution, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetSolution() must be called before BRGN regularizer construction");
  PetscCall(VecGetLocalSize(tao->solution, &n));
  PetscCall(VecGetSize(tao->solution, &N));

  switch (gn->reg_type) {
  case TAOBRGN_REGULARIZATION_L2PURE:
  case TAOBRGN_REGULARIZATION_L2PROX:
    PetscCall(TaoTermCreateHalfL2Squared(PetscObjectComm((PetscObject)tao), n, N, reg_term));
    break;
  case TAOBRGN_REGULARIZATION_L1DICT:
    if (gn->D) {
      PetscInt mloc, M;
      PetscCall(MatGetLocalSize(gn->D, &mloc, NULL));
      PetscCall(MatGetSize(gn->D, &M, NULL));
      PetscCall(TaoTermCreateL1(PetscObjectComm((PetscObject)tao), mloc, M, gn->epsilon, reg_term));
      PetscCall(PetscObjectReference((PetscObject)gn->D));
      *map = gn->D;
    } else {
      PetscCall(TaoTermCreateL1(PetscObjectComm((PetscObject)tao), n, N, gn->epsilon, reg_term));
    }
    break;
  case TAOBRGN_REGULARIZATION_LM:
    PetscCall(TaoTermCreateBRGNLMDamping(gn_term, reg_term));
    break;
  case TAOBRGN_REGULARIZATION_USER:
    PetscCall(TaoBRGNCreateUserShimTerm(tao, reg_term));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "unknown TaoBRGNRegularizationType %d", (int)gn->reg_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Apply LM-specific subterm settings: mask out objective and gradient (damping is
   step-regularization only); set unit line search. */
static PetscErrorCode TaoBRGNApplyLMSettings(Tao tao)
{
  TAO_BRGN     *gn = (TAO_BRGN *)tao->data;
  TaoLineSearch ls;

  PetscFunctionBegin;
  PetscCall(TaoTermSumSetTermMask(gn->subsolver->objective_term.term, 1, (TaoTermMask)(TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_GRADIENT)));
  PetscCall(TaoGetLineSearch(gn->subsolver, &ls));
  PetscCall(TaoLineSearchSetType(ls, TAOLINESEARCHUNIT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNInstallRegularizerSubterm(Tao tao, TaoTerm term, Mat map)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoTermSumSetTerm(gn->subsolver->objective_term.term, 1, BRGN_REG_PREFIX, gn->lambda, term, map));
  if (gn->reg_type == TAOBRGN_REGULARIZATION_LM) PetscCall(TaoBRGNApplyLMSettings(tao));
  else PetscCall(TaoTermSumSetTermMask(gn->subsolver->objective_term.term, 1, TAOTERM_MASK_NONE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ----- Public API ----- */

/*@
  TaoBRGNGetRegularizationType - Get the `TaoBRGNRegularizationType` of a `TAOBRGN`

  Not collective

  Input Parameter:
. tao - a `Tao` of type `TAOBRGN`

  Output Parameter:
. type - the `TaoBRGNRegularizationType`

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TAOBRGN`, `TaoBRGNRegularizationType`, `TaoBRGNSetRegularizationType()`
@*/
PetscErrorCode TaoBRGNGetRegularizationType(Tao tao, TaoBRGNRegularizationType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscUseMethod((PetscObject)tao, "TaoBRGNGetRegularizationType_C", (Tao, TaoBRGNRegularizationType *), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetRegularizationType_BRGN(Tao tao, TaoBRGNRegularizationType *type)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  *type = gn->reg_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNSetRegularizationType - Set the `TaoBRGNRegularizationType` of a `TAOBRGN`

  Logically collective

  Input Parameters:
+ tao  - a `Tao` of type `TAOBRGN`
- type - the `TaoBRGNRegularizationType`

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TAOBRGN`, `TaoBRGNRegularizationType`, `TaoBRGNGetRegularizationType()`
@*/
PetscErrorCode TaoBRGNSetRegularizationType(Tao tao, TaoBRGNRegularizationType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(tao, type, 2);
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetRegularizationType_C", (Tao, TaoBRGNRegularizationType), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizationType_BRGN(Tao tao, TaoBRGNRegularizationType type)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  if (type == gn->reg_type) PetscFunctionReturn(PETSC_SUCCESS);
  gn->reg_type = type;
  if (tao->setupcalled) {
    TaoTerm gn_term, reg_term;
    Mat     map;

    PetscCall(TaoTermSumGetTerm(gn->subsolver->objective_term.term, 0, NULL, NULL, &gn_term, NULL));
    PetscCall(TaoBRGNCreateDefaultRegularizer(tao, gn_term, &reg_term, &map));
    PetscCall(TaoBRGNInstallRegularizerSubterm(tao, reg_term, map));
    PetscCall(TaoTermDestroy(&reg_term));
    PetscCall(MatDestroy(&map));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetDampingVector - Get the damping vector $\mathrm{diag}(J^T J)$ from a `TAOBRGN` with `TAOBRGN_REGULARIZATION_LM` regularization

  Collective

  Input Parameter:
. tao - a `Tao` of type `TAOBRGN` with `TAOBRGN_REGULARIZATION_LM` regularization

  Output Parameter:
. d - the damping vector

  Level: developer

.seealso: [](ch_tao), `Tao`, `TAOBRGN`, `TaoBRGNRegularzationTypes`
@*/
PetscErrorCode TaoBRGNGetDampingVector(Tao tao, Vec *d)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(d, 2);
  PetscUseMethod((PetscObject)tao, "TaoBRGNGetDampingVector_C", (Tao, Vec *), (tao, d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetDampingVector_BRGN(Tao tao, Vec *d)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;
  TaoTerm   reg;

  PetscFunctionBegin;
  PetscCheck(gn->reg_type == TAOBRGN_REGULARIZATION_LM, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "Damping vector is only available if regularization type is lm.");
  PetscCheck(tao->setupcalled, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called before TaoBRGNGetDampingVector()");
  PetscCall(TaoTermSumGetTerm(gn->subsolver->objective_term.term, 1, NULL, NULL, &reg, NULL));
  if (!gn->damping) PetscCall(VecDuplicate(tao->solution, &gn->damping));
  PetscCall(TaoTermBRGNLMDampingCopyDiagonal(reg, gn->damping));
  *d = gn->damping;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetSubsolver - Get the pointer to the subsolver inside a `TAOBRGN`

  Collective

  Input Parameter:
. tao - the Tao solver context

  Output Parameter:
. subsolver - the `Tao` sub-solver context

  Level: advanced

.seealso: `Tao`, `Mat`, `TAOBRGN`
@*/
PetscErrorCode TaoBRGNGetSubsolver(Tao tao, Tao *subsolver)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscUseMethod((PetscObject)tao, "TaoBRGNGetSubsolver_C", (Tao, Tao *), (tao, subsolver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetSubsolver_BRGN(Tao tao, Tao *subsolver)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  *subsolver = gn->subsolver;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNSetRegularizerWeight - Set the regularizer weight for the Gauss-Newton least-squares algorithm

  Collective

  Input Parameters:
+ tao    - the `Tao` solver context
- lambda - the scalar weight $\lambda$ multiplying the regularizer

  Level: beginner

.seealso: `Tao`, `Mat`, `TAOBRGN`
@*/
PetscErrorCode TaoBRGNSetRegularizerWeight(Tao tao, PetscReal lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, lambda, 2);
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetRegularizerWeight_C", (Tao, PetscReal), (tao, lambda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizerWeight_BRGN(Tao tao, PetscReal lambda)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  gn->lambda = lambda;
  if (tao->setupcalled) {
    const char *name;
    TaoTerm     term;
    Mat         map;

    PetscCall(TaoTermSumGetTerm(gn->subsolver->objective_term.term, 1, &name, NULL, &term, &map));
    PetscCall(TaoTermSumSetTerm(gn->subsolver->objective_term.term, 1, name, lambda, term, map));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNSetL1SmoothEpsilon - Set the L1-norm smooth approximation parameter for L1-regularized least-squares algorithm

  Collective

  Input Parameters:
+ tao     - the `Tao` solver context
- epsilon - L1-norm smooth approximation parameter

  Level: advanced

.seealso: `Tao`, `Mat`, `TAOBRGN`
@*/
PetscErrorCode TaoBRGNSetL1SmoothEpsilon(Tao tao, PetscReal epsilon)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, epsilon, 2);
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetL1SmoothEpsilon_C", (Tao, PetscReal), (tao, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetL1SmoothEpsilon_BRGN(Tao tao, PetscReal epsilon)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  gn->epsilon = epsilon;
  if (tao->setupcalled && gn->reg_type == TAOBRGN_REGULARIZATION_L1DICT) {
    TaoTerm reg;

    PetscCall(TaoTermSumGetTerm(gn->subsolver->objective_term.term, 1, NULL, NULL, &reg, NULL));
    PetscCall(TaoTermL1SetEpsilon(reg, epsilon));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNSetDictionaryMatrix - bind the dictionary matrix from user application context to the L1DICT regularizer

  Input Parameters:
+ tao  - the `Tao` context
- dict - the user specified dictionary matrix.  We allow to set a `NULL` dictionary, which means identity matrix by default

  Level: advanced

.seealso: `Tao`, `Mat`, `TAOBRGN`
@*/
PetscErrorCode TaoBRGNSetDictionaryMatrix(Tao tao, Mat dict)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetDictionaryMatrix_C", (Tao, Mat), (tao, dict));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetDictionaryMatrix_BRGN(Tao tao, Mat dict)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  if (dict) {
    PetscValidHeaderSpecific(dict, MAT_CLASSID, 2);
    PetscCheckSameComm(tao, 1, dict, 2);
    PetscCall(PetscObjectReference((PetscObject)dict));
  }
  PetscCall(MatDestroy(&gn->D));
  gn->D = dict;
  if (tao->setupcalled && gn->reg_type == TAOBRGN_REGULARIZATION_L1DICT) {
    TaoTerm gn_term, reg_term;
    Mat     map;

    PetscCall(TaoTermSumGetTerm(gn->subsolver->objective_term.term, 0, NULL, NULL, &gn_term, NULL));
    PetscCall(TaoBRGNCreateDefaultRegularizer(tao, gn_term, &reg_term, &map));
    PetscCall(TaoBRGNInstallRegularizerSubterm(tao, reg_term, map));
    PetscCall(TaoTermDestroy(&reg_term));
    PetscCall(MatDestroy(&map));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoBRGNSetRegularizerObjectiveAndGradientRoutine - Sets the user-defined regularizer call-back
  function into the algorithm.

  Input Parameters:
+ tao  - the Tao context
. func - function pointer for the regularizer value and gradient evaluation
- ctx  - user context for the regularizer

  Calling sequence:
+ tao - the `Tao` context
. u   - the location at which to compute the objective and gradient
. val - location to store objective function value
. g   - location to store gradient
- ctx - user context for the regularizer Hessian

  Level: advanced

.seealso: `Tao`, `Mat`, `TAOBRGN`, `TaoBRGNSetRegularizerTerm()`
@*/
PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine(Tao tao, PetscErrorCode (*func)(Tao tao, Vec u, PetscReal *val, Vec g, PetscCtx ctx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetRegularizerObjectiveAndGradientRoutine_C", (Tao, PetscErrorCode (*)(Tao, Vec, PetscReal *, Vec, PetscCtx), PetscCtx), (tao, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizerObjectiveAndGradientRoutine_BRGN(Tao tao, PetscErrorCode (*func)(Tao tao, Vec u, PetscReal *val, Vec g, PetscCtx ctx), PetscCtx ctx)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  if (ctx) gn->user_objgrad_ctx = ctx;
  if (func) gn->user_objgrad = func;
  gn->reg_type = TAOBRGN_REGULARIZATION_USER;
  if (tao->setupcalled) {
    TaoTerm gn_term, reg_term;
    Mat     map;

    PetscCall(TaoTermSumGetTerm(gn->subsolver->objective_term.term, 0, NULL, NULL, &gn_term, NULL));
    PetscCall(TaoBRGNCreateDefaultRegularizer(tao, gn_term, &reg_term, &map));
    PetscCall(TaoBRGNInstallRegularizerSubterm(tao, reg_term, map));
    if (gn->user_hessian_mat) PetscCall(TaoTermSumSetTermHessianMatrices(gn->subsolver->objective_term.term, 1, gn->user_hessian_mat, gn->user_hessian_mat, NULL, NULL));
    PetscCall(TaoTermDestroy(&reg_term));
    PetscCall(MatDestroy(&map));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoBRGNSetRegularizerHessianRoutine - Sets the user-defined regularizer call-back
  function into the algorithm.

  Input Parameters:
+ tao  - the `Tao` context
. Hreg - user-created matrix for the Hessian of the regularization term
. func - function pointer for the regularizer Hessian evaluation
- ctx  - user context for the regularizer Hessian

  Calling sequence:
+ tao  - the `Tao` context
. u    - the location at which to compute the Hessian
. Hreg - user-created matrix for the Hessian of the regularization term
- ctx  - user context for the regularizer Hessian

  Level: advanced

.seealso: `Tao`, `Mat`, `TAOBRGN`, `TaoBRGNSetRegularizerTerm()`
@*/
PetscErrorCode TaoBRGNSetRegularizerHessianRoutine(Tao tao, Mat Hreg, PetscErrorCode (*func)(Tao tao, Vec u, Mat Hreg, PetscCtx ctx), PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetRegularizerHessianRoutine_C", (Tao, Mat, PetscErrorCode (*)(Tao, Vec, Mat, PetscCtx), PetscCtx), (tao, Hreg, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizerHessianRoutine_BRGN(Tao tao, Mat Hreg, PetscErrorCode (*func)(Tao tao, Vec u, Mat Hreg, PetscCtx ctx), PetscCtx ctx)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCheck(Hreg, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "NULL Hessian detected! User must provide valid Hessian for the regularizer.");
  PetscValidHeaderSpecific(Hreg, MAT_CLASSID, 2);
  PetscCheckSameComm(tao, 1, Hreg, 2);
  if (ctx) gn->user_hessian_ctx = ctx;
  if (func) gn->user_hessian = func;
  PetscCall(PetscObjectReference((PetscObject)Hreg));
  PetscCall(MatDestroy(&gn->user_hessian_mat));
  gn->user_hessian_mat = Hreg;
  gn->reg_type         = TAOBRGN_REGULARIZATION_USER;
  if (tao->setupcalled) {
    TaoTerm reg;

    PetscCall(TaoTermSumGetTerm(gn->subsolver->objective_term.term, 1, NULL, NULL, &reg, NULL));
    PetscCall(TaoTermShellSetHessian(reg, TaoBRGNUserShimHessian));
    PetscCall(TaoTermSumSetTermHessianMatrices(gn->subsolver->objective_term.term, 1, Hreg, Hreg, NULL, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNSetRegularizerTerm - Set the regularization term in the `TAOBRGN` solver in the form $\lambda g(Ax;p)$

  Collective

  Input Parameters:
+ tao    - a `Tao` of type `TAOBRGN`
. scale  - the scalar $\lambda$ multiplying the regularization term
. term   - the `TaoTerm` $g$
. params - the parameters $p$ of the `TaoTerm` (NULL if the term has no parameters, see `TaoTermGetParametersMode()`)
- map    - the map $A$ of the regularization term (NULL if the map is the identity)

  Level: advanced

.seealso: [](ch_tao), [](sec_tao_term), `Tao`, `TAOBRGN`, `TaoTerm`, `TaoBRGNGetRegularizerTerm()`
@*/
PetscErrorCode TaoBRGNSetRegularizerTerm(Tao tao, PetscReal scale, TaoTerm term, Vec params, Mat map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 3);
  if (params) PetscValidHeaderSpecific(params, VEC_CLASSID, 4);
  if (map) PetscValidHeaderSpecific(map, MAT_CLASSID, 5);
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetRegularizerTerm_C", (Tao, PetscReal, TaoTerm, Vec, Mat), (tao, scale, term, params, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizerTerm_BRGN(Tao tao, PetscReal scale, TaoTerm term, Vec params, Mat map)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCheck(tao->setupcalled, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called before TaoBRGNSetRegularizerTerm()");
  gn->reg_type = TAOBRGN_REGULARIZATION_USER;
  gn->lambda   = scale;
  PetscCall(TaoTermSumSetTerm(gn->subsolver->objective_term.term, 1, BRGN_REG_PREFIX, scale, term, map));
  PetscCall(TaoTermSumSetTermMask(gn->subsolver->objective_term.term, 1, TAOTERM_MASK_NONE));
  if (params) {
    Vec sub_params[2] = {NULL, NULL};
    if (gn->subsolver->objective_parameters) PetscCall(TaoTermSumParametersUnpack(gn->subsolver->objective_term.term, &gn->subsolver->objective_parameters, sub_params));
    if (sub_params[1] != params) PetscCall(VecDestroy(&sub_params[1]));
    PetscCall(PetscObjectReference((PetscObject)params));
    sub_params[1] = params;
    PetscCall(TaoTermSumParametersPack(gn->subsolver->objective_term.term, sub_params, &gn->subsolver->objective_parameters));
    PetscCall(VecDestroy(&sub_params[0]));
    PetscCall(VecDestroy(&sub_params[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetRegularizerTerm - Get the regularization term in the `TAOBRGN` solver in the form $\lambda g(Ax;p)$

  Not collective

  Input Parameter:
. tao - a `Tao` of type `TAOBRGN`

  Output Parameters:
+ scale  - (optional) the scalar $\lambda$ multiplying the regularization term
. term   - (optional) the `TaoTerm` $g$
. params - (optional) the parameters $p$ of the `TaoTerm` (NULL if the term has no parameters, see `TaoTermGetParametersMode()`)
- map    - (optional) the map $A$ of the regularization term (NULL if the map is the identity)

  Level: advanced

.seealso: [](ch_tao), [](sec_tao_term), `Tao`, `TAOBRGN`, `TaoTerm`, `TaoBRGNSetRegularizerTerm()`
@*/
PetscErrorCode TaoBRGNGetRegularizerTerm(Tao tao, PetscReal *scale, TaoTerm *term, Vec *params, Mat *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscUseMethod((PetscObject)tao, "TaoBRGNGetRegularizerTerm_C", (Tao, PetscReal *, TaoTerm *, Vec *, Mat *), (tao, scale, term, params, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetRegularizerTerm_BRGN(Tao tao, PetscReal *scale, TaoTerm *term, Vec *params, Mat *map)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCheck(tao->setupcalled, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called before TaoBRGNGetRegularizerTerm()");
  PetscCall(TaoTermSumGetTerm(gn->subsolver->objective_term.term, 1, NULL, scale, term, map));
  if (params) {
    *params = NULL;
    if (gn->subsolver->objective_parameters) PetscCall(VecNestGetTaoTermSumParameters(gn->subsolver->objective_parameters, 1, params));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ----- Update hook + solve + setup ----- */

static PetscErrorCode GNHookFunction(Tao tao, PetscInt iter, PetscCtx ctx)
{
  TAO_BRGN *gn = (TAO_BRGN *)ctx;

  PetscFunctionBegin;
  /* Forward subsolver counters/state to the parent. */
  gn->parent->objective_term.term->nobj     = tao->objective_term.term->nobj;
  gn->parent->objective_term.term->ngrad    = tao->objective_term.term->ngrad;
  gn->parent->objective_term.term->nobjgrad = tao->objective_term.term->nobjgrad;
  gn->parent->objective_term.term->nhess    = tao->objective_term.term->nhess;
  gn->parent->nres                          = tao->nres;
  gn->parent->niter                         = tao->niter;
  gn->parent->ksp_its                       = tao->ksp_its;
  gn->parent->ksp_tot_its                   = tao->ksp_tot_its;
  gn->parent->fc                            = tao->fc;
  PetscCall(TaoGetConvergedReason(tao, &gn->parent->reason));
  if (iter > 0) PetscCall(VecCopy(tao->solution, gn->parent->solution));
  PetscCall(VecCopy(tao->gradient, gn->parent->gradient));

  /* LM lambda update. */
  if (gn->reg_type == TAOBRGN_REGULARIZATION_LM) {
    if (iter > 0) {
      const char *name;
      TaoTerm     reg;
      Mat         map;
      PetscReal   factor = (gn->fc_old > tao->fc) ? gn->downhill_lambda_change : gn->uphill_lambda_change;

      gn->lambda *= factor;
      PetscCall(TaoTermSumGetTerm(tao->objective_term.term, 1, &name, NULL, &reg, &map));
      PetscCall(TaoTermSumSetTerm(tao->objective_term.term, 1, name, gn->lambda, reg, map));
    }
    gn->fc_old = tao->fc;
  }

  /* L2PROX bias update. */
  if (gn->reg_type == TAOBRGN_REGULARIZATION_L2PROX && tao->objective_parameters) {
    Vec bias = NULL;

    PetscCall(VecNestGetTaoTermSumParameters(tao->objective_parameters, 1, &bias));
    if (bias) {
      if (iter == 0) PetscCall(VecZeroEntries(bias));
      else PetscCall(VecCopy(tao->solution, bias));
    }
  }

  PetscTryTypeMethod(gn->parent, update, gn->parent->niter, gn->parent->user_update);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_BRGN(Tao tao)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoSolve(gn->subsolver));
  tao->objective_term.term->nobj     = gn->subsolver->objective_term.term->nobj;
  tao->objective_term.term->ngrad    = gn->subsolver->objective_term.term->ngrad;
  tao->objective_term.term->nobjgrad = gn->subsolver->objective_term.term->nobjgrad;
  tao->objective_term.term->nhess    = gn->subsolver->objective_term.term->nhess;
  tao->nres                          = gn->subsolver->nres;
  tao->niter                         = gn->subsolver->niter;
  tao->ksp_its                       = gn->subsolver->ksp_its;
  tao->ksp_tot_its                   = gn->subsolver->ksp_tot_its;
  PetscCall(TaoGetConvergedReason(gn->subsolver, &tao->reason));
  PetscCall(VecCopy(gn->subsolver->solution, tao->solution));
  PetscCall(VecCopy(gn->subsolver->gradient, tao->gradient));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_BRGN(Tao tao, PetscOptionItems PetscOptionsObject)
{
  TAO_BRGN                 *gn               = (TAO_BRGN *)tao->data;
  PetscReal                 lambda           = gn->lambda;
  PetscReal                 epsilon          = gn->epsilon;
  TaoBRGNRegularizationType reg_type         = gn->reg_type;
  PetscBool                 mat_explicit_set = PETSC_FALSE;
  PetscBool                 mat_explicit     = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "least-squares problems with regularizer: ||f(x)||^2 + lambda*g(x), g(x) = ||xk-xkm1||^2 or ||Dx||_1 or user defined function.");
  PetscCall(PetscOptionsBool("-tao_brgn_mat_explicit", "(Removed) switches the Hessian construction to be an explicit matrix rather than MATSHELL", "TaoBRGN", mat_explicit, &mat_explicit, &mat_explicit_set));
  PetscCall(PetscOptionsReal("-tao_brgn_regularizer_weight", "regularizer weight (default 1e-4)", "TaoBRGNSetRegularizerWeight", lambda, &lambda, NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_l1_smooth_epsilon", "L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)", "TaoBRGNSetL1SmoothEpsilon", epsilon, &epsilon, NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_lm_downhill_lambda_change", "Factor to decrease trust region by on downhill steps", "", gn->downhill_lambda_change, &gn->downhill_lambda_change, NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_lm_uphill_lambda_change", "Factor to increase trust region by on uphill steps", "", gn->uphill_lambda_change, &gn->uphill_lambda_change, NULL));
  PetscCall(PetscOptionsEnum("-tao_brgn_regularization_type", "regularization type", "TaoBRGNSetRegularizationType", TaoBRGNRegularizationTypes, (PetscEnum)reg_type, (PetscEnum *)&reg_type, NULL));
  PetscOptionsHeadEnd();

  if (mat_explicit_set) PetscCall(PetscInfo(tao, "-tao_brgn_mat_explicit was removed. Use -<prefix>brgn_gauss_newton_tao_term_hessian_mat_type aij (or similar) instead.\n"));

  PetscCall(TaoBRGNSetRegularizerWeight(tao, lambda));
  if (epsilon != gn->epsilon) PetscCall(TaoBRGNSetL1SmoothEpsilon(tao, epsilon));
  if (reg_type != gn->reg_type) PetscCall(TaoBRGNSetRegularizationType(tao, reg_type));

  PetscCall(TaoSetFromOptions(gn->subsolver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_BRGN(Tao tao, PetscViewer viewer)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Regularizer weight: %g\n", (double)gn->lambda));
    PetscCall(PetscViewerASCIIPrintf(viewer, "BRGN Regularization Type: %s\n", TaoBRGNRegularizationTypes[gn->reg_type]));
    switch (gn->reg_type) {
    case TAOBRGN_REGULARIZATION_L1DICT:
      PetscCall(PetscViewerASCIIPrintf(viewer, "L1 smooth epsilon: %g\n", (double)gn->epsilon));
      break;
    case TAOBRGN_REGULARIZATION_LM:
      PetscCall(PetscViewerASCIIPrintf(viewer, "Downhill trust region decrease factor:: %g\n", (double)gn->downhill_lambda_change));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Uphill trust region increase factor:: %g\n", (double)gn->uphill_lambda_change));
      break;
    default:
      break;
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(TaoView(gn->subsolver, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_BRGN(Tao tao)
{
  TAO_BRGN   *gn = (TAO_BRGN *)tao->data;
  TaoTerm     gn_term, reg_term;
  Mat         map = NULL;
  const char *prefix;

  PetscFunctionBegin;
  PetscCheck(tao->ls_res, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetResidualRoutine() must be called before setup!");
  PetscCheck(tao->ls_jac, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetResidualJacobianRoutine() must be called before setup!");
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));

  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tao, &prefix));

  /* Build the Gauss-Newton term over the parent. */
  PetscCall(TaoTermCreateGaussNewton(tao, &gn_term));
  PetscCall(PetscObjectSetName((PetscObject)gn_term, "BRGN Gauss-Newton term"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)gn_term, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)gn_term, BRGN_GN_PREFIX));
  PetscCall(TaoAddTerm(gn->subsolver, BRGN_GN_PREFIX, 1.0, gn_term, NULL, NULL));

  /* Build the regularizer subterm. */
  PetscCall(TaoBRGNCreateDefaultRegularizer(tao, gn_term, &reg_term, &map));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)reg_term, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)reg_term, "brgn_regularizer_"));
  PetscCall(TaoAddTerm(gn->subsolver, BRGN_REG_PREFIX, gn->lambda, reg_term, NULL, map));
  if (gn->reg_type == TAOBRGN_REGULARIZATION_LM) PetscCall(TaoBRGNApplyLMSettings(tao));
  if (gn->reg_type == TAOBRGN_REGULARIZATION_USER && gn->user_hessian_mat) PetscCall(TaoTermSumSetTermHessianMatrices(gn->subsolver->objective_term.term, 1, gn->user_hessian_mat, gn->user_hessian_mat, NULL, NULL));

  PetscCall(TaoTermDestroy(&gn_term));
  PetscCall(TaoTermDestroy(&reg_term));
  PetscCall(MatDestroy(&map));

  /* Subsolver wiring. */
  PetscCall(TaoSetUpdate(gn->subsolver, GNHookFunction, gn));
  PetscCall(TaoSetSolution(gn->subsolver, tao->solution));
  if (tao->bounded) PetscCall(TaoSetVariableBounds(gn->subsolver, tao->XL, tao->XU));
  PetscCall(TaoSetTolerances(gn->subsolver, tao->gatol, tao->grtol, tao->gttol));
  PetscCall(TaoSetMaximumIterations(gn->subsolver, tao->max_it));
  PetscCall(TaoSetMaximumFunctionEvaluations(gn->subsolver, tao->max_funcs));
  PetscCall(TaoSetUp(gn->subsolver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_BRGN(Tao tao)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&tao->gradient));
  PetscCall(VecDestroy(&gn->damping));
  PetscCall(MatDestroy(&gn->D));
  PetscCall(MatDestroy(&gn->user_hessian_mat));
  PetscCall(TaoDestroy(&gn->subsolver));
  gn->parent = NULL;
  PetscCall(PetscFree(tao->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizationType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizationType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetDampingVector_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetDictionaryMatrix_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetSubsolver_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerWeight_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetL1SmoothEpsilon_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerObjectiveAndGradientRoutine_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerHessianRoutine_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerTerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizerTerm_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOBRGN - Bounded Regularized Gauss-Newton method for solving nonlinear least-squares
            problems with bound constraints. This algorithm is a thin wrapper around `TAOBNLS`
            that constructs the Gauss-Newton problem from the user-provided least-squares
            residual and Jacobian (set with `TaoSetResidualRoutine()` and
            `TaoSetJacobianResidualRoutine()`) and composes it with a regularizer term using
            `TaoAddTerm()` on the subsolver. The algorithm offers an L2-norm ("l2pure"),
            L2-norm proximal point ("l2prox") regularizer, and L1-norm dictionary regularizer
            ("l1dict"), where we approximate the L1-norm $\|x\|_1$ by
            $\sum_i(\sqrt{x_i^2+\epsilon^2}-\epsilon)$ with a small positive number $\epsilon$.
            Also offered is the "lm" regularizer which uses a scaled diagonal of $J^T J$.
            With the "lm" regularizer, `TAOBRGN` is a Levenberg-Marquardt optimizer.
            The user can also provide an arbitrary `TaoTerm` regularizer with
            `TaoBRGNSetRegularizerTerm()` (preferred) or use the legacy callback setters
            `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()` and
            `TaoBRGNSetRegularizerHessianRoutine()`.

  Options Database Keys:
+ -tao_brgn_regularization_type - regularization type ("user", "l2prox", "l2pure", "l1dict", "lm") (default "l2prox")
. -tao_brgn_regularizer_weight  - regularizer weight (default 1e-4)
- -tao_brgn_l1_smooth_epsilon   - L1-norm smooth approximation parameter (default 1e-6)

  Level: beginner

  Note:
  Internally `TAOBRGN` builds two `TaoTerm`s on the subsolver: a `TAOTERMGAUSSNEWTON`
  for $\tfrac{1}{2}\|R(x)\|_2^2$ and a regularizer term scaled by lambda.  Use
  `-<prefix>brgn_gauss_newton_tao_term_hessian_mat_type aij` to assemble the
  Gauss-Newton Hessian explicitly (the previous `-tao_brgn_mat_explicit` flag has
  been removed).

.seealso: `Tao`, `TaoBRGNGetSubsolver()`, `TaoBRGNSetRegularizerWeight()`, `TaoBRGNSetL1SmoothEpsilon()`, `TaoBRGNSetDictionaryMatrix()`,
          `TaoBRGNSetRegularizerTerm()`, `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()`, `TaoBRGNSetRegularizerHessianRoutine()`,
          `TAOTERMGAUSSNEWTON`
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BRGN(Tao tao)
{
  TAO_BRGN   *gn;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gn));

  tao->ops->destroy        = TaoDestroy_BRGN;
  tao->ops->setup          = TaoSetUp_BRGN;
  tao->ops->setfromoptions = TaoSetFromOptions_BRGN;
  tao->ops->view           = TaoView_BRGN;
  tao->ops->solve          = TaoSolve_BRGN;
  tao->uses_gradient       = PETSC_TRUE;

  PetscCall(TaoParametersInitialize(tao));

  tao->data                  = gn;
  gn->reg_type               = TAOBRGN_REGULARIZATION_L2PROX;
  gn->lambda                 = 1e-4;
  gn->epsilon                = 1e-6;
  gn->downhill_lambda_change = 1. / 5.;
  gn->uphill_lambda_change   = 1.5;
  gn->parent                 = tao;

  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tao, &prefix));
  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &gn->subsolver));
  PetscCall(TaoSetType(gn->subsolver, TAOBNLS));
  PetscCall(TaoSetOptionsPrefix(gn->subsolver, prefix));
  PetscCall(TaoAppendOptionsPrefix(gn->subsolver, "tao_brgn_subsolver_"));

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizationType_C", TaoBRGNGetRegularizationType_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizationType_C", TaoBRGNSetRegularizationType_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetDampingVector_C", TaoBRGNGetDampingVector_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetDictionaryMatrix_C", TaoBRGNSetDictionaryMatrix_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetSubsolver_C", TaoBRGNGetSubsolver_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerWeight_C", TaoBRGNSetRegularizerWeight_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetL1SmoothEpsilon_C", TaoBRGNSetL1SmoothEpsilon_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerObjectiveAndGradientRoutine_C", TaoBRGNSetRegularizerObjectiveAndGradientRoutine_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerHessianRoutine_C", TaoBRGNSetRegularizerHessianRoutine_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerTerm_C", TaoBRGNSetRegularizerTerm_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizerTerm_C", TaoBRGNGetRegularizerTerm_BRGN));
  PetscFunctionReturn(PETSC_SUCCESS);
}
