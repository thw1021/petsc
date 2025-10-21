#include <../src/tao/leastsquares/impls/brgn/brgn.h> /*I "petsctao.h" I*/

const char *const TaoBRGNRegularizationTypes[] = {"user", "l2prox", "l2pure", "l1dict", "lm", "TaoBRGNRegularizationType", "TAOBRGN_REGULARIZATION_", NULL};

// A TAOTERMSHELL for the Gauss-Newton term
typedef struct _n_TaoTerm_GaussNewton TaoTerm_GaussNewton;

struct _n_TaoTerm_GaussNewton {
  Tao      tao;   // weak-reference
  PetscInt refct; // because this can be reference from the Gauss-Newton and the Levenberg-Marquardt terms

  PetscObjectId    x_id; // data for recomputing the residual Jacobian J
  PetscObjectState x_state;

  PetscObjectId    lm_x_id; // data for recomputing diag(J^T J) for the Levenberg-Marquardt regularizer
  PetscObjectState lm_x_state;

  Vec r_work; // intermediate result vector for storing Jv when computing (J^T J)v
};

static PetscErrorCode TaoTermDestroy_GaussNewton(void **ctx)
{
  TaoTerm_GaussNewton *gnterm = (TaoTerm_GaussNewton *)*ctx;

  PetscFunctionBegin;
  if (--gnterm->refct > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecDestroy(&gnterm->r_work));
  PetscCall(PetscFree(gnterm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// When x has changed, recompute the Jacobian matrix of the residual
static PetscErrorCode TaoTermGaussNewtonGetJacobian(TaoTerm_GaussNewton *gnterm, Vec x, Mat *ls_jac)
{
  PetscObjectId    x_id;
  PetscObjectState x_state;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (x_id != gnterm->x_id || x_state != gnterm->x_state) {
    gnterm->x_id    = x_id;
    gnterm->x_state = x_state;
    PetscCall(TaoComputeResidualJacobian(gnterm->tao, x, gnterm->tao->ls_jac, gnterm->tao->ls_jac_pre));
  }
  *ls_jac = gnterm->tao->ls_jac;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// When x has changed, recompute diag(J^T J)
static PetscErrorCode TaoTermLevenbergMarquardtComputeDiagonal(TaoTerm_GaussNewton *gnterm, Vec x, Vec diag)
{
  PetscObjectId    x_id;
  PetscObjectState x_state;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (x_id != gnterm->lm_x_id || x_state != gnterm->lm_x_state) {
    Mat          ls_jac;
    PetscInt     n, cstart, cend;
    PetscReal   *cnorms;
    PetscScalar *diag_ary;

    gnterm->lm_x_id    = x_id;
    gnterm->lm_x_state = x_state;
    PetscCall(TaoTermGaussNewtonGetJacobian(gnterm, x, &ls_jac));
    PetscCall(MatGetSize(ls_jac, NULL, &n));
    PetscCall(PetscMalloc1(n, &cnorms));
    PetscCall(MatGetColumnNorms(ls_jac, NORM_2, cnorms));
    PetscCall(MatGetOwnershipRangeColumn(ls_jac, &cstart, &cend));
    PetscCall(VecGetArray(diag, &diag_ary));
    for (PetscInt i = 0; i < cend - cstart; i++) {
      PetscReal v = cnorms[cstart + i] * cnorms[cstart + i];

      diag_ary[i] = PetscClipInterval(v, PETSC_SQRT_MACHINE_EPSILON, PetscSqrtReal(PETSC_MAX_REAL));
    }
    PetscCall(VecRestoreArray(diag, &diag_ary));
    PetscCall(PetscFree(cnorms));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// (1/2) || R(x) ||_2^2, where R is the residual function
static PetscErrorCode TaoTermComputeObjectiveAndGradient_GaussNewton(TaoTerm term, Vec x, Vec _params, PetscReal *value, Vec g)
{
  PetscScalar          sval;
  TaoTerm_GaussNewton *gnterm;
  Tao                  tao;
  Mat                  ls_jac;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, (void *)&gnterm));
  tao = gnterm->tao;
  PetscCall(TaoComputeResidual(tao, x, tao->ls_res));
  PetscCall(VecDot(tao->ls_res, tao->ls_res, &sval));
  *value = 0.5 * PetscRealPart(sval);
  PetscCall(TaoTermGaussNewtonGetJacobian(gnterm, x, &ls_jac));
  PetscCall(MatMultTranspose(ls_jac, tao->ls_res, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Form H = (J^T J)
static PetscErrorCode TaoTermComputeHessian_GaussNewton_Internal(TaoTerm term, Vec x, Vec _params, Mat H)
{
  TaoTerm_GaussNewton *gnterm;
  Mat                  ls_jac = NULL;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, (void *)&gnterm));
  PetscCall(TaoTermGaussNewtonGetJacobian(gnterm, x, &ls_jac));
  PetscCall(MatTransposeMatMult(ls_jac, ls_jac, MAT_REUSE_MATRIX, PETSC_DETERMINE, &H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_GaussNewton(TaoTerm term, Vec x, Vec _params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  if (H) PetscCall(TaoTermComputeHessian_GaussNewton_Internal(term, x, _params, H));
  if (Hpre && Hpre != H) {
    if (H) PetscCall(MatCopy(H, Hpre, UNKNOWN_NONZERO_PATTERN));
    else PetscCall(TaoTermComputeHessian_GaussNewton_Internal(term, x, _params, Hpre));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateGaussNewton(Tao tao, TaoTerm *term)
{
  TaoTerm_GaussNewton *gnterm;
  TaoTerm              _term;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gnterm));
  gnterm->tao     = tao;
  gnterm->refct   = 1;
  gnterm->x_id    = 0;
  gnterm->x_state = 0;
  PetscCall(TaoTermCreateShell(PetscObjectComm((PetscObject)tao), (void *)gnterm, TaoTermDestroy_GaussNewton, &_term));
  PetscCall(TaoTermSetParametersMode(_term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermShellSetCreateHessianMatrices(_term, TaoTermCreateHessianMatricesDefault));
  PetscCall(TaoTermSetCreateHessianMode(_term, PETSC_TRUE, MATSHELL, MATSHELL));
  PetscCall(TaoTermShellSetObjectiveAndGradient(_term, TaoTermComputeObjectiveAndGradient_GaussNewton));
  PetscCall(TaoTermShellSetHessian(_term, TaoTermComputeHessian_GaussNewton));
  *term = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* For the Levenberg-Marquardt regularizer diag(J^T * J), we create a quadratic TaoTerm
   and update the diagonal by overriding the Hessian routines */
static PetscErrorCode TaoTermUpdateLevenbergMarquardt(TaoTerm term, Vec x)
{
  PetscContainer       gncontainer;
  TaoTerm_GaussNewton *gnterm;
  Mat                  diag_mat;
  Vec                  diag_vec;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)term, "__TaoTermCreateLevenbergMarquardt", (PetscObject *)&gncontainer));
  PetscCall(PetscContainerGetPointer(gncontainer, (void **)&gnterm));
  PetscCall(TaoTermQuadraticGetMat(term, &diag_mat));
  PetscCall(MatDiagonalGetDiagonal(diag_mat, &diag_vec));
  PetscCall(TaoTermLevenbergMarquardtComputeDiagonal(gnterm, x, diag_vec));
  PetscCall(MatDiagonalRestoreDiagonal(diag_mat, &diag_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_LevenbergMarquardt(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  PetscCall(TaoTermUpdateLevenbergMarquardt(term, x));
  PetscCall(TaoTermComputeHessian_Quadratic(term, x, params, H, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateLevenbergMarquardt(TaoTerm gn, TaoTerm *lm)
{
  PetscContainer       gncontainer;
  TaoTerm_GaussNewton *gnterm;
  TaoTerm              _term;
  Mat                  diag_mat;
  Vec                  diag_vec;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(gn, (void *)&gnterm));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)gn), &gncontainer));
  gnterm->refct++;
  PetscCall(PetscContainerSetPointer(gncontainer, (void *)gnterm));
  PetscCall(PetscContainerSetCtxDestroy(gncontainer, TaoTermDestroy_GaussNewton));
  PetscCall(TaoTermDuplicate(gn, TAOTERM_DUPLICATE_SIZEONLY, &_term));
  PetscCall(TaoTermSetType(_term, TAOTERMQUADRATIC));
  PetscCall(VecDuplicate(gnterm->tao->solution, &diag_vec));
  PetscCall(MatCreateDiagonal(diag_vec, &diag_mat));
  PetscCall(VecDestroy(&diag_vec));
  PetscCall(TaoTermQuadraticSetMat(_term, diag_mat));
  PetscCall(MatDestroy(&diag_mat));
  PetscCall(PetscObjectCompose((PetscObject)_term, "__TaoTermCreateLevenbergMarquardt", (PetscObject)gncontainer));
  PetscCall(PetscContainerDestroy(&gncontainer));
  _term->ops->hessian     = TaoTermComputeHessian_LevenbergMarquardt;
  *lm                     = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

.seealso: [](ch_tao), `Tao`, `TAOBRGN`, `TaoBRGNRegularizationType`, `TaoBRGNGetRegularizationType`
@*/
PetscErrorCode TaoBRGNSetRegularizationType(Tao tao, TaoBRGNRegularizationType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetRegularizationType_C", (Tao, TaoBRGNRegularizationType), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNCreateRegularizerTerm(Tao tao, TaoTerm *term, Mat *map, Mat *H)
{
  char        name[256];
  const char *prefix;
  TAO_BRGN   *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  *H = NULL;
  switch (gn->reg_type) {
  case TAOBRGN_REGULARIZATION_L1DICT:
    PetscCall(PetscObjectReference((PetscObject)gn->D));
    *map = gn->D;
    if (gn->D) {
      Vec output;

      PetscCall(MatCreateVecs(gn->D, NULL, &output));
      PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), term));
      PetscCall(TaoTermSetSolutionTemplate(*term, output));
      PetscCall(VecDestroy(&output));
    } else {
      PetscCall(TaoTermDuplicate(tao->orig_callbacks, TAOTERM_DUPLICATE_SIZEONLY, term));
    }
    PetscCall(TaoTermSetType(*term, TAOTERML1));
    PetscCall(TaoTermL1SetEpsilon(*term, gn->epsilon));
    break;
  case TAOBRGN_REGULARIZATION_L2PURE:
  case TAOBRGN_REGULARIZATION_L2PROX:
    PetscCall(TaoTermDuplicate(tao->orig_callbacks, TAOTERM_DUPLICATE_SIZEONLY, term));
    PetscCall(TaoTermSetType(*term, TAOTERMHALFL2SQUARED));
    *map = NULL;
    break;
  case TAOBRGN_REGULARIZATION_LM: {
    TaoTerm gnterm;
    Mat     mat_diag;

    PetscCall(TaoTermSumGetSubterm(gn->subsolver->objective_term.term, 0, NULL, NULL, &gnterm, NULL));
    PetscCall(TaoTermCreateLevenbergMarquardt(gnterm, term));
    PetscCall(TaoTermQuadraticGetMat(*term, &mat_diag));
    PetscCall(PetscObjectReference((PetscObject)mat_diag));
    *H   = mat_diag;
    *map = NULL;
  } break;
  default:
    break;
  }
  PetscCall(PetscSNPrintf(name, PETSC_STATIC_ARRAY_LENGTH(name), "BRGN regularizer %s", TaoBRGNRegularizationTypes[gn->reg_type]));
  PetscCall(PetscObjectSetName((PetscObject)*term, name));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tao, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*term, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)*term, "brgn_regularizer_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizationType_BRGN(Tao tao, TaoBRGNRegularizationType type)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  if (type != gn->reg_type && type != TAOBRGN_REGULARIZATION_USER) {
    const char *name;
    TaoTerm     reg_term;
    Mat         reg_map, Hreg;
    PetscReal   scale;

    gn->reg_type = type;
    PetscCall(TaoBRGNCreateRegularizerTerm(tao, &reg_term, &reg_map, &Hreg));
    PetscCall(TaoTermSumGetSubterm(gn->subsolver->objective_term.term, 1, &name, &scale, NULL, NULL));
    PetscCall(TaoTermSumSetSubterm(gn->subsolver->objective_term.term, 1, name, scale, reg_term, reg_map));
    PetscCall(TaoTermSumSetSubtermHessianMatrices(gn->subsolver->objective_term.term, 1, Hreg, Hreg, NULL, NULL));
    PetscCall(TaoTermDestroy(&reg_term));
    PetscCall(MatDestroy(&reg_map));
    PetscCall(MatDestroy(&Hreg));
  }
  if (gn->reg_type == TAOBRGN_REGULARIZATION_LM) {
    TaoLineSearch ls;

    PetscCall(TaoGetLineSearch(gn->subsolver, &ls));
    PetscCall(TaoLineSearchSetType(ls, TAOLINESEARCHUNIT));
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
  PetscUseMethod((PetscObject)tao, "TaoBRGNGetDampingVector_C", (Tao, Vec *), (tao, d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetDampingVector_BRGN(Tao tao, Vec *d)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;
  TaoTerm   reg;
  Mat       reg_mat;
  Vec       reg_vec;

  PetscFunctionBegin;
  PetscCheck(gn->reg_type == TAOBRGN_REGULARIZATION_LM, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "Damping vector is only available if regularization type is lm.");
  PetscCall(TaoBRGNGetRegularizerTerm(tao, NULL, &reg, NULL, NULL));
  PetscCall(TaoTermQuadraticGetMat(reg, &reg_mat));
  PetscCall(MatDiagonalGetDiagonal(reg_mat, &reg_vec));
  if (!gn->damping) PetscCall(VecDuplicate(reg_vec, &gn->damping));
  PetscCall(VecCopy(reg_vec, gn->damping));
  *d = gn->damping;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetRegularizerWeight(Tao tao, PetscReal *lambda)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoTermSumGetSubterm(gn->subsolver->objective_term.term, 1, NULL, lambda, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetRegularizerTerm - Get the regularization term in the `TAOBRGN` solver in the form $\lambda g(Ax;p)$

  Not collective

  Input Parameter:
. tao - a `Tao` of type `TAOBRGN`

  Output Parameters:
+ scale  - the scalar $\lambda$ multiplying the regularization term
. term   - the `TaoTerm` $g$
. params - the parameters $p$ of the `TaoTerm` (NULL if the term has no parameters, see `TaoTermGetParametersMode()`)
- map    - the map $A$ of the regularization term (NULL if the map is the identity)

  Level: advanced

  Note:
  The parameters $p$ have different interpretations depending on $g$.  If $g$
  is one of the built-in terms `TAOTERMHALFL2SQUARED`, `TAOTERML1`, or
  `TAOTERMQUADRATIC`, then $p$ the bias of the regularizer, i.e. the penalty is
  $\tfrac{\lambda}{2}\|x - p\|_2^2$, $\lambda \|x - p\|_1$, or
  $\tfrac{\lambda}{2}(x-p)^T Q (x - p)$, respectively.

.seealso: [](ch_tao), [](sec_tao_term), `Tao`, `TAOBRGN`, `TaoTerm`, `TaoBRGNSetRegularizerTerm()`
@*/
PetscErrorCode TaoBRGNGetRegularizerTerm(Tao tao, PetscReal *scale, TaoTerm *term, Vec *params, Mat *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscUseMethod(tao, "TaoBRGNGetRegularizerTerm_C", (Tao, PetscReal *, TaoTerm *, Vec *, Mat *), (tao, scale, term, params, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetRegularizerTerm_BRGN(Tao tao, PetscReal *scale, TaoTerm *term, Vec *params, Mat *map)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoTermSumGetSubterm(gn->subsolver->objective_term.term, 1, NULL, scale, term, map));
  if (params) {
    *params = NULL;

    if (gn->subsolver->objective_parameters) PetscCall(VecNestGetTaoTermSumSubParameters(gn->subsolver->objective_parameters, 1, params));
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
  PetscUseMethod(tao, "TaoBRGNSetRegularizerTerm_C", (Tao, PetscReal, TaoTerm, Vec, Mat), (tao, scale, term, params, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizerTerm_Internal(Tao tao, PetscReal scale, TaoTerm term, Vec params, Mat map, PetscBool user_call)
{
  TAO_BRGN   *gn = (TAO_BRGN *)tao->data;
  const char *name;
  TaoTerm     current_term;
  Vec         sub_params[2] = {NULL, NULL};

  PetscFunctionBegin;
  PetscCall(TaoTermSumGetSubterm(gn->subsolver->objective_term.term, 1, &name, NULL, &current_term, NULL));
  if (user_call && term != current_term) gn->reg_type = TAOBRGN_REGULARIZATION_USER;
  PetscCall(TaoTermSumSetSubterm(gn->subsolver->objective_term.term, 1, name, scale, term, map));
  if (gn->subsolver->objective_parameters) PetscCall(TaoTermSumParametersUnpack(gn->subsolver->objective_term.term, &gn->subsolver->objective_parameters, sub_params));
  if (sub_params[1] != params) PetscCall(VecDestroy(&sub_params[1]));
  sub_params[1] = params;
  if (sub_params[0] || sub_params[1]) PetscCall(TaoTermSumParametersPack(gn->subsolver->objective_term.term, sub_params, &gn->subsolver->objective_parameters));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizerTerm_BRGN(Tao tao, PetscReal scale, TaoTerm term, Vec params, Mat map)
{
  PetscFunctionBegin;
  PetscCall(TaoBRGNSetRegularizerTerm_Internal(tao, scale, term, params, map, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetSubsolver - Get the pointer to the subsolver inside a `TAOBRGN`

  Collective

  Input Parameters:
+ tao       - the Tao solver context
- subsolver - the `Tao` sub-solver context

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
- lambda - L1-norm regularizer weight

  Level: beginner

.seealso: `Tao`, `Mat`, `TAOBRGN`
@*/
PetscErrorCode TaoBRGNSetRegularizerWeight(Tao tao, PetscReal lambda)
{
  PetscFunctionBegin;
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetRegularizerWeight_C", (Tao, PetscReal), (tao, lambda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetRegularizerWeight_BRGN(Tao tao, PetscReal lambda)
{
  TAO_BRGN   *gn = (TAO_BRGN *)tao->data;
  const char *prefix;
  TaoTerm     term;
  Mat         map;

  /* Initialize lambda here */

  PetscFunctionBegin;
  PetscCall(TaoTermSumGetSubterm(gn->subsolver->objective_term.term, 1, &prefix, NULL, &term, &map));
  PetscCall(TaoTermSumSetSubterm(gn->subsolver->objective_term.term, 1, prefix, lambda, term, map));
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
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetL1SmoothEpsilon_C", (Tao, PetscReal), (tao, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetL1SmoothEpsilon_BRGN(Tao tao, PetscReal epsilon)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;
  TaoTerm   term;

  /* Initialize epsilon here */
  PetscFunctionBegin;
  PetscCall(TaoBRGNGetRegularizerTerm(tao, NULL, &term, NULL, NULL));
  gn->epsilon = epsilon;
  PetscCall(TaoTermL1SetEpsilon(term, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNSetDictionaryMatrix - bind the dictionary matrix from user application context to gn->D, for compressed sensing (with least-squares problem)

  Input Parameters:
+ tao  - the `Tao` context
- dict - the user specified dictionary matrix.  We allow to set a `NULL` dictionary, which means identity matrix by default

  Level: advanced

.seealso: `Tao`, `Mat`, `TAOBRGN`
@*/
PetscErrorCode TaoBRGNSetDictionaryMatrix(Tao tao, Mat dict)
{
  PetscFunctionBegin;
  PetscTryMethod((PetscObject)tao, "TaoBRGNSetDictionaryMatrix_C", (Tao, Mat), (tao, dict));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetDictionaryMatrix_BRGN(Tao tao, Mat dict)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (dict) {
    PetscValidHeaderSpecific(dict, MAT_CLASSID, 2);
    PetscCheckSameComm(tao, 1, dict, 2);
    PetscCall(PetscObjectReference((PetscObject)dict));
  }
  PetscCall(MatDestroy(&gn->D));
  gn->D = dict;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_BRGN(Tao tao)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&tao->gradient));
  PetscCall(VecDestroy(&gn->damping));
  PetscCall(VecDestroy(&gn->y));
  PetscCall(MatDestroy(&gn->D));
  PetscCall(MatDestroy(&gn->Hreg));
  PetscCall(TaoDestroy(&gn->subsolver));
  gn->parent = NULL;
  PetscCall(PetscFree(tao->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizerTerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerTerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizationType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizationType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetDampingVector_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetSubsolver_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerWeight_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetL1SmoothEpsilon_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetDictionaryMatrix_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GNHookFunction(Tao tao, PetscInt iter, void *ctx)
{
  TAO_BRGN *gn = (TAO_BRGN *)ctx;

  PetscFunctionBegin;
  /* Update basic tao information from the subsolver */
  gn->parent->nfuncs      = tao->nfuncs;
  gn->parent->ngrads      = tao->ngrads;
  gn->parent->nfuncgrads  = tao->nfuncgrads;
  gn->parent->nhess       = tao->nhess;
  gn->parent->niter       = tao->niter;
  gn->parent->ksp_its     = tao->ksp_its;
  gn->parent->ksp_tot_its = tao->ksp_tot_its;
  gn->parent->fc          = tao->fc;
  PetscCall(TaoGetConvergedReason(tao, &gn->parent->reason));
  /* Update the solution vectors */
  if (iter > 0) PetscCall(VecCopy(tao->solution, gn->parent->solution));
  /* Update the gradient */
  PetscCall(VecCopy(tao->gradient, gn->parent->gradient));

  /* Update damping parameter for LM */
  if (gn->reg_type == TAOBRGN_REGULARIZATION_LM) {
    if (iter > 0) {
      PetscReal lambda;

      PetscCall(TaoBRGNGetRegularizerWeight(gn->parent, &lambda));
      if (gn->fc_old > tao->fc) {
        lambda *= gn->downhill_lambda_change;
      } else {
        /* uphill step */
        lambda *= gn->uphill_lambda_change;
      }
      PetscCall(TaoBRGNSetRegularizerWeight(gn->parent, lambda));
    }
    gn->fc_old = tao->fc;
  }

  if (gn->reg_type == TAOBRGN_REGULARIZATION_L2PROX) {
    Vec bias;

    PetscCall(TaoBRGNGetRegularizerTerm(gn->parent, NULL, NULL, &bias, NULL));
    if (iter == 0) {
      PetscCall(VecZeroEntries(bias));
    } else {
      PetscCall(VecCopy(tao->solution, bias));
    }
  }

  /* Call general purpose update function */
  if (gn->parent->ops->update) PetscCall((*gn->parent->ops->update)(gn->parent, gn->parent->niter, gn->parent->user_update));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_BRGN(Tao tao)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoSolve(gn->subsolver));
  /* Update basic tao information from the subsolver */
  tao->nfuncs      = gn->subsolver->nfuncs;
  tao->ngrads      = gn->subsolver->ngrads;
  tao->nfuncgrads  = gn->subsolver->nfuncgrads;
  tao->nhess       = gn->subsolver->nhess;
  tao->niter       = gn->subsolver->niter;
  tao->ksp_its     = gn->subsolver->ksp_its;
  tao->ksp_tot_its = gn->subsolver->ksp_tot_its;
  PetscCall(TaoGetConvergedReason(gn->subsolver, &tao->reason));
  /* Update vectors */
  PetscCall(VecCopy(gn->subsolver->solution, tao->solution));
  PetscCall(VecCopy(gn->subsolver->gradient, tao->gradient));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_BRGN(Tao tao, PetscOptionItems PetscOptionsObject)
{
  TAO_BRGN                 *gn = (TAO_BRGN *)tao->data;
  TaoLineSearch             ls;
  PetscReal                 lambda;
  PetscReal                 epsilon  = gn->epsilon;
  TaoBRGNRegularizationType reg_type = gn->reg_type;

  PetscFunctionBegin;
  PetscCall(TaoBRGNGetRegularizerWeight(tao, &lambda));
  PetscOptionsHeadBegin(PetscOptionsObject, "least-squares problems with regularizer: ||f(x)||^2 + lambda*g(x), g(x) = ||xk-xkm1||^2 or ||Dx||_1 or user defined function.");
  PetscCall(PetscOptionsReal("-tao_brgn_regularizer_weight", "regularizer weight (default 1e-4)", "", lambda, &lambda, NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_l1_smooth_epsilon", "L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)", "", epsilon, &epsilon, NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_lm_downhill_lambda_change", "Factor to decrease trust region by on downhill steps", "", gn->downhill_lambda_change, &gn->downhill_lambda_change, NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_lm_uphill_lambda_change", "Factor to increase trust region by on uphill steps", "", gn->uphill_lambda_change, &gn->uphill_lambda_change, NULL));
  PetscCall(PetscOptionsEnum("-tao_brgn_regularization_type", "regularization type", "", TaoBRGNRegularizationTypes, (PetscEnum)reg_type, (PetscEnum *)&reg_type, NULL));
  PetscOptionsHeadEnd();
  PetscCall(TaoBRGNSetRegularizerWeight(tao, lambda));
  /* set unit line search direction as the default when using the lm regularizer */
  if (reg_type != gn->reg_type) PetscCall(TaoBRGNSetRegularizationType(tao, reg_type));
  if (gn->reg_type == TAOBRGN_REGULARIZATION_LM) {
    PetscCall(TaoGetLineSearch(gn->subsolver, &ls));
    PetscCall(TaoLineSearchSetType(ls, TAOLINESEARCHUNIT));
  }
  if (epsilon != gn->epsilon) PetscCall(TaoBRGNSetL1SmoothEpsilon(tao, epsilon));

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
    PetscCall(PetscViewerASCIIPrintf(viewer, "BRGN Regularization Type: %s\n", TaoBRGNRegularizationTypes[gn->reg_type]));
    switch (gn->reg_type) {
    case TAOBRGN_REGULARIZATION_L1DICT:
      PetscCall(PetscViewerASCIIPrintf(viewer, "L1 smooth epsilon: %g\n", (double)gn->epsilon));
      break;
    case TAOBRGN_REGULARIZATION_LM:
      PetscCall(PetscViewerASCIIPrintf(viewer, "Downhill trust region decrease factor:: %g\n", (double)gn->downhill_lambda_change));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Uphill trust region increase factor:: %g\n", (double)gn->uphill_lambda_change));
      break;
    case TAOBRGN_REGULARIZATION_L2PROX:
    case TAOBRGN_REGULARIZATION_L2PURE:
    case TAOBRGN_REGULARIZATION_USER:
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
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;
  Mat       H;

  PetscFunctionBegin;
  PetscCheck(tao->ls_res, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetResidualRoutine() must be called before setup!");
  PetscCheck(!gn->subsolver->uses_hessian_matrices || tao->ls_jac, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetResidualJacobianRoutine() must be called before setup!");
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));

  PetscCall(TaoTermSetSolutionTemplate(gn->subsolver->objective_term.term, tao->solution));
  for (PetscInt i = 0; i < 2; i++) {
    TaoTerm term;
    Mat     map;

    PetscCall(TaoTermSumGetSubterm(gn->subsolver->objective_term.term, i, NULL, NULL, &term, &map));
    if (!map) PetscCall(TaoTermSetSolutionTemplate(term, tao->solution));
  }

  if (gn->reg_type == TAOBRGN_REGULARIZATION_LM) PetscCall(TaoTermSumSetSubtermMask(gn->subsolver->objective_term.term, 1, (TaoTermMask)(TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_GRADIENT)));
  else PetscCall(TaoTermSumSetSubtermMask(gn->subsolver->objective_term.term, 1, TAOTERM_MASK_NONE));

  if (gn->reg_type == TAOBRGN_REGULARIZATION_L2PROX) {
    Vec       x_old;
    TaoTerm   term;
    PetscReal scale;
    Mat       map;

    PetscCall(VecDuplicate(tao->solution, &x_old));
    PetscCall(VecZeroEntries(x_old));
    PetscCall(TaoBRGNGetRegularizerTerm(tao, &scale, &term, NULL, &map));
    PetscCall(TaoBRGNSetRegularizerTerm_Internal(tao, scale, term, x_old, map, PETSC_FALSE));
    PetscCall(VecDestroy(&x_old));
  }

  /* Hessian setup */
  {
    TaoTerm   reg_term;
    MatType   H_type, Hpre_type;
    PetscBool H_is_Hpre, is_shell;

    PetscCall(TaoComputeResidualJacobian(tao, tao->solution, tao->ls_jac, tao->ls_jac_pre));
    PetscCall(MatTransposeMatMult(tao->ls_jac, tao->ls_jac, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &H));
    PetscCall(TaoBRGNGetRegularizerTerm(tao, NULL, &reg_term, NULL, NULL));
    PetscCall(TaoTermGetCreateHessianMode(reg_term, &H_is_Hpre, &H_type, &Hpre_type));
    PetscCall(PetscStrcmp(H_type, MATSHELL, &is_shell));
    if (is_shell) H_type = Hpre_type;
    if (H_type) PetscCall(PetscStrcmp(H_type, MATSHELL, &is_shell));
    if (!H_type || is_shell) {
      switch (gn->reg_type) {
      case TAOBRGN_REGULARIZATION_L1DICT:
        H_type = MATDIAGONAL;
        break;
      case TAOBRGN_REGULARIZATION_L2PROX:
      case TAOBRGN_REGULARIZATION_L2PURE:
        H_type = MATCONSTANTDIAGONAL;
        break;
      default:
        H_type = MATAIJ;
        break;
      }
    }
    PetscCall(TaoTermSetCreateHessianMode(reg_term, H_is_Hpre, H_type, Hpre_type));
  }
  PetscCall(MatSetUp(H));
  PetscCall(TaoSetHessianMatrices(gn->subsolver, H, H));//TODO
  //PetscCall(TaoTermSumSetSubtermHessianMatrices(gn->subsolver->objective_term.term, 1, H, H, NULL, NULL));
  PetscCall(MatDestroy(&H));
  /* Subsolver setup,include initial vector and dictionary D */
  PetscCall(TaoSetUpdate(gn->subsolver, GNHookFunction, gn));
  PetscCall(TaoSetSolution(gn->subsolver, tao->solution));
  if (tao->bounded) PetscCall(TaoSetVariableBounds(gn->subsolver, tao->XL, tao->XU));
  /* Propagate some options down */
  PetscCall(TaoSetTolerances(gn->subsolver, tao->gatol, tao->grtol, tao->gttol));
  PetscCall(TaoSetMaximumIterations(gn->subsolver, tao->max_it));
  PetscCall(TaoSetMaximumFunctionEvaluations(gn->subsolver, tao->max_funcs));
  for (PetscInt i = 0; i < tao->numbermonitors; ++i) {
    PetscCall(TaoMonitorSet(gn->subsolver, tao->monitor[i], tao->monitorcontext[i], tao->monitordestroy[i]));
    PetscCall(PetscObjectReference((PetscObject)tao->monitorcontext[i]));
  }
  PetscCall(TaoSetUp(gn->subsolver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOBRGN - Bounded Regularized Gauss-Newton method for solving nonlinear least-squares
            problems with bound constraints. This algorithm is a thin wrapper around `TAOBNTL`
            that constructs the Gauss-Newton problem with the user-provided least-squares
            residual and Jacobian. The algorithm offers an L2-norm ("l2pure"), L2-norm proximal point ("l2prox")
            regularizer, and L1-norm dictionary regularizer ("l1dict"), where we approximate the
            L1-norm ||x||_1 by sum_i(sqrt(x_i^2+epsilon^2)-epsilon) with a small positive number epsilon.
            Also offered is the "lm" regularizer which uses a scaled diagonal of J^T J.
            With the "lm" regularizer, `TAOBRGN` is a Levenberg-Marquardt optimizer.
            The user can also provide own regularization function.

  Options Database Keys:
+ -tao_brgn_regularization_type - regularization type ("user", "l2prox", "l2pure", "l1dict", "lm") (default "l2prox")
. -tao_brgn_regularizer_weight  - regularizer weight (default 1e-4)
- -tao_brgn_l1_smooth_epsilon   - L1-norm smooth approximation parameter: ||x||_1 = sum(sqrt(x.^2+epsilon^2)-epsilon) (default 1e-6)

  Level: beginner

.seealso: `Tao`, `TaoBRGNGetSubsolver()`, `TaoBRGNSetRegularizerWeight()`, `TaoBRGNSetL1SmoothEpsilon()`, `TaoBRGNSetDictionaryMatrix()`, `TaoBRGNSetRegularizerTerm()`
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BRGN(Tao tao)
{
  TAO_BRGN   *gn;
  TaoTerm     gauss_newton_term;
  TaoTerm     regularizer_term;
  Mat         map, Hreg;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gn));

  tao->ops->destroy        = TaoDestroy_BRGN;
  tao->ops->setup          = TaoSetUp_BRGN;
  tao->ops->setfromoptions = TaoSetFromOptions_BRGN;
  tao->ops->view           = TaoView_BRGN;
  tao->ops->solve          = TaoSolve_BRGN;

  PetscCall(TaoParametersInitialize(tao));

  tao->data                  = gn;
  gn->reg_type               = TAOBRGN_REGULARIZATION_L2PROX;
  gn->epsilon                = 1e-6;
  gn->downhill_lambda_change = 1. / 5.;
  gn->uphill_lambda_change   = 1.5;
  gn->parent                 = tao;

  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tao, &prefix));
  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &gn->subsolver));
  PetscCall(TaoSetType(gn->subsolver, TAOBNLS));
  PetscCall(TaoSetOptionsPrefix(gn->subsolver, prefix));
  PetscCall(TaoAppendOptionsPrefix(gn->subsolver, "tao_brgn_subsolver_"));

  PetscCall(TaoTermCreateGaussNewton(tao, &gauss_newton_term));
  PetscCall(PetscObjectSetName((PetscObject)gauss_newton_term, "BRGN Gauss-Newton term"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)gauss_newton_term, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)gauss_newton_term, "brgn_gauss_newton_"));
  PetscCall(TaoAddTerm(gn->subsolver, "brgn_gauss_newton_", 1.0, gauss_newton_term, NULL, NULL));
  PetscCall(TaoTermDestroy(&gauss_newton_term));

  PetscCall(TaoBRGNCreateRegularizerTerm(tao, &regularizer_term, &map, &Hreg));
  PetscCall(TaoAddTerm(gn->subsolver, "regularizer_", 1.e-4, regularizer_term, NULL, map));
  PetscCall(TaoTermSumSetSubtermHessianMatrices(gn->subsolver->objective_term.term, 1, Hreg, Hreg, NULL, NULL));
  PetscCall(TaoTermDestroy(&regularizer_term));
  PetscCall(MatDestroy(&map));
  PetscCall(MatDestroy(&Hreg));

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizerTerm_C", TaoBRGNGetRegularizerTerm_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerTerm_C", TaoBRGNSetRegularizerTerm_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizationType_C", TaoBRGNGetRegularizationType_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizationType_C", TaoBRGNSetRegularizationType_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetDampingVector_C", TaoBRGNGetDampingVector_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetSubsolver_C", TaoBRGNGetSubsolver_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetRegularizerWeight_C", TaoBRGNSetRegularizerWeight_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetL1SmoothEpsilon_C", TaoBRGNSetL1SmoothEpsilon_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetDictionaryMatrix_C", TaoBRGNSetDictionaryMatrix_BRGN));
  PetscFunctionReturn(PETSC_SUCCESS);
}
