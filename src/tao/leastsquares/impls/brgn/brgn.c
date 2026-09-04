#include <../src/tao/leastsquares/impls/brgn/brgn.h> /*I "petsctao.h" I*/

static const char *const TaoBRGNPresets[] = {"l2prox", "l2pure", "l1dict", "TaoBRGNPreset", "TAOBRGN_PRESET_", NULL};

static PetscErrorCode TaoBRGNGetDataTerm(Tao tao, TaoTerm *term, Vec *params)
{
  PetscBool is_sum;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tao->objective_term.term, TAOTERMSUM, &is_sum));
  if (is_sum) {
    PetscReal scale;
    Mat       map;

    PetscCall(TaoTermSumGetTerm(tao->objective_term.term, 0, NULL, &scale, term, &map));
    PetscCheck(!map, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "TAOBRGN does not support a mapping on its data term");
    PetscCheck(scale == 1.0, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "TAOBRGN does not support a scale other than 1 on its data term");
    if (params) {
      *params = NULL;
      if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumParameters(tao->objective_parameters, 0, params));
    }
  } else {
    *term = tao->objective_term.term;
    if (params) *params = tao->objective_parameters;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetNumberRegularizers(Tao tao, PetscInt *nregularizers)
{
  PetscBool is_sum;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tao->objective_term.term, TAOTERMSUM, &is_sum));
  if (is_sum) {
    PetscInt nterms;

    PetscCall(TaoTermSumGetNumberTerms(tao->objective_term.term, &nterms));
    *nregularizers = nterms - 1;
  } else *nregularizers = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNAddObjectiveRegularizer(Tao tao, const char prefix[], PetscReal scale, TaoTerm term, Vec params, Mat map)
{
  PetscFunctionBegin;
  if (!tao->term_set) {
    TaoTerm     data_sum;
    const char *tao_prefix;
    Vec         data_params = tao->objective_parameters;

    PetscCall(TaoTermDuplicate(tao->objective_term.term, TAOTERM_DUPLICATE_SIZEONLY, &data_sum));
    PetscCall(TaoTermSetType(data_sum, TAOTERMSUM));
    PetscCall(TaoGetOptionsPrefix(tao, &tao_prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data_sum, tao_prefix));
    PetscCall(TaoTermSumSetNumberTerms(data_sum, 1));
    PetscCall(TaoTermSumSetTerm(data_sum, 0, "data_", 1.0, tao->objective_term.term, NULL));
    PetscCall(TaoTermMappingReset(&tao->objective_term));
    PetscCall(TaoTermMappingSetData(&tao->objective_term, NULL, 1.0, data_sum, NULL));
    tao->objective_parameters = NULL;
    if (data_params) {
      Vec subparams[1];

      subparams[0] = data_params;
      PetscCall(TaoTermSumParametersPack(data_sum, subparams, &tao->objective_parameters));
      PetscCall(VecDestroy(&data_params));
    }
    PetscCall(TaoTermDestroy(&data_sum));
    tao->num_terms = 1;
    tao->term_set  = PETSC_TRUE;
  }
  PetscCall(TaoAddTerm(tao, prefix, scale, term, params, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNEvaluateResidual(TAO_BRGN *gn, Vec x)
{
  Tao       tao = gn->parent;
  TaoTerm   term;
  Vec       params;
  PetscBool has_residual;

  PetscFunctionBegin;
  PetscCall(TaoBRGNGetDataTerm(tao, &term, &params));
  PetscCall(TaoTermIsResidualDefined(term, &has_residual));
  if (has_residual) {
    PetscCall(TaoTermComputeResidual(term, x, params, tao->ls_res));
    tao->nres++;
  } else PetscCall(TaoComputeResidual(tao, x, tao->ls_res));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNEvaluateJacobianResidual(TAO_BRGN *gn, Vec x)
{
  Tao       tao = gn->parent;
  TaoTerm   term;
  Vec       params;
  PetscBool has_jacobian;

  PetscFunctionBegin;
  PetscCall(TaoBRGNGetDataTerm(tao, &term, &params));
  PetscCall(TaoTermIsJacobianResidualDefined(term, &has_jacobian));
  if (has_jacobian) {
    PetscCall(TaoTermComputeJacobianResidual(term, x, params, tao->ls_jac, tao->ls_jac_pre));
    tao->njac++;
  } else PetscCall(TaoComputeResidualJacobian(tao, x, tao->ls_jac, tao->ls_jac_pre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNCreateProduct(Mat J, MatReuse reuse, Mat *JtJ)
{
  Mat       Jassembled = NULL;
  PetscBool needs_assembly, is_shell;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATCOMPOSITE, &needs_assembly));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)J, MATSHELL, &is_shell));
  needs_assembly = (PetscBool)(needs_assembly || is_shell);
  if (needs_assembly) {
    PetscCall(MatComputeOperator(J, MATAIJ, &Jassembled));
    J = Jassembled;
  }
  PetscCall(MatTransposeMatMult(J, J, reuse, PETSC_DETERMINE, JtJ));
  PetscCall(MatDestroy(&Jassembled));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNCacheJacobian(TAO_BRGN *gn, Vec x)
{
  PetscBool current = PETSC_FALSE;

  PetscFunctionBegin;
  if (gn->hessian_x) PetscCall(VecEqual(x, gn->hessian_x, &current));
  if (!current) {
    PetscCall(TaoBRGNEvaluateJacobianResidual(gn, x));
    if (!gn->hessian_x) PetscCall(VecDuplicate(x, &gn->hessian_x));
    PetscCall(VecCopy(x, gn->hessian_x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNComputeHessianMult(Tao subsolver, Vec x, Vec v, Vec Hv, PetscCtx ctx)
{
  TAO_BRGN *gn = (TAO_BRGN *)ctx;

  PetscFunctionBegin;
  PetscCall(TaoBRGNCacheJacobian(gn, x));
  PetscCall(MatMult(gn->parent->ls_jac, v, gn->r_work));
  PetscCall(MatMultTranspose(gn->parent->ls_jac, gn->r_work, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNApplyLM(TAO_BRGN *gn, Mat H)
{
  Mat          J = gn->parent->ls_jac;
  PetscInt     n, cstart, cend;
  PetscReal   *norms;
  PetscScalar *array;
  PetscBool    is_diagonal;

  PetscFunctionBegin;
  if (!gn->use_lm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectTypeCompare((PetscObject)J, MATDIAGONAL, &is_diagonal));
  if (is_diagonal) {
    Vec diagonal;

    PetscCall(MatDiagonalGetDiagonal(J, &diagonal));
    PetscCall(VecPointwiseMult(gn->damping, diagonal, diagonal));
    PetscCall(MatDiagonalRestoreDiagonal(J, &diagonal));
  } else {
    PetscCall(MatGetSize(J, NULL, &n));
    PetscCall(PetscMalloc1(n, &norms));
    PetscCall(MatGetColumnNorms(J, NORM_2, norms));
    PetscCall(MatGetOwnershipRangeColumn(J, &cstart, &cend));
    PetscCall(VecGetArray(gn->damping, &array));
    for (PetscInt i = 0; i < cend - cstart; i++) array[i] = norms[cstart + i] * norms[cstart + i];
    PetscCall(VecRestoreArray(gn->damping, &array));
    PetscCall(PetscFree(norms));
  }
  PetscCall(VecGetArray(gn->damping, &array));
  PetscCall(VecGetLocalSize(gn->damping, &n));
  for (PetscInt i = 0; i < n; i++) {
    PetscReal value = PetscRealPart(array[i]);

    array[i] = gn->lm_lambda * PetscClipInterval(value, PETSC_SQRT_MACHINE_EPSILON, PetscSqrtReal(PETSC_MAX_REAL));
  }
  PetscCall(VecRestoreArray(gn->damping, &array));
  PetscCall(MatDiagonalSet(H, gn->damping, ADD_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNObjectiveAndGradient(Tao subsolver, Vec x, PetscReal *value, Vec g, PetscCtx ctx)
{
  TAO_BRGN *gn  = (TAO_BRGN *)ctx;
  Tao       tao = gn->parent;
  TaoTerm   term;
  Vec       params;
  PetscBool has_obj, has_objgrad, has_grad;

  PetscFunctionBegin;
  PetscCall(TaoBRGNGetDataTerm(tao, &term, &params));
  PetscCall(TaoTermIsObjectiveDefined(term, &has_obj));
  PetscCall(TaoTermIsObjectiveAndGradientDefined(term, &has_objgrad));
  PetscCall(TaoTermIsGradientDefined(term, &has_grad));
  if (has_objgrad || (has_obj && has_grad)) {
    PetscCall(TaoTermComputeObjectiveAndGradient(term, x, params, value, g));
  } else {
    PetscScalar dot;

    PetscCall(TaoBRGNEvaluateResidual(gn, x));
    PetscCall(VecDot(tao->ls_res, tao->ls_res, &dot));
    *value = 0.5 * PetscRealPart(dot);
    PetscCall(TaoBRGNEvaluateJacobianResidual(gn, x));
    if (gn->matrix_free) {
      if (!gn->hessian_x) PetscCall(VecDuplicate(x, &gn->hessian_x));
      PetscCall(VecCopy(x, gn->hessian_x));
    }
    PetscCall(MatMultTranspose(tao->ls_jac, tao->ls_res, g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNComputeHessian(Tao subsolver, Vec x, Mat H, Mat Hpre, PetscCtx ctx)
{
  TAO_BRGN *gn  = (TAO_BRGN *)ctx;
  Tao       tao = gn->parent;
  TaoTerm   term;
  Vec       params;
  PetscBool has_hessian;

  PetscFunctionBegin;
  PetscCall(TaoBRGNGetDataTerm(tao, &term, &params));
  PetscCall(TaoTermIsHessianDefined(term, &has_hessian));
  if (has_hessian) {
    PetscCall(TaoTermComputeHessian(term, x, params, H, Hpre));
    if (gn->use_lm) {
      PetscCall(TaoBRGNEvaluateJacobianResidual(gn, x));
      PetscCall(TaoBRGNApplyLM(gn, H));
      if (Hpre && Hpre != H) PetscCall(TaoBRGNApplyLM(gn, Hpre));
    }
  } else {
    PetscCall(TaoBRGNEvaluateJacobianResidual(gn, x));
    PetscCall(TaoBRGNCreateProduct(tao->ls_jac, MAT_REUSE_MATRIX, &H));
    PetscCall(TaoBRGNApplyLM(gn, H));
    if (Hpre && Hpre != H) PetscCall(MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNUpdateRegularizer(TaoTerm term, Tao tao, PetscInt iter)
{
  PetscBool is_sum;

  PetscFunctionBegin;
  if (!term) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoTermL2ProxUpdate_Private(term, tao, iter));
  PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERMSUM, &is_sum));
  if (is_sum) {
    PetscInt nterms;

    PetscCall(TaoTermSumGetNumberTerms(term, &nterms));
    for (PetscInt i = 0; i < nterms; i++) {
      TaoTerm subterm;

      PetscCall(TaoTermSumGetTerm(term, i, NULL, NULL, &subterm, NULL));
      PetscCall(TaoBRGNUpdateRegularizer(subterm, tao, iter));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNUpdate(Tao subsolver, PetscInt iter, PetscCtx ctx)
{
  TAO_BRGN *gn = (TAO_BRGN *)ctx;
  PetscInt  nregularizers;

  PetscFunctionBegin;
  gn->parent->objective_term.term->nobj     = subsolver->objective_term.term->nobj;
  gn->parent->objective_term.term->ngrad    = subsolver->objective_term.term->ngrad;
  gn->parent->objective_term.term->nobjgrad = subsolver->objective_term.term->nobjgrad;
  gn->parent->objective_term.term->nhess    = subsolver->objective_term.term->nhess;
  gn->parent->niter                         = subsolver->niter;
  gn->parent->ksp_its                       = subsolver->ksp_its;
  gn->parent->ksp_tot_its                   = subsolver->ksp_tot_its;
  gn->parent->fc                            = subsolver->fc;
  PetscCall(TaoGetConvergedReason(subsolver, &gn->parent->reason));
  if (iter > 0) PetscCall(VecCopy(subsolver->solution, gn->parent->solution));
  PetscCall(VecCopy(subsolver->gradient, gn->parent->gradient));
  PetscCall(TaoBRGNGetNumberRegularizers(gn->parent, &nregularizers));
  for (PetscInt i = 0; i < nregularizers; i++) {
    TaoTerm regularizer;

    PetscCall(TaoTermSumGetTerm(gn->parent->objective_term.term, i + 1, NULL, NULL, &regularizer, NULL));
    PetscCall(TaoBRGNUpdateRegularizer(regularizer, subsolver, iter));
  }
  PetscTryTypeMethod(gn->parent, update, gn->parent->niter, gn->parent->user_update);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNCreatePresetRegularizer(Tao tao)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;
  TaoTerm   term;
  PetscInt  n, N;

  PetscFunctionBegin;
  PetscCheck(tao->solution, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetSolution() must be called before TaoSetUp()");
  PetscCall(VecGetLocalSize(tao->solution, &n));
  PetscCall(VecGetSize(tao->solution, &N));
  switch (gn->preset) {
  case TAOBRGN_PRESET_L2PROX:
    PetscCall(TaoTermCreateL2Prox(PetscObjectComm((PetscObject)tao), n, N, &term));
    break;
  case TAOBRGN_PRESET_L2PURE:
    PetscCall(TaoTermCreateHalfL2Squared(PetscObjectComm((PetscObject)tao), n, N, &term));
    break;
  case TAOBRGN_PRESET_L1DICT:
    PetscCall(TaoTermCreateL1(PetscObjectComm((PetscObject)tao), n, N, gn->preset_l1_epsilon, &term));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "Unknown TAOBRGN regularizer preset");
  }
  PetscCall(TaoBRGNAddObjectiveRegularizer(tao, "regularizer_", gn->preset_weight, term, NULL, NULL));
  PetscCall(TaoTermDestroy(&term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetSubsolver - Get the subsolver used by `TAOBRGN`

  Collective

  Input Parameter:
. tao - the `TAOBRGN`

  Output Parameter:
. subsolver - the subsolver

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TAOBRGN`, `TaoBRGNAddRegularizerTerm()`
@*/
PetscErrorCode TaoBRGNGetSubsolver(Tao tao, Tao *subsolver)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(subsolver, 2);
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
  TaoBRGNAddRegularizerTerm - Add a regularizer $\alpha g(Ax;p)$ to `TAOBRGN`

  Collective

  Input Parameters:
+ tao    - the `TAOBRGN`
. prefix - options prefix for the regularizer
. scale  - the coefficient $\alpha$
. term   - the regularizer $g$
. params - optional parameters $p$
- map    - optional map $A$

  Level: advanced

.seealso: [](ch_tao), [](sec_tao_term), `TAOBRGN`, `TaoBRGNGetRegularizerTerm()`, `TaoAddTerm()`
@*/
PetscErrorCode TaoBRGNAddRegularizerTerm(Tao tao, const char prefix[], PetscReal scale, TaoTerm term, Vec params, Mat map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (prefix) PetscAssertPointer(prefix, 2);
  PetscValidLogicalCollectiveReal(tao, scale, 3);
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 4);
  PetscCheckSameComm(tao, 1, term, 4);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 5);
    PetscCheckSameComm(tao, 1, params, 5);
  }
  if (map) {
    PetscValidHeaderSpecific(map, MAT_CLASSID, 6);
    PetscCheckSameComm(tao, 1, map, 6);
  }
  PetscUseMethod((PetscObject)tao, "TaoBRGNAddRegularizerTerm_C", (Tao, const char[], PetscReal, TaoTerm, Vec, Mat), (tao, prefix, scale, term, params, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNAddRegularizerTerm_BRGN(Tao tao, const char prefix[], PetscReal scale, TaoTerm term, Vec params, Mat map)
{
  PetscFunctionBegin;
  PetscCheck(!tao->setupcalled, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TaoBRGNAddRegularizerTerm() must be called before TaoSetUp() or TaoSolve()");
  PetscCall(TaoBRGNAddObjectiveRegularizer(tao, prefix, scale, term, params, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetRegularizerTerm - Get the objective sum containing the regularizers used by `TAOBRGN`

  Not Collective

  Input Parameter:
. tao - the `TAOBRGN`

  Output Parameter:
. term - the objective `TAOTERMSUM`, whose terms after index 0 are regularizers, or `NULL` if there are no regularizers

  Level: advanced

.seealso: [](ch_tao), [](sec_tao_term), `TAOBRGN`, `TaoBRGNAddRegularizerTerm()`, `TAOTERMSUM`
@*/
PetscErrorCode TaoBRGNGetRegularizerTerm(Tao tao, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(term, 2);
  PetscUseMethod((PetscObject)tao, "TaoBRGNGetRegularizerTerm_C", (Tao, TaoTerm *), (tao, term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetRegularizerTerm_BRGN(Tao tao, TaoTerm *term)
{
  PetscInt nregularizers;

  PetscFunctionBegin;
  PetscCall(TaoBRGNGetNumberRegularizers(tao, &nregularizers));
  *term = nregularizers ? tao->objective_term.term : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNSetUseLM - Set whether `TAOBRGN` uses Levenberg-Marquardt damping

  Logically Collective

  Input Parameters:
+ tao    - the `TAOBRGN`
- use_lm - whether to use Levenberg-Marquardt damping

  Level: advanced

.seealso: [](ch_tao), `TAOBRGN`, `TaoBRGNGetUseLM()`, `TaoBRGNSetLMLambda()`
@*/
PetscErrorCode TaoBRGNSetUseLM(Tao tao, PetscBool use_lm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, use_lm, 2);
  PetscUseMethod((PetscObject)tao, "TaoBRGNSetUseLM_C", (Tao, PetscBool), (tao, use_lm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetUseLM_BRGN(Tao tao, PetscBool use_lm)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  gn->use_lm = use_lm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetUseLM - Get whether `TAOBRGN` uses Levenberg-Marquardt damping

  Not Collective

  Input Parameter:
. tao - the `TAOBRGN`

  Output Parameter:
. use_lm - whether Levenberg-Marquardt damping is enabled

  Level: advanced

.seealso: [](ch_tao), `TAOBRGN`, `TaoBRGNSetUseLM()`, `TaoBRGNGetLMLambda()`
@*/
PetscErrorCode TaoBRGNGetUseLM(Tao tao, PetscBool *use_lm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(use_lm, 2);
  PetscUseMethod((PetscObject)tao, "TaoBRGNGetUseLM_C", (Tao, PetscBool *), (tao, use_lm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetUseLM_BRGN(Tao tao, PetscBool *use_lm)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  *use_lm = gn->use_lm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNSetLMLambda - Set the Levenberg-Marquardt damping coefficient

  Logically Collective

  Input Parameters:
+ tao    - the `TAOBRGN`
- lambda - the nonnegative damping coefficient

  Level: advanced

.seealso: [](ch_tao), `TAOBRGN`, `TaoBRGNGetLMLambda()`, `TaoBRGNSetUseLM()`
@*/
PetscErrorCode TaoBRGNSetLMLambda(Tao tao, PetscReal lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, lambda, 2);
  PetscCheck(lambda >= 0.0, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "LM lambda must be nonnegative");
  PetscUseMethod((PetscObject)tao, "TaoBRGNSetLMLambda_C", (Tao, PetscReal), (tao, lambda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNSetLMLambda_BRGN(Tao tao, PetscReal lambda)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  gn->lm_lambda = lambda;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoBRGNGetLMLambda - Get the Levenberg-Marquardt damping coefficient

  Not Collective

  Input Parameter:
. tao - the `TAOBRGN`

  Output Parameter:
. lambda - the damping coefficient

  Level: advanced

.seealso: [](ch_tao), `TAOBRGN`, `TaoBRGNSetLMLambda()`, `TaoBRGNGetUseLM()`
@*/
PetscErrorCode TaoBRGNGetLMLambda(Tao tao, PetscReal *lambda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(lambda, 2);
  PetscUseMethod((PetscObject)tao, "TaoBRGNGetLMLambda_C", (Tao, PetscReal *), (tao, lambda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBRGNGetLMLambda_BRGN(Tao tao, PetscReal *lambda)
{
  TAO_BRGN *gn = (TAO_BRGN *)tao->data;

  PetscFunctionBegin;
  *lambda = gn->lm_lambda;
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
  TAO_BRGN     *gn     = (TAO_BRGN *)tao->data;
  TaoBRGNPreset preset = gn->preset;
  PetscBool     preset_set;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Bounded regularized Gauss-Newton options");
  PetscCall(PetscOptionsEnum("-tao_brgn_regularization_type", "regularizer preset", "TaoBRGNAddRegularizerTerm", TaoBRGNPresets, (PetscEnum)preset, (PetscEnum *)&preset, &preset_set));
  PetscCall(PetscOptionsReal("-tao_brgn_regularizer_weight", "regularizer weight", "TaoBRGNAddRegularizerTerm", gn->preset_weight, &gn->preset_weight, NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_l1_smooth_epsilon", "L1 smoothing parameter", "TaoTermL1SetEpsilon", gn->preset_l1_epsilon, &gn->preset_l1_epsilon, NULL));
  PetscCall(PetscOptionsBool("-tao_brgn_use_lm", "use Levenberg-Marquardt damping", "TaoBRGNSetUseLM", gn->use_lm, &gn->use_lm, NULL));
  PetscCall(PetscOptionsReal("-tao_brgn_lm_lambda", "Levenberg-Marquardt damping coefficient", "TaoBRGNSetLMLambda", gn->lm_lambda, &gn->lm_lambda, NULL));
  PetscOptionsHeadEnd();
  gn->preset     = preset;
  gn->preset_set = (PetscBool)(gn->preset_set || preset_set);
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "Levenberg-Marquardt damping: %s", gn->use_lm ? "enabled" : "disabled"));
    if (gn->use_lm) PetscCall(PetscViewerASCIIPrintf(viewer, " (lambda %g)", (double)gn->lm_lambda));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
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
  TaoTerm   data_term;
  PetscInt  nregularizers;
  PetscBool has_residual, has_jacobian, has_obj, has_objgrad, has_grad, has_hessian, exact_objective, use_residual, is_l2, is_shell, is_composite;

  PetscFunctionBegin;
  PetscCheck(!tao->objective_term.map, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "TAOBRGN does not support an outer mapping on its residual model");
  PetscCheck(tao->objective_term.scale == 1.0, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "TAOBRGN does not support an outer scale on its residual model");
  PetscCall(TaoBRGNGetDataTerm(tao, &data_term, NULL));
  PetscCall(TaoTermIsResidualDefined(data_term, &has_residual));
  PetscCall(TaoTermIsJacobianResidualDefined(data_term, &has_jacobian));
  PetscCall(TaoTermIsObjectiveDefined(data_term, &has_obj));
  PetscCall(TaoTermIsObjectiveAndGradientDefined(data_term, &has_objgrad));
  PetscCall(TaoTermIsGradientDefined(data_term, &has_grad));
  PetscCall(TaoTermIsHessianDefined(data_term, &has_hessian));
  exact_objective = (PetscBool)((has_objgrad || (has_obj && has_grad)) && has_hessian);
  use_residual    = (PetscBool)(!exact_objective || gn->use_lm);
  if (use_residual && (has_residual || has_jacobian)) {
    PetscCheck(has_residual && has_jacobian, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TAOBRGN residual TaoTerm must define both residual and residual Jacobian operations");
    PetscCall(PetscObjectTypeCompare((PetscObject)data_term, TAOTERMHALFL2SQUARED, &is_l2));
    if (is_l2 && (!data_term->residual || !data_term->jacobian_residual)) {
      Vec residual;
      Mat J, Jpre;

      PetscCall(VecDuplicate(tao->solution, &residual));
      PetscCall(TaoTermCreateHessianMatrices(data_term, &J, &Jpre));
      PetscCall(TaoTermSetResidual_Internal(data_term, residual, data_term->ops->residual));
      PetscCall(TaoTermSetJacobianResidual_Internal(data_term, J, Jpre, data_term->ops->jacobianresidual));
      PetscCall(VecDestroy(&residual));
      PetscCall(MatDestroy(&J));
      PetscCall(MatDestroy(&Jpre));
    }
    PetscCheck(data_term->residual && data_term->jacobian_residual, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOBRGN residual TaoTerm must provide residual and residual Jacobian storage");
    PetscCall(PetscObjectReference((PetscObject)data_term->residual));
    PetscCall(VecDestroy(&tao->ls_res));
    tao->ls_res = data_term->residual;
    PetscCall(PetscObjectReference((PetscObject)data_term->jacobian_residual));
    PetscCall(MatDestroy(&tao->ls_jac));
    tao->ls_jac = data_term->jacobian_residual;
    PetscCall(PetscObjectReference((PetscObject)data_term->jacobian_residual_pre));
    PetscCall(MatDestroy(&tao->ls_jac_pre));
    tao->ls_jac_pre = data_term->jacobian_residual_pre;
  }
  if (use_residual) {
    PetscCheck(tao->ls_res, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetResidual() or a residual-capable TaoTerm with storage must be configured before TaoSetUp()");
    PetscCheck(tao->ls_jac, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetJacobianResidual() or a residual-Jacobian-capable TaoTerm with storage must be configured before TaoSetUp()");
  }
  if (gn->use_lm) PetscCheck(tao->ls_jac, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "Levenberg-Marquardt damping requires residual Jacobian storage");
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)tao->ls_jac, MATSHELL, &is_shell));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao->ls_jac, MATCOMPOSITE, &is_composite));
  gn->matrix_free = (PetscBool)(!exact_objective && !gn->use_lm && (is_shell || is_composite));
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  PetscCall(TaoBRGNGetNumberRegularizers(tao, &nregularizers));
  PetscCheck(!gn->preset_set || !nregularizers, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Cannot combine -tao_brgn_regularization_type with regularizer terms added through TaoAddTerm() or TaoBRGNAddRegularizerTerm(); drop the preset option or the added regularizer terms");
  if (!nregularizers && !gn->use_lm) {
    PetscCall(TaoBRGNCreatePresetRegularizer(tao));
    PetscCall(TaoBRGNGetNumberRegularizers(tao, &nregularizers));
  }
  if (gn->use_lm && !gn->damping) PetscCall(MatCreateVecs(tao->ls_jac, &gn->damping, NULL));
  if (exact_objective) {
    if (!gn->H) {
      Mat Hpre;

      PetscCall(TaoTermCreateHessianMatrices(data_term, &gn->H, &Hpre));
      PetscCheck(Hpre == gn->H, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "TAOBRGN exact data term must use the same matrix for its Hessian and preconditioner");
      PetscCall(MatDestroy(&Hpre));
    }
    PetscCheck(gn->H, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "TAOBRGN data term has an exact Hessian operation but cannot create Hessian matrix storage");
    PetscCall(TaoBRGNComputeHessian(gn->subsolver, tao->solution, gn->H, gn->H, gn));
  } else if (!gn->matrix_free) {
    PetscCall(TaoBRGNEvaluateJacobianResidual(gn, tao->solution));
    PetscCall(TaoBRGNCreateProduct(tao->ls_jac, MAT_INITIAL_MATRIX, &gn->H));
    PetscCall(TaoBRGNApplyLM(gn, gn->H));
  } else {
    PetscCall(MatCreateVecs(tao->ls_jac, NULL, &gn->r_work));
  }
  PetscCall(TaoSetObjectiveAndGradient(gn->subsolver, NULL, TaoBRGNObjectiveAndGradient, gn));
  PetscCall(TaoSetHessian(gn->subsolver, gn->H, gn->H, TaoBRGNComputeHessian, gn));
  if (gn->matrix_free) PetscCall(TaoSetHessianMult(gn->subsolver, TaoBRGNComputeHessianMult, gn));
  for (PetscInt i = 0; i < nregularizers; i++) {
    const char *prefix;
    TaoTerm     regularizer;
    Vec         regularizer_params = NULL;
    Mat         regularizer_map;
    PetscReal   regularizer_scale;

    PetscCall(TaoTermSumGetTerm(tao->objective_term.term, i + 1, &prefix, &regularizer_scale, &regularizer, &regularizer_map));
    if (tao->objective_parameters) PetscCall(VecNestGetTaoTermSumParameters(tao->objective_parameters, i + 1, &regularizer_params));
    PetscCall(TaoAddTerm(gn->subsolver, prefix, regularizer_scale, regularizer, regularizer_params, regularizer_map));
  }
  if (gn->matrix_free) PetscCall(TaoTermSetCreateHessianMode(gn->subsolver->objective_term.term, PETSC_TRUE, MATSHELL, NULL));
  PetscCall(TaoSetUpdate(gn->subsolver, TaoBRGNUpdate, gn));
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
  PetscCall(VecDestroy(&gn->hessian_x));
  PetscCall(VecDestroy(&gn->r_work));
  PetscCall(MatDestroy(&gn->H));
  PetscCall(TaoDestroy(&gn->subsolver));
  PetscCall(PetscFree(tao->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetSubsolver_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNAddRegularizerTerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizerTerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetUseLM_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetUseLM_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetLMLambda_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetLMLambda_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOBRGN - Bounded regularized Gauss-Newton method

  Options Database Keys:
+ -tao_brgn_regularization_type (l2prox|l2pure|l1dict) - create a built-in regularizer (default l2prox)
. -tao_brgn_regularizer_weight lambda                  - regularizer weight (default 1e-4)
. -tao_brgn_l1_smooth_epsilon epsilon                 - smoothing for the built-in L1 regularizer (default 1e-6)
. -tao_brgn_use_lm                                    - use Levenberg-Marquardt damping
- -tao_brgn_lm_lambda lambda                          - Levenberg-Marquardt damping coefficient

  Level: beginner

  Notes:
  Term 0 of the objective is the least-squares data term. Later terms are regularizers.
  Use `TaoAddTerm()`, `TaoBRGNAddRegularizerTerm()`, or `-tao_add_terms` to add arbitrary
  regularizers. The `-tao_brgn_regularization_type` preset applies only when no regularizer
  terms have been added. Levenberg-Marquardt damping is solver policy and can be enabled
  independently of the objective regularizers.

.seealso: [](ch_tao), `Tao`, `TaoBRGNAddRegularizerTerm()`, `TaoBRGNGetRegularizerTerm()`, `TaoBRGNSetUseLM()`, `TAOTERML2PROX`
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
  tao->data                = gn;
  gn->parent               = tao;
  gn->preset               = TAOBRGN_PRESET_L2PROX;
  gn->preset_weight        = 1e-4;
  gn->preset_l1_epsilon    = 1e-6;
  gn->lm_lambda            = 1e-4;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tao, &prefix));
  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao), &gn->subsolver));
  PetscCall(TaoSetType(gn->subsolver, TAOBNLS));
  PetscCall(TaoSetOptionsPrefix(gn->subsolver, prefix));
  PetscCall(TaoAppendOptionsPrefix(gn->subsolver, "tao_brgn_subsolver_"));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetSubsolver_C", TaoBRGNGetSubsolver_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNAddRegularizerTerm_C", TaoBRGNAddRegularizerTerm_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetRegularizerTerm_C", TaoBRGNGetRegularizerTerm_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetUseLM_C", TaoBRGNSetUseLM_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetUseLM_C", TaoBRGNGetUseLM_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNSetLMLambda_C", TaoBRGNSetLMLambda_BRGN));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoBRGNGetLMLambda_C", TaoBRGNGetLMLambda_BRGN));
  PetscFunctionReturn(PETSC_SUCCESS);
}
