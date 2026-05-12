#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsctaoterm.h>          /*I "petsctaoterm.h" I*/

typedef struct _n_TaoTerm_GaussNewton TaoTerm_GaussNewton;

struct _n_TaoTerm_GaussNewton {
  Tao tao; // weak reference to the Tao that owns the residual machinery (ls_res, ls_jac)

  PetscObjectId    jac_x_id;
  PetscObjectState jac_x_state;
  PetscBool        jac_valid;

  Vec r_work; // workspace for (J^T J) v evaluation
};

/*
  Cached accessor for the residual Jacobian J(x) of the wrapped Tao.  All of this term's
  ops (objective, gradient, Hessian, Hessian-mult) go through this routine so that
  successive evaluations at the same iterate share a single `TaoComputeResidualJacobian()`
  call.  Companion terms that share the same Jacobian (e.g. `TAOTERMSHELL`-based
  Levenberg-Marquardt damping in `taotermbrgnlm.c`) also enter here via the public
  `TaoTermGaussNewtonGetJacobian()`.

  Input Parameters:
+ term - a `TAOTERMGAUSSNEWTON`
- x    - the current iterate; cache key is `(PetscObjectId, PetscObjectState)` of `x`

  Output Parameter:
. ls_jac - the cached residual Jacobian (alias of `tao->ls_jac`, not a copy)

  Notes:
  Errors out if either `tao` or its residual Jacobian have not been set yet, since the
  cache cannot be built without them.  The cache is invalidated automatically by
  `TaoTermGaussNewtonSetTao_GaussNewton()` whenever the wrapped `Tao` changes.
*/
static PetscErrorCode TaoTermGaussNewtonGetJacobian_Internal(TaoTerm term, Vec x, Mat *ls_jac)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;
  PetscObjectId        x_id;
  PetscObjectState     x_state;

  PetscFunctionBegin;
  PetscCheck(gn->tao, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao not set, call TaoTermGaussNewtonSetTao() first");
  PetscCheck(gn->tao->ls_jac, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao residual Jacobian not set, call TaoSetJacobianResidualRoutine() first");
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (!gn->jac_valid || gn->jac_x_id != x_id || gn->jac_x_state != x_state) {
    PetscCall(TaoComputeResidualJacobian(gn->tao, x, gn->tao->ls_jac, gn->tao->ls_jac_pre));
    gn->jac_x_id    = x_id;
    gn->jac_x_state = x_state;
    gn->jac_valid   = PETSC_TRUE;
  }
  *ls_jac = gn->tao->ls_jac;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_GaussNewton(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;
  PetscScalar          sval;

  PetscFunctionBegin;
  PetscCheck(gn->tao, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao not set, call TaoTermGaussNewtonSetTao() first");
  PetscCheck(gn->tao->ls_res, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao residual not set, call TaoSetResidualRoutine() first");
  PetscCall(TaoComputeResidual(gn->tao, x, gn->tao->ls_res));
  PetscCall(VecDot(gn->tao->ls_res, gn->tao->ls_res, &sval));
  *value = 0.5 * PetscRealPart(sval);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeGradient_GaussNewton(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;
  Mat                  ls_jac;

  PetscFunctionBegin;
  PetscCheck(gn->tao, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao not set, call TaoTermGaussNewtonSetTao() first");
  PetscCheck(gn->tao->ls_res, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao residual not set, call TaoSetResidualRoutine() first");
  PetscCall(TaoComputeResidual(gn->tao, x, gn->tao->ls_res));
  PetscCall(TaoTermGaussNewtonGetJacobian_Internal(term, x, &ls_jac));
  PetscCall(MatMultTranspose(ls_jac, gn->tao->ls_res, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjectiveAndGradient_GaussNewton(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;
  PetscScalar          sval;
  Mat                  ls_jac;

  PetscFunctionBegin;
  PetscCheck(gn->tao, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao not set, call TaoTermGaussNewtonSetTao() first");
  PetscCheck(gn->tao->ls_res, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao residual not set, call TaoSetResidualRoutine() first");
  PetscCall(TaoComputeResidual(gn->tao, x, gn->tao->ls_res));
  PetscCall(VecDot(gn->tao->ls_res, gn->tao->ls_res, &sval));
  *value = 0.5 * PetscRealPart(sval);
  PetscCall(TaoTermGaussNewtonGetJacobian_Internal(term, x, &ls_jac));
  PetscCall(MatMultTranspose(ls_jac, gn->tao->ls_res, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_GaussNewton_Single(TaoTerm term, Vec x, Mat M)
{
  Mat            ls_jac;
  PetscContainer marker = NULL;

  PetscFunctionBegin;
  PetscCall(TaoTermGaussNewtonGetJacobian_Internal(term, x, &ls_jac));
  /* On first use of M, do MAT_INITIAL_MATRIX so MatProduct picks the right
     algorithm for the (J, J) types; thereafter MAT_REUSE_MATRIX refreshes the
     existing product. A marker on M tracks initialization. */
  PetscCall(PetscObjectQuery((PetscObject)M, "__TaoTermGaussNewtonProduct", (PetscObject *)&marker));
  if (!marker) {
    PetscCall(MatTransposeMatMult(ls_jac, ls_jac, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &M));
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &marker));
    PetscCall(PetscObjectCompose((PetscObject)M, "__TaoTermGaussNewtonProduct", (PetscObject)marker));
    PetscCall(PetscContainerDestroy(&marker));
  } else {
    PetscCall(MatTransposeMatMult(ls_jac, ls_jac, MAT_REUSE_MATRIX, PETSC_DETERMINE, &M));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_GaussNewton(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  /* Shell H/Hpre have already been preprocessed (state-updated and NULL'd) by
     TaoTermPreprocessHessianShells() upstream of this op; so if both are NULL
     the only thing we needed to do was advance the cached Jacobian, and
     TaoTermComputeHessianMult_GaussNewton() will do that on demand. */
  if (!H && !Hpre) PetscFunctionReturn(PETSC_SUCCESS);
  if (H) PetscCall(TaoTermComputeHessian_GaussNewton_Single(term, x, H));
  if (Hpre && Hpre != H) PetscCall(TaoTermComputeHessian_GaussNewton_Single(term, x, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessianMult_GaussNewton(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;
  Mat                  ls_jac;

  PetscFunctionBegin;
  PetscCall(TaoTermGaussNewtonGetJacobian_Internal(term, x, &ls_jac));
  if (!gn->r_work) PetscCall(MatCreateVecs(ls_jac, NULL, &gn->r_work));
  PetscCall(MatMult(ls_jac, v, gn->r_work));
  PetscCall(MatMultHermitianTranspose(ls_jac, gn->r_work, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateHessianMatrices_GaussNewton(TaoTerm term, Mat *H, Mat *Hpre)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;
  Mat                  ls_jac;
  Mat                  prototype = NULL;
  PetscBool            shell_H, shell_Hpre;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(term->H_mattype, MATSHELL, &shell_H));
  PetscCall(PetscStrcmp(term->Hpre_mattype, MATSHELL, &shell_Hpre));
  if (shell_H && shell_Hpre) {
    /* All-shell mode: use the default shell-creating helper. */
    PetscCall(TaoTermCreateHessianMatricesDefault(term, H, Hpre));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(gn->tao && gn->tao->solution, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Tao solution not set; TaoSetSolution() must be called before TAOTERMGAUSSNEWTON Hessian matrices are created");
  /* Seed the assembled J^T J from the current iterate so the sparsity / type is
     determined naturally by MatProduct (AtB) on the residual Jacobian. */
  PetscCall(TaoTermGaussNewtonGetJacobian_Internal(term, gn->tao->solution, &ls_jac));
  PetscCall(MatTransposeMatMult(ls_jac, ls_jac, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &prototype));
  {
    PetscContainer marker;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &marker));
    PetscCall(PetscObjectCompose((PetscObject)prototype, "__TaoTermGaussNewtonProduct", (PetscObject)marker));
    PetscCall(PetscContainerDestroy(&marker));
  }
  if (H && Hpre) {
    if (term->Hpre_is_H) {
      *H = prototype;
      PetscCall(PetscObjectReference((PetscObject)prototype));
      *Hpre = prototype;
    } else {
      *H    = prototype;
      *Hpre = NULL;
      PetscCall(MatTransposeMatMult(ls_jac, ls_jac, MAT_INITIAL_MATRIX, PETSC_DETERMINE, Hpre));
      {
        PetscContainer marker;
        PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &marker));
        PetscCall(PetscObjectCompose((PetscObject)*Hpre, "__TaoTermGaussNewtonProduct", (PetscObject)marker));
        PetscCall(PetscContainerDestroy(&marker));
      }
    }
  } else if (H) {
    *H = prototype;
  } else if (Hpre) {
    *Hpre = prototype;
  } else {
    PetscCall(MatDestroy(&prototype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsComputeHessianFDPossible_GaussNewton(TaoTerm term, PetscBool3 *ispossible)
{
  PetscFunctionBegin;
  *ispossible = PETSC_BOOL3_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_GaussNewton(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;
  PetscBool            iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Gauss-Newton term wrapping a Tao residual%s\n", gn->tao ? "" : " (Tao not set)"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermDestroy_GaussNewton(TaoTerm term)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&gn->r_work));
  /* gn->tao is a weak reference: do not destroy. */
  PetscCall(PetscFree(gn));
  term->data = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermGaussNewtonSetTao_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermGaussNewtonGetTao_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermGaussNewtonGetJacobian_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGaussNewtonSetTao_GaussNewton(TaoTerm term, Tao tao)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;

  PetscFunctionBegin;
  /* Weak reference: do not Reference/Destroy. The Tao owns the residual
     machinery and will outlive the term in normal use. */
  if (gn->tao != tao) {
    gn->tao       = tao;
    gn->jac_valid = PETSC_FALSE;
    PetscCall(VecDestroy(&gn->r_work));
  }
  if (tao && tao->solution) PetscCall(TaoTermSetSolutionTemplate(term, tao->solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGaussNewtonGetTao_GaussNewton(TaoTerm term, Tao *tao)
{
  TaoTerm_GaussNewton *gn = (TaoTerm_GaussNewton *)term->data;

  PetscFunctionBegin;
  *tao = gn->tao;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGaussNewtonGetJacobian_GaussNewton(TaoTerm term, Vec x, Mat *J)
{
  PetscFunctionBegin;
  PetscCall(TaoTermGaussNewtonGetJacobian_Internal(term, x, J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMGAUSSNEWTON - A `TaoTerm` that wraps the residual machinery of a `Tao` (its
  residual function $R(x)$ and residual Jacobian $J(x)$, set via `TaoSetResidualRoutine()`
  and `TaoSetJacobianResidualRoutine()`) as the smooth least-squares term
  $\tfrac{1}{2}\|R(x)\|_2^2$.  The Hessian is the Gauss-Newton approximation $J(x)^T J(x)$.

  Level: advanced

  Notes:
  This term is `TAOTERM_PARAMETERS_NONE`.

  The default Hessian creation mode is `MATSHELL` for both `H` and `Hpre`. To assemble
  $J^T J$ explicitly, call `TaoTermSetCreateHessianMode()` with a concrete `MatType`
  (e.g. `MATAIJ`), or pass `-<prefix>tao_term_hessian_mat_type aij` at the command line.

  This term caches the residual Jacobian by `(x_id, x_state)` so successive
  objective+gradient and Hessian calls at the same iterate do not redundantly
  invoke the user's Jacobian routine.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreateGaussNewton()`,
          `TaoTermGaussNewtonSetTao()`,
          `TaoTermGaussNewtonGetJacobian()`,
          `TAOBRGN`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Gaussnewton(TaoTerm term)
{
  TaoTerm_GaussNewton *gn;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gn));
  term->data = (void *)gn;

  PetscCall(PetscFree(term->H_mattype));
  PetscCall(PetscFree(term->Hpre_mattype));
  /* Default to MATAIJ so the assembled J^T J integrates with bound-constrained
     solvers that need MatCreateSubMatrix on the Hessian.  Override via
     TaoTermSetCreateHessianMode() (or -<prefix>tao_term_hessian_mat_type) for
     a MATSHELL Hessian when only HessianMult is required. */
  PetscCall(PetscStrallocpy(MATAIJ, (char **)&term->H_mattype));
  PetscCall(PetscStrallocpy(MATAIJ, (char **)&term->Hpre_mattype));
  term->Hpre_is_H = PETSC_TRUE;

  term->ops->destroy                    = TaoTermDestroy_GaussNewton;
  term->ops->view                       = TaoTermView_GaussNewton;
  term->ops->objective                  = TaoTermComputeObjective_GaussNewton;
  term->ops->gradient                   = TaoTermComputeGradient_GaussNewton;
  term->ops->objectiveandgradient       = TaoTermComputeObjectiveAndGradient_GaussNewton;
  term->ops->hessian                    = TaoTermComputeHessian_GaussNewton;
  term->ops->hessianmult                = TaoTermComputeHessianMult_GaussNewton;
  term->ops->createhessianmatrices      = TaoTermCreateHessianMatrices_GaussNewton;
  term->ops->iscomputehessianfdpossible = TaoTermIsComputeHessianFDPossible_GaussNewton;

  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermGaussNewtonSetTao_C", TaoTermGaussNewtonSetTao_GaussNewton));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermGaussNewtonGetTao_C", TaoTermGaussNewtonGetTao_GaussNewton));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermGaussNewtonGetJacobian_C", TaoTermGaussNewtonGetJacobian_GaussNewton));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGaussNewtonSetTao - Set the `Tao` whose residual machinery a `TAOTERMGAUSSNEWTON` wraps.

  Logically collective

  Input Parameters:
+ term - a `TaoTerm` of type `TAOTERMGAUSSNEWTON`
- tao  - a `Tao` whose residual function and Jacobian have been set with
         `TaoSetResidualRoutine()` and `TaoSetJacobianResidualRoutine()`

  Level: advanced

  Note:
  The term holds a weak reference to `tao`. The caller is responsible for ensuring `tao`
  outlives the term.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMGAUSSNEWTON`,
          `TaoTermGaussNewtonGetTao()`,
          `TaoTermCreateGaussNewton()`
@*/
PetscErrorCode TaoTermGaussNewtonSetTao(TaoTerm term, Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (tao) PetscValidHeaderSpecific(tao, TAO_CLASSID, 2);
  PetscTryMethod(term, "TaoTermGaussNewtonSetTao_C", (TaoTerm, Tao), (term, tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGaussNewtonGetTao - Get the `Tao` set by `TaoTermGaussNewtonSetTao()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERMGAUSSNEWTON`

  Output Parameter:
. tao - the wrapped `Tao` (may be `NULL`)

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMGAUSSNEWTON`,
          `TaoTermGaussNewtonSetTao()`
@*/
PetscErrorCode TaoTermGaussNewtonGetTao(TaoTerm term, Tao *tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(tao, 2);
  *tao = NULL;
  PetscTryMethod(term, "TaoTermGaussNewtonGetTao_C", (TaoTerm, Tao *), (term, tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGaussNewtonGetJacobian - Get the residual Jacobian $J(x)$ at the current iterate $x$.

  Collective

  Input Parameters:
+ term - a `TaoTerm` of type `TAOTERMGAUSSNEWTON`
- x    - the current iterate

  Output Parameter:
. J - the residual Jacobian (the same `Mat` returned by `TaoComputeResidualJacobian()`)

  Level: advanced

  Note:
  This routine recomputes $J$ via `TaoComputeResidualJacobian()` only when $x$ has
  changed since the last call (detected via `PetscObjectId` and `PetscObjectState`).
  Companion routines that share the same Jacobian (for example a Levenberg-Marquardt
  damping term) should call this rather than `TaoComputeResidualJacobian()` directly,
  so that the cache is shared.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMGAUSSNEWTON`,
          `TaoTermGaussNewtonSetTao()`,
          `TaoComputeResidualJacobian()`
@*/
PetscErrorCode TaoTermGaussNewtonGetJacobian(TaoTerm term, Vec x, Mat *J)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(J, 3);
  *J = NULL;
  PetscUseMethod(term, "TaoTermGaussNewtonGetJacobian_C", (TaoTerm, Vec, Mat *), (term, x, J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateGaussNewton - Create a `TAOTERMGAUSSNEWTON` over a given `Tao`.

  Collective

  Input Parameter:
. tao - a `Tao` whose residual function and Jacobian have been set with
        `TaoSetResidualRoutine()` and `TaoSetJacobianResidualRoutine()`

  Output Parameter:
. term - a `TaoTerm` that computes $\tfrac{1}{2}\|R(x)\|_2^2$ with Gauss-Newton Hessian $J^T J$

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMGAUSSNEWTON`,
          `TaoTermGaussNewtonSetTao()`,
          `TAOBRGN`
@*/
PetscErrorCode TaoTermCreateGaussNewton(Tao tao, TaoTerm *term)
{
  TaoTerm _term;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(term, 2);
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)tao), &_term));
  PetscCall(TaoTermSetType(_term, TAOTERMGAUSSNEWTON));
  PetscCall(TaoTermGaussNewtonSetTao(_term, tao));
  *term = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}
