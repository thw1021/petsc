#include <../src/tao/leastsquares/impls/brgn/brgn.h>
#include <petscmat.h>

/*
  Internal helper for TAOBRGN: a TAOTERMSHELL companion to TAOTERMGAUSSNEWTON
  that produces the Levenberg-Marquardt damping `diag(J^T J)`.

  The damping is a function of the iterate `x` only via the Gauss-Newton term's
  cached residual Jacobian: this term holds a reference to the GN term and uses
  TaoTermGaussNewtonGetJacobian() so that successive evaluations at the same x
  do not re-trigger TaoComputeResidualJacobian().

  Mathematically the term is the quadratic form (1/2) v^T diag(J^T J) v with
  Hessian diag(J^T J). In the BRGN setup the objective and gradient
  contributions are masked off (TAOTERM_MASK_OBJECTIVE | TAOTERM_MASK_GRADIENT)
  because the damping is a step regularizer, not part of the objective; the
  obj/grad implementations here are provided for self-consistency.

  This is not a public PETSc symbol: it is used only by brgn.c.
*/

typedef struct {
  TaoTerm gn;       // strong reference to the companion TAOTERMGAUSSNEWTON
  Mat     diag_mat; // MATDIAGONAL holding the damping diagonal
  Vec     diag;     // alias of MatDiagonalGetDiagonal(diag_mat); held while in scope

  PetscObjectId    diag_x_id;
  PetscObjectState diag_x_state;
  PetscBool        diag_valid;
} TaoTerm_BRGNLM;

static PetscErrorCode TaoTermBRGNLMUpdateDiagonal(TaoTerm term, Vec x)
{
  TaoTerm_BRGNLM  *lm = NULL;
  Mat              ls_jac;
  PetscObjectId    x_id;
  PetscObjectState x_state;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, &lm));
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (lm->diag_valid && lm->diag_x_id == x_id && lm->diag_x_state == x_state) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(TaoTermGaussNewtonGetJacobian(lm->gn, x, &ls_jac));
  {
    PetscInt     n, cstart, cend;
    PetscReal   *cnorms;
    Vec          dvec;
    PetscScalar *darr;

    PetscCall(MatGetSize(ls_jac, NULL, &n));
    PetscCall(PetscMalloc1(n, &cnorms));
    PetscCall(MatGetColumnNorms(ls_jac, NORM_2, cnorms));
    PetscCall(MatGetOwnershipRangeColumn(ls_jac, &cstart, &cend));
    PetscCall(MatDiagonalGetDiagonal(lm->diag_mat, &dvec));
    PetscCall(VecGetArray(dvec, &darr));
    for (PetscInt i = 0; i < cend - cstart; i++) {
      PetscReal v = cnorms[cstart + i];
      v *= v;
      darr[i] = PetscClipInterval(v, PETSC_SQRT_MACHINE_EPSILON, PetscSqrtReal(PETSC_MAX_REAL));
    }
    PetscCall(VecRestoreArray(dvec, &darr));
    PetscCall(MatDiagonalRestoreDiagonal(lm->diag_mat, &dvec));
    PetscCall(PetscFree(cnorms));
  }
  lm->diag_x_id    = x_id;
  lm->diag_x_state = x_state;
  lm->diag_valid   = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBRGNLMObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_BRGNLM *lm = NULL;
  PetscScalar     dot;

  PetscFunctionBegin;
  PetscCall(TaoTermBRGNLMUpdateDiagonal(term, x));
  PetscCall(TaoTermShellGetContext(term, &lm));
  PetscCall(MatMult(lm->diag_mat, x, g));
  PetscCall(VecDot(x, g, &dot));
  *value = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBRGNLMHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_BRGNLM *lm = NULL;

  PetscFunctionBegin;
  PetscCall(TaoTermBRGNLMUpdateDiagonal(term, x));
  PetscCall(TaoTermShellGetContext(term, &lm));
  if (H) PetscCall(MatCopy(lm->diag_mat, H, UNKNOWN_NONZERO_PATTERN));
  if (Hpre && Hpre != H) PetscCall(MatCopy(lm->diag_mat, Hpre, UNKNOWN_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBRGNLMHessianMult(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  TaoTerm_BRGNLM *lm = NULL;

  PetscFunctionBegin;
  PetscCall(TaoTermBRGNLMUpdateDiagonal(term, x));
  PetscCall(TaoTermShellGetContext(term, &lm));
  PetscCall(MatMult(lm->diag_mat, v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBRGNLMCreateHessianMatrices(TaoTerm term, Mat *H, Mat *Hpre)
{
  TaoTerm_BRGNLM *lm = NULL;
  Vec             dvec;

  PetscFunctionBegin;
  PetscCall(TaoTermShellGetContext(term, &lm));
  PetscCall(MatDiagonalGetDiagonal(lm->diag_mat, &dvec));
  if (H) {
    Vec d_copy;
    PetscCall(VecDuplicate(dvec, &d_copy));
    PetscCall(MatCreateDiagonal(d_copy, H));
    PetscCall(VecDestroy(&d_copy));
  }
  if (Hpre) {
    if (H) {
      PetscCall(PetscObjectReference((PetscObject)*H));
      *Hpre = *H;
    } else {
      Vec d_copy;
      PetscCall(VecDuplicate(dvec, &d_copy));
      PetscCall(MatCreateDiagonal(d_copy, Hpre));
      PetscCall(VecDestroy(&d_copy));
    }
  }
  PetscCall(MatDiagonalRestoreDiagonal(lm->diag_mat, &dvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermBRGNLMDestroy(PetscCtxRt ctx_arg)
{
  TaoTerm_BRGNLM **slot = (TaoTerm_BRGNLM **)ctx_arg;
  TaoTerm_BRGNLM  *lm   = *slot;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&lm->diag_mat));
  PetscCall(TaoTermDestroy(&lm->gn));
  PetscCall(PetscFree(lm));
  *slot = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermCreateBRGNLMDamping - create a TAOTERMSHELL implementing the Levenberg-Marquardt
  damping diag(J^T J) for a companion TAOTERMGAUSSNEWTON.

  The created term holds a strong reference to `gn` and uses
  TaoTermGaussNewtonGetJacobian() to share J's caching.

  Default Hessian creation mode is MATDIAGONAL (matched to MATDIAGONAL).
*/
PETSC_INTERN PetscErrorCode TaoTermCreateBRGNLMDamping(TaoTerm gn, TaoTerm *lm)
{
  TaoTerm         _term;
  TaoTerm_BRGNLM *ctx;
  Tao             tao;
  Vec             dvec;
  PetscBool       is_gn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(gn, TAOTERM_CLASSID, 1);
  PetscAssertPointer(lm, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)gn, TAOTERMGAUSSNEWTON, &is_gn));
  PetscCheck(is_gn, PetscObjectComm((PetscObject)gn), PETSC_ERR_ARG_WRONGSTATE, "TaoTermCreateBRGNLMDamping requires a TAOTERMGAUSSNEWTON companion");
  PetscCall(TaoTermGaussNewtonGetTao(gn, &tao));
  PetscCheck(tao && tao->solution, PetscObjectComm((PetscObject)gn), PETSC_ERR_ARG_WRONGSTATE, "TAOTERMGAUSSNEWTON companion must have a Tao with TaoSetSolution() called");

  PetscCall(PetscNew(&ctx));
  PetscCall(PetscObjectReference((PetscObject)gn));
  ctx->gn = gn;
  PetscCall(VecDuplicate(tao->solution, &dvec));
  PetscCall(VecSet(dvec, 0.0));
  PetscCall(MatCreateDiagonal(dvec, &ctx->diag_mat));
  PetscCall(VecDestroy(&dvec));

  PetscCall(TaoTermCreateShell(PetscObjectComm((PetscObject)gn), (PetscCtx)ctx, TaoTermBRGNLMDestroy, &_term));
  PetscCall(TaoTermSetParametersMode(_term, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionTemplate(_term, tao->solution));
  PetscCall(TaoTermShellSetObjectiveAndGradient(_term, TaoTermBRGNLMObjectiveAndGradient));
  PetscCall(TaoTermShellSetHessian(_term, TaoTermBRGNLMHessian));
  PetscCall(TaoTermShellSetHessianMult(_term, TaoTermBRGNLMHessianMult));
  PetscCall(TaoTermShellSetCreateHessianMatrices(_term, TaoTermBRGNLMCreateHessianMatrices));
  PetscCall(TaoTermSetCreateHessianMode(_term, PETSC_TRUE, MATDIAGONAL, MATDIAGONAL));

  *lm = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermBRGNLMDampingCopyDiagonal - copy the current LM damping diagonal into a
  caller-provided Vec compatible with the LM term's solution layout.
*/
PETSC_INTERN PetscErrorCode TaoTermBRGNLMDampingCopyDiagonal(TaoTerm lm, Vec dst)
{
  TaoTerm_BRGNLM *ctx = NULL;
  Vec             dvec;
  PetscBool       is_shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lm, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(dst, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)lm, TAOTERMSHELL, &is_shell));
  PetscCheck(is_shell, PetscObjectComm((PetscObject)lm), PETSC_ERR_ARG_WRONG, "TaoTermBRGNLMDampingCopyDiagonal requires the BRGN LM-damping shell term");
  PetscCall(TaoTermShellGetContext(lm, &ctx));
  PetscCall(MatDiagonalGetDiagonal(ctx->diag_mat, &dvec));
  PetscCall(VecCopy(dvec, dst));
  PetscCall(MatDiagonalRestoreDiagonal(ctx->diag_mat, &dvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}
