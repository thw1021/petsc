#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_L1 TaoTerm_L1;

struct _n_TaoTerm_L1 {
  PetscReal        epsilon;
  PetscBool        epsilon_warning;
  Vec              diff; // caches $x - p$
  Vec              d;    // caches $d_i = sqrt((diff_i)^2 + epsilon^2)
  PetscReal        d_epsilon;
  PetscReal        d_sum;
  PetscObjectId    d_x_id, d_p_id;
  PetscObjectState d_x_state, d_p_state;
  Vec              diag;
  PetscObjectId    diag_x_id, diag_p_id;
  PetscObjectState diag_x_state, diag_p_state;
  PetscReal        diag_epsilon;
};

static PetscErrorCode TaoTermDestroy_L1(TaoTerm term)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  term->data = NULL;
  PetscCall(VecDestroy(&l1->diff));
  PetscCall(VecDestroy(&l1->d));
  PetscCall(VecDestroy(&l1->diag));
  PetscCall(PetscFree(l1));
  PetscCall(TaoTermDestroy_ElementwiseDivergence_Internal(term));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermL1SetEpsilon_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermL1GetEpsilon_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1ComputeData(TaoTerm term, Vec x, Vec params, Vec *_diff, Vec *d)
{
  TaoTerm_L1      *l1 = (TaoTerm_L1 *)term->data;
  PetscObjectId    x_id, p_id       = 0;
  PetscObjectState x_state, p_state = 0;
  Vec              diff = x;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (params) {
    PetscCall(PetscObjectGetId((PetscObject)params, &p_id));
    PetscCall(PetscObjectStateGet((PetscObject)params, &p_state));
  }
  if (l1->d_x_id != x_id || l1->d_x_state != x_state || l1->d_p_id != p_id || l1->d_p_state != p_state || l1->d_epsilon != l1->epsilon) {
    l1->d_x_id    = x_id;
    l1->d_x_state = x_state;
    l1->d_p_id    = p_id;
    l1->d_p_state = p_state;
    l1->d_epsilon = l1->epsilon;
    diff          = x;
    if (params) {
      if (!l1->diff) PetscCall(VecDuplicate(x, &l1->diff));
      PetscCall(VecWAXPY(l1->diff, -1.0, params, x));
      diff = l1->diff;
    }
    if (l1->epsilon != 0.0) {
      if (!l1->d) PetscCall(VecDuplicate(x, &l1->d));
      PetscCall(VecPointwiseMult(l1->d, diff, diff));
      PetscCall(VecShift(l1->d, l1->epsilon * l1->epsilon));
      PetscCall(VecSqrtAbs(l1->d));
    }
  }
  *_diff = diff;
  *d     = l1->d;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1ComputeDiag(TaoTerm term, Vec x, Vec params, Vec *diag)
{
  Vec              diff, d;
  TaoTerm_L1      *l1 = (TaoTerm_L1 *)term->data;
  PetscObjectId    x_id, p_id       = 0;
  PetscObjectState x_state, p_state = 0;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeData(term, x, params, &diff, &d));
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (params) {
    PetscCall(PetscObjectGetId((PetscObject)params, &p_id));
    PetscCall(PetscObjectStateGet((PetscObject)params, &p_state));
  }
  if (l1->diag_x_id != x_id || l1->diag_x_state != x_state || l1->diag_p_id != p_id || l1->diag_p_state != p_state || l1->diag_epsilon != l1->epsilon) {
    l1->diag_x_id    = x_id;
    l1->diag_x_state = x_state;
    l1->diag_p_id    = p_id;
    l1->diag_p_state = p_state;
    l1->diag_epsilon = l1->epsilon;
    if (!l1->diag) PetscCall(VecDuplicate(x, &l1->diag));
    PetscCall(VecCopy(d, l1->diag));
    PetscCall(VecPointwiseMult(l1->diag, l1->diag, d));
    PetscCall(VecPointwiseMult(l1->diag, l1->diag, d));
    PetscCall(VecReciprocal(l1->diag));
    PetscCall(VecScale(l1->diag, l1->epsilon * l1->epsilon));
  }
  *diag = l1->diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_L1_Internal(TaoTerm term, Vec diff, Vec d, PetscReal *value)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  if (l1->epsilon == 0.0) {
    PetscCall(VecNorm(diff, NORM_1, value));
  } else {
    PetscScalar sum;
    PetscInt    n;

    PetscCall(VecGetSize(d, &n));
    PetscCall(VecSum(d, &sum));
    *value = PetscRealPart(sum) - n * l1->epsilon;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_L1(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  Vec diff, d;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeData(term, x, params, &diff, &d));
  PetscCall(TaoTermObjective_L1_Internal(term, diff, d, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_L1_Internal(TaoTerm term, Vec diff, Vec d, Vec g)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  if (l1->epsilon == 0.0) {
    if (!l1->epsilon_warning) {
      l1->epsilon_warning = PETSC_TRUE;
      PetscCall(PetscInfo(term, "Asking for gradient of l1 norm, which is not smooth.  Consider smoothing the TaoTerm with TaoTermL1SetEpsilon() or using a Tao that does not require gradients\n"));
    }
    PetscCall(VecCopy(diff, g));
    PetscCall(VecSign(g));
  } else {
    PetscCall(VecPointwiseDivide(g, diff, d));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_L1(TaoTerm term, Vec x, Vec params, Vec g)
{
  Vec diff, d;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeData(term, x, params, &diff, &d));
  PetscCall(TaoTermGradient_L1_Internal(term, diff, d, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_L1(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  Vec diff, d;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeData(term, x, params, &diff, &d));
  PetscCall(TaoTermObjective_L1_Internal(term, diff, d, value));
  PetscCall(TaoTermGradient_L1_Internal(term, diff, d, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_L1_Internal(TaoTerm term, Vec diag, Mat H)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  if (l1->epsilon == 0.0) {
    if (!l1->epsilon_warning) {
      l1->epsilon_warning = PETSC_TRUE;
      PetscCall(PetscInfo(term, "Asking for Hessian of l1 norm, which is not smooth.  Consider smoothing the TaoTerm with TaoTermL1SetEpsilon() or using a Tao that does not require Hessians\n"));
    }
    PetscCall(MatZeroEntries(H));
  } else {
    PetscCall(MatDiagonalSet(H, diag, INSERT_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_L1(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  Vec diag;

  PetscFunctionBegin;
  PetscCall(TaoTermUpdateHessianShells(term, x, params, &H, &Hpre));
  if (!H && !Hpre) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoTermL1ComputeDiag(term, x, params, &diag));
  if (H) PetscCall(TaoTermHessian_L1_Internal(term, diag, H));
  if (Hpre && Hpre != H) PetscCall(TaoTermHessian_L1_Internal(term, diag, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessianMult_L1_Internal(TaoTerm term, Vec diag, Vec v, Vec Hv)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  if (l1->epsilon == 0.0) {
    if (!l1->epsilon_warning) {
      l1->epsilon_warning = PETSC_TRUE;
      PetscCall(PetscInfo(term, "Asking for Hessian of l1 norm, which is not smooth.  Consider smoothing the TaoTerm with TaoTermL1SetEpsilon() or using a Tao that does not require Hessians\n"));
    }
    PetscCall(VecZeroEntries(Hv));
  } else {
    PetscCall(VecPointwiseMult(Hv, v, diag));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessianMult_L1(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  Vec diag;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeDiag(term, x, params, &diag));
  PetscCall(TaoTermHessianMult_L1_Internal(term, diag, v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermL1SetEpsilon - Set an $\epsilon$ smoothing parameter.

  Logically collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERML1`
- epsilon - a real number $\geq 0$

  Options Database Keys:
. -taoterm_l1_epsilon <real> - $\epsilon$

  Level: advanced

  Note:
  If $\epsilon = 0$ (the default), then `term` computes $\|x - p\|_1$, but if $\epsilon > 0$, then it computes
  $\sum_{i=0}^n | \sqrt((x_i-p_i)^2 + epsilon^2) - \epsilon|$.

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERML1`, `TaoTermL1GetEpsilon()`
@*/
PetscErrorCode TaoTermL1SetEpsilon(TaoTerm term, PetscReal epsilon)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(term, epsilon, 2);
  PetscTryMethod(term, "TaoTermL1SetEpsilon_C", (TaoTerm, PetscReal), (term, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1SetEpsilon_L1(TaoTerm term, PetscReal epsilon)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  PetscCheck(epsilon > 0, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_OUTOFRANGE, "L1 epsilon (%g) cannot be < 0.0", (double)epsilon);
  l1->epsilon = epsilon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermL1GetEpsilon - Get the $\epsilon$ smoothing parameter set by `TaoTermL1SetEpsilon()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERML1`

  Output Parameter:
. epsilon - the smoothing parameter

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERML1`
@*/
PetscErrorCode TaoTermL1GetEpsilon(TaoTerm term, PetscReal *epsilon)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(epsilon, 2);
  PetscUseMethod(term, "TaoTermL1GetEpsilon_C", (TaoTerm, PetscReal *), (term, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1GetEpsilon_L1(TaoTerm term, PetscReal *epsilon)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  *epsilon = l1->epsilon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_L1(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;
  PetscBool   is_ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &is_ascii));
  if (is_ascii) PetscCall(PetscViewerASCIIPrintf(viewer, "epsilon (taoterm_l1_epsilon): %g\n", (double)l1->epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetFromOptions_L1(TaoTerm term, PetscOptionItems *PetscOptionsObject)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoTerm l1 options");
  PetscCall(PetscOptionsBoundedReal("-taoterm_l1_epsilon", "smoothing parameter", "TaoTermL1SetEpsilon", l1->epsilon, &l1->epsilon, NULL, 0.0));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERML1 - A `TaoTerm` that computes $\|x - p\|_1$, for solution $x$ and parameters $p$.

  Level: intermediate

  Options Database Keys:
. -taoterm_l1_epsilon <real> - (default 0.0) a smoothing parameter (see `TaoTermL1SetEpsilon()`)

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMHALFL2SQUARED`
MC*/
PETSC_INTERN PetscErrorCode TaoTermCreate_L1(TaoTerm term)
{
  TaoTerm_L1 *l1;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&l1));
  term->data = (void *)l1;

  term->ops->destroy               = TaoTermDestroy_L1;
  term->ops->view                  = TaoTermView_L1;
  term->ops->setfromoptions        = TaoTermSetFromOptions_L1;
  term->ops->objective             = TaoTermObjective_L1;
  term->ops->gradient              = TaoTermGradient_L1;
  term->ops->objectiveandgradient  = TaoTermObjectiveAndGradient_L1;
  term->ops->hessian               = TaoTermHessian_L1;
  term->ops->hessianmult           = TaoTermHessianMult_L1;
  term->ops->createhessianmatrices = TaoTermCreateHessianMatricesDefault;

  if (!term->H_mattype) PetscCall(PetscStrallocpy(MATSHELL, &term->H_mattype));
  if (!term->Hpre_mattype) PetscCall(PetscStrallocpy(MATDIAGONAL, &term->Hpre_mattype));

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermL1SetEpsilon_C", TaoTermL1SetEpsilon_L1));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermL1GetEpsilon_C", TaoTermL1GetEpsilon_L1));
  PetscFunctionReturn(PETSC_SUCCESS);
}
