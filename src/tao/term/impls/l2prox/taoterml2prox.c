#include <petsc/private/taoimpl.h> /*I "petsctaoterm.h" I*/

typedef struct {
  PetscObjectId    id;
  PetscObjectState state;
  PetscBool        valid;
} TaoTermL2ProxHessianState;

typedef struct {
  Vec                       center;
  Vec                       work;
  TaoTermL2ProxHessianState H_state;
  TaoTermL2ProxHessianState Hpre_state;
} TaoTerm_L2Prox;

static PetscErrorCode TaoTermDestroy_L2Prox(TaoTerm term)
{
  TaoTerm_L2Prox *prox = (TaoTerm_L2Prox *)term->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&prox->center));
  PetscCall(VecDestroy(&prox->work));
  PetscCall(PetscFree(prox));
  term->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetUp_L2Prox(TaoTerm term)
{
  TaoTerm_L2Prox *prox = (TaoTerm_L2Prox *)term->data;

  PetscFunctionBegin;
  if (!prox->center) {
    PetscCall(TaoTermCreateSolutionVec(term, &prox->center));
    PetscCall(VecZeroEntries(prox->center));
  }
  if (!prox->work) PetscCall(VecDuplicate(prox->center, &prox->work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_L2Prox(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_L2Prox *prox = (TaoTerm_L2Prox *)term->data;
  PetscScalar     dot;

  PetscFunctionBegin;
  PetscCall(VecWAXPY(prox->work, -1.0, prox->center, x));
  PetscCall(VecDot(prox->work, prox->work, &dot));
  *value = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeGradient_L2Prox(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_L2Prox *prox = (TaoTerm_L2Prox *)term->data;

  PetscFunctionBegin;
  PetscCall(VecWAXPY(g, -1.0, prox->center, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjectiveAndGradient_L2Prox(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  PetscScalar dot;

  PetscFunctionBegin;
  PetscCall(TaoTermComputeGradient_L2Prox(term, x, params, g));
  PetscCall(VecDot(g, g, &dot));
  *value = 0.5 * PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL2ProxSetIdentity(Mat H, TaoTermL2ProxHessianState *cached)
{
  PetscObjectId    id;
  PetscObjectState state;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)H, &id));
  PetscCall(MatGetState(H, &state));
  if (cached->valid && cached->id == id && cached->state == state) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatZeroEntries(H));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(H, 1.0));
  PetscCall(MatGetState(H, &cached->state));
  cached->id    = id;
  cached->valid = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_L2Prox(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_L2Prox *prox = (TaoTerm_L2Prox *)term->data;

  PetscFunctionBegin;
  if (H) PetscCall(TaoTermL2ProxSetIdentity(H, &prox->H_state));
  if (Hpre && Hpre != H) PetscCall(TaoTermL2ProxSetIdentity(Hpre, &prox->Hpre_state));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessianMult_L2Prox(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(v, Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateHessianMatrices_L2Prox(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreateHessianMatricesDefault(term, H, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermL2ProxUpdate_Private(TaoTerm term, Tao tao, PetscInt iter)
{
  TaoTerm_L2Prox *prox;
  PetscBool       is_prox;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)term, TAOTERML2PROX, &is_prox));
  if (!is_prox) PetscFunctionReturn(PETSC_SUCCESS);
  prox = (TaoTerm_L2Prox *)term->data;
  if (iter > 0) PetscCall(VecCopy(tao->solution, prox->center));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERML2PROX - A proximal term $\frac{1}{2}\|x-x_k\|_2^2$ used by `TAOBRGN`

  Level: advanced

  Notes:
  The initial center is the zero vector. `TAOBRGN` updates the center to the current
  accepted solution at the beginning of subsequent iterations.

  This term is `TAOTERM_PARAMETERS_NONE`.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermCreateL2Prox()`, `TAOBRGN`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_L2Prox(TaoTerm term)
{
  TaoTerm_L2Prox *prox;

  PetscFunctionBegin;
  PetscCall(PetscNew(&prox));
  term->data = prox;
  PetscCall(PetscFree(term->H_mattype));
  PetscCall(PetscFree(term->Hpre_mattype));
  PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->H_mattype));
  PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->Hpre_mattype));
  term->Hpre_is_H                  = PETSC_TRUE;
  term->ops->destroy               = TaoTermDestroy_L2Prox;
  term->ops->setup                 = TaoTermSetUp_L2Prox;
  term->ops->objective             = TaoTermComputeObjective_L2Prox;
  term->ops->gradient              = TaoTermComputeGradient_L2Prox;
  term->ops->objectiveandgradient  = TaoTermComputeObjectiveAndGradient_L2Prox;
  term->ops->hessian               = TaoTermComputeHessian_L2Prox;
  term->ops->hessianmult           = TaoTermComputeHessianMult_L2Prox;
  term->ops->createhessianmatrices = TaoTermCreateHessianMatrices_L2Prox;
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_NONE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateL2Prox - Create a proximal regularizer for `TAOBRGN`

  Collective

  Input Parameters:
+ comm - the communicator
. n    - the local solution size
- N    - the global solution size

  Output Parameter:
. term - the `TAOTERML2PROX`

  Level: advanced

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERML2PROX`, `TaoBRGNAddRegularizerTerm()`
@*/
PetscErrorCode TaoTermCreateL2Prox(MPI_Comm comm, PetscInt n, PetscInt N, TaoTerm *term)
{
  TaoTerm prox;

  PetscFunctionBegin;
  PetscAssertPointer(term, 4);
  PetscCall(TaoTermCreate(comm, &prox));
  PetscCall(TaoTermSetType(prox, TAOTERML2PROX));
  PetscCall(TaoTermSetSolutionSizes(prox, n, N, 1));
  *term = prox;
  PetscFunctionReturn(PETSC_SUCCESS);
}
