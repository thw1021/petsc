#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petscmath.h>

static PetscErrorCode TaoSolve_Prox(Tao tao)
{
  DM        dm0, dm1;
  PetscReal step;
  Vec       y;

  PetscFunctionBegin;

  PetscCheck(tao->num_terms == 2, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Number of DMTao has to be two.");
  dm0 = tao->dms[0];
  dm1 = tao->dms[1];

  PetscCall(DMTaoGetCentralVector(dm1, &y));
  PetscCall(DMTaoGetScale(dm1, &step));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);
    /* Solve */
    PetscCall(DMTaoApplyProximalMap(dm0, dm1, step, y, tao->solution, NULL));
    /* TODO is there better converged_reason than user?
     * Also, if Method is iterative using subtao, how to deal with it wrt converged reason and monitor?  */
    tao->reason = TAO_CONVERGED_USER;
    /* Note: Changes to stepsize / VM should be done outside TAOPROX */
    tao->niter++;
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_Prox(Tao tao)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_Prox(Tao tao)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_Prox(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Proximal algorithm optimization");
  /* Not supporting options to change prox type, as it doesn't makes sense */
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_Prox(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     TAOPROX -  Proximal algorithm.

   Solves proximal operator. User gives function, f(x), and this algorithm solves

   min_x f(x) + g(x,y), where f(x) is tao->pds[0], and g(x,y) is tao->pds[1].

  Notes:
     Prox formulas are:
  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_Prox(Tao tao)
{
  PetscFunctionBegin;

  tao->ops->setup          = TaoSetUp_Prox;
  tao->ops->solve          = TaoSolve_Prox;
  tao->ops->view           = TaoView_Prox;
  tao->ops->setfromoptions = TaoSetFromOptions_Prox;
  tao->ops->destroy        = TaoDestroy_Prox;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;
  PetscFunctionReturn(PETSC_SUCCESS);
}
