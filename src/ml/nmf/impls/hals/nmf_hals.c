#include <../src/ml/nmf/impls/hals/nmf_hals.h> /*I "petscnmf.h" I*/
#include <petsctao.h>
#include <petsc/private/nmfimpl.h>

static PetscErrorCode PetscNMFView_HALS(PetscNMF nmf, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNMFSetFromOptions_HALS(PetscNMF nmf, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscNMF HALS Options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNMFFit_HALS(PetscNMF nmf)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNMFSetUp_HALS(PetscNMF nmf)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNMFDestroy_HALS(PetscNMF nmf)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(nmf->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCNMFADMM - Alternating objective ADMM to solve NMF

  Level: beginner

.seealso: PetscNMFCreate(), PetscNMF, PetscNMFSetType()
M*/

PETSC_EXTERN PetscErrorCode  PetscNMFCreate_HALS(PetscNMF nmf)
{
  PetscNMF_HALS *hals;

  PetscFunctionBegin;
  PetscCall(PetscNew(&hals));
  nmf->data = (void *)hals;

  nmf->ops->fit            = PetscNMFFit_HALS;
  nmf->ops->setup          = PetscNMFSetUp_HALS;
  nmf->ops->setfromoptions = PetscNMFSetFromOptions_HALS;
  nmf->ops->view           = PetscNMFView_HALS;
  nmf->ops->destroy        = PetscNMFDestroy_HALS;
  PetscFunctionReturn(PETSC_SUCCESS);
}
