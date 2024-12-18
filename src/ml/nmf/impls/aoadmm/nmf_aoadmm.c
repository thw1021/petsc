#include <../src/ml/nmf/impls/aoadmm/nmf_aoadmm.h> /*I "petscnmf.h" I*/
#include <petsctao.h>
#include <petsc/private/nmfimpl.h>

static PetscErrorCode PetscNMFView_AOADMM(PetscNMF nmf, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNMFSetFromOptions_AOADMM(PetscNMF nmf, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscNMF AOADMM Options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNMFFit_AOADMM(PetscNMF nmf)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNMFSetUp_AOADMM(PetscNMF nmf)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscNMFDestroy_AOADMM(PetscNMF nmf)
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

PETSC_EXTERN PetscErrorCode  PetscNMFCreate_AOADMM(PetscNMF nmf)
{
  PetscNMF_AOADMM *aoadmm;

  PetscFunctionBegin;
  PetscCall(PetscNew(&aoadmm));
  nmf->data = (void *)aoadmm;

  nmf->ops->fit            = PetscNMFFit_AOADMM;
  nmf->ops->setup          = PetscNMFSetUp_AOADMM;
  nmf->ops->setfromoptions = PetscNMFSetFromOptions_AOADMM;
  nmf->ops->view           = PetscNMFView_AOADMM;
  nmf->ops->destroy        = PetscNMFDestroy_AOADMM;
  PetscFunctionReturn(PETSC_SUCCESS);
}
