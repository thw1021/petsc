#include <petsc/private/imbasicimpl.h>

PetscErrorCode IMCreate_Basic(IM m)
{
  IM_Basic       *mb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &mb);CHKERRQ(ierr);
  ierr = IMInitializeBasic_Private(mb);CHKERRQ(ierr);
  m->data = (void *)mb;
  m->ops->destroy = IMDestroy_Basic;
  m->ops->setup = IMSetup_Basic;
  m->ops->permute = IMPermute_Basic;
  m->ops->getindices = IMGetIndices_Basic;
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy_Basic(IM m)
{
  IM_Basic       *mb = (IM_Basic *) m->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMDestroyBasic_Private(mb);CHKERRQ(ierr);
  ierr = PetscFree(mb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetup_Basic(IM m)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetIndices_Basic(IM m, const PetscInt *idx[])
{
  PetscFunctionBegin;
  *idx = m->idx;
  PetscFunctionReturn(0);
}

/* Must take IM_BASIC as input for both, and m must be IM_ARRAY since permuting interval makes no sense here */
PetscErrorCode IMPermute_Basic(IM m, IM pm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
