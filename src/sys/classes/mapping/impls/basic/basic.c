#include <petsc/private/imbasicimpl.h>

PetscErrorCode IMCreate_Basic(IM m)
{
  IM_Basic       *imb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &imb);CHKERRQ(ierr);
  m->data = (void *)imb;
  m->ops->destroy = IMDestroy_Basic;
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy_Basic(IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*m)->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
