#include <petsc/private/imtrivialimpl.h>

PetscErrorCode IMCreate_Trivial(IM m)
{
  IM_Trivial     *pmt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &pmt);CHKERRQ(ierr);
  m->data = (void *)pmt;
  m->ops->destroy = IMDestroy_Trivial;
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy_Trivial(IM *m)
{
  IM_Trivial *pmt = (IM_Trivial *)(*m)->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pmt->idx);CHKERRQ(ierr);
  ierr = PetscFree((*m)->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
