#include <petsc/private/immapimpl.h>

PetscErrorCode IMCreate_Map(IM m)
{
  IM_Map         *immap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &immap);CHKERRQ(ierr);
  m->data = (void *)immap;
  m->ops->destroy = IMDestroy_Map;
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy_Map(IM *m)
{
  IM_Map          *immap = (IM_Map *)(*m)->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(immap->idx);CHKERRQ(ierr);
  ierr = PetscFree((*m)->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
