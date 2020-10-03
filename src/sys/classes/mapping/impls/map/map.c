#include <petsc/private/immapimpl.h>

PetscErrorCode IMCreate_Map(IM m)
{
  IM_Map         *mm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &mm);CHKERRQ(ierr);
  m->data = (void *)mm;
  m->ops->destroy = IMDestroy_Map;
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy_Map(IM m)
{
  IM_Map          *mm = (IM_Map *) m->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mm->idx);CHKERRQ(ierr);
  ierr = PetscFree(mm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
