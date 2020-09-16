#include <petsc/private/mappingtrivialimpl.h>

PetscErrorCode PetscMappingCreate_Trivial(PetscMapping m)
{
  PetscMapping_Trivial *pmt;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &pmt);CHKERRQ(ierr);
  m->data = (void *)pmt;
  m->ops->destroy = PetscMappingDestroy_Trivial;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingDestroy_Trivial(PetscMapping m)
{
  PetscMapping_Trivial *pmt = (PetscMapping_Trivial *)m->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pmt->idx);CHKERRQ(ierr);
  ierr = PetscFree(m->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
