#include <petsc/private/imbasicimpl.h>

PetscErrorCode IMCreate_Basic(IM m)
{
  IM_Basic       *imb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &imb);CHKERRQ(ierr);
  ierr = IMResetBasic_Private(imb);CHKERRQ(ierr);
  m->data = (void *)imb;
  m->ops->destroy = IMDestroy_Basic;
  m->ops->setup = IMSetup_Basic;
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy_Basic(IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*m)->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetup_Basic(IM m)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)m, &comm);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(comm, &m->nKeys[IM_LOCAL], &m->nKeys[IM_GLOBAL]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "%D %D\n", m->nKeys[IM_LOCAL], m->nKeys[IM_GLOBAL]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
