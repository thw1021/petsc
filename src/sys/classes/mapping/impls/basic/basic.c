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

PetscErrorCode IMSetup_Basic(PETSC_UNUSED IM m)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode IMBasicCreateFromSizes(MPI_Comm comm, IMState state, PetscInt n, PetscInt N, IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMCreate(comm, m);CHKERRQ(ierr);
  ierr = IMSetType(*m, IMBASIC);CHKERRQ(ierr);
  ierr = IMSetKeyStateAndSizes(*m, state, n, N);CHKERRQ(ierr);
  ierr = IMSetUp(*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
