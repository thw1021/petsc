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
  IM_Basic *mb = (IM_Basic *) m->data;

  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"TODO");
  mb->locked = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Must take IM_BASIC as input for both, and m must be IM_ARRAY since permuting interval makes no sense here */
PetscErrorCode IMPermute_Basic(IM m, IM pm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"TODO");
#if (0)
  switch (m->kstorage) {
  case IM_INTERVAL:
    ierr = PetscInfo1(m,"Input index is %s, makes no sense to permute\n", IMStates[m->kstorage]);CHKERRQ(ierr);
    PetscFunctionReturn(0);
    break;
  case IM_ARRAY:
  {
    IMState  ostate = pm->kstorage;
    PetscInt *tmp, *tmp2;
    PetscInt i;

    ierr = IMConvertKeyState(pm, IM_ARRAY);CHKERRQ(ierr);
    ierr = IMSetUp(pm);CHKERRQ(ierr);
    /* Need to do this separately since they are freed separately */
    ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &tmp);CHKERRQ(ierr);
    ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &tmp2);CHKERRQ(ierr);
    for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) {
      tmp[i] = m->array->keys[pm->discontig->keys[i]];
      tmp2[i] = m->array->keyIndexGlobal[pm->discontig->keys[i]];
    }
    if (m->discontig->alloced) {
      ierr = PetscFree(m->discontig->keys);CHKERRQ(ierr);
    }
    ierr = PetscFree(m->discontig->keyIndexGlobal);CHKERRQ(ierr);
    m->discontig->keys = tmp;
    m->discontig->keyIndexGlobal = tmp2;
    m->discontig->alloced = PETSC_TRUE;
    ierr = IMConvertKeyState(pm, ostate);CHKERRQ(ierr);
    ierr = IMSetUp(pm);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown IMState");
    break;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode IMBasicCreateFromSizes(MPI_Comm comm, IMState state, PetscInt n, PetscInt N, IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMGenerateDefault(comm, IMBASIC, state, n, N, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMBasicCreateFromRanges(MPI_Comm comm, IMState state, const PetscInt ranges[], PetscCopyMode mode, IM *m)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"TODO");
  ierr = IMCreate(comm, m);CHKERRQ(ierr);
  ierr = IMSetType(*m, IMBASIC);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = IMSetKeyInterval(*m, ranges[2*rank], ranges[2*rank+1]);CHKERRQ(ierr);
  ierr = IMConvertKeyState(*m, state);CHKERRQ(ierr);
  ierr = IMSetUp(*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
