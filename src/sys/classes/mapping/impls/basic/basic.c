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
  IM_Basic       *mb = (IM_Basic *) m->data;
  PetscInt       min, max, send[2];
  PetscMPIInt    size;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = PetscCalloc1(2*size, &(mb->globalRanges));CHKERRQ(ierr);
  switch (m->kstorage) {
  case IM_INTERVAL:
    min = m->interval->keyStart;
    max = m->interval->keyEnd;
    break;
  case IM_ARRAY:
    if (m->sorted[IM_LOCAL]) {
      min = m->array->keys[0];
      max = m->array->keys[m->nKeys[IM_LOCAL]-1];
    } else {
      PetscInt *keycopy;

      ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &keycopy);CHKERRQ(ierr);
      ierr = PetscArraycpy(keycopy, m->array->keys, m->nKeys[IM_LOCAL]);CHKERRQ(ierr);
      ierr = PetscIntSortSemiOrdered(m->nKeys[IM_LOCAL], keycopy);CHKERRQ(ierr);
      min = keycopy[0];
      max = keycopy[m->nKeys[IM_LOCAL]-1];
      ierr = PetscFree(keycopy);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid IMState");
    break;
  }
  send[0] = min, send[1] = max;
  ierr = MPI_Allgather(send, 2, MPIU_INT, mb->globalRanges, 2, MPIU_INT, comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetIndices_Basic(IM m, const PetscInt *idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMConvertKeyState(m, IM_ARRAY);CHKERRQ(ierr);
  *idx = m->array->keys;
  PetscFunctionReturn(0);
}

/* Must take IM_BASIC as input for both, and m must be IM_ARRAY since permuting interval makes no sense here */
PetscErrorCode IMPermute_Basic(IM m, IM pm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (m->kstorage) {
  case IM_INTERVAL:
    if (PetscDefined(USE_DEBUG)) {
      ierr = PetscInfo1(m,"Input index is %s, makes no sense to permute\n", IMStates[m->kstorage]);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
    break;
  case IM_ARRAY:
  {
    PetscInt *tmp;
    PetscInt i;

    ierr = IMConvertKeyState(pm, IM_ARRAY);CHKERRQ(ierr);
    ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &tmp);CHKERRQ(ierr);
    for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) tmp[i] = m->array->keys[pm->array->keys[i]];
    ierr = PetscArraycpy(m->array->keys, tmp, m->nKeys[IM_LOCAL]);CHKERRQ(ierr);
    ierr = PetscFree(tmp);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown IMState");
    break;
  }
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
