#include <petsc/private/imlayoutimpl.h>

PetscErrorCode IMCreate_Layout(IM m)
{
  IM_Layout       *ml;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &ml);CHKERRQ(ierr);
  ierr = IMInitializeLayout_Private(ml);CHKERRQ(ierr);
  m->echelon = 1234; /* need better way of telling it not to make a map */
  m->data = (void *)ml;
  m->ops->destroy = IMDestroy_Layout;
  m->ops->setup = IMSetUp_Layout;
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy_Layout(IM m)
{
  IM_Layout       *ml = (IM_Layout *) m->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMDestroyLayout_Private(ml);CHKERRQ(ierr);
  ierr = PetscFree(ml);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetUp_Layout(IM m)
{
  IM_Layout      *ml = (IM_Layout *) m->data;
  PetscInt       minmax[2];
  const PetscInt nk = m->nIdx[IM_LOCAL];
  PetscMPIInt    size, rank;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*size, &ml->ranges);CHKERRQ(ierr);
  if (m->sorted[IM_LOCAL]) {
    minmax[0] = m->idx[0];
    minmax[1] = m->idx[nk-1];
  } else {
    PetscInt *tmp;

    ierr = PetscMalloc1(nk, &tmp);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmp, m->idx, nk);CHKERRQ(ierr);
    ierr = PetscIntSortSemiOrdered(nk, tmp);CHKERRQ(ierr);
    minmax[0] = tmp[0];
    minmax[1] = tmp[nk-1];
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  }
  ierr = MPI_Allgather(minmax, 2, MPIU_INT, ml->ranges, 2, MPIU_INT, comm);CHKERRQ(ierr);
  ml->rstart = ml->ranges[2*rank];
  ml->rend   = ml->ranges[2*rank+1];
  PetscFunctionReturn(0);
}

PetscErrorCode IMLayoutCreate(MPI_Comm comm, IM *m)
{
  IM             mt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(m,2);
  ierr = IMCreate(comm, &mt);CHKERRQ(ierr);
  ierr = IMSetType(mt, IMLAYOUT);CHKERRQ(ierr);
  *m = mt;
  PetscFunctionReturn(0);
}

PetscErrorCode IMLayoutSetFromMapping(IM m, PetscInt n, const PetscInt glob[], PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(m,IM_CLASSID,1,IMLAYOUT);
  ierr = IMSetIndices(m, n, glob, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMLayoutSetFromSizes(IM m, PetscInt n, PetscInt N)
{
  PetscInt       *arr;
  PetscInt       i, sum = 0;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(m,IM_CLASSID,1,IMLAYOUT);
  ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(comm, &n, &N);CHKERRQ(ierr);
  ierr = MPI_Exscan(&n, &sum, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &arr);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) arr[i] = sum+i;
  ierr = IMSetIndices(m, n, arr, PETSC_OWN_POINTER);CHKERRQ(ierr);
  m->nIdx[IM_GLOBAL] = N;
  m->sorted[IM_LOCAL] = IM_TRUE;
  m->sorted[IM_GLOBAL] = IM_TRUE;
  PetscFunctionReturn(0);
}
