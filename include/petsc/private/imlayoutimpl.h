#if !defined(PETSCIMLAYOUTIMPL_H)
#define PETSCIMLAYOUTIMPL_H

#include <petsc/private/imimpl.h> /* I petscim.h */

typedef struct {
  PetscInt  stride, bs;
  PetscInt  rstart, rend;
  PetscInt  *ranges;
} IM_Layout;

PETSC_INTERN PetscErrorCode IMCreate_Layout(IM);
PETSC_INTERN PetscErrorCode IMDestroy_Layout(IM);
PETSC_INTERN PetscErrorCode IMSetUp_Layout(IM);

PETSC_STATIC_INLINE PetscErrorCode IMInitializeLayout_Private(IM_Layout *ml)
{
  PetscFunctionBegin;
  ml->stride = -1;
  ml->bs     = -1;
  ml->rstart = PETSC_DECIDE;
  ml->rend   = PETSC_DECIDE;
  ml->ranges = NULL;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode IMDestroyLayout_Private(IM_Layout *ml)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ml->stride = -1;
  ml->bs     = -1;
  ml->rstart = PETSC_DECIDE;
  ml->rend   = PETSC_DECIDE;
  ierr = PetscFree(ml->ranges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSCIMLAYOUT_H */
