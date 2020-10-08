#if !defined(PETSCIMBASICIMPL_H)
#define PETSCIMBASICIMPL_H

#include <petsc/private/imimpl.h>
#include <petscim.h>

typedef struct {
  PetscInt  *globalRanges;
  PetscInt  rstart, rend;
} IM_Basic;

PETSC_INTERN PetscErrorCode IMCreate_Basic(IM);
PETSC_INTERN PetscErrorCode IMSetup_Basic(IM);
PETSC_INTERN PetscErrorCode IMDestroy_Basic(IM);
PETSC_INTERN PetscErrorCode IMGetIndices_Basic(IM,const PetscInt*[]);
PETSC_INTERN PetscErrorCode IMPermute_Basic(IM,IM);

PETSC_STATIC_INLINE PetscErrorCode IMInitializeBasic_Private(PETSC_UNUSED IM_Basic *mb)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode IMDestroyBasic_Private(IM_Basic *mb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mb->globalRanges);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSCIMBASICIMPL_H */
