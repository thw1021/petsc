#if !defined(PETSCIMBASICIMPL_H)
#define PETSCIMBASICIMPL_H

#include <petsc/private/imimpl.h>
#include <petscim.h>

typedef struct {
  PetscInt  bs, stride;
} IM_Basic;

PETSC_INTERN PetscErrorCode IMCreate_Basic(IM);
PETSC_INTERN PetscErrorCode IMSetup_Basic(IM);
PETSC_INTERN PetscErrorCode IMDestroy_Basic(IM);
PETSC_INTERN PetscErrorCode IMGetIndices_Basic(IM,const PetscInt*[]);
PETSC_INTERN PetscErrorCode IMPermute_Basic(IM,IM);

PETSC_STATIC_INLINE PetscErrorCode IMInitializeBasic_Private(IM_Basic *mb)
{
  PetscFunctionBegin;
  mb->bs     = PETSC_DECIDE;
  mb->stride = PETSC_DECIDE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode IMDestroyBasic_Private(IM_Basic *mb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMInitializeBasic_Private(mb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSCIMBASICIMPL_H */
