#if !defined(PETSCIMBASICIMPL_H)
#define PETSCIMBASICIMPL_H

#include <petsc/private/imimpl.h>
#include <petscim.h>

typedef struct {
  PetscBool locked;
  PetscInt  *globalRanges;
} IM_Basic;

PETSC_EXTERN PetscErrorCode IMCreate_Basic(IM);
PETSC_EXTERN PetscErrorCode IMSetup_Basic(IM);
PETSC_EXTERN PetscErrorCode IMDestroy_Basic(IM);
PETSC_EXTERN PetscErrorCode IMPermute_Basic(IM,IM);

PETSC_STATIC_INLINE PetscErrorCode IMInitializeBasic_Private(IM_Basic *mb)
{
  PetscFunctionBegin;
  mb->locked = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode IMDestroyBasic_Private(IM_Basic *mb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(mb->globalRanges);CHKERRQ(ierr);
  mb->locked = PETSC_FALSE;
  PetscFunctionReturn(0);
}
#endif /* PETSCIMBASICIMPL_H */
