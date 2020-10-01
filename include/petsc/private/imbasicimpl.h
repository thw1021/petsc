#if !defined(_PETSCIMBASICIMPL_H)
#define _PETSCIMBASICIMPL_H

#include <petsc/private/imimpl.h>
#include <petscim.h>

typedef struct {
  PetscBool locked;
} IM_Basic;

PETSC_EXTERN PetscErrorCode IMCreate_Basic(IM);
PETSC_EXTERN PetscErrorCode IMDestroy_Basic(IM*);
PETSC_EXTERN PetscErrorCode IMSetup_Basic(IM);
PETSC_EXTERN PetscErrorCode IMPermute_Basic(IM,IM);

PETSC_STATIC_INLINE PetscErrorCode IMResetBasic_Private(IM_Basic *m)
{
  PetscFunctionBegin;
  m->locked = PETSC_FALSE;
  PetscFunctionReturn(0);
}
#endif
