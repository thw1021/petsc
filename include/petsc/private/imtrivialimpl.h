#if !defined(_PETSCIMTRIVIALIMPL_H)
#define _PETSCIMTRIVIALIMPL_H

#include <petsc/private/imimpl.h>
#include <petscim.h>

typedef struct {
  PetscInt  *idx;
  PetscInt   n;
  PetscInt   bs;
} IM_Trivial;

PETSC_EXTERN PetscErrorCode IMCreate_Trivial(IM);
PETSC_EXTERN PetscErrorCode IMDestroy_Trivial(IM*);
#endif
