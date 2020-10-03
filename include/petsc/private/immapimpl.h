#if !defined(PETSCIMMAPIMPL_H)
#define PETSCIMMAPIMPL_H

#include <petsc/private/imimpl.h>
#include <petscim.h>

typedef struct {
  PetscInt  *idx;
  PetscInt   n;
  PetscInt   bs;
} IM_Map;

PETSC_EXTERN PetscErrorCode IMCreate_Map(IM);
PETSC_EXTERN PetscErrorCode IMDestroy_Map(IM);
#endif /* PETSCIMMAPIMPL_H */
