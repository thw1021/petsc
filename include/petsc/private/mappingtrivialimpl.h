#if !defined(_MAPPINGTRIVIALIMPL_H)
#define _MAPPINGTRIVIALIMPL_H

#include <petscmapping.h>
#include <petsc/private/mappingimpl.h>

typedef struct {
  PetscInt  *idx;
  PetscInt   n;
} PetscMapping_Trivial;

#endif
