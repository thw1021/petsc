#if !defined(PETSCMAPPINGIMPL_H)
#define PETSCMAPPINGIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscmapping.h>

struct _p_PetscMapping {
  PETSCHEADER(int);
  PetscMapping *maps;
  PetscInt     *indices;
  PetscInt     *offsets;
  PetscInt     *idx2mapidx; /* maps maps to entries in indices */
  PetscInt     maxNumChildMaps;
  PetscInt     numChildMaps;
  PetscInt     size;
  PetscBool    allocated;
};

#endif
