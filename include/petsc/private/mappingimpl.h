#if !defined(PETSCMAPPINGIMPL_H)
#define PETSCMAPPINGIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscmapping.h>

struct _p_PetscMapping {
  PETSCHEADER(int);
  PetscMapping      *maps;  /* n-recurvise map objects */
  PetscInt          *keys;  /* "keys" to depth 1 "values" */
  PetscInt          *cidx;  /* cached depth 1 "values" */
  PetscInt          *dof;   /* number of depth 1 "values" per "key" */
  PetscInt          nidx;   /* number of total depth 1 "values" in the map */
  PetscInt          nblade; /* number of "keys" to the map */
  PetscMappingState valid;  /* are maps more up to date than cidx or vice versa */
  PetscBool         iallocated; /* are cidx copied */
  PetscBool         kallocated; /* are keys copied */
  PetscBool         mallocated; /* are maps copied */
};

#endif
