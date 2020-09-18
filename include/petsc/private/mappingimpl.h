#if !defined(PETSCMAPPINGIMPL_H)
#define PETSCMAPPINGIMPL_H

#include <petscmapping.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool PetscMappingRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscMappingRegisterAll(void);

typedef struct _PetscMappingOps *PetscMappingOps;
struct _PetscMappingOps {
  PetscErrorCode (*create)(MPI_Comm,PetscMapping*);
  PetscErrorCode (*destroy)(PetscMapping*);
  PetscErrorCode (*view)(PetscMapping,PetscViewer);
  PetscErrorCode (*setup)(PetscMapping);
  PetscErrorCode (*setfromoptions)(PetscMapping);
  PetscErrorCode (*sort)(PetscMapping);
  PetscErrorCode (*sorted)(PetscMapping);
  PetscErrorCode (*getkeys)(PetscMapping,PetscInt*,const PetscInt*[]);
  PetscErrorCode (*getvalues)(PetscMapping,PetscInt*,const PetscInt*[]);
  PetscErrorCode (*restorekeys)(PetscMapping,PetscInt,const PetscInt*[]);
  PetscErrorCode (*restorevalues)(PetscMapping,PetscInt,const PetscInt*[]);
};

typedef struct _n_PetscMappingContiguous *PetscMappingContiguous;
struct {
  PetscInt  keyStart, keyEnd;  /* section-like start end */
  PetscInt  keyStartGlobal;    /* offset into global key array */
} _n_PetscMappingContiguous;

typedef struct _n_PetscMappingNonContiguous *PetscMappingNonContiguous;
struct {
  PetscInt  *keys;             /* local key array */
  PetscInt  *keyIndexGlobal;   /* locations of keys in global key array */
} _n_PetscMappingNonContiguous;

struct _p_PetscMapping {
  PETSCHEADER(struct _PetscMappingOps);
  PetscMapping map;                      /* Map object to processor */
  PetscMapping permutation;              /* permutation of keys */
  union {
    PetscMappingContiguous    contig;    /* section-like use */
    PetscMappingNonContiguous noncontig; /* multi-map-like use */
  };
  PetscInt     nKeysLocal;               /* always cached regardless of contig or not */
  PetscBool    contiguous;               /* are the keys contiguous? */
  PetscBool    sorted;
  void         *data;                    /* impls, contains *idx */
};

#if (0)
struct _p_PetscMapping {
  PETSCHEADER(struct _PetscMappingOps);
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
  PetscBool         setup;
  void              *data;
};
#endif

#endif
