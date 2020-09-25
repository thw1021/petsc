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
  PetscErrorCode (*getvalues)(PetscMapping,PetscInt*,const PetscInt*[]);
  PetscErrorCode (*restorevalues)(PetscMapping,PetscInt,const PetscInt*[]);
};

typedef struct _n_PetscMappingContiguous *PetscMappingContiguous;
struct _n_PetscMappingContiguous {
  PetscInt  keyStart, keyEnd;  /* section-like start end */
  PetscInt  keyStartGlobal;    /* offset into global key array */
};

typedef struct _n_PetscMappingDisContiguous *PetscMappingDisContiguous;
struct _n_PetscMappingDisContiguous {
  PetscInt  *keys;             /* local key array */
  PetscInt  *keyIndexGlobal;   /* locations of keys in global key array */
  PetscBool alloced;
};

struct _p_PetscMapping {
  PETSCHEADER(struct _PetscMappingOps);
  PetscMapping map;                      /* Map object to processor */
  PetscMapping permutation;              /* permutation of keys */
  union {
    PetscMappingContiguous    contig;    /* section-like use */
    PetscMappingDisContiguous discontig; /* multi-map-like use */
  };
  PetscInt     nKeysLocal;               /* always cached regardless of contig or not */
  PetscInt     nKeysGlobal;               /* always cached regardless of contig or not */
  PetscMappingState kstorage;               /* are the keys contiguous? */
  PetscBool    sorted;
  PetscBool    setup;
  void         *data;                    /* impls, contains *idx */
};
#endif
