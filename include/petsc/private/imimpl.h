#if !defined(PETSCIMIMPL_H)
#define PETSCIMIMPL_H

#include <petscim.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool IMRegisterAllCalled;
PETSC_EXTERN PetscErrorCode IMRegisterAll(void);

typedef struct _IMOps *IMOps;
struct _IMOps {
  PetscErrorCode (*create)(MPI_Comm,IM*);
  PetscErrorCode (*destroy)(IM*);
  PetscErrorCode (*view)(IM,PetscViewer);
  PetscErrorCode (*setup)(IM);
  PetscErrorCode (*setfromoptions)(IM);
  PetscErrorCode (*sort)(IM,IMOpMode);
  PetscErrorCode (*getvalues)(IM,PetscInt*,const PetscInt*[]);
  PetscErrorCode (*restorevalues)(IM,PetscInt,const PetscInt*[]);
  PetscErrorCode (*permute)(IM);
};

typedef struct _n_IMContiguous *IMContiguous;
struct _n_IMContiguous {
  PetscInt  keyStart, keyEnd;  /* section-like start end */
  PetscInt  keyStartGlobal;    /* offset into global key array */
};

typedef struct _n_IMDisContiguous *IMDisContiguous;
struct _n_IMDisContiguous {
  PetscInt  *keys;             /* local key array */
  PetscInt  *keyIndexGlobal;   /* locations of keys in global key array */
  PetscBool alloced;
};

struct _p_IM {
  PETSCHEADER(struct _IMOps);
  IM                 map;         /* Map object to processor */
  IM                 permutation; /* permutation of keys */
  union {
    IMContiguous     contig;      /* section-like use */
    IMDisContiguous  discontig;   /* multi-map-like use */
  };
  PetscInt           nKeys[2];    /* local/global always cached regardless of contig or not */
  IMState            kstorage;    /* are the keys contiguous? */
  PetscBool          sorted[2];   /* are local/global keys sorted */
  PetscBool          setup;
  void              *data;        /* impls, contains *idx */
};
#endif
