#if !defined(PETSCIMIMPL_H)
#define PETSCIMIMPL_H

#include <petscim.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      IMRegisterAllCalled;
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
  IMContiguous       contig;      /* section-like use */
  IMDisContiguous    discontig;   /* multi-map-like use */
  PetscInt           nKeys[2];    /* local/global always cached regardless of contig or not */
  IMState            kstorage;    /* are the keys contiguous? */
  PetscBool          sorted[2];   /* are local/global keys sorted */
  PetscBool          setup;
  void               *data;       /* impls, contains *idx */
};

PETSC_STATIC_INLINE PetscErrorCode IMResetBase_Private(IM *m)
{
  PetscFunctionBegin;
  (*m)->map = NULL;
  (*m)->permutation = NULL;
  (*m)->nKeys[IM_LOCAL] = PETSC_DECIDE;
  (*m)->nKeys[IM_GLOBAL] = PETSC_DECIDE;
  (*m)->sorted[IM_LOCAL] = PETSC_FALSE;
  (*m)->sorted[IM_GLOBAL] = PETSC_FALSE;
  if ((*m)->kstorage) {
    PetscErrorCode ierr;
    if ((*m)->kstorage == IM_CONTIGUOUS) {ierr = PetscFree((*m)->contig);CHKERRQ(ierr);}
    else {ierr = PetscFree((*m)->discontig);CHKERRQ(ierr);}
  }
  (*m)->kstorage = IM_INVALID;
  (*m)->contig = NULL;
  (*m)->discontig = NULL;
  (*m)->data = NULL;
  (*m)->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode IMCheckKeyState_Private(IM m, IMState intendedstate)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    MPI_Comm       comm;
    PetscErrorCode ierr;

    ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
    if (m->setup) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on already setup map");
    if (m->kstorage && (m->kstorage != intendedstate)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage has already been set");
  }
  PetscFunctionReturn(0);
}
#endif
