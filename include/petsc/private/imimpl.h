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
  PetscInt  *globalRanges;     /* global keyStart/End, position is implicit by rank */
};

typedef struct _n_IMArray *IMArray;
struct _n_IMArray {
  PetscInt  *keys;             /* local key array */
  PetscInt  *keyIndexGlobal;   /* locations of keys in global key array */
  PetscBool alloced;
};

struct _p_IM {
  PETSCHEADER(struct _IMOps);
  IM              permutation;  /* permutation of keys */
  union {
    IMContiguous  contig;       /* section-like use */
    IMArray       discontig;    /* multi-map-like use */
  };
  PetscInt        nKeys[2];     /* local/global always cached regardless of contig or not */
  IMState         kstorage;     /* are the keys contiguous? */
  PetscBool       sorted[2];    /* are local/global keys sorted */
  PetscBool       keysetcalled; /* keys were explicitly set, not from sizes */
  PetscBool       setupcalled;  /* is  __everything__ setup, locks everything */
  void           *data;         /* impls, contains *idx */
};

PETSC_STATIC_INLINE PetscErrorCode IMClearKeyState_Private(IM m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (m->contig) {
    ierr = PetscFree(m->contig->globalRanges);CHKERRQ(ierr);
    ierr = PetscFree(m->contig);CHKERRQ(ierr);
  } else if (m->discontig) {
    if (m->discontig->alloced) {
      ierr = PetscFree(m->discontig->keys);CHKERRQ(ierr);
    }
    ierr = PetscFree(m->discontig->keyIndexGlobal);CHKERRQ(ierr);
    ierr = PetscFree(m->discontig);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
 If kstorage == newstate and newstate is valid, frees discontig keys. Sets m->sorted true for contig, false for
 arrray. Sets nKeys to PETSC_DECIDE for local and global, and kstorage.
*/
PETSC_STATIC_INLINE PetscErrorCode IMSetupKeyState_Private(IM m, IMState newstate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (m->kstorage == newstate) {
    /* If m->kstorage is set these are guaranteed to exist */
    if (m->kstorage == IM_ARRAY) {
      if (m->discontig->alloced) {
        ierr = PetscFree(m->discontig->keys);CHKERRQ(ierr);
      }
      ierr = PetscFree(m->discontig->keyIndexGlobal);CHKERRQ(ierr);
      m->sorted[IM_LOCAL] = PETSC_FALSE;
    } else if (m->kstorage == IM_CONTIGUOUS) {
      m->sorted[IM_LOCAL] = PETSC_TRUE;
    }
  } else {
    ierr = IMClearKeyState_Private(m);CHKERRQ(ierr);
    /* Is newstate not IM_INVALID */
    if (newstate) {
      if (newstate == IM_CONTIGUOUS) {
        ierr = PetscNewLog(m, &(m->contig));CHKERRQ(ierr);
        m->sorted[IM_LOCAL] = PETSC_TRUE;
      } else if (newstate == IM_ARRAY) {
        ierr = PetscNewLog(m, &(m->discontig));CHKERRQ(ierr);
        m->sorted[IM_LOCAL] = PETSC_FALSE;
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown IMState");
      }
    }
  }
  m->nKeys[IM_LOCAL] = PETSC_DECIDE;
  m->nKeys[IM_GLOBAL] = PETSC_DECIDE;
  m->kstorage = newstate;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode IMResetBase_Private(IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  (*m)->permutation       = NULL;
  ierr = IMClearKeyState_Private(*m);CHKERRQ(ierr);
  (*m)->contig            = NULL;
  (*m)->discontig         = NULL;
  (*m)->nKeys[IM_LOCAL]   = PETSC_DECIDE;
  (*m)->nKeys[IM_GLOBAL]  = PETSC_DECIDE;
  (*m)->kstorage          = IM_INVALID;
  (*m)->sorted[IM_LOCAL]  = PETSC_FALSE;
  (*m)->sorted[IM_GLOBAL] = PETSC_FALSE;
  (*m)->keysetcalled      = PETSC_FALSE;
  (*m)->setupcalled       = PETSC_FALSE;
  (*m)->data              = NULL;
  PetscFunctionReturn(0);
}
#endif
