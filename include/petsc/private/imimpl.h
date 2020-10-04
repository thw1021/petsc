#if !defined(PETSCIMIMPL_H)
#define PETSCIMIMPL_H

#include <petscim.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      IMRegisterAllCalled;
PETSC_EXTERN PetscErrorCode IMRegisterAll(void);

typedef struct _IMOps *IMOps;
struct _IMOps {
  PetscErrorCode (*create)(IM);
  PetscErrorCode (*destroy)(IM);
  PetscErrorCode (*view)(IM,PetscViewer);
  PetscErrorCode (*setup)(IM);
  PetscErrorCode (*setfromoptions)(IM);
  PetscErrorCode (*sort)(IM,IMOpMode);
  PetscErrorCode (*convertkeys)(IM,IMState);
  PetscErrorCode (*getindices)(IM,const PetscInt*[]);
  PetscErrorCode (*restorevalues)(IM,PetscInt,const PetscInt*[]);
  PetscErrorCode (*permute)(IM,IM);
};

typedef struct _n_IMInterval *IMInterval;
struct _n_IMInterval {
  PetscInt         keyStart, keyEnd; /* local section-like start end */
  PetscObjectState state;
};

typedef struct _n_IMArray *IMArray;
struct _n_IMArray {
  PetscInt         *keys;       /* local key array */
  PetscBool         alloced;
  PetscObjectState  state;
};

struct _p_IM {
  PETSCHEADER(struct _IMOps);
  IMInterval  interval;         /* section-like use */
  IMArray     array;            /* multi-map-like use */
  void       *data;             /* impls, contains *idx */
  IM          map;              /* map entries to processes */
  PetscInt    nKeys[2];         /* local/global always cached regardless of contig or not */
  PetscInt    bs;               /* blocksize */
  IMState     kstorage;         /* which storage variant is most up to date? */
  PetscBool   sorted[2];        /* are local/global keys sorted */
  PetscBool   generated;        /* were the keys generated from defaults */
  PetscBool   setupcalled;      /* is  __everything__ setup, locks everything */
};

PETSC_STATIC_INLINE PetscErrorCode IMInitializeBase_Private(IM *m)
{
  PetscObjectState  state;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &((*m)->interval));CHKERRQ(ierr);
  ierr = PetscNewLog(m, &((*m)->array));CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject) *m, &state);CHKERRQ(ierr);
  (*m)->interval->state   = state;
  (*m)->array->state      = state;
  (*m)->data              = NULL;
  (*m)->permutation       = NULL;
  (*m)->nKeys[IM_LOCAL]   = PETSC_DECIDE;
  (*m)->nKeys[IM_GLOBAL]  = PETSC_DECIDE;
  (*m)->bs                = -1;
  (*m)->kstorage          = IM_INVALID;
  (*m)->sorted[IM_LOCAL]  = PETSC_FALSE;
  (*m)->sorted[IM_GLOBAL] = PETSC_FALSE;
  (*m)->generated         = PETSC_FALSE;
  (*m)->setupcalled       = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode IMDestroyBase_Private(IM m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(m->interval);CHKERRQ(ierr);
  if (m->array->alloced) {
    ierr = PetscFree(m->array->keys);CHKERRQ(ierr);
  }
  ierr = PetscFree(m->array);CHKERRQ(ierr);
  PetscObjectStateIncrease((PetscObject) m);
  m->kstorage = IM_INVALID;
  PetscFunctionReturn(0);
}
#endif /* PETSCIMIMPL_H */
