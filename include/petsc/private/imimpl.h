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
  PetscErrorCode (*getsizes)(IM,PetscInt*,PetscInt*);
  PetscErrorCode (*getindices)(IM,const PetscInt*[]);
  PetscErrorCode (*restorevalues)(IM,PetscInt,const PetscInt*[]);
  PetscErrorCode (*permute)(IM,IM);
};

typedef enum {
  IM_UNKNOWN = -1,
  IM_FALSE = PETSC_FALSE,
  IM_TRUE = PETSC_TRUE
} IMBool;

struct _p_IM {
  PETSCHEADER(struct _IMOps);
  void      *data;              /* impls, contains *idx */
  PetscInt  *idx;               /* local key array */
  PetscInt   nIdx[IM_MAX_MODE]; /* local/global keys */
  IMBool     sorted[IM_MAX_MODE];
  IM         map;
  PetscBool  allowedMap;
  PetscBool  alloced;
  PetscBool  setupcalled;       /* is  __everything__ setup, locks everything */
  PetscObjectState cstate;
};

PETSC_STATIC_INLINE PetscErrorCode IMInitializeBase_Private(IM m)
{
  PetscFunctionBegin;
  m->data              = NULL;
  m->idx               = NULL;
  m->nIdx[IM_LOCAL]    = PETSC_DECIDE;
  m->nIdx[IM_GLOBAL]   = PETSC_DECIDE;
  m->sorted[IM_LOCAL]  = IM_UNKNOWN;
  m->sorted[IM_GLOBAL] = IM_UNKNOWN;
  m->map               = NULL;
  m->allowedMap        = PETSC_FALSE;
  m->alloced           = PETSC_TRUE;
  m->setupcalled       = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode IMDestroyBase_Private(IM m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(m->idx);CHKERRQ(ierr);
  ierr = IMDestroy(&m->map);CHKERRQ(ierr);
  ierr = IMInitializeBase_Private(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSCIMIMPL_H */
