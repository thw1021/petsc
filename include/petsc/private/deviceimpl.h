#if !defined(PETSC_DEVICEIMPL_H)
#define PETSC_DEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

/* non-error returning version for the inline macro */
PETSC_STATIC_INLINE PetscBool PetscDeviceContextTypeCompare_Internal(const char type_ref[], const char type_name[])
{
  if (!type_ref && !type_name) {
    return PETSC_TRUE;
  } else if (!type_ref || !type_name) {
    return PETSC_FALSE;
  } else if (strcmp(type_ref,type_name)) {
    return PETSC_FALSE;
  }
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextTypeCompare(const char type_ref[], const char type_name[], PetscBool *same)
{
  PetscFunctionBegin;
  PetscValidCharPointer(type_ref,1);
  PetscValidCharPointer(type_name,2);
  PetscValidBoolPointer(same,3);
  *same = PetscDeviceContextTypeCompare_Internal(type_ref,type_name);
  PetscFunctionReturn(0);
}

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename T>
void PetscValidStreamType(T,int);
template <typename T>
void PetscValidStreamTypeSpecific(T,int,const char[]);
template <typename T>
void PetscValidSameStreamType(T,int,T,int);
#else /* PETSC_CLANG_STATIC_ANALYZER */
#if PetscDefined(USE_DEBUG)
#define PetscValidStreamType(_p_strm__,_p_arg__)                        \
  do {                                                                  \
    PetscValidPointer(_p_strm__,_p_arg__);                              \
    PetscValidCharPointer((_p_strm__)->type,_p_arg__);                  \
    if (PetscUnlikely(!(_p_strm__)->type)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_TYPENOTSET,"PetscDeviceContextType is not set: Argument #%d",(_p_arg__)); \
  } while (0)

#define PetscValidStreamTypeSpecific(_p_strm__,_p_arg__,_p_type__)      \
  do {                                                                  \
    PetscBool      _type_same_strm_;                                    \
    PetscErrorCode _strm_ierr_;                                         \
    PetscValidStreamType(_p_strm__,_p_arg__);                           \
    _strm_ierr_=PetscDeviceContextTypeCompare((_p_strm__)->type,(_p_type__),&_type_same_strm_);CHKERRQ(_strm_ierr_); \
    if (PetscUnlikely(!_type_same_strm_)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"PetscDeviceContextType %s is incompatible: Argument # %d",(_p_type__),(_p_arg__)); \
  } while (0)

#define PetscCheckValidSameStreamType(_p_strm1__,_p_arg1__,_p_strm2__,_p_arg2__) \
  do {                                                                  \
    PetscValidStreamType(_p_strm1__,_p_arg1__);                         \
    PetscValidStreamType(_p_strm2__,_p_arg2__);                         \
    if (PetscUnlikely(!PetscDeviceContextTypeCompare_Internal((_p_strm1__)->type,(_p_strm2__)->type))) { \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"PetscDeviceContextType %s is incompatible with PetscDeviceContextType %s: Arguments # %d and # %d",((_p_strm1__)->type),((_p_strm2__)->type),(_p_arg1__),(_p_arg2__)); \
    }                                                                   \
  } while (0)
#else /* PETSC_USE_DEBUG */
#define PetscValidStreamType(_p_strm__,_p_arg__)                   ((void)(_p_strm__))
#define PetscValidStreamTypeSpecific(_p_strm__,_p_arg__,_p_type__) ((void)(_p_strm__))
#define PetscCheckValidSameStreamType(_p_strm1__,_p_arg1__,_p_strm2__,_p_arg2__) (void)(_p_strm1__),(void)(_p_strm2__)
#endif /* PETSC_USE_DEBUG */
#endif /* PETSC_CLANG_STATIC_ANALYZER */

typedef struct _DeviceContextOps *DeviceContextOps;
struct _DeviceContextOps {
  PetscErrorCode (*create)(PetscDeviceContext);
  PetscErrorCode (*destroy)(PetscDeviceContext);
  PetscErrorCode (*changestreamtype)(PetscDeviceContext,PetscStreamType);
  PetscErrorCode (*setup)(PetscDeviceContext);
  PetscErrorCode (*query)(PetscDeviceContext,PetscBool*);
  PetscErrorCode (*waitforctx)(PetscDeviceContext,PetscDeviceContext);
  PetscErrorCode (*synchronize)(PetscDeviceContext);
};

struct _n_PetscDeviceContext {
  struct _DeviceContextOps  ops[1];
  char                     *type;
  void                     *data;            /* solver contexts, event, stream */
  PetscBool                 idle;            /* does this context think it has work? this value non-binding in debug mode */
  PetscInt                 *childIDs;        /* array containing ids of context forked from this one */
  PetscInt                  id;              /* unique id per created context */
  PetscInt                  numChildren;     /* how many children does this context expect to destroy */
  PetscInt                  maxNumChildren;  /* how many children can this context have room for without realloc'ing */
  PetscStreamType           streamType;
  PetscBool                 setup;
};

/* Called in debug-mode when a context claims it is idle to check that it isn't lying. A no-op when debugging is
 disabled */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextValidateIdle_Internal(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool      idle;
    PetscErrorCode ierr;

    ierr = (*dctx->ops->query)(dctx,&idle);CHKERRQ(ierr);
    if (PetscUnlikely(dctx->idle && !idle)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContext cache corrupted, stream thought it was idle when it still had work");
  }
  PetscFunctionReturn(0);
}
#endif /* PETSC_DEVICEIMPL_H */
