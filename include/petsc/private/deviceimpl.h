#if !defined(DEVICEIMPL_H)
#define DEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

PETSC_EXTERN PetscErrorCode PetscDeviceRegisterAll(void);

PETSC_STATIC_INLINE PetscErrorCode PetscStreamTypeCompare(const char type_ref[], const char type_name[], PetscBool *same)
{
  PetscFunctionBegin;
  if (!type_ref && !type_name) *same = PETSC_TRUE;
  else if (!type_ref || !type_name) *same = PETSC_FALSE;
  else {
    PetscErrorCode ierr;
    ierr = PetscStrcmp(type_ref,type_name,same);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if PetscDefined(USE_DEBUG)
#define PetscValidStreamType(_p_strm__,_p_arg__)                        \
  do {                                                                  \
    if (PetscUnlikely(!(_p_strm__))) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Object is NULL: Argument #%d",(_p_arg__)); \
    if (PetscUnlikely(!(_p_strm__)->type)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_TYPENOTSET,"PetscStreamType is not set: Argument #%d",(_p_arg__)); \
  } while (0)

#define PetscValidStreamTypeSpecific(_p_strm__,_p_arg__,_p_type__) \
  do {                                                                  \
    PetscBool      _type_same_strm_;                                    \
    PetscErrorCode _strm_ierr_;                                         \
    PetscValidStreamType(_p_strm__,_p_arg__);                           \
    _strm_ierr_=PetscStreamTypeCompare((_p_strm__)->type,(_p_type__),&_type_same_strm_);CHKERRQ(_strm_ierr_); \
    if (PetscUnlikely(!_type_same_strm_)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"PetscStreamType %s is incompatible: Argument # %d",(_p_type__),(_p_arg__)); \
  } while (0)

#define PetscCheckValidSameStreamType(_p_strm1__,_p_arg1__,_p_strm2__,_p_arg2__) \
  do {                                                                  \
    PetscBool      _type_same_strm_2_;                                  \
    PetscErrorCode _strm_ierr_2_;                                       \
    PetscValidStreamType(_p_strm1__,_p_arg1__);                         \
    PetscValidStreamType(_p_strm2__,_p_arg2__);                         \
    _strm_ierr_2_=PetscStreamTypeCompare((_p_strm1__)->type,(_p_strm2__)->type,&_type_same_strm_2_);CHKERRQ(_strm_ierr_2_); \
    if (PetscUnlikely(!_type_same_strm_2_)) {                           \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"PetscStreamType %s is incompatible with PetscStreamType %s: Arguments # %d and # %d",((_p_strm1__)->type),((_p_strm2__)->type),(_p_arg1__),(_p_arg2__)); \
    }                                                                   \
  } while (0)
#else
#define PetscValidStreamType(_p_strm__,_p_arg__)                   ((void)(_p_strm__))
#define PetscValidStreamTypeSpecific(_p_strm__,_p_arg__,_p_type__) ((void)(_p_strm__))
#define PetscCheckValidSameStreamType(_p_strm1__,_p_arg1__,_p_strm2__,_p_arg2__) do {(void)(_p_strm1__);(void)(_p_strm2__);} while (0)
#endif

typedef struct _DeviceContextOps *DeviceContextOps;
struct _DeviceContextOps {
  PetscErrorCode (*create)(PetscDeviceContext);
  PetscErrorCode (*destroy)(PetscDeviceContext);
  PetscErrorCode (*setup)(PetscDeviceContext);
  PetscErrorCode (*getblashandle)(PetscDeviceContext,void*);
  PetscErrorCode (*restoreblashandle)(PetscDeviceContext,void*);
  PetscErrorCode (*getsolverhandle)(PetscDeviceContext,void*);
  PetscErrorCode (*restoresolverhandle)(PetscDeviceContext,void*);
  PetscErrorCode (*query)(PetscDeviceContext,PetscBool*);
  PetscErrorCode (*waitforctx)(PetscDeviceContext,PetscDeviceContext);
  PetscErrorCode (*synchronize)(PetscDeviceContext);
};

struct _n_PetscDeviceContext {
  struct _DeviceContextOps  ops[1];
  char                     *type;
  void                     *data; // solver contexts, event, stream
  PetscBool                 idle;
  PetscInt                 *childIDs;
  PetscInt                  id,numChildren,maxNumChildren;
  PetscStreamMode           mode;
  PetscBool                 setup;
};

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
#endif /* DEVICEIMPL_H */
