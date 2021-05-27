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

typedef struct _EventOps *EventOps;
struct _EventOps {
  PetscErrorCode (*create)(PetscEvent);
  PetscErrorCode (*destroy)(PetscEvent);
  PetscErrorCode (*setup)(PetscEvent);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscEvent);
  PetscErrorCode (*synchronize)(PetscEvent);
  PetscErrorCode (*query)(PetscEvent,PetscBool*);
};

struct _n_PetscEvent {
  struct _EventOps ops[1];
  char             *type;
  void             *data;
  unsigned int     eventFlags, waitFlags;
  PetscInt         laststreamid;
  PetscBool        idle;
  PetscBool        setup;
  PetscBool        setfromoptionscalled;
};

typedef struct _DeviceContextOps *DeviceContextOps;
struct _DeviceContextOps {
  PetscErrorCode (*create)(PetscDeviceContext);
  PetscErrorCode (*destroy)(PetscDeviceContext);
  PetscErrorCode (*setup)(PetscDeviceContext);
  PetscErrorCode (*getstream)(PetscDeviceContext,void*);
  PetscErrorCode (*restorestream)(PetscDeviceContext,void*);
  PetscErrorCode (*getblashandle)(PetscDeviceContext,void*);
  PetscErrorCode (*restoreblashandle)(PetscDeviceContext,void*);
  PetscErrorCode (*query)(PetscDeviceContext,PetscBool*);
  PetscErrorCode (*waitforctx)(PetscDeviceContext,PetscDeviceContext);
  PetscErrorCode (*join)(PetscDeviceContext);
};

struct _n_PetscDeviceContext {
  struct _DeviceContextOps  ops[1];
  char                     *type;
  void                     *data; // solver contexts, event, stream
  PetscBool                 idle;
  PetscInt                 *childIDs;
  PetscInt                  id,numChildren,maxNumChildren;
  PetscStreamMode           mode;
  PetscEvent                event;
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

PETSC_STATIC_INLINE PetscErrorCode PetscEventValidateIdle_Internal(PetscEvent event)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool      idle;
    PetscErrorCode ierr;

    ierr = (*event->ops->query)(event,&idle);CHKERRQ(ierr);
    if (PetscUnlikely(event->idle && !idle)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscEvent cache corrupted, event thought it was idle when it still had work");
  }
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_STATIC_INLINE PetscErrorCode PetscCUBLASSetStream_Internal(cublasHandle_t cublasv2handle, cudaStream_t cstrm)
{
  cudaStream_t        cublasStrm;
  cublasPointerMode_t mode;
  cublasStatus_t      cberr;

  PetscFunctionBegin;
  /* We get an check these since setting these blindly would "reset the workspace". It is not clear whether cublas
   checks for equality internally. */
  cberr = cublasGetStream(cublasv2handle,&cublasStrm);CHKERRCUBLAS(cberr);
  if (cstrm != cublasStrm) {cberr = cublasSetStream(cublasv2handle,cstrm);CHKERRCUBLAS(cberr);}
  cberr = cublasGetPointerMode(cublasv2handle,&mode);CHKERRCUBLAS(cberr);
  if (mode != CUBLAS_POINTER_MODE_DEVICE) {
    cberr = cublasSetPointerMode(cublasv2handle,CUBLAS_POINTER_MODE_DEVICE);CHKERRCUBLAS(cberr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscCUBLASSetHost_Internal(cublasHandle_t cublasv2handle)
{
  cudaStream_t        cublasStrm;
  cublasPointerMode_t mode;
  cublasStatus_t      cberr;

  PetscFunctionBegin;
  cberr = cublasGetStream(cublasv2handle,&cublasStrm);CHKERRCUBLAS(cberr);
  if (cublasStrm) {cberr = cublasSetStream(cublasv2handle,NULL);CHKERRCUBLAS(cberr);}
  cberr = cublasGetPointerMode(cublasv2handle,&mode);CHKERRCUBLAS(cberr);
  if (mode != CUBLAS_POINTER_MODE_HOST) {
    cberr = cublasSetPointerMode(cublasv2handle,CUBLAS_POINTER_MODE_HOST);CHKERRCUBLAS(cberr);
  }
  PetscFunctionReturn(0);

}
#endif
#endif /* DEVICEIMPL_H */
