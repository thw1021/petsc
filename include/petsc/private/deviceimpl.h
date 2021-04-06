#if !defined(DEVICEIMPL_H)
#define DEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

PETSC_EXTERN PetscErrorCode PetscStreamRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscEventRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscStreamScalarRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscStreamGraphRegisterAll(void);

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

typedef struct _StreamOps *StreamOps;
struct _StreamOps {
  PetscErrorCode (*create)(PetscStream);
  PetscErrorCode (*destroy)(PetscStream);
  PetscErrorCode (*setup)(PetscStream);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscStream);
  PetscErrorCode (*getstream)(PetscStream,void*);
  PetscErrorCode (*restorestream)(PetscStream,void*);
  PetscErrorCode (*recordevent)(PetscStream,PetscEvent);
  PetscErrorCode (*waitevent)(PetscStream,PetscEvent);
  PetscErrorCode (*synchronize)(PetscStream);
  PetscErrorCode (*query)(PetscStream,PetscBool*);
  PetscErrorCode (*capturebegin)(PetscStream);
  PetscErrorCode (*captureend)(PetscStream,PetscStreamGraph);
  PetscErrorCode (*waitforstream)(PetscStream,PetscStream);
};

struct _n_PetscStream {
  struct _StreamOps ops[1];
  char              *type;
  void              *data;
  PetscInt          id;
  PetscBool         idle;
  PetscStreamMode   mode;
  PetscBool         setup;
  PetscBool         setfromoptionscalled;
};

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

typedef struct _ScalOps *ScalOps;
struct _ScalOps {
  PetscErrorCode (*create)(PetscStreamScalar);
  PetscErrorCode (*destroy)(PetscStreamScalar);
  PetscErrorCode (*setup)(PetscStreamScalar);
  PetscErrorCode (*setvalue)(PetscStreamScalar,const PetscScalar*,PetscMemType,PetscStream);
  PetscErrorCode (*await)(PetscStreamScalar,PetscScalar*,PetscStream);
  PetscErrorCode (*getdevice)(PetscStreamScalar,PetscScalar**,PetscBool,PetscStream);
  PetscErrorCode (*restoredevice)(PetscStreamScalar,PetscScalar**,PetscStream);
  PetscErrorCode (*realpart)(PetscStreamScalar,PetscStream);
  PetscErrorCode (*axty)(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
  PetscErrorCode (*aydx)(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
};

typedef enum {
  PSS_FALSE = 0,
  PSS_UNKNOWN = 1,
  PSS_TRUE = 2
} PSSCacheBool;

struct _n_PetscStreamScalar {
  struct _ScalOps  ops[1];
  PetscBool        setup;
  PetscOffloadMask omask;
  char             *type;
  PetscEvent       event;
  PetscScalar      *host;
  PetscScalar      *device;
  PetscInt         poolID;
  PSSCacheBool     cache[PSS_CACHE_MAX];
};

struct _GraphOps {
  PetscErrorCode (*create)(PetscStreamGraph);
  PetscErrorCode (*destroy)(PetscStreamGraph);
  PetscErrorCode (*setup)(PetscStreamGraph);
  PetscErrorCode (*assemble)(PetscStreamGraph,PetscGraphAssemblyType);
  PetscErrorCode (*exec)(PetscStreamGraph,PetscStream);
  PetscErrorCode (*duplicate)(PetscStreamGraph,PetscStreamGraph);
  PetscErrorCode (*getgraph)(PetscStreamGraph,void*);
  PetscErrorCode (*restoregraph)(PetscStreamGraph,void*);
};

struct _n_PetscStreamGraph {
  struct _GraphOps ops[1];
  PetscBool        setup;
  PetscBool        assembled;
  char             *type;
  PetscInt         capStrmId;
  void             *data;
};

PETSC_STATIC_INLINE PetscErrorCode PetscStreamValidateIdle_Internal(PetscStream strm)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscBool      idle;
    PetscErrorCode ierr;

    ierr = (*strm->ops->query)(strm,&idle);CHKERRQ(ierr);
    if (PetscUnlikely(strm->idle && !idle)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscStream cache corrupted, stream thought it was idle when it still had work");
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

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarUpdateCache_Internal(PetscStreamScalar pscal, const PetscScalar *val, PetscMemType mtype)
{
  PetscFunctionBegin;
  if (val) {
    if (PetscMemTypeHost(mtype)) {
      const PetscScalar deref = *val;
      if (deref == (PetscScalar)0.0) {
        pscal->cache[PSS_ZERO] = PSS_TRUE;
        pscal->cache[PSS_ONE] = PSS_FALSE;
        pscal->cache[PSS_INF] = PSS_FALSE;
        pscal->cache[PSS_NAN] = PSS_FALSE;
      } else if (deref == (PetscScalar)1.0) {
        pscal->cache[PSS_ZERO] = PSS_FALSE;
        pscal->cache[PSS_ONE] = PSS_TRUE;
        pscal->cache[PSS_INF] = PSS_FALSE;
        pscal->cache[PSS_NAN] = PSS_FALSE;
      } else {
        pscal->cache[PSS_ZERO] = PSS_FALSE;
        pscal->cache[PSS_ONE] = PSS_FALSE;
        pscal->cache[PSS_INF] = PetscIsInfScalar(deref) ? PSS_TRUE : PSS_FALSE;
        pscal->cache[PSS_NAN] = PetscIsNanScalar(deref) ? PSS_TRUE : PSS_FALSE;
      }
    } else {
      for (int i = 0; i < PSS_CACHE_MAX; ++i) pscal->cache[i] = PSS_UNKNOWN;
    }
  } else {
    pscal->cache[PSS_ZERO] = PSS_TRUE;
    pscal->cache[PSS_ONE] = PSS_FALSE;
    pscal->cache[PSS_INF] = PSS_FALSE;
    pscal->cache[PSS_NAN] = PSS_FALSE;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSetCache_Internal(PetscStreamScalar pscal, PSSCacheType ctype, PSSCacheBool val)
{
  PetscFunctionBegin;
  pscal->cache[ctype] = val;
  switch (ctype) {
  case PSS_ZERO:
    pscal->cache[PSS_ONE] = val ? PSS_FALSE : PSS_UNKNOWN;
    pscal->cache[PSS_INF] = val ? PSS_FALSE : PSS_UNKNOWN;
    pscal->cache[PSS_NAN] = val ? PSS_FALSE : PSS_UNKNOWN;
    break;
  case PSS_ONE:
    pscal->cache[PSS_ZERO] = val ? PSS_FALSE : PSS_UNKNOWN;
    pscal->cache[PSS_INF] = val ? PSS_FALSE : PSS_UNKNOWN;
    pscal->cache[PSS_NAN] = val ? PSS_FALSE : PSS_UNKNOWN;
    break;
  case PSS_INF:
    pscal->cache[PSS_ZERO] = val ? PSS_FALSE : PSS_UNKNOWN;
    pscal->cache[PSS_ONE] = val ? PSS_FALSE : PSS_UNKNOWN;
    pscal->cache[PSS_NAN] = val ? PSS_FALSE : PSS_UNKNOWN;
    break;
  case PSS_NAN:
    pscal->cache[PSS_ZERO] = val ? PSS_FALSE : PSS_UNKNOWN;
    pscal->cache[PSS_ONE] = val ? PSS_FALSE : PSS_UNKNOWN;
    pscal->cache[PSS_INF] = val ? PSS_FALSE : PSS_UNKNOWN;
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarCheckCache_Internal(PetscStreamScalar pscal, PetscScalar assertval, PetscStream pstream)
{
  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  {
    PetscErrorCode ierr;
    PetscScalar    alpha;

    ierr = PetscStreamScalarAwait(pscal,&alpha,pstream);CHKERRQ(ierr);
    if (PetscUnlikely(alpha != assertval)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bug in PetscStreamScalar cache, assumed %g but was %g",(double)PetscRealPart(assertval),(double)PetscRealPart(alpha));
  }
#endif
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
