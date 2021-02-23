#if !defined(DEVICEIMPL_H)
#define DEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

PETSC_EXTERN PetscBool PetscStreamRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscStreamRegisterAll(void);

typedef struct _StreamOps *StreamOps;
struct _StreamOps {
  PetscErrorCode (*create)(PetscStream);
  PetscErrorCode (*destroy)(PetscStream);
  PetscErrorCode (*setup)(PetscStream);
  PetscErrorCode (*getstream)(PetscStream,void*);
  PetscErrorCode (*restorestream)(PetscStream,void*);
  PetscErrorCode (*recordevent)(PetscStream,PetscEvent);
  PetscErrorCode (*waitevent)(PetscStream,PetscEvent);
  PetscErrorCode (*synchronize)(PetscStream);
  PetscErrorCode (*query)(PetscStream,PetscBool*);
};

struct _n_PetscStream {
  struct _StreamOps ops[1];
  PetscInt          id;
  PetscBool         idle;
  PetscBool         setup;
  PetscStreamType   type;
  PetscStreamMode   mode;
  void              *data;
};

#define PetscValidStreamType(_p_strm__,_p_arg__)                        \
  do {                                                                  \
    if (PetscUnlikelyDebug((_p_strm__)->type == PETSC_STREAM_INVALID)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_TYPENOTSET,"PetscStreamType is not set: Argument # %d",_p_arg__); \
  } while (0)

#define PetscValidStreamTypeSpecific(_p_strm__,_p_arg__,_p_type__,_v_type__) \
  do {                                                                  \
    PetscValidStreamType(_p_strm__,_p_arg__);                           \
    if (PetscUnlikelyDebug((_p_strm__)->type != (_p_type__))) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"PetscStreamType %d arg #%d is incompatible with vectype %s",(int)(_p_type__),(_p_arg__),(_v_type__)); \
  } while (0)

#define PetscCheckValidSameStreamType(_p_strm1__,_p_arg1__,_p_strm2__,_p_arg2__) \
  do {                                                                  \
    PetscValidStreamType(_p_strm1__,_p_arg1__);                         \
    PetscValidStreamType(_p_strm2__,_p_arg2__);                         \
    if (PetscUnlikelyDebug((_p_strm1__)->type != (_p_strm2__)->type)) { \
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"PetscStreamType %d is incompatible with other PetscStreamType %d in arguments #%d and #%d",(int)((_p_strm1__)->type),(int)((_p_strm2__)->type),(_p_arg1__),(_p_arg2__)); \
    }                                                                   \
} while (0)

typedef struct _EventOps *EventOps;
struct _EventOps {
  PetscErrorCode (*create)(PetscEvent);
  PetscErrorCode (*destroy)(PetscEvent);
  PetscErrorCode (*setup)(PetscEvent);
  PetscErrorCode (*synchronize)(PetscEvent);
  PetscErrorCode (*query)(PetscEvent,PetscBool*);
};

struct _n_PetscEvent {
  struct _EventOps ops[1];
  PetscInt         id;
  PetscBool        idle;
  PetscBool        setup;
  PetscStreamType  type;
  unsigned int     eventFlags, waitFlags;
  void             *data;
};

typedef struct _ScalOps *ScalOps;
struct _ScalOps {
  PetscErrorCode (*create)(PetscStreamScalar);
  PetscErrorCode (*destroy)(PetscStreamScalar);
  PetscErrorCode (*setup)(PetscStreamScalar);
  PetscErrorCode (*setvalue)(PetscStreamScalar,const PetscScalar*,PetscMemType,PetscStream);
  PetscErrorCode (*gethost)(PetscStreamScalar,PetscScalar**,PetscBool,PetscStream);
  PetscErrorCode (*restorehost)(PetscStreamScalar,PetscScalar**,PetscStream);
  PetscErrorCode (*getdevice)(PetscStreamScalar,PetscScalar**,PetscBool,PetscStream);
  PetscErrorCode (*restoredevice)(PetscStreamScalar,PetscScalar**,PetscStream);
  PetscErrorCode (*axty)(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
  PetscErrorCode (*aydx)(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
  PetscErrorCode (*accumop)(PetscStreamScalar,PetscInt,PetscStreamScalar[],PetscStreamComputeOp,PetscStreamComputeOp,PetscStream);
};

typedef enum {
  PSS_FALSE = 0,
  PSS_UNKNOWN = 1,
  PSS_TRUE = 2
} PSSCacheBool;

typedef enum {
  PSS_ZERO = 0,
  PSS_ONE,
  PSS_INF,
  PSS_NAN,
  PSSCACHE_MAX
} PSSCacheType;

struct _n_PetscStreamScalar {
  struct _ScalOps  ops[1];
  PetscBool        setup;
  PetscOffloadMask omask;
  PetscStreamType  type;
  PetscEvent       event;
  PetscScalar      *host;
  PetscScalar      *device;
  PetscInt         poolID;
  PSSCacheBool     cache[PSSCACHE_MAX];
};

struct _GraphOps {
  PetscErrorCode (*create)(PetscStreamGraph);
  PetscErrorCode (*destroy)(PetscStreamGraph);
  PetscErrorCode (*setup)(PetscStreamGraph);
  PetscErrorCode (*assemble)(PetscStreamGraph);
  PetscErrorCode (*exec)(PetscStreamGraph,PetscStream);
  PetscErrorCode (*getgraph)(PetscStreamGraph,void*);
  PetscErrorCode (*restoregraph)(PetscStreamGraph,void*);
};

struct _n_PetscStreamGraph {
  struct _GraphOps ops[1];
  PetscBool        setup;
  PetscBool        assembled;
  PetscStreamType  type;
  void             *data;
};

PETSC_INTERN PetscErrorCode PetscStreamCreate_CUDA(PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamCreate_HIP(PetscStream);
PETSC_INTERN PetscErrorCode PetscEventCreate_CUDA(PetscEvent);
PETSC_INTERN PetscErrorCode PetscEventCreate_HIP(PetscEvent);
PETSC_INTERN PetscErrorCode PetscStreamScalarCreate_CUDA(PetscStreamScalar);

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
      for (int i = 0; i < PSSCACHE_MAX; ++i) pscal->cache[i] = PSS_UNKNOWN;
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
    const PetscScalar *alpha;
    PetscErrorCode    ierr;

    ierr = PetscStreamScalarGetHostRead(pscal, &alpha, pstream);CHKERRQ(ierr);
    if (PetscUnlikely(*alpha != assertval)) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bug in PetscStreamScalar cache, assumed %f but was %f",assertval,*alpha);
    }
  }
#endif
  PetscFunctionReturn(0);
}
#endif /* DEVICEIMPL_H */
