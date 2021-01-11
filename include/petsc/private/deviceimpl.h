#if !defined(DEVICEIMPL_H)
#define DEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

struct _n_PetscStream {
  PetscStreamType type;
  PetscStreamMode mode;
#if PetscDefined(HAVE_CUDA)
  cudaStream_t    cstream;
#endif /* PETSC_HAVE_CUDA */
#if PetscDefined(HAVE_HIP)
  hipStream_t     hstream;
#endif /* PETSC_HAVE_HIP */
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

PETSC_STATIC_INLINE PetscErrorCode PetscStreamAssembleHIP_Internal(PetscStream strm)
{
  PetscFunctionBeginHot;
  switch (strm->mode) {
#if PetscDefined(HAVE_HIP)
    hipError_t herr;
#endif
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
#if PetscDefined(HAVE_HIP)
    if (!(strm->hstream)) {herr = hipStreamCreate(&strm->hstream);CHKERRHIP(herr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_HIP)
    if (!(strm->hstream)) {
      herr = hipStreamCreateWithFlags(&strm->hstream, hipStreamNonBlocking);CHKERRHIP(herr);
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamAssembleCUDA_Internal(PetscStream strm)
{
  PetscFunctionBeginHot;
  switch (strm->mode) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t cerr;
#endif
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (!(strm->cstream)) {cerr = cudaStreamCreate(&strm->cstream);CHKERRCUDA(cerr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (!(strm->cstream)) {
      cerr = cudaStreamCreateWithFlags(&strm->cstream, cudaStreamNonBlocking);CHKERRCUDA(cerr);
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamDisassembleHIP_Internal(PetscStream strm)
{
  PetscFunctionBeginHot;
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_HIP)
    if (strm->hstream) {
      hipError_t herr;

      herr = hipStreamDestroy(strm->hstream);CHKERRHIP(herr);
      strm->hstream = NULL;
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamDisassembleCUDA_Internal(PetscStream strm)
{
  PetscFunctionBeginHot;
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (strm->cstream) {
      cudaError_t cerr;

      cerr = cudaStreamDestroy(strm->cstream);CHKERRCUDA(cerr);
      strm->cstream = NULL;
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

struct _n_PetscEvent {
  PetscBool       setup;
  PetscStreamType type;
#if PetscDefined(HAVE_CUDA)
  cudaEvent_t     cevent;
#endif /* PETSC_HAVE_CUDA */
#if PetscDefined(HAVE_HIP)
  hipEvent_t      hevent;
#endif /* PETSC_HAVE_HIP */
  unsigned int    eventFlags, waitFlags;
};

struct _n_PetscStreamScalar {
  PetscOffloadMask omask;
  PetscStreamType  type;
  PetscEvent       event;
  PetscScalar      *host;
  PetscScalar      *device;
  PetscBool        isZero, isOne;
};

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSynchronizeDevice_Internal(PetscStreamScalar pscal, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
  if (pscal->omask == PETSC_OFFLOAD_CPU) {
    switch (pscal->type) {
    case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    {
      cudaError_t    cerr;
      cudaStream_t   cstream;

      ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
      cerr = cudaMemcpyAsync(pscal->device, pscal->host, sizeof(PetscScalar), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
      ierr = PetscStreamRestoreStream(pstream, cstream, PETSC_FALSE);CHKERRQ(ierr);
    }
#endif
    break;
    case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    {
      hipError_t     herr;
      hipStream_t    hstream;

      ierr = PetscStreamGetStream(pstream, &hstream);CHKERRQ(ierr);
      herr = hipMemcpyAsync(pscal->device, pscal->host, sizeof(PetscScalar), hipMemcpyHostToDevice, hstream);CHKERRHIP(herr);
      ierr = PetscStreamRestoreStream(pstream, hstream, PETSC_FALSE);CHKERRQ(ierr);
    }
#endif
    default:
      break;
    }
    ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
  }
  /* Host is up to date, check on the respective values and update cache */
  pscal->isZero = (PetscBool)(*pscal->host == (PetscScalar)0.0);
  pscal->isOne = (PetscBool)(*pscal->host == (PetscScalar)1.0);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSynchronizeHost_Internal(PetscStreamScalar pscal, PetscStream pstream, PetscBool *sync)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidBoolPointer(sync,3);
  *sync = PETSC_FALSE;
  ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
  if (pscal->omask == PETSC_OFFLOAD_GPU) {
    switch (pscal->type) {
    case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    {
      cudaError_t    cerr;
      cudaStream_t   cstream;

      ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
      cerr = cudaMemcpyAsync(pscal->host, pscal->device, sizeof(PetscScalar), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
      ierr = PetscStreamRestoreStream(pstream, cstream, PETSC_FALSE);CHKERRQ(ierr);
    }
#endif
    break;
    case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    {
      hipError_t     herr;
      hipStream_t    hstream;

      ierr = PetscStreamGetStream(pstream, &hstream);CHKERRQ(ierr);
      herr = hipMemcpyAsync(pscal->host, pscal->device, sizeof(PetscScalar), hipMemcpyDeviceToHost, hstream);CHKERRHIP(herr);
      ierr = PetscStreamRestoreStream(pstream, hstream, PETSC_FALSE);CHKERRQ(ierr);
    }
#endif
    default:
      break;
    }
    ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
    *sync = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarGetDevice_Internal(PetscStreamScalar pscal, PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarSynchronizeDevice_Internal(pscal, pstream);CHKERRQ(ierr);
  *ptr = pscal->device;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarGetHost_Internal(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscBool      sync;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarSynchronizeHost_Internal(pscal, pstream, &sync);CHKERRQ(ierr);
  if (sync) {ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);}
  pscal->isZero = (PetscBool)(*pscal->host == (PetscScalar)0.0);
  pscal->isOne = (PetscBool)(*pscal->host == (PetscScalar)1.0);
  *val = pscal->host;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarRestoreHost_Internal(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscScalar    *valptr = *val;
  PetscBool      eventbusy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If they hold the same value, everything stays. Note it is possible that copies from device to host are currently
   in flight, so this may have to be synced */
  ierr = PetscEventQuery(pscal->event, &eventbusy);CHKERRQ(ierr);
  if (eventbusy) {
    PetscBool streambusy;
    /* Event is doing something, so we must sync on it, now we have a choice of either syncing on stream or syncing on the
     event. We check stream in the hopes it has no work, otherwise sync on event. */
    ierr = PetscStreamQuery(pstream, &streambusy);CHKERRQ(ierr);
    if (streambusy) {
      ierr = PetscEventSynchronize(pscal->event);CHKERRQ(ierr);
    } else {
      ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
      ierr = PetscStreamSynchronize(pstream);CHKERRQ(ierr);
    }
    /* Now history is clean, we can update our values */
    *pscal->host = *valptr;
    pscal->omask = PETSC_OFFLOAD_CPU;
  } else if (*valptr != *pscal->host) {
    /* Event is clear, but values don't match, update value and flag */
    *pscal->host = *valptr;
    pscal->omask = PETSC_OFFLOAD_CPU;
  }
  pscal->isZero = (PetscBool)(*pscal->host == (PetscScalar)0.0);
  pscal->isOne = (PetscBool)(*pscal->host == (PetscScalar)1.0);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarCheckCache_Internal(PetscStreamScalar pscal, PetscScalar assertval, PetscStream pstream)
{
  PetscFunctionBegin;
#if PetscDefined(USE_DEBUG)
  {
    PetscScalar    *alpha;
    PetscErrorCode ierr;

    ierr = PetscStreamScalarGetHost_Internal(pscal, &alpha, pstream);CHKERRQ(ierr);
    if (PetscUnlikely(*alpha != assertval)) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bug in PetscStreamScalar cache, assumed %f but was %f\n",assertval,*alpha);
    }
  }
#endif
  PetscFunctionReturn(0);
}
#endif /* DEVICEIMPL_H */
