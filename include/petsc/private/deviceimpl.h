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
  PetscScalar      host;
  PetscScalar      *device;
  PetscBool        isZero, isOne;
};

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSynchronizeDevice_Internal(PetscStreamScalar pscal, PetscStream pstream)
{
  PetscFunctionBegin;
  if (pscal->omask == PETSC_OFFLOAD_CPU) {
    PetscErrorCode ierr;

    switch (pscal->type) {
    case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    {
      cudaError_t    cerr;
      cudaStream_t   cstream;

      ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
      cerr = cudaMemcpyAsync(pscal->device, &pscal->host, sizeof(PetscScalar), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
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
      herr = hipMemcpyAsync(pscal->device, &pscal->host, sizeof(PetscScalar), hipMemcpyHostToDevice, hstream);CHKERRHIP(herr);
      ierr = PetscStreamRestoreStream(pstream, hstream, PETSC_FALSE);CHKERRQ(ierr);
    }
#endif
    default:
      break;
    }
    ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_BOTH;
    pscal->isZero = pscal->host == (PetscScalar)0.0;
    pscal->isOne = pscal->host == (PetscScalar)1.0;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarSynchronizeHost_Internal(PetscStreamScalar pscal, PetscStream pstream, PetscBool *sync)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(sync,3);
  *sync = PETSC_FALSE;
  if (pscal->omask == PETSC_OFFLOAD_GPU) {
    PetscErrorCode ierr;

    switch (pscal->type) {
    case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    {
      cudaError_t    cerr;
      cudaStream_t   cstream;

      ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
      cerr = cudaMemcpyAsync(&pscal->host, pscal->device, sizeof(PetscScalar), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
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
      herr = hipMemcpyAsync(&pscal->host, pscal->device, sizeof(PetscScalar), hipMemcpyDeviceToHost, hstream);CHKERRHIP(herr);
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
#endif /* DEVICEIMPL_H */
