#include <petsc/private/streamimpl.h>

PetscErrorCode PetscStreamCreate(PetscStream *strm)
{
  PetscStream    s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(strm,1);
  /* Setting to null taken from VecCreate(), why though? */
  *strm = NULL;
  ierr = PetscNew(&s);CHKERRQ(ierr);
#if PetscDefined(HAVE_CUDA)
  s->cstream = NULL;
  s->cevent = NULL;
#if PetscDefined(USE_DEBUG)
  s->gotCUDA = PETSC_FALSE;
#endif
#endif
#if PetscDefined(HAVE_HIP)
  s->hstream = NULL;
  s->hevent = NULL;
#if PetscDefined(USE_DEBUG)
  s->gotHIP  = PETSC_FALSE;
#endif
#endif
  s->mode = PETSC_STREAM_DEFAULT_BLOCKING;
  *strm = s;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamDestroy(PetscStream *strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*strm) PetscFunctionReturn(0);
  PetscValidPointer(strm,1);
#if PetscDefined(HAVE_CUDA)
#if PetscDefined(USE_DEBUG)
  if (PetscUnlikely((*strm)->gotCUDA)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"CUDA stream object is checked out. Restore it via PetscSctreamRestoreStream() before destroying\n");
  }
#endif
  {
    cudaError_t cerr;

    if ((*strm)->cstream) {cerr = cudaStreamDestroy((*strm)->cstream);CHKERRCUDA(cerr);}
    if ((*strm)->cevent)  {cerr = cudaEventDestroy((*strm)->cevent);CHKERRCUDA(cerr);}
  }
#endif
#if PetscDefined(HAVE_HIP)
#if PetscDefined(USE_DEBUG)
  if (PetscUnlikely((*strm)->gotHIP)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"HIP stream object is checked out. Restore it via PetscSctreamRestoreStream() before destroying\n");
  }
#endif
  {
    hipError_t herr;

    if ((*strm)->hstream) {herr = hipStreamDestroy((*strm)->hstream);CHKERRHIP(herr);}
    if ((*strm)->hevent)  {herr = hipEventDestroy((*strm)->hevent);CHKERRHIP(herr);}
  }
#endif
  ierr = PetscFree(*strm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSetMode(PetscStream strm, PetscStreamMode mode)
{
  PetscFunctionBegin;
  strm->mode = mode;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGetMode(PetscStream strm, PetscStreamMode *mode)
{
  PetscFunctionBegin;
  PetscValidPointer(mode,2);
  *mode = strm->mode;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSetupHIP_Private(PetscStream strm)
{
  PetscFunctionBegin;
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
    if (!(strm->hevent))  {herr = hipEventCreate(&strm->hevent);CHKERRHIP(herr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_HIP)
    if (!(strm->hstream)) {
      herr = hipStreamCreateWithFlags(&strm->hstream, hipStreamNonBlocking);CHKERRHIP(herr);
    }
    if (!(strm->hevent))  {herr = hipEventCreate(&strm->hevent);CHKERRHIP(herr);}
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSetupCUDA_Private(PetscStream strm)
{
  PetscFunctionBegin;
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
    if (!(strm->cevent))  {cerr = cudaEventCreate(&strm->cevent);CHKERRCUDA(cerr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (!(strm->cstream)) {
      cerr = cudaStreamCreateWithFlags(&strm->cstream, cudaStreamNonBlocking);CHKERRCUDA(cerr);
    }
    if (!(strm->cevent))  {cerr = cudaEventCreate(&strm->cevent);CHKERRCUDA(cerr);}
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSetup_Private(PetscStream strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamSetupCUDA_Private(strm);CHKERRQ(ierr);
  ierr = PetscStreamSetupHIP_Private(strm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGetStream(PetscStream strm, PetscStreamType type, void *dstrm)
{
  PetscFunctionBegin;
  PetscValidPointer(dstrm,3);
  switch (type) {
    PetscErrorCode ierr;
  case PETSC_STREAM_CUDA:
    ierr = PetscStreamSetupCUDA_Private(strm);CHKERRQ(ierr);
#if PetscDefined(HAVE_CUDA)
#if PetscDefined(USE_DEBUG)
    if (PetscUnlikely(strm->gotCUDA)) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already checked out CUDA stream, call PetscStreamRestoreStream() before checking out another stream\n");
    }
    strm->gotCUDA = PETSC_TRUE;
#endif
    *((cudaStream_t*) dstrm) = strm->cstream;
#endif
    break;
  case PETSC_STREAM_HIP:
    ierr = PetscStreamSetupHIP_Private(strm);CHKERRQ(ierr);
#if PetscDefined(HAVE_HIP)
#if PetscDefined(USE_CUDA)
    if (PetscUnlikely(strm->gotHIP)) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already checked out HIP stream, call PetscStreamRestoreStream() before checking out another stream\n");
    }
    strm->gotHIP = PETSC_TRUE;
#endif
    *((hipStream_t*) dstrm) = strm->hstream;
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRestoreStream(PetscStream strm, PetscStreamType type, void *dstrm)
{
  PetscFunctionBegin;
  PetscValidPointer(dstrm,3);
  switch (type) {
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
  {
    cudaError_t cerr;

#if PetscDefined(USE_DEBUG)
    if (PetscUnlikely(!(strm->gotCUDA))) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"CUDA stream is not checked out\n");
    }
    if (PetscUnlikely(*((cudaStream_t*) dstrm) != strm->cstream)) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"CUDA stream is not the same as the one that was checked out\n");
    }
    strm->gotCUDA = PETSC_FALSE;
#endif
    cerr = cudaStreamDestroy(strm->cstream);CHKERRCUDA(cerr);
    strm->cstream = NULL;
  }
#endif
  break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
  {
    hipError_t herr;

#if PetscDefined(USE_DEBUG)
    if (PetscUnlikely(!(strm->gotHIP))) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"HIP stream is not checked out\n");
    }
    if (PetscUnlikely(*((hipStream_t*) dstrm) != strm->hstream)) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"HIP stream is not the same as the one that was checked out\n");
    }
    strm->gotHIP = PETSC_FALSE;
#endif
    herr = hipStreamDestroy(strm->hstream);CHKERRHIP(herr);
    strm->hstream = NULL;
  }
#endif
  break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSynchronizeBegin(PetscStream strm, PetscStreamType type, void *dstrm)
{
  PetscFunctionBegin;
#error TODO, make begin record event, and end wait on event
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSynchronizeHost(PetscStream strm)
{
  PetscFunctionBegin;
  switch (strm->mode) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t     cerr;
#endif
#if PetscDefined(HAVE_HIP)
    hipError_t      herr;
#endif
    PetscErrorCode  ierr;

  case PETSC_STREAM_GLOBAL_BLOCKING:
#if PetscDefined(HAVE_CUDA)
    cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
#endif
    /* No need to sync twice, ordering should be reviewed */
#if PetscDefined(HAVE_HIP) && !PetscDefined(HAVE_CUDA)
    herr = hipDeviceSynchronize();CHKERRHIP(herr);
#endif
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    ierr = PetscStreamSetup_Private(strm);CHKERRQ(ierr);
#if PetscDefined(HAVE_CUDA)
    cerr = cudaStreamSynchronize(strm->cstream);CHKERRCUDA(cerr);
    cerr = cudaStreamDestroy(strm->cstream);CHKERRCUDA(cerr);
    strm->cstream = NULL;
#endif
#if PetscDefined(HAVE_HIP) && !PetscDefined(HAVE_CUDA)
    herr = hipStreamSynchronize(strm->hstream);CHKERRHIP(herr);
    herr = hipStreamDestroy(strm->hstream);CHKERRHIP(herr);
    strm->hstream = NULL;
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSynchronizeDevice(PetscStream strm, PetscStreamType type, void *event)
{
  PetscFunctionBegin;
  if (event) {PetscValidPointer(event,3);}
  switch (type) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t     cerr;
#endif
#if PetscDefined(HAVE_HIP)
    hipError_t      herr;
#endif
    PetscErrorCode  ierr;

  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    ierr = PetscStreamSetupCUDA_Private(strm);CHKERRQ(ierr);
    cerr = cudaEventRecord(event ? *((cudaEvent_t*) event) : strm->cevent, )
    cerr = cudaStreamWaitEvent(strm->cstream, event ? *((cudaEvent_t*) event) : strm->cevent, 0);CHKERRCUDA(cerr);
    cerr = cudaStreamDestroy(strm->cstream);CHKERRCUDA(cerr);
    strm->cstream = NULL;
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    ierr = PetscStreamSetupHIP_Private(strm);CHKERRQ(ierr);
    herr = hipStreamWaitEvent(strm->hstream, event ? *((hipEvent_t*) event) : strm->hevent, 0);CHKERRHIP(herr);
    herr = hipStreamDestroy(strm->hstream);CHKERRHIP(herr);
    strm->hstream = NULL;
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}
