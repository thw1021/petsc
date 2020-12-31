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
  s->setup = PETSC_FALSE;
  if (PetscDefined(HAVE_CUDA)) {
    s->cstream = NULL;
    if (PetscDefined(USE_DEBUG)) {s->gotCUDA = PETSC_FALSE;}
  }
  if (PetscDefined(HAVE_HIP)) {
    s->hstream = NULL;
    if (PetscDefined(USE_DEBUG)) {s->gotHIP  = PETSC_FALSE;}
  }
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
  if (PetscDefined(HAVE_CUDA)) {
    if (PetscUnlikelyDebug((*strm)->gotCUDA)) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"CUDA stream object is checked out. Restore it via PetscSctreamRestoreStream() before destroying\n");
    }
    if ((*strm)->cstream) {
      cudaError_t cerr;
      cerr = cudaStreamDestroy((*strm)->cstream);CHKERRCUDA(cerr);
    }
  }
  if (PetscDefined(HAVE_HIP)) {
    if (PetscUnlikelyDebug((*strm)->gotHIP)) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"HIP stream object is checked out. Restore it via PetscSctreamRestoreStream() before destroying\n");
    }
    if ((*strm)->hstream) {
      hipError_t herr;
      herr = hipStreamDestroy((*strm)->hstream);CHKERRHIP(herr);
    }
  }
  ierr = PetscFree(*strm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSetup(PetscStream strm)
{
#if defined(PETSC_HAVE_CUDA)
  cudaError_t cerr;
#endif
#if defined(PETSC_HAVE_HIP)
  hipError_t  herr;
#endif

  PetscFunctionBegin;
  if (PetscDefined(HAVE_DEVICE)) {
    switch (strm->mode) {
    case PETSC_STREAM_GLOBAL_BLOCKING:
      /* NULL stream always blocks */
      break;
    case PETSC_STREAM_DEFAULT_BLOCKING:
      if (PetscDefined(HAVE_CUDA)) {cerr = cudaStreamCreate(&strm->cstream);CHKERRCUDA(cerr);}
      if (PetscDefined(HAVE_HIP)) {herr = hipStreamCreate(&strm->hstream);CHKERRHIP(herr);}
      break;
    case PETSC_STREAM_GLOBAL_NONBLOCKING:
      if (PetscDefined(HAVE_CUDA)) {
        cerr = cudaStreamCreateWithFlags(&strm->cstream, cudaStreamNonBlocking);CHKERRCUDA(cerr);
      }
      if (PetscDefined(HAVE_HIP)) {
        herr = hipStreamCreateWithFlags(&strm->hstream, hipStreamNonBlocking);CHKERRHIP(herr);
      }
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
      break;
    }
  }
  strm->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSetMode(PetscStream strm, PetscStreamMode mode)
{
  PetscFunctionBegin;
  if (PetscUnlikely(strm->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscStreamSetup() has already been called\n");
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

PetscErrorCode PetscStreamGetStream(PetscStream strm, PetscStreamType type, void *dstrm)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(dstrm,3);
  ierr = PetscStreamSetup(strm);CHKERRQ(ierr);
  switch (type) {
  case PETSC_STREAM_CUDA:
    if (PetscDefined(HAVE_CUDA)) {
      if (PetscUnlikelyDebug(strm->gotCUDA)) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already checked out CUDA stream, call PetscStreamRestoreStream() before checking out another stream\n");
      }
      *((cudaStream_t*) dstrm) = strm->cstream;
      if (PetscDefined(USE_DEBUG)) strm->gotCUDA = PETSC_TRUE;
    }
    break;
  case PETSC_STREAM_HIP:
    if (PetscDefined(HAVE_HIP)) {
      if (PetscUnlikelyDebug(strm->gotHIP)) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already checked out HIP stream, call PetscStreamRestoreStream() before checking out another stream\n");
      }
      *((hipStream_t*) dstrm) = strm->hstream;
      if (PetscDefined(USE_DEBUG)) strm->gotHIP = PETSC_TRUE;
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRestoreStream(PetscStream str, PetscStreamType type, void *dstrm)
{
  PetscFunctionBegin;
  PetscValidPointer(dstrm,3);
  switch (type) {
  case PETSC_STREAM_CUDA:
    if (PetscDefined(HAVE_CUDA)) {
      cudaError_t cerr;

      if (PetscUnlikelyDebug(!(strm->gotCUDA))) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"CUDA stream is not checked out\n");
      }
      if (PetscUnlikelyDebug(*((cudaStream_t*) dstrm) != strm->cstream)) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"CUDA stream is not the same as the one that was checked out\n");
      }
      if (PetscDefined(USE_DEBUG)) strm->gotCUDA = PETSC_FALSE;
      cerr = cudaStreamDestroy(strm->cstream);CHKERRCUDA(cerr);
      strm->cstream = NULL;
    }
    break;
  case PETSC_STREAM_HIP:
    if (PetscDefined(HAVE_HIP)) {
      hipError_t herr;

      if (PetscUnlikelyDebug(!(strm->gotHIP))) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"HIP stream is not checked out\n");
      }
      if (PetscUnlikelyDebug(*((hipStream_t*) dstrm) != strm->hstream)) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"HIP stream is not the same as the one that was checked out\n");
      }
      if (PetscDefined(USE_DEBUG)) strm->gotHIP = PETSC_FALSE;
      herr = hipStreamDestroy(strm->hstream);CHKERRHIP(herr);
      strm->hstream = NULL;
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSynchronize(PetscStream strm)
{
#if defined(PETSC_HAVE_CUDA)
  cudaError_t     cerr;
#endif
#if defined(PETSC_HAVE_HIP)
  hipError_t      herr;
#endif
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscStreamSetup(strm);CHKERRQ(ierr);
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    if (PetscDefined(HAVE_CUDA)) {cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);}
    /* No need to sync twice, ordering should be reviewed */
    if (PetscDefined(HAVE_HIP) && !PetscDefined(HAVE_CUDA)) {
      herr = hipDeviceSynchronize();CHKERRHIP(herr);
    }
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    if (PetscDefined(HAVE_CUDA)) {cerr = cudaStreamSynchronize(strm->cstream);CHKERRCUDA(cerr);}
    if (PetscDefined(HAVE_HIP) && !PetscDefined(HAVE_CUDA)) {
      herr = hipStreamSynchronize(strm->hstream);CHKERRHIP(herr);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}
