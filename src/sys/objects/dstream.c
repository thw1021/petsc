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
  for (PetscInt i = 0; i < PETSC_MAX_STREAMS; ++i) {
    s->cstream[i] = NULL;
#if PetscDefined(USE_DEBUG)
    s->gotCUDA[i] = PETSC_FALSE;
#endif
  }
  s->cevent = NULL;
#endif
#if PetscDefined(HAVE_HIP)
  for (PetscInt i = 0; i < PETSC_MAX_STREAMS; ++i) {
    s->hstream[i] = NULL;
#if PetscDefined(USE_DEBUG)
    s->gotHIP[i] = PETSC_FALSE;
#endif
  }
  s->hevent = NULL;
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
  {
    cudaError_t cerr;

    for (PetscInt i = 0; i < PETSC_MAX_STREAMS; ++i) {
#if PetscDefined(USE_DEBUG)
      if (PetscUnlikely((*strm)->gotCUDA[i])) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"CUDA stream #%d is checked out. Restore it via PetscSctreamRestoreStream() before destroying\n", i);
      }
#endif
      if ((*strm)->cstream[i]) {cerr = cudaStreamDestroy((*strm)->cstream[i]);CHKERRCUDA(cerr);}
    }
    if ((*strm)->cevent) {cerr = cudaEventDestroy((*strm)->cevent);CHKERRCUDA(cerr);}
  }
#endif
#if PetscDefined(HAVE_HIP)
  {
    hipError_t herr;

    for (PetscInt i = 0; i < PETSC_MAX_STREAMS; ++i) {
#if PetscDefined(USE_DEBUG)
      if (PetscUnlikely((*strm)->gotHIP[i])) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"HIP stream #%d is checked out. Restore it via PetscSctreamRestoreStream() before destroying\n", i);
      }
#endif
      if ((*strm)->hstream[i]) {herr = hipStreamDestroy((*strm)->hstream[i]);CHKERRHIP(herr);}
    }
    if ((*strm)->hevent) {herr = hipEventDestroy((*strm)->hevent);CHKERRHIP(herr);}
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

PetscErrorCode PetscStreamAssemble(PetscStream strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStreamAssembleCUDA_Private(strm);CHKERRQ(ierr);
  ierr = PetscStreamAssembleHIP_Private(strm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGetStream(PetscStream strm, PetscStreamType type, void *dstrm)
{
  PetscFunctionBegin;
  PetscValidPointer(dstrm,3);
  switch (type) {
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
    PetscErrorCode ierr;
    PetscInt       i = 0;
#endif
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    /* Global blocking = NULL stream, whihc would fail checks below */
    if (strm->mode != PETSC_STREAM_GLOBAL_BLOCKING) {
      do {
        if (!(strm->cstream[i])) break;
      } while (++i < PETSC_MAX_STREAMS);
      if (PetscUnlikelyDebug(i >= PETSC_MAX_STREAMS)) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not enough CUDA streams available, restore unused streams first\n");
      }
    }
    ierr = PetscStreamAssembleCUDA_Private(strm, i);CHKERRQ(ierr);
    *((cudaStream_t*) dstrm) = strm->cstream[i];
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    if (strm->mode != PETSC_STREAM_GLOBAL_BLOCKING) {
      do {
        if (!(strm->hstream[i])) break;
      } while (++i < PETSC_MAX_STREAMS);
      if (PetscUnlikelyDebug(i >= PETSC_MAX_STREAMS)) {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not enough HIP streams available, restore unused streams first\n");
      }
    }
    ierr = PetscStreamAssembleHIP_Private(strm, i);CHKERRQ(ierr);
    *((hipStream_t*) dstrm) = strm->hstream[i];
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
  }
#endif
  break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRecordEvent(PetscStream strm, PetscStreamType type)
{
  PetscFunctionBegin;
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
    ierr = PetscStreamAssembleCUDA_Private(strm);CHKERRQ(ierr);
    cerr = cudaEventRecord(strm->cevent, strm->cstream);CHKERRCUDA(cerr);
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    ierr = PetscStreamAssembleHIP_Private(strm);CHKERRQ(ierr);
    herr = hipEventRecord(strm->hevent, strm->hstream);CHKERRHIP(herr);
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamWaitEvent(PetscStream strm, PetscStreamType type, unsigned int flags)
{
  PetscFunctionBegin;
  switch (type) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t     cerr;
#endif
#if PetscDefined(HAVE_HIP)
    hipError_t      herr;
#endif

  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    cerr = cudaStreamWaitEvent(strm->cstream, strm->cevent, flags);CHKERRCUDA(cerr);
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    herr = hipStreamWaitEvent(strm->hstream, strm->hevent, flags);CHKERRHIP(herr);
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSynchronize(PetscStream strm)
{
  PetscFunctionBegin;
  switch (strm->mode) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t     cerr;
#endif
#if PetscDefined(HAVE_HIP)
    hipError_t      herr;
#endif

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
#if PetscDefined(HAVE_CUDA)
    cerr = cudaStreamSynchronize(strm->cstream);CHKERRCUDA(cerr);
#endif
#if PetscDefined(HAVE_HIP) && !PetscDefined(HAVE_CUDA)
    herr = hipStreamSynchronize(strm->hstream);CHKERRHIP(herr);
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamFinalizeStream(PetscStream strm)
{
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}
