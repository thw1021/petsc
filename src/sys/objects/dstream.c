#include <petsc/private/deviceimpl.h>

PetscErrorCode PetscStreamCreate(PetscStream *strm)
{
  PetscStream    s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(strm,1);
  /* Setting to null taken from VecCreate(), why though? */
  *strm = NULL;
  ierr = PetscNew(&s);CHKERRQ(ierr);
  s->mode = PETSC_STREAM_DEFAULT_BLOCKING;
#if PetscDefined(HAVE_CUDA)
  s->cstream = NULL;
#endif
#if PetscDefined(HAVE_HIP)
  s->hstream = NULL;
#endif
  *strm = s;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamDestroy(PetscStream *strm)
{
#if PetscDefined(HAVE_CUDA)
  cudaError_t cerr;
#endif
#if PetscDefined(HAVE_HIP)
  hipError_t herr;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*strm) PetscFunctionReturn(0);
  PetscValidPointer(strm,1);
#if PetscDefined(HAVE_CUDA)
  if ((*strm)->cstream) {cerr = cudaStreamDestroy((*strm)->cstream);CHKERRCUDA(cerr);}
#endif
#if PetscDefined(HAVE_HIP)
  if ((*strm)->hstream) {herr = hipStreamDestroy((*strm)->hstream);CHKERRHIP(herr);}
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

PetscErrorCode PetscStreamGetStream(PetscStream strm, PetscStreamType type, void *dstrm)
{
  PetscFunctionBegin;
  PetscValidPointer(dstrm,3);
  switch (type) {
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
    PetscErrorCode ierr;
#endif
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    ierr = PetscStreamAssembleCUDA_Internal(strm);CHKERRQ(ierr);
    *((cudaStream_t*) dstrm) = strm->cstream;
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    ierr = PetscStreamAssembleHIP_Internal(strm);CHKERRQ(ierr);
    *((hipStream_t*) dstrm) = strm->hstream;
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRestoreStream(PetscStream strm, PetscStreamType type, void *dstrm, PetscBool destroy)
{
  PetscFunctionBegin;
  PetscValidPointer(dstrm,3);
  switch (type) {
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
    PetscErrorCode ierr;
#endif
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    if (PetscUnlikelyDebug(*((cudaStream_t*) dstrm) != strm->cstream)) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"CUDA stream is not the same as the one that was checked out\n");
    }
    if (destroy) {ierr = PetscStreamDisassembleCUDA_Internal(strm);CHKERRQ(ierr);}
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    if (PetscUnlikelyDebug(*((hipStream_t*) dstrm) != strm->hstream)) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"HIP stream is not the same as the one that was checked out\n");
    }
    if (destroy) {ierr = PetscStreamDisassembleHIP_Internal(strm);CHKERRQ(ierr);}
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSplitBegin(PetscStream strm, PetscStreamType type, void *dstrm)
{
  PetscFunctionBegin;
  switch (type) {
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    if (strm->mode != PETSC_STREAM_GLOBAL_BLOCKING) {
      cudaError_t cerr;

      cerr = cudaStreamCreate((cudaStream_t*) dstrm);CHKERRCUDA(cerr);
    } else {
      *((cudaStream_t*) dstrm) = NULL;
    }
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    if (strm->mode != PETSC_STREAM_GLOBAL_BLOCKING) {
      hipError_t herr;

      herr = hipStreamCreate((hipStream_t*) dstrm);CHKERRHIP(herr);
    } else {
      *((hipStream_t*) dstrm) = NULL;
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSplitEnd(PetscStream strm, PetscStreamType type, void *dstrm, PetscBool destroy)
{
  PetscFunctionBegin;
  switch (type) {
#if PetscDefined(HAVE_CUDA) || PetscDefined(HAVE_HIP)
    PetscErrorCode ierr;
#endif
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    if (strm->mode != PETSC_STREAM_GLOBAL_BLOCKING) {
      cudaError_t cerr;
      cudaEvent_t cevent;

      ierr = PetscStreamAssembleCUDA_Internal(strm);CHKERRQ(ierr);
      /* Turn off timing, as it is purely about sync */
      cerr = cudaEventCreateWithFlags(&cevent, cudaEventDisableTiming);CHKERRCUDA(cerr);
      cerr = cudaEventRecord(cevent, *((cudaStream_t*) dstrm));CHKERRCUDA(cerr);
      ierr = cudaStreamWaitEvent(strm->cstream, cevent, 0);CHKERRCUDA(cerr);
      /* Safe to destroy, event won't actually be destroyed on device before it occurs */
      cerr = cudaEventDestroy(cevent);CHKERRCUDA(cerr);
      cerr = cudaStreamDestroy(*((cudaStream_t*) dstrm));CHKERRCUDA(cerr);
      if (destroy) {ierr = PetscStreamDisassembleCUDA_Internal(strm);CHKERRQ(ierr);}
    }
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    if (strm->mode != PETSC_STREAM_GLOBAL_BLOCKING) {
      hipError_t herr;
      hipEvent_t hevent;

      ierr = PetscStreamAssembleHIP_Internal(strm);CHKERRQ(ierr);
      /* Turn off timing, as it is purely about sync */
      herr = hipEventCreateWithFlags(&hevent, hipEventDisableTiming);CHKERRHIP(herr);
      herr = hipEventRecord(hevent, *((hipStream_t*) dstrm));CHKERRHIP(herr);
      ierr = hipStreamWaitEvent(strm->hstream, hevent, 0);CHKERRHIP(herr);
      /* Safe to destroy, event won't actually be destroyed on device before it occurs */
      herr = hipEventDestroy(hevent);CHKERRHIP(herr);
      herr = hipStreamDestroy(*((hipStream_t*) dstrm));CHKERRHIP(herr);
      if (destroy) {ierr = PetscStreamDisassembleHIP_Internal(strm);CHKERRQ(ierr);}
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRecordEvent(PetscStream strm, PetscStreamType type, PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscEventSetup(event);CHKERRQ(ierr);
  switch (type) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t     cerr;
#endif
#if PetscDefined(HAVE_HIP)
    hipError_t      herr;
#endif

  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    ierr = PetscStreamAssembleCUDA_Internal(strm);CHKERRQ(ierr);
    /* It is honestly baffling that nvidia do not have a "since version X" note on their functions to make this
     easier. Instead you must find and *manually* search each version of the documentation until you find the version in
     which they introduced some change. $330 BILLION market cap and they can't hire some intern to do this???? */
#if defined(CUDART_VERSION)&& (CUDART_VERSION >= 11010) /* 11.1.0 */
    cerr = cudaEventRecordWithFlags(event->cevent, strm->cstream, event->waitFlags);CHKERRCUDA(cerr);
#else
    cerr = cudaEventRecord(event->cevent, strm->cstream);CHKERRCUDA(cerr);
#endif /* CUDART_VERSION >= 11010 */
#endif /* HAVE_CUDA */
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    ierr = PetscStreamAssembleHIP_Internal(strm);CHKERRQ(ierr);
    herr = hipEventRecordWithFlags(event->hevent, strm->hstream, event->waitFlags);CHKERRHIP(herr);
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType in argument 2\n");
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamWaitEvent(PetscStream strm, PetscEvent event, PetscStreamType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscEventSetup(event);CHKERRQ(ierr);
  switch (type) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t cerr;
#endif
#if PetscDefined(HAVE_HIP)
    hipError_t  herr;
#endif

  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    ierr = PetscStreamAssembleCUDA_Internal(strm);CHKERRQ(ierr);
    cerr = cudaStreamWaitEvent(strm->cstream, event->cevent, event->waitFlags);CHKERRCUDA(cerr);
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    ierr = PetscStreamAssembleHIP_Internal(strm);CHKERRQ(ierr);
    herr = hipStreamWaitEvent(strm->hstream, event->hevent, event->waitFlags);CHKERRHIP(herr);
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscStreamType\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSynchronizeStream_Private(PetscStream strm, PetscStreamType type)
{
  PetscFunctionBegin;
  switch (type) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t    cerr;
#endif
#if PetscDefined(HAVE_HIP)
    hipError_t     herr;
#endif
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    cerr = cudaStreamSynchronize(strm->cstream);CHKERRCUDA(cerr);
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    herr = hipStreamSynchronize(strm->hstream);CHKERRHIP(herr);
#endif
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unhandled PetscStreamType %D\n",(PetscInt)type);
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSynchronizeDevice_Private(PetscStreamType type)
{
#if PetscDefined(HAVE_CUDA)
  cudaError_t cerr;
#endif
#if PetscDefined(HAVE_HIP)
  hipError_t herr;
#endif

  PetscFunctionBegin;
  switch (type) {
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    herr = hipDeviceSynchronize();CHKERRHIP(herr);
#endif
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unhandled PetscStreamType %D\n",(PetscInt)type);
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSynchronize(PetscStream strm, PetscStreamType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    ierr = PetscStreamSynchronizeDevice_Private(type);CHKERRQ(ierr);
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    ierr = PetscStreamSynchronizeStream_Private(strm, type);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode %D\n",(PetscInt)(strm->mode));
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventCreate(PetscEvent *event)
{
  PetscEvent     e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(event,1);
  /* Setting to null taken from VecCreate(), why though? */
  *event = NULL;
  ierr = PetscNew(&e);CHKERRQ(ierr);
  e->setup = PETSC_FALSE;
#if PetscDefined(HAVE_CUDA)
  e->cevent = NULL;
#endif
#if PetscDefined(HAVE_HIP)
  e->hevent = NULL;
#endif
  e->eventFlags = 0;
  e->waitFlags = 0;
  *event = e;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventDestroy(PetscEvent *event)
{
#if PetscDefined(HAVE_CUDA)
  cudaError_t cerr;
#endif
#if PetscDefined(HAVE_HIP)
  hipError_t herr;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*event) PetscFunctionReturn(0);
  PetscValidPointer(event,1);
#if PetscDefined(HAVE_CUDA)
  if ((*event)->cevent) {cerr = cudaEventDestroy((*event)->cevent);CHKERRCUDA(cerr);}
#endif
#if PetscDefined(HAVE_HIP)
  if ((*event)->hevent) {herr = hipEventDestroy((*event)->hevent);CHKERRHIP(herr);}
#endif
  ierr = PetscFree(*event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventSetFlags(PetscEvent event, unsigned int eventFlags, unsigned int waitFlags)
{
  PetscFunctionBegin;
  event->eventFlags = eventFlags;
  event->waitFlags = waitFlags;
  event->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventGetFlags(PetscEvent event, unsigned int *eventFlags, unsigned int *waitFlags)
{
  PetscFunctionBegin;
  if (eventFlags) *eventFlags = event->eventFlags;
  if (waitFlags)  *waitFlags  = event->waitFlags;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventSetup(PetscEvent event)
{
#if PetscDefined(HAVE_CUDA)
  cudaError_t cerr;
#endif
#if PetscDefined(HAVE_HIP)
  hipError_t  herr;
#endif

  PetscFunctionBegin;
  if (event->setup) PetscFunctionReturn(0);
#if PetscDefined(HAVE_CUDA)
  if (event->cevent) {cerr = cudaEventDestroy(event->cevent);CHKERRCUDA(cerr);}
  cerr = cudaEventCreateWithFlags(&event->cevent, event->eventFlags);CHKERRCUDA(cerr);
#endif
#if PetscDefined(HAVE_HIP)
  if (event->hevent) {herr = hipEventDestroy(event->hevent);CHKERRHIP(herr);}
  herr = hipEventCreateWithFlags(&event->hevent, event->eventFlags);CHKERRHIP(herr);
#endif
  event->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventSynchronize(PetscEvent event, PetscStreamType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscEventSetup(event);CHKERRQ(ierr);
  switch (type) {
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
  {
    cudaError_t cerr;

    cerr = cudaEventSynchronize(event->cevent);CHKERRCUDA(cerr);
  }
#endif
  break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
  {
    hipError_t herr;

    herr = hipEventSynchronize(event->hevent);CHKERRHIP(herr);
  }
#endif
  break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unhandled PetscStreamType %D\n",(PetscInt)type);
    break;
  }
  PetscFunctionReturn(0);
}
