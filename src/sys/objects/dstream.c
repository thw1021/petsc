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
  s->type = PETSC_STREAM_INVALID;
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

PetscErrorCode PetscStreamSetType(PetscStream strm, PetscStreamType type)
{
  PetscFunctionBegin;
  strm->type = type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGetType(PetscStream strm, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidPointer(type,2);
  *type = strm->type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGetStream(PetscStream strm, void *dstrm)
{
  PetscFunctionBegin;
  PetscStreamValidType(strm,1);
  PetscValidPointer(dstrm,2);
  switch (strm->type) {
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
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRestoreStream(PetscStream strm, void *dstrm, PetscBool destroy)
{
  PetscFunctionBegin;
  PetscStreamValidType(strm,1);
  PetscValidPointer(dstrm,2);
  switch (strm->type) {
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
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSplitBegin(PetscStream strm, void *dstrm)
{
  PetscFunctionBegin;
  PetscStreamValidType(strm,1);
  PetscValidPointer(dstrm,2);
  switch (strm->type) {
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
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSplitEnd(PetscStream strm, void *dstrm, PetscBool destroy)
{
  PetscFunctionBegin;
  PetscStreamValidType(strm,1);
  PetscValidPointer(dstrm,2);
  switch (strm->type) {
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
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRecordEvent(PetscStream strm, PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscStreamValidType(strm,1);
  ierr = PetscEventSetup(event);CHKERRQ(ierr);
  switch (strm->type) {
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
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamWaitEvent(PetscStream strm, PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscStreamValidType(strm,1);
  ierr = PetscEventSetup(event);CHKERRQ(ierr);
  switch (strm->type) {
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
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSynchronizeStream_Private(PetscStream strm)
{
  PetscFunctionBegin;
  switch (strm->type) {
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
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSynchronizeDevice_Private(PetscStream strm)
{
#if PetscDefined(HAVE_CUDA)
  cudaError_t cerr;
#endif
#if PetscDefined(HAVE_HIP)
  hipError_t herr;
#endif

  PetscFunctionBegin;
  switch (strm->type) {
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    herr = hipDeviceSynchronize();CHKERRHIP(herr);
#endif
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSynchronize(PetscStream strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscStreamValidType(strm,1);
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    ierr = PetscStreamSynchronizeDevice_Private(strm);CHKERRQ(ierr);
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    ierr = PetscStreamSynchronizeStream_Private(strm);CHKERRQ(ierr);
  default:
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
  e->type = PETSC_STREAM_INVALID;
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
  if (PetscUnlikelyDebug(!event)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Input event is NULL!\n");
  PetscStreamValidType(event,1);
  if (event->setup) PetscFunctionReturn(0);
  switch (event->type) {
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    if (event->cevent) {cerr = cudaEventDestroy(event->cevent);CHKERRCUDA(cerr);}
    cerr = cudaEventCreateWithFlags(&event->cevent, event->eventFlags);CHKERRCUDA(cerr);
#endif
    break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    if (event->hevent) {herr = hipEventDestroy(event->hevent);CHKERRHIP(herr);}
    herr = hipEventCreateWithFlags(&event->hevent, event->eventFlags);CHKERRHIP(herr);
#endif
  default:
    break;
  }
  event->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventSynchronize(PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscEventSetup(event);CHKERRQ(ierr);
  switch (event->type) {
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
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarCreate(PetscScalar val, PetscStream pstream, PetscStreamScalar *pscal)
{
  PetscStreamScalar s;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscStreamValidType(pstream,2);
  PetscValidPointer(pscal,3);
  /* Setting to null taken from VecCreate(), why though? */
  *pscal = NULL;
  ierr = PetscNew(&s);CHKERRQ(ierr);
  s->omask = PETSC_OFFLOAD_BOTH;
  s->pstream = pstream;
  s->host = val;
  switch (pstream->type) {
  case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
  {
    cudaError_t  cerr;
    cudaStream_t cstream;

    cerr = cudaMalloc((void**) &s->device, sizeof(PetscScalar));CHKERRCUDA(cerr);
    ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
    cerr = cudaMemcpyAsync(s->device, &s->host, sizeof(PetscScalar), cudaMemcpyHostToDevice, cstream);CHKERRCUDA(cerr);
    ierr = PetscStreamRestoreStream(pstream, cstream, PETSC_FALSE);CHKERRQ(ierr);
  }
#endif
  break;
  case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
  {
    hipError_t  herr;
    hipStream_t hstream;

    herr = hipMalloc((void**) &s->device, sizeof(PetscScalar));CHKERRHIP(herr);
    ierr = PetscStreamGetStream(pstream, &hstream);CHKERRQ(ierr);
    herr = hipMemcpyAsync(s->device, &s->host, sizeof(PetscScalar), hipMemcpyHostToDevice, hstream);CHKERRHIP(herr);
    ierr = PetscStreamRestoreStream(pstream, hstream, PETSC_FALSE);CHKERRQ(ierr);
  }
#endif
  default:
    break;
  }
  *pscal = s;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarDestroy(PetscStreamScalar *pscal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*pscal) PetscFunctionReturn(0);
  PetscValidPointer(pscal,1);
  PetscStreamValidType((*pscal)->pstream,1);
  switch ((*pscal)->pstream->type) {
  case PETSC_STREAM_CUDA:
  {
#if PetscDefined(HAVE_CUDA)
    cudaError_t cerr;

    cerr = cudaFree((*pscal)->device);CHKERRCUDA(cerr);
#endif
  }
  break;
  case PETSC_STREAM_HIP:
  {
#if PetscDefined(HAVE_HIP)
    hipError_t herr;

    herr = hipFree((*pscal)->device);CHKERRHIP(herr);
#endif
  }
  default:
    break;
  }
  ierr = PetscFree(*pscal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarGetHostValue(PetscStreamScalar pscal, PetscScalar *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscStreamValidType(pscal->pstream,1);
  PetscValidScalarPointer(val,2);
  if (pscal->omask == PETSC_OFFLOAD_GPU) {
    switch (pscal->pstream->type) {
    case PETSC_STREAM_CUDA:
#if PetscDefined(HAVE_CUDA)
    {
      cudaError_t  cerr;
      cudaStream_t cstream;

      ierr = PetscStreamGetStream(pscal->pstream, &cstream);CHKERRQ(ierr);
      cerr = cudaMemcpyAsync(&pscal->host, pscal->device, sizeof(PetscScalar), cudaMemcpyDeviceToHost, cstream);CHKERRCUDA(cerr);
      ierr = PetscStreamRestoreStream(pscal->pstream, cstream, PETSC_FALSE);CHKERRQ(ierr);
    }
#endif
    break;
    case PETSC_STREAM_HIP:
#if PetscDefined(HAVE_HIP)
    {
      hipError_t  herr;
      hipStream_t hstream;

      ierr = PetscStreamGetStream(pscal->pstream, &hstream);CHKERRQ(ierr);
      herr = hipMemcpyAsync(&pscal->host, pscal->device, sizeof(PetscScalar), hipMemcpyDeviceToHost, hstream);CHKERRHIP(herr);
      ierr = PetscStreamRestoreStream(pscal->pstream, hstream, PETSC_FALSE);CHKERRQ(ierr);
    }
#endif
    default:
      break;
    }
    pscal->omask = PETSC_OFFLOAD_BOTH;
    ierr = PetscStreamSynchronize(pscal->pstream);CHKERRQ(ierr);
  }
  *val = pscal->host;
  PetscFunctionReturn(0);
}
