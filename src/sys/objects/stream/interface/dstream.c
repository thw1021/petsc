#include <petsc/private/deviceimpl.h>

static PetscInt streamID = 0;

PetscErrorCode PetscStreamCreate(PetscStream *strm)
{
  PetscStream    s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(strm,1);
  ierr = PetscStreamRegisterAll();CHKERRQ(ierr);
  /* Setting to null taken from VecCreate(), why though? */
  *strm = NULL;
  ierr = PetscNew(&s);CHKERRQ(ierr);
  s->type = PETSC_STREAM_INVALID;
  s->mode = PETSC_STREAM_DEFAULT_BLOCKING;
  s->id = streamID++;
  *strm = s;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamDestroy(PetscStream *strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*strm) PetscFunctionReturn(0);
  PetscValidPointer(strm,1);
  ierr = (*(*strm)->ops->destroy)(*strm);CHKERRQ(ierr);
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

PetscErrorCode PetscStreamSetUp(PetscStream strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  if (strm->setup) PetscFunctionReturn(0);
  ierr = (*strm->ops->setup)(strm);CHKERRQ(ierr);
  strm->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGetStream(PetscStream strm, void *dstrm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(dstrm,2);
  ierr = (*strm->ops->getstream)(strm, dstrm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRestoreStream(PetscStream strm, void *dstrm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(dstrm,2);
  ierr = (*strm->ops->restorestream)(strm, dstrm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSplitBegin(PetscStream strm, void *dstrm)
{
  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
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

PetscErrorCode PetscStreamSplitEnd(PetscStream strm, void *dstrm)
{
  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(dstrm,2);
#if 0
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
      ierr = PetscStreamDisassembleCUDA_Internal(strm);CHKERRQ(ierr);
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
      ierr = PetscStreamDisassembleHIP_Internal(strm);CHKERRQ(ierr);
    }
#endif
  default:
    break;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRecordEvent(PetscStream strm, PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(strm,1,event,2);
  ierr = (*strm->ops->recordevent)(strm, event);CHKERRQ(ierr);
  /* Imprint on the event the id of the stream, so subsequent waits need not check */
  event->id = strm->id;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamWaitEvent(PetscStream strm, PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(strm,1,event,2);
  /* Last stream to interact with this event was this stream, no need to wait */
  if (strm->id == event->id) PetscFunctionReturn(0);
  ierr = (*strm->ops->waitevent)(strm, event);CHKERRQ(ierr);
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
  PetscValidStreamType(strm,1);
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    ierr = PetscStreamSynchronizeDevice_Private(strm);CHKERRQ(ierr);
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    ierr = (*strm->ops->synchronize)(strm);CHKERRQ(ierr);
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamQuery(PetscStream strm, PetscBool *busy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidBoolPointer(busy,2);
  ierr = (*strm->ops->query)(strm, busy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
