#include "streamcuda.h" /*I "petscdevice.h" I*/

PETSC_STATIC_INLINE PetscErrorCode PetscStreamDestroy_CUDA(PetscStream strm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  if (psc->cstream) {cerr = cudaStreamDestroy(psc->cstream);CHKERRCUDA(cerr);}
  ierr = PetscFree(strm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSetUp_CUDA(PetscStream strm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  cerr = cudaStreamCreate(&psc->cstream);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamGetStream_CUDA(PetscStream strm, void *dstrm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;

  PetscFunctionBegin;
  *((cudaStream_t*) dstrm) = psc->cstream;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamRestoreStream_CUDA(PetscStream strm, void *dstrm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(*((cudaStream_t*) dstrm) != psc->cstream)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"CUDA stream is not the same as the one that was checked out");
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamRecordEvent_CUDA(PetscStream strm, PetscEvent event)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  PetscEvent_CUDA  *pec = (PetscEvent_CUDA *)event->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  /* It is honestly baffling that nvidia do not have a "since version X" note on their functions to make this
   easier. Instead you must find and *manually* search each version of the documentation until you find the version in
   which they introduced some change. $330 BILLION market cap and they can't hire some intern to do this???? */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11010) /* 11.1.0 */
  cerr = cudaEventRecordWithFlags(pec->cevent,psc->cstream,event->waitFlags);CHKERRCUDA(cerr);
#else
  cerr = cudaEventRecord(pec->cevent,psc->cstream);CHKERRCUDA(cerr);
#endif /* CUDART_VERSION >= 11010 */
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamWaitEvent_CUDA(PetscStream strm, PetscEvent event)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  PetscEvent_CUDA  *pec = (PetscEvent_CUDA *)event->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  cerr = cudaStreamWaitEvent(psc->cstream,pec->cevent,event->waitFlags);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSynchronize_CUDA(PetscStream strm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  cerr = cudaStreamSynchronize(psc->cstream);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamQuery_CUDA(PetscStream strm, PetscBool *idle)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;

  PetscFunctionBegin;
  *idle = cudaStreamQuery(psc->cstream) == cudaErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamCaptureBegin_CUDA(PetscStream strm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  cerr = cudaStreamBeginCapture(psc->cstream,cudaStreamCaptureModeGlobal);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamCaptureEnd_CUDA(PetscStream strm, PetscStreamGraph sgraph)
{
  PetscStream_CUDA      *psc = (PetscStream_CUDA *)strm->data;
  PetscStreamGraph_CUDA *psgc = (PetscStreamGraph_CUDA *)sgraph->data;
  cudaError_t           cerr;

  PetscFunctionBegin;
  if (psgc->cgraph) {cerr = cudaGraphDestroy(psgc->cgraph);CHKERRCUDA(cerr);}
  cerr = cudaStreamEndCapture(psc->cstream,&psgc->cgraph);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static cudaEvent_t waitEvent   = NULL;
static PetscBool   waitCreated = PETSC_FALSE;

PETSC_STATIC_INLINE PetscErrorCode PetscStreamCUDADestroyWaitEvent(void)
{
  cudaError_t cerr;

  PetscFunctionBegin;
  cerr = cudaEventDestroy(waitEvent);CHKERRCUDA(cerr);
  waitEvent   = NULL;
  waitCreated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamWaitForStream_CUDA(PetscStream strmx, PetscStream strmy)
{
  PetscStream_CUDA *pscx = (PetscStream_CUDA *)strmx->data;
  PetscStream_CUDA *pscy = (PetscStream_CUDA *)strmy->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  if (!waitCreated) {
    PetscErrorCode ierr;

    cerr = cudaEventCreateWithFlags(&waitEvent,cudaEventDisableTiming);CHKERRCUDA(cerr);
    ierr = PetscRegisterFinalize(PetscStreamCUDADestroyWaitEvent);CHKERRQ(ierr);
    waitCreated = PETSC_TRUE;
  }
  cerr = cudaEventRecord(waitEvent,pscy->cstream);CHKERRCUDA(cerr);
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11011) /* 11.1.1 */
  cerr = cudaStreamWaitEvent(pscx->cstream,waitEvent,cudaEventWaitDefault);CHKERRCUDA(cerr);
#else
  cerr = cudaStreamWaitEvent(pscx->cstream,waitEvent,0);CHKERRCUDA(cerr);
#endif /* 11.1.1 */
  PetscFunctionReturn(0);
}

static const struct _StreamOps cuops = {
  PetscStreamCreate_CUDA,
  PetscStreamDestroy_CUDA,
  PetscStreamSetUp_CUDA,
  NULL,
  PetscStreamGetStream_CUDA,
  PetscStreamRestoreStream_CUDA,
  PetscStreamRecordEvent_CUDA,
  PetscStreamWaitEvent_CUDA,
  PetscStreamSynchronize_CUDA,
  PetscStreamQuery_CUDA,
  PetscStreamCaptureBegin_CUDA,
  PetscStreamCaptureEnd_CUDA,
  PetscStreamWaitForStream_CUDA
};

PetscErrorCode PetscStreamCreate_CUDA(PetscStream strm)
{
  PetscStream_CUDA *psc;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&psc);CHKERRQ(ierr);
  strm->data = (void *)psc;
  ierr = PetscMemcpy(strm->ops,&cuops,sizeof(cuops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
