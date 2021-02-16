#include "streamcuda.h"

#if PetscDefined(HAVE_CUDA)
static PetscErrorCode PetscStreamDestroy_CUDA(PetscStream strm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  if (psc->cstream) {cerr = cudaStreamDestroy(psc->cstream);CHKERRCUDA(cerr);}
  ierr = PetscFree(strm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamSetUp_CUDA(PetscStream strm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  cerr = cudaStreamCreate(&psc->cstream);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamGetStream_CUDA(PetscStream strm, void *dstrm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;

  PetscFunctionBegin;
  *((cudaStream_t*) dstrm) = psc->cstream;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamRestoreStream_CUDA(PetscStream strm, void *dstrm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(*((cudaStream_t*) dstrm) != psc->cstream)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"CUDA stream is not the same as the one that was checked out\n");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamRecordEvent_CUDA(PetscStream strm, PetscEvent event)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  PetscEvent_CUDA  *pec = (PetscEvent_CUDA *)event->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  /* It is honestly baffling that nvidia do not have a "since version X" note on their functions to make this
   easier. Instead you must find and *manually* search each version of the documentation until you find the version in
   which they introduced some change. $330 BILLION market cap and they can't hire some intern to do this???? */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11010) /* 11.1.0 */
  cerr = cudaEventRecordWithFlags(pec->cevent, psc->cstream, event->waitFlags);CHKERRCUDA(cerr);
#else
  cerr = cudaEventRecord(pec->cevent, psc->cstream);CHKERRCUDA(cerr);
#endif /* CUDART_VERSION >= 11010 */
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamWaitEvent_CUDA(PetscStream strm, PetscEvent event)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  PetscEvent_CUDA  *pec = (PetscEvent_CUDA *)event->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  cerr = cudaStreamWaitEvent(psc->cstream, pec->cevent, event->waitFlags);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamSynchronize_CUDA(PetscStream strm)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;
  cudaError_t      cerr;

  PetscFunctionBegin;
  cerr = cudaStreamSynchronize(psc->cstream);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscStreamQuery_CUDA(PetscStream strm, PetscBool *busy)
{
  PetscStream_CUDA *psc = (PetscStream_CUDA *)strm->data;

  PetscFunctionBegin;
  *busy = PETSC_FALSE;
  if (cudaStreamQuery(psc->cstream) == cudaErrorNotReady) {
    *busy = PETSC_TRUE;
    cudaGetLastError();
  }
  PetscFunctionReturn(0);
}

static const struct _StreamOps cuops = {
  PetscStreamCreate_CUDA,
  PetscStreamDestroy_CUDA,
  PetscStreamSetUp_CUDA,
  PetscStreamGetStream_CUDA,
  PetscStreamRestoreStream_CUDA,
  PetscStreamRecordEvent_CUDA,
  PetscStreamWaitEvent_CUDA,
  PetscStreamSynchronize_CUDA,
  PetscStreamQuery_CUDA
};
#endif /* HAVE_CUDA */

PETSC_EXTERN PetscErrorCode PetscStreamCreate_CUDA(PetscStream strm)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_CUDA)
  {
    PetscStream_CUDA *psc;
    PetscErrorCode   ierr;

    ierr = PetscNew(&psc);CHKERRQ(ierr);
    strm->data = (void *)psc;
    ierr = PetscMemcpy(strm->ops, &cuops, sizeof(cuops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with CUDA support\n");
#endif
  PetscFunctionReturn(0);
}
