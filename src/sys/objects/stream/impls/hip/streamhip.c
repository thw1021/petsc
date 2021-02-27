#include "streamhip.h"

#if PetscDefined(HAVE_HIP)
PETSC_STATIC_INLINE PetscErrorCode PetscStreamDestroy_HIP(PetscStream strm)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;
  PetscErrorCode   ierr;
  hipError_t       herr;

  PetscFunctionBegin;
  if (psh->cstream) {herr = hipStreamDestroy(psh->hstream);CHKERRHIP(herr);}
  ierr = PetscFree(strm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSetUp_HIP(PetscStream strm)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;
  hipError_t      herr;

  PetscFunctionBegin;
  herr = hipStreamCreate(&psh->hstream);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamGetStream_HIP(PetscStream strm, void *dstrm)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;

  PetscFunctionBegin;
  *((hipStream_t*) dstrm) = psh->hstream;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamRestoreStream_HIP(PetscStream strm, void *dstrm)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(*((hipStream_t*) dstrm) != psh->hstream)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"HIP stream is not the same as the one that was checked out");
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamRecordEvent_HIP(PetscStream strm, PetscEvent event)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;
  hipError_t      herr;

  PetscFunctionBegin;
  herr = hipEventRecordWithFlags(event->hevent, psh->hstream, event->waitFlags);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamWaitEvent_HIP(PetscStream strm, PetscEvent event)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;
  hipError_t      herr;

  PetscFunctioBegin;
  herr = hipStreamWaitEvent(psh->hstream, event->hevent, event->waitFlags);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSynchronize_HIP(PetscStream strm)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;
  hipError_t      herr;

  PetscFunctionBegin;
  herr = hipStreamSynchronize(psh->hstream);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamQuery_HIP(PetscStream strm, PetscBool *busy)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;

  PetscFunctionBegin;
  *busy = PETSC_FALSE;
  if (hipStreamQuery(psh->Hstream) == hipErrorNotReady) {
    *busy = PETSC_TRUE;
    hipGetLastError();
  }
  PetscFunctionReturn(0);
}

static const struct _StreamOps hipops = {
  PetscStreamCreate_HIP,
  PetscStreamDestroy_HIP,
  PetscStreamSetUp_HIP,
  PetscStreamGetStream_HIP,
  PetscStreamRestoreStream_HIP,
  PetscStreamRecordEvent_HIP,
  PetscStreamWaitEvent_HIP,
  PetscStreamSynchronize_HIP,
  PetscStreamQuery_HIP
};
#endif /* HAVE_HIP */

PetscErrorCode PetscStreamCreate_HIP(PetscStream strm)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_HIP)
  {
    PetscStream_HIP *psh;
    PetscErrorCode   ierr;

    ierr = PetscNew(&psh);CHKERRQ(ierr);
    strm->data = (void *)psh;
    ierr = PetscMemcpy(strm->ops, &hipops, sizeof(hipops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with HIP support");
#endif
  PetscFunctionReturn(0);
}
