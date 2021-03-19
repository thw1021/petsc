#include "streamhip.h" /*I "petscdevice.h" I*/

PETSC_STATIC_INLINE PetscErrorCode PetscStreamDestroy_HIP(PetscStream strm)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;
  PetscErrorCode   ierr;
  hipError_t       herr;

  PetscFunctionBegin;
  if (psh->hstream) {herr = hipStreamDestroy(psh->hstream);CHKERRHIP(herr);}
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
  PetscEvent_HIP  *peh = (PetscEvent_HIP *)event->data;
  hipError_t      herr;

  PetscFunctionBegin;
  /* HIP does not have hipEventRecordWithFlags atm */
  herr = hipEventRecord(peh->hevent,psh->hstream);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamWaitEvent_HIP(PetscStream strm, PetscEvent event)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;
  PetscEvent_HIP  *peh = (PetscEvent_HIP *)event->data;
  hipError_t      herr;

  PetscFunctionBegin;
  herr = hipStreamWaitEvent(psh->hstream,peh->hevent,event->waitFlags);CHKERRHIP(herr);
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

PETSC_STATIC_INLINE PetscErrorCode PetscStreamQuery_HIP(PetscStream strm, PetscBool *idle)
{
  PetscStream_HIP *psh = (PetscStream_HIP *)strm->data;

  PetscFunctionBegin;
  *idle = hipStreamQuery(psh->hstream) == hipErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

static hipEvent_t waitEvent   = NULL;
static PetscBool  waitCreated = PETSC_FALSE;

PETSC_STATIC_INLINE PetscErrorCode PetscStreamHIPDestroyWaitEvent(void)
{
  hipError_t herr;

  PetscFunctionBegin;
  herr = hipEventDestroy(waitEvent);CHKERRHIP(herr);
  waitEvent   = NULL;
  waitCreated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamWaitForStream_HIP(PetscStream strmx, PetscStream strmy)
{
  PetscStream_HIP *pshx = (PetscStream_HIP *)strmx->data;
  PetscStream_HIP *pshy = (PetscStream_HIP *)strmy->data;
  hipError_t      herr;

  PetscFunctionBegin;
  if (!waitCreated) {
    PetscErrorCode ierr;

    herr = hipEventCreateWithFlags(&waitEvent,hipEventDisableTiming);CHKERRHIP(herr);
    ierr = PetscRegisterFinalize(PetscStreamHIPDestroyWaitEvent);CHKERRQ(ierr);
    waitCreated = PETSC_TRUE;
  }
  herr = hipEventRecord(waitEvent,pshy->hstream);CHKERRHIP(herr);
  herr = hipStreamWaitEvent(pshx->hstream,waitEvent,0);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

static const struct _StreamOps hipops = {
  PetscStreamCreate_HIP,
  PetscStreamDestroy_HIP,
  PetscStreamSetUp_HIP,
  NULL,
  PetscStreamGetStream_HIP,
  PetscStreamRestoreStream_HIP,
  PetscStreamRecordEvent_HIP,
  PetscStreamWaitEvent_HIP,
  PetscStreamSynchronize_HIP,
  PetscStreamQuery_HIP,
  NULL,
  NULL,
  PetscStreamWaitForStream_HIP
};

PetscErrorCode PetscStreamCreate_HIP(PetscStream strm)
{
  PetscStream_HIP *psh;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&psh);CHKERRQ(ierr);
  strm->data = (void *)psh;
  ierr = PetscMemcpy(strm->ops,&hipops,sizeof(hipops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
