#include "streamcuda.h"

#if PetscDefined(HAVE_CUDA)
static PetscErrorCode PetscEventDestroy_CUDA(PetscEvent event)
{
  PetscEvent_CUDA *pec = (PetscEvent_CUDA *)event->data;
  PetscErrorCode  ierr;
  cudaError_t     cerr;

  PetscFunctionBegin;
  if (pec->cevent) {cerr = cudaEventDestroy(pec->cevent);CHKERRCUDA(cerr);}
  ierr = PetscFree(event->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscEventSetup_CUDA(PetscEvent event)
{
  PetscEvent_CUDA *pec = (PetscEvent_CUDA *)event->data;
  cudaError_t     cerr;

  PetscFunctionBegin;
  cerr = cudaEventCreateWithFlags(&pec->cevent, event->eventFlags);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscEventSynchronize_CUDA(PetscEvent event)
{
  PetscEvent_CUDA *pec = (PetscEvent_CUDA *)event->data;
  cudaError_t     cerr;

  PetscFunctionBegin;
  cerr = cudaEventSynchronize(pec->cevent);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscEventQuery_CUDA(PetscEvent event, PetscBool *busy)
{
  PetscEvent_CUDA *pec = (PetscEvent_CUDA *)event->data;

  PetscFunctionBegin;
  *busy = PETSC_FALSE;
  if (cudaEventQuery(pec->cevent) == cudaErrorNotReady) {
    *busy = PETSC_TRUE;
    cudaGetLastError();
  }
  PetscFunctionReturn(0);
}

static const struct _EventOps ecuops = {
  PetscEventCreate_CUDA,
  PetscEventDestroy_CUDA,
  PetscEventSetup_CUDA,
  PetscEventSynchronize_CUDA,
  PetscEventQuery_CUDA
};
#endif /* HAVE_CUDA */

PETSC_EXTERN PetscErrorCode PetscEventCreate_CUDA(PetscEvent event)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_CUDA)
  {
    PetscEvent_CUDA *pec;
    PetscErrorCode  ierr;

    ierr = PetscNew(&pec);CHKERRQ(ierr);
    event->data = (void *)pec;
    ierr = PetscMemcpy(event->ops, &ecuops, sizeof(ecuops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with CUDA support\n");
#endif
  PetscFunctionReturn(0);
}
