#include "streamcuda.h" /*I "petscdevice.h" I*/

PETSC_STATIC_INLINE PetscErrorCode PetscEventDestroy_CUDA(PetscEvent event)
{
  PetscEvent_CUDA *pec = (PetscEvent_CUDA *)event->data;
  PetscErrorCode  ierr;
  cudaError_t     cerr;

  PetscFunctionBegin;
  if (pec->cevent) {cerr = cudaEventDestroy(pec->cevent);CHKERRCUDA(cerr);}
  ierr = PetscFree(event->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscEventSetup_CUDA(PetscEvent event)
{
  PetscEvent_CUDA *pec = (PetscEvent_CUDA *)event->data;
  cudaError_t     cerr;

  PetscFunctionBegin;
  cerr = cudaEventCreateWithFlags(&pec->cevent,event->eventFlags);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscEventSynchronize_CUDA(PetscEvent event)
{
  PetscEvent_CUDA *pec = (PetscEvent_CUDA *)event->data;
  cudaError_t     cerr;

  PetscFunctionBegin;
  cerr = cudaEventSynchronize(pec->cevent);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscEventQuery_CUDA(PetscEvent event, PetscBool *idle)
{
  PetscEvent_CUDA *pec = (PetscEvent_CUDA *)event->data;

  PetscFunctionBegin;
  *idle = cudaEventQuery(pec->cevent) == cudaErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

static const struct _EventOps ecuops = {
  PetscEventCreate_CUDA,
  PetscEventDestroy_CUDA,
  PetscEventSetup_CUDA,
  NULL,
  PetscEventSynchronize_CUDA,
  PetscEventQuery_CUDA
};

PetscErrorCode PetscEventCreate_CUDA(PetscEvent event)
{
  PetscEvent_CUDA *pec;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&pec);CHKERRQ(ierr);
  event->data = (void *)pec;
  ierr = PetscMemcpy(event->ops,&ecuops,sizeof(ecuops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
