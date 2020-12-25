#include "streamhip.h" /*I "petscdevice.h" I*/

PETSC_STATIC_INLINE PetscErrorCode PetscEventDestroy_HIP(PetscEvent event)
{
  PetscEvent_HIP *peh = (PetscEvent_HIP *)event->data;
  PetscErrorCode ierr;
  hipError_t     herr;

  PetscFunctionBegin;
  if (peh->hevent) {herr = hipEventDestroy(peh->hevent);CHKERRHIP(herr);}
  ierr = PetscFree(event->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscEventSetup_HIP(PetscEvent event)
{
  PetscEvent_HIP *peh = (PetscEvent_HIP *)event->data;
  hipError_t     herr;

  PetscFunctionBegin;
  herr = hipEventCreateWithFlags(&peh->hevent,event->eventFlags);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscEventSynchronize_HIP(PetscEvent event)
{
  PetscEvent_HIP *peh = (PetscEvent_HIP *)event->data;
  hipError_t     herr;

  PetscFunctionBegin;
  herr = hipEventSynchronize(peh->hevent);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscEventQuery_HIP(PetscEvent event, PetscBool *idle)
{
  PetscEvent_HIP *peh = (PetscEvent_HIP *)event->data;

  PetscFunctionBegin;
  *idle = hipEventQuery(peh->hevent) == hipErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

static const struct _EventOps ehops = {
  PetscEventCreate_HIP,
  PetscEventDestroy_HIP,
  PetscEventSetup_HIP,
  NULL,
  PetscEventSynchronize_HIP,
  PetscEventQuery_HIP
};

PetscErrorCode PetscEventCreate_HIP(PetscEvent event)
{
  PetscEvent_HIP *peh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&peh);CHKERRQ(ierr);
  event->data = (void *)peh;
  ierr = PetscMemcpy(event->ops,&ehops,sizeof(ehops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
