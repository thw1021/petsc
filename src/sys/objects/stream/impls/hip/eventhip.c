#include "streamhip.h"

#if PetscDefined(HAVE_HIP)
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
  herr = hipEventCreateWithFlags(&peh->hevent, event->eventFlags);CHKERRHIP(herr);
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
  PetcsEventSynchronize_HIP,
  PetscEventQuery_HIP
};
#endif /* HAVE_HIP */

PetscErrorCode PetscEventCreate_HIP(PetscEvent event)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_HIP)
  {
    PetscEvent_HIP *peh;
    PetscErrorCode ierr;

    ierr = PetscNew(&peh);CHKERRQ(ierr);
    event->data = (void *)peh;
    ierr = PetscMemcpy(event->ops, &ehops, sizeof(ehops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with HIP support");
#endif
  PetscFunctionReturn(0);
}
