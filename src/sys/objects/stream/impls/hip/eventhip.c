#include "streamhip.h"

#if PetscDefined(HAVE_HIP)
static PetscErrorCode PetscEventDestroy_HIP(PetscEvent event)
{
  PetscEvent_HIP *peh = (PetscEvent_HIP *)event->data;
  PetscErrorCode ierr;
  hipError_t     herr;

  PetscFunctionBegin;
  if (peh->hevent) {herr = hipEventDestroy(peh->hevent);CHKERRHIP(herr);}
  ierr = PetscFree(event->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscEventSetup_HIP(PetscEvent event)
{
  PetscEvent_HIP *peh = (PetscEvent_HIP *)event->data;
  hipError_t     herr;

  PetscFunctionBegin;
  herr = hipEventCreateWithFlags(&peh->hevent, event->eventFlags);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscEventSynchronize_HIP(PetscEvent event)
{
  PetscEvent_HIP *peh = (PetscEvent_HIP *)event->data;
  hipError_t     herr;

  PetscFunctionBegin;
  herr = hipEventSynchronize(peh->hevent);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscEventQuery_HIP(PetscEvent event, PetscBool *busy)
{
  PetscEvent_HIP *peh = (PetscEvent_HIP *)event->data;

  PetscFunctionBegin;
  *busy = PETSC_FALSE;
  if (hipEventQuery(peh->hevent) == cudaErrorNotReady) {
    *busy = PETSC_TRUE;
    hipGetLastError();
  }
  PetscFunctionReturn(0);
}

static struct _EventOps ehops = {
  PetscEventCreate_HIP,
  PetscEventDestroy_HIP,
  PetscEventSetup_HIP,
  PetcsEventSynchronize_HIP,
  PetscEventQuery_HIP
};
#endif /* HAVE_HIP */

PETSC_EXTERN PetscErrorCode PetscEventCreate_HIP(PetscEvent event)
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
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with HIP support\n");
#endif
  PetscFunctionReturn(0);
}
