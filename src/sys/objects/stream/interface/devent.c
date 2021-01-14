#include <petsc/private/deviceimpl.h>

PetscErrorCode PetscEventCreate(PetscEvent *event)
{
  PetscEvent     e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(event,1);
  ierr = PetscStreamRegisterAll();CHKERRQ(ierr);
  /* Setting to null taken from VecCreate(), why though? */
  *event = NULL;
  ierr = PetscNew(&e);CHKERRQ(ierr);
  e->setup = PETSC_FALSE;
  e->type = PETSC_STREAM_INVALID;
  e->eventFlags = 0;
  e->waitFlags = 0;
  *event = e;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventDestroy(PetscEvent *event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!event) PetscFunctionReturn(0);
  PetscValidPointer(event,1);
  ierr = (*(*event)->ops->destroy)(*event);CHKERRQ(ierr);
  ierr = PetscFree(*event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventSetFlags(PetscEvent event, unsigned int eventFlags, unsigned int waitFlags)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(event->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change flags on already setup event\n");
  event->eventFlags = eventFlags;
  event->waitFlags = waitFlags;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventGetFlags(PetscEvent event, unsigned int *eventFlags, unsigned int *waitFlags)
{
  PetscFunctionBegin;
  if (eventFlags) *eventFlags = event->eventFlags;
  if (waitFlags)  *waitFlags  = event->waitFlags;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventSetup(PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  if (event->setup) PetscFunctionReturn(0);
  ierr = (*event->ops->setup)(event);CHKERRQ(ierr);
  event->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventSynchronize(PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  ierr = (*event->ops->synchronize)(event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventQuery(PetscEvent event, PetscBool *busy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  PetscValidBoolPointer(busy,2);
  ierr = (*event->ops->query)(event, busy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
