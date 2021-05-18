#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

/*@C
  PetscEventCreate - Creates an empty PetscEvent object. The type can then be set with PetscEventSetType().

  Not Collective

  Output Parameter:
. event  - The allocated PetscEvent object

  Notes:
  You must set the stream type before using the PetscEvenr object, otherwise an error is generated on debug builds.

  Level: beginner

.seealso: PetscEventDestroy(), PetscEventSetType(), PetscEventSetFlags(), PetscEventSetUp()
@*/
PetscErrorCode PetscEventCreate(PetscEvent *event)
{
  PetscEvent     e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(event,1);
  ierr = PetscEventInitializePackage();CHKERRQ(ierr);
  /* Setting to null taken from VecCreate(), why though? */
  *event = NULL;
  ierr = PetscNew(&e);CHKERRQ(ierr);
  e->laststreamid = PETSC_DEFAULT;
  e->idle = PETSC_TRUE;
  *event = e;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventDestroy - Destroys a PetscEvent

  Not Collective

  Input Parameter:
. event  - The PetscEvent to destroy

  Level: beginner

.seealso: PetscEventCreate(), PetscEventSetType(), PetscEventSetFlags(), PetscEventSetUp()
@*/
PetscErrorCode PetscEventDestroy(PetscEvent *event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*event) PetscFunctionReturn(0);
  PetscValidStreamType(*event,1);
  ierr = (*(*event)->ops->destroy)(*event);CHKERRQ(ierr);
  ierr = PetscFree((*event)->type);CHKERRQ(ierr);
  ierr = PetscFree(*event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscEventSetFlags - Set flags which determine the behavior of PetscEvent and wait calls

  Not Collective

  Input Parameters:
+ event - The PetscEvent object
. eventFlags - Flags governing event object itself
- waitFlags - Flags governing how wait operations are performed

  Notes:
  The particular flag values should be obtained from documentation for the respective underlying stream
  implementation. For example for a PetscEvent of type PETSCSTREAMCUDA this routine serves to collect the flags
  normally passed to cudaEventCreateWithFlags() and cudaEventRecordWithFlags() respectively.

  Level: intermediate

.seealso: PetscEventDestroy(), PetscEventSetType(), PetscEventGetFlags(), PetscEventSetUp()
@*/
PetscErrorCode PetscEventSetFlags(PetscEvent event, unsigned int eventFlags, unsigned int waitFlags)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(event->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change flags on already setup event");
  event->eventFlags = eventFlags;
  event->waitFlags = waitFlags;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventGetFlags - Get flags which determine the behavior of event and wait calls

  Not Collective

  Input Parameter:
. event - The PetscEvent object

  Output Parameters:
+ eventFlags - Flags governing event object itself
- waitFlags - Flags governing how wait operations are performed

  Notes:
  Pass NULL for either argument if not needed

  Level: intermediate

.seealso: PetscEventDestroy(), PetscEventSetType(), PetscEventSetFlags(), PetscEventSetUp()
@*/
PetscErrorCode PetscEventGetFlags(PetscEvent event, unsigned int *eventFlags, unsigned int *waitFlags)
{
  PetscFunctionBegin;
  if (eventFlags) *eventFlags = event->eventFlags;
  if (waitFlags)  *waitFlags  = event->waitFlags;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventSetUp - Sets up and finalizes internal data structures

  Not Collective

  Input Parameter:
. event  - The PetscEvent to set up

  Level: beginner

.seealso: PetscEventCreate(), PetscEventSetType(), PetscEventDestroy()
@*/
PetscErrorCode PetscEventSetUp(PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  if (event->setup) PetscFunctionReturn(0);
  ierr = (*event->ops->setup)(event);CHKERRQ(ierr);
  event->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventSynchronize - Blocks the calling host thread until all work captured by the PetscEvent has finished

  Not Collective

  Input Parameter:
. event  - The PetscEvent to synchronize on

  Notes:
  All work is guaranteed to have been completed only after the host thread returns from this function. As it hard-stops
  the host thread, this routine should only be used as a last resort between asynchronous calls, or at the end of a set
  of asynchronous calls.
 The user should almost never have reason to call this routine directly, as any asynchronous routines may invoke it if
  necessary.

  Level: beginner

.seealso: PetscEventCreate(), PetscEventQuery(), PetscStreamWaitEvent(), PetscStreamSynchronize()
@*/
PetscErrorCode PetscEventSynchronize(PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  if (PetscUnlikelyDebug(!event->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscEventSetUp() before using it");
  if (event->idle) {
    ierr = PetscEventValidateIdle_Internal(event);CHKERRQ(ierr);
  } else {
    ierr = (*event->ops->synchronize)(event);CHKERRQ(ierr);
    event->idle = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscEventQuery - Returns whether a PetscEvent is idle

  Not Collective

  Input Parameter:
. event  - The PetscEvent object

  Output Parameter:
. idle - PETSC_TRUE if the PetscEvent is idle, PETSC_FALSE otherwise

  Notes:
  Results of this routine are cached on return, allowing this function to be called repeatedly in an efficient
  manner.

  Level: intermediate

.seealso: PetscEventCreate(), PetscEventSynchronize(), PetscStreamWaitEvent(), PetscStreamSynchronize()
@*/
PetscErrorCode PetscEventQuery(PetscEvent event, PetscBool *idle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  PetscValidBoolPointer(idle,2);
  if (PetscUnlikelyDebug(!event->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscEventSetUp() before calling this routine");
  if (event->idle) {
    *idle = PETSC_TRUE;
    ierr = PetscEventValidateIdle_Internal(event);CHKERRQ(ierr);
  } else {
    ierr = (*event->ops->query)(event,idle);CHKERRQ(ierr);
    event->idle = *idle;
  }
  PetscFunctionReturn(0);
}
