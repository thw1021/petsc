#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

const char *const PetscStreamModes[] = {"global_blocking","default_blocking","global_nonblocking","MAX_MODE","PetscStreamMode","PETSC_STREAM_",NULL};
static PetscInt streamID = 0;

/*@C
  PetscStreamCreate - Creates an empty PetscStream object. The type can then be set with PetscStreamSetType().

  Not Collective

  Output Parameter:
. strm  - The allocated PetscStream object

  Notes:
  You must set the stream type before using the PetscStream object, otherwise an error is generated on debug builds.

  Level: beginner

.seealso: PetscStreamDestroy(), PetscStreamSetType(), PetscStreamSetMode(), PetscStreamSetUp()
@*/
PetscErrorCode PetscStreamCreate(PetscStream *strm)
{
  PetscStream    s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(strm,1);
  ierr = PetscStreamInitializePackage();CHKERRQ(ierr);
  /* Setting to null taken from VecCreate(), why though? */
  *strm = NULL;
  ierr = PetscNew(&s);CHKERRQ(ierr);
  s->id = streamID++;
  s->idle = PETSC_TRUE;
  s->mode = PETSC_STREAM_DEFAULT_BLOCKING;
  *strm = s;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamDestroy - Destroys a PetscStream

  Not Collective

  Input Parameter:
. strm - The PetscStream object

  Level: beginner

.seealso: PetscStreamCreate(), PetscStreamSetType(), PetscStreamSetMode(), PetscStreamSetUp()
@*/
PetscErrorCode PetscStreamDestroy(PetscStream *strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*strm) PetscFunctionReturn(0);
  PetscValidPointer(strm,1);
  ierr = (*(*strm)->ops->destroy)(*strm);CHKERRQ(ierr);
  ierr = PetscFree((*strm)->type);CHKERRQ(ierr);
  ierr = PetscFree(*strm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamSetMode - Sets the stream-synchronization mode for a particular PetscStream

  Not Collective

  Input Parameters:
+ strm - The PetscStream object
- mode - The PetscStreamMode to set

  Notes:
  See PetscStreamMode for available stream modes

  Level: beginner

.seealso: PetscStreamCreate(), PetscStreamSetType(), PetscStreamGetMode(), PetscStreamSetUp()
@*/
PetscErrorCode PetscStreamSetMode(PetscStream strm, PetscStreamMode mode)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(mode >= PETSC_STREAM_MAX_MODE) || PetscUnlikelyDebug(mode < 0)) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscStreamMode %d is invalid, out of range of [0,%d)",(int)mode,(int)PETSC_STREAM_MAX_MODE);
  }
  strm->mode = mode;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGetMode - Gets the stream-synchronization mode for a particular PetscStream

  Not Collective

  Input Parameter:
. strm - The PetscStream object

  Output Paramater:
. mode - The PetscStreamMode

  Notes:
  See PetscStreamMode for available stream modes

  Level: beginner

.seealso: PetscStreamCreate(), PetscStreamSetType(), PetscStreamSetMode(), PetscStreamSetUp()
@*/
PetscErrorCode PetscStreamGetMode(PetscStream strm, PetscStreamMode *mode)
{
  PetscFunctionBegin;
  PetscValidPointer(mode,2);
  *mode = strm->mode;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamSetUp - Sets up and finalizes internal data structures for later use.

  Not Collective

  Input Parameter:
. strm - The PetscStream object

  Level: beginner

.seealso: PetscStreamCreate(), PetscStreamSetType(), PetscStreamSetMode(), PetscStreamGetStream()
@*/
PetscErrorCode PetscStreamSetUp(PetscStream strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  if (strm->setup) PetscFunctionReturn(0);
  ierr = (*strm->ops->setup)(strm);CHKERRQ(ierr);
  strm->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamDuplicate - Duplicates a PetscStream object

  Not Collective

  Input Parameter:
. strm - The PetscStream object to duplicate

  Output Paramter:
. strmdup - The duplicated PetscStream

  Level: beginner

.seealso: PetscStreamCreate(), PetscStreamSetType(), PetscStreamSetMode()
@*/
PetscErrorCode PetscStreamDuplicate(PetscStream strm, PetscStream *strmdup)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(strmdup,2);
  ierr = PetscStreamCreate(strmdup);CHKERRQ(ierr);
  ierr = PetscStreamSetMode(*strmdup,strm->mode);CHKERRQ(ierr);
  ierr = PetscStreamSetType(*strmdup,strm->type);CHKERRQ(ierr);
  ierr = PetscStreamSetUp(*strmdup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGetStream - Retrieves the implementation specific stream

  Not Collective

  Input Parameter:
. strm - The PetscStream object

  Output Parameter:
. dstrm - The device stream

  Notes:
  This is a borrowed reference, the user should not destroy it themselves

  Level: advanced

.seealso: PetscStreamCreate(), PetscStreamSetType(), PetscStreamSetMode(), PetscStreamRestoreStream()
@*/
PetscErrorCode PetscStreamGetStream(PetscStream strm, void *dstrm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(dstrm,2);
  if (PetscUnlikelyDebug(!strm->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscStream is not setup yet, must call PetscStreamSetUp()");
  ierr = (*strm->ops->getstream)(strm, dstrm);CHKERRQ(ierr);
  /* Assume the stream will get work */
  strm->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamRestoreStream - Restores the implementation specific stream

  Not Collective

  Input Parameter:
+ strm - The PetscStream object
- dstrm - The device stream

  Notes:
  The restored stream must be the same stream that was checked out via PetscStreamGetStream()

  Level: advanced

.seealso: PetscStreamCreate(), PetscStreamSetType(), PetscStreamSetMode(), PetscStreamGetStream()
@*/
PetscErrorCode PetscStreamRestoreStream(PetscStream strm, void *dstrm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(dstrm,2);
  ierr = (*strm->ops->restorestream)(strm, dstrm);CHKERRQ(ierr);
  /* In case the stream is checked out, sync'ed while checked out, then work queued onto stream */
  strm->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamRecordEvent - Records the contents of a PetscStream in a PetscEvent

  Not Collective

  Input Parameter:
+ strm - The PetscStream object
- event - The PetscEvent object

  Notes:
  Records in the PetscEvent the state of the PetscStream at the time of this call. Subsequent uses of the PetscStream
  will not affect the PetscEvent. PetscStreamRecordEvent() can be called an arbitrary number of times on the same
 PetscEvent, with subsequent calls overwriting the previous state. Thus the content of the PetscEvent will only
 represent the most recently recorded state.

  Level: intermediate

.seealso: PetscStreamCreate(), PetscEventCreate(), PetscStreamWaitEvent(), PetscStreamWaitForStream()
@*/
PetscErrorCode PetscStreamRecordEvent(PetscStream strm, PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(strm,1,event,2);
  ierr = (*strm->ops->recordevent)(strm,event);CHKERRQ(ierr);
  /* Imprint on the event the id of the stream, so subsequent waits need not check */
  event->laststreamid = strm->id;
  /* Assume the event has work */
  event->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamWaitEvent - Makes a PetscStream wait for all work captured in a PetscEvent to be completed

  Collective on PetscEvent

  Input Parameter:
+ strm - The PetscStream object
- event - The PetscEvent object

  Notes:
  Each event stores an internal identifier of the last PetscStream to wait on it, allowing this routine to be
  efficiently called repeatedly for the same PetscStream and PetscEvent.

  Usage:
$ // Enqueue some work onto strm1
$ ierr = VecNormAsync(vec1,NORM_2,&res1,strm1);CHKERRQ(ierr);
$ // Record the state of strm1
$ ierr = PetscStreamRecordEvent(strm1,event);CHKERRQ(ierr);
$ // Concurrently enqueue other work onto strm2
$ ierr = VecDotAsync(vec1,vec2,res2,strm2);CHKERRQ(ierr);
$ // Make strm2 asynchronously wait on the completion of strm1
$ ierr = PetscStreamWaitEvent(strm2,event);CHKERRQ(ierr);
$ // Enqueue more work onto strm2 using results from strm1
$ ierr = VecScaleAsync(vec1,res1);CHKERRQ(ierr);

  Level: intermediate

.seealso: PetscStreamCreate(), PetscEventCreate(), PetscStreamRecordEvent(), PetscStreamWaitForStream()
@*/
PetscErrorCode PetscStreamWaitEvent(PetscStream strm, PetscEvent event)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(strm,1,event,2);
  /* Last stream to interact with this event was this stream, no need to wait */
  if (event->laststreamid == strm->id) PetscFunctionReturn(0);
  ierr = (*strm->ops->waitevent)(strm, event);CHKERRQ(ierr);
  event->laststreamid = strm->id;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamSynchronizeDevice_Private(PetscStream strm)
{
#if PetscDefined(HAVE_CUDA)
  cudaError_t cerr;
#endif
#if PetscDefined(HAVE_HIP)
  hipError_t herr;
#endif

  PetscFunctionBegin;
#if PetscDefined(HAVE_CUDA)
  cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
#endif
#if PetscDefined(HAVE_HIP)
  herr = hipDeviceSynchronize();CHKERRHIP(herr);
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamSynchronize - Block the calling host thread until all work enqueued in the PetscStream has finished

  Not Collective

  Input Parameter:
. strm - The PetscStream object

  Notes:
  All work is guaranteed to have been completed only after the host thread returns from this function. As it hard-stops
  the host thread, this routine should only be used as a last resort between asynchronous calls, or at the end of a set
  of asynchronous calls.

  Level: advanced

.seealso: PetscStreamCreate(), PetscStreamQuery(), PetscEventSynchronize(), PetscStreamWaitForStream()
@*/
PetscErrorCode PetscStreamSynchronize(PetscStream strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    ierr = PetscStreamSynchronizeDevice_Private(strm);CHKERRQ(ierr);
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    if (strm->idle) {
      ierr = PetscStreamValidateIdle_Internal(strm);CHKERRQ(ierr);
    } else {
      ierr = (*strm->ops->synchronize)(strm);CHKERRQ(ierr);
      strm->idle = PETSC_TRUE;
    }
  default:
    break;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamQuery - Returns whether or not a PetscStream is idle

  Not Collective

  Input Parameter:
. strm - The PetscStream object

  Output Parameter:
. idle - PETSC_TRUE if PetscStream has NO work, PETSC_FALSE if it has work

  Notes:
  Results of PetscStreamQuery() are cached on return, allowing this function to be called repeatedly in an efficient
  manner.

  Level: advanced

.seealso: PetscStreamCreate(), PetscStreamQuery(), PetscEventSynchronize()
@*/
PetscErrorCode PetscStreamQuery(PetscStream strm, PetscBool *idle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidBoolPointer(idle,2);
  if (strm->idle) {
    *idle = PETSC_TRUE;
    ierr = PetscStreamValidateIdle_Internal(strm);CHKERRQ(ierr);
  } else {
    ierr = (*strm->ops->query)(strm,idle);CHKERRQ(ierr);
    strm->idle = *idle;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamWaitForStream - Make one stream wait for another stream to finish

  Not Collective, Asynchronous

  Input Parameters:
+ strmx - The PetscStream object that is waiting
- strmy - The PetscStream object that is being waited on

  Notes:
  This routine is a more stream-lined version of PetscStreamRecordEvent() -> PetscStreamWaitEvent() chain for the case
  of serializing two streams. If one is synchronizing multiple streams however, it is recommended that one use the
  aforementioned event recording chain. This routine uses only the state of strmy at the moment this routine was
  called, so any future work queued will not affect strmx. It is safe to pass the same stream to both arguments.

  Level: beginner

.seealso: PetscStreamCreate(), PetscStreamQuery(), PetscStreamRecordEvent(), PetscStreamWaitEvent()
@*/
PetscErrorCode PetscStreamWaitForStream(PetscStream strmx, PetscStream strmy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(strmx,1,strmy,2);
  if (strmx == strmy) PetscFunctionReturn(0);
  if (strmy->idle) {
    ierr = PetscStreamValidateIdle_Internal(strmy);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = (*strmx->ops->waitforstream)(strmx,strmy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
