#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

const char *const PetscStreamModes[] = {"global_blocking","default_blocking","global_nonblocking","MAX_MODE","PetscStreamMode","PETSC_STREAM_",NULL};

static PetscInt PetscDeviceContextID = 0;

PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *dctx)
{
  PetscDeviceContext dc;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  ierr  = PetscDeviceInitializePackage();CHKERRQ(ierr);
  *dctx = NULL;
  ierr  = PetscNew(&dc);CHKERRQ(ierr);
  dc->id   = PetscDeviceContextID++;
  dc->idle = PETSC_TRUE;
  dc->mode = PETSC_STREAM_DEFAULT_BLOCKING;
  *dctx = dc;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dctx) PetscFunctionReturn(0);
  PetscValidPointer(dctx,1);
  if (!*dctx) PetscFunctionReturn(0);
  if (PetscUnlikelyDebug((*dctx)->numChildren)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Device context still has %D un-restored children, must call PetscDeviceContextRestore() on all children before destroying",(*dctx)->numChildren);
  ierr = (*(*dctx)->ops->destroy)(*dctx);CHKERRQ(ierr);
  ierr = PetscFree((*dctx)->type);CHKERRQ(ierr);
  ierr = PetscFree((*dctx)->childIDs);CHKERRQ(ierr);
  ierr = PetscFree(*dctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* See PetscStreamMode in petscdevicetypes.h */
PetscErrorCode PetscDeviceContextSetMode(PetscDeviceContext dctx, PetscStreamMode mode)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(mode >= PETSC_STREAM_MAX_MODE) || PetscUnlikelyDebug(mode < 0)) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscStreamMode %d is invalid, out of range of [0,%d)",(int)mode,(int)PETSC_STREAM_MAX_MODE);
  }
  dctx->mode = mode;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextGetMode(PetscDeviceContext dctx, PetscStreamMode *mode)
{
  PetscFunctionBegin;
  PetscValidPointer(mode,2);
  *mode = dctx->mode;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextSetup(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  if (dctx->setup) PetscFunctionReturn(0);
  ierr = PetscEventCreate(&dctx->event);CHKERRQ(ierr);
  ierr = PetscEventSetType(dctx->event,dctx->type);CHKERRQ(ierr);
  ierr = PetscEventSetUp(dctx->event);CHKERRQ(ierr);
  ierr = (*dctx->ops->setup)(dctx);CHKERRQ(ierr);
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDuplicate - Duplicates a PetscDeviceContext object

  Not Collective

  Input Parameter:
. dctx - The PetscDeviceContext object to duplicate

  Output Paramter:
. strmdup - The duplicated PetscDeviceContext

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(), PetscDeviceContextSetMode()
@*/
PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext dctx, PetscDeviceContext *dctxdup)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dctxdup,2);
  ierr = PetscDeviceContextCreate(dctxdup);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetType(*dctxdup,dctx->type);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetMode(*dctxdup,dctx->mode);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetUp(*dctxdup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetStream - Retrieves the implementation specific stream

  Not Collective

  Input Parameter:
. dctx - The PetscDeviceContext object

  Output Parameter:
. dstrm - The device stream

  Notes:
  This is a borrowed reference, the user should not destroy it themselves

  Level: advanced

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(), PetscDeviceContextSetMode(), PetscDeviceContextRestoreStream()
@*/
PetscErrorCode PetscDeviceContextGetStream(PetscDeviceContext dctx, void *dstrm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dstrm,2);
  if (PetscUnlikelyDebug(!dctx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscDeviceContext is not setup yet, must call PetscDeviceContextSetUp()");
  ierr = (*dctx->ops->getstream)(dctx,dstrm);CHKERRQ(ierr);
  /* Assume the stream will get work */
  dctx->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextRestoreStream - Restores the implementation specific stream

  Not Collective

  Input Parameter:
+ dctx - The PetscDeviceContext object
- dstrm - The device stream

  Notes:
  The restored stream must be the same stream that was checked out via PetscDeviceContextGetStream()

  Level: advanced

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(), PetscDeviceContextSetMode(), PetscDeviceContextGetStream()
@*/
PetscErrorCode PetscDeviceContextRestoreStream(PetscDeviceContext dctx, void *dstrm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dstrm,2);
  ierr = (*dctx->ops->restorestream)(dctx,dstrm);CHKERRQ(ierr);
  /* In case the stream is checked out, sync'ed while checked out, then work queued onto stream */
  dctx->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextGetBLASHandle(PetscDeviceContext dctx, void *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(handle,2);
  if (PetscUnlikelyDebug(!dctx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscDeviceContext is not setup yet, must call PetscDeviceContextSetUp()");
  ierr = (*dctx->ops->getblashandle)(dctx,handle);CHKERRQ(ierr);
  dctx->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextRestoreBLASHandle(PetscDeviceContext dctx, void *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(handle,2);
  ierr = (*dctx->ops->restoreblashandle)(dctx,handle);CHKERRQ(ierr);
  /* In case the handle is checked out, sync'ed while checked out, then work queued onto stream */
  dctx->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextQuery - Returns whether or not a PetscDeviceContext is idle

  Not Collective

  Input Parameter:
. dctx - The PetscDeviceContext object

  Output Parameter:
. idle - PETSC_TRUE if PetscDeviceContext has NO work, PETSC_FALSE if it has work

  Notes:
  This routine only refers a singular context and does NOT take any of its children into
  account. That is, if dctx is idle but has dependents who do have work, this routine
  still returns PETSC_TRUE.

  Results of PetscDeviceContextQuery() are cached on return, allowing this function to be
  called repeatedly in an efficient manner.

  Level: advanced

.seealso: PetscDeviceContextCreate(), PetscDeviceContextWaitForContext()
@*/
PetscErrorCode PetscDeviceContextQuery(PetscDeviceContext dctx, PetscBool *idle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidBoolPointer(idle,2);
  if (dctx->idle) {
    *idle = PETSC_TRUE;
    ierr = PetscDeviceContextValidateIdle_Internal(dctx);CHKERRQ(ierr);
  } else {
    ierr = (*dctx->ops->query)(dctx,idle);CHKERRQ(ierr);
    dctx->idle = *idle;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextWaitForContext - Make one context wait for another context to finish

  Not Collective, Asynchronous

  Input Parameters:
+ dctxa - The PetscDeviceContext object that is waiting
- dctxb - The PetscDeviceContext object that is being waited on

  Notes:
  This routine is a more stream-lined version of PetscDeviceContextRecordEvent() -> PetscDeviceContextWaitEvent() chain for the case
  of serializing two streams. If one is synchronizing multiple streams however, it is recommended that one use the
  aforementioned event recording chain. This routine uses only the state of dctxb at the moment this routine was
  called, so any future work queued will not affect dctxa. It is safe to pass the same context to both arguments.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextQuery(), PetscDeviceContextRecordEvent(), PetscDeviceContextWaitEvent()
@*/
PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(dctxa,1,dctxb,2);
  if (dctxa == dctxb) PetscFunctionReturn(0);
  if (dctxb->idle) {
    /* No need to do the extra function lookup and event record if the stream were waiting on isn't doing anything */
    ierr = PetscDeviceContextValidateIdle_Internal(dctxb);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = (*dctxa->ops->waitforctx)(dctxa,dctxb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Make n edges in the DAG from node dctx, dctx expects to restore these children before it is freed */
PetscErrorCode PetscDeviceContextSplit(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub)
{
  PetscDeviceContext *dsubTmp = NULL;
  PetscInt           i = 0;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dsub,3);
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts requested %D < 0",n);
  /* update new child totals */
  dctx->numChildren += n;
  /* now to find out if we have room */
  if (dctx->numChildren > dctx->maxNumChildren) {
    /* no room, either from having too many kids or not having any */
    if (dctx->childIDs) {
      /* have existing children, must reallocate them */
      ierr = PetscRealloc(dctx->numChildren*sizeof(*(dctx->childIDs)),&dctx->childIDs);CHKERRQ(ierr);
    } else {
      /* have no children */
      ierr = PetscCalloc1(dctx->numChildren,&dctx->childIDs);CHKERRQ(ierr);
    }
    /* update total number of children */
    dctx->maxNumChildren = dctx->numChildren;
  }
  ierr = PetscMalloc1(n,&dsubTmp);CHKERRQ(ierr);
  while (n) {
    /* empty child slot */
    if (!(dctx->childIDs[i])) {
      /* create the child context in the image of its parent */
      ierr = PetscDeviceContextCreate(dsubTmp+i);CHKERRQ(ierr);
      ierr = PetscDeviceContextSetType(dsubTmp[i],dctx->type);CHKERRQ(ierr);
      ierr = PetscDeviceContextSetMode(dsubTmp[i],dctx->mode);CHKERRQ(ierr);
      ierr = PetscDeviceContextSetUp(dsubTmp[i]);CHKERRQ(ierr);
      ierr = PetscDeviceContextWaitForContext(dsubTmp[i],dctx);CHKERRQ(ierr);
       /* register the child with its parent */
      dctx->childIDs[i] = dsubTmp[i]->id;
      --n;
    }
    ++i;
  }
  /* pass the children back to caller */
  *dsub = dsubTmp;
  PetscFunctionReturn(0);
}

/* Reconverge n edges into dctx, need not be it children! */
PetscErrorCode PetscDeviceContextMerge(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext *dsub)
{
  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dsub,3);
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts merged %D < 0",n);
  for (PetscInt i = 0; i < n; ++i) {
    PetscErrorCode ierr;

    PetscCheckValidSameStreamType(dctx,1,dsub[i],3);
    ierr = PetscDeviceContextWaitForContext(dctx,dsub[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Simply cleanup the edges, does not merge */
PetscErrorCode PetscDeviceContextRestore(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub)
{
  PetscInt       i = 0,j = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dsub,3);
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts merged %D < 0",n);
  if (PetscUnlikelyDebug(n > dctx->numChildren)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to restore %D children to a parent context that only has %D children, likely trying to restore to wrong parent",n,dctx->numChildren);
  /* update child count while it's still fresh in memory */
  dctx->numChildren -= n;
  while (i != dctx->maxNumChildren) {
    if (dctx->childIDs[i] && (dctx->childIDs[i] == (*dsub)[j]->id)) {
      /* child is one of ours, can destroy it */
      PetscCheckValidSameStreamType(dctx,1,(*dsub)[j],3);
      ierr = PetscDeviceContextDestroy((*dsub)+j);CHKERRQ(ierr);
      /* reset the child slot */
      dctx->childIDs[i] = 0;
      if (++j == n) break;
    }
    ++i;
  }
  /* gone through the loop but did not find every child, if this triggers (or well, doesn't) on perf-builds we leak the remaining contexts memory */
  if (PetscUnlikelyDebug(j != n)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"%D DeviceContexts still remain after restore, this may be because you are trying to restore to the wrong parent context, or the device contexts are not in the same order as they were checkout out in.",n-j);
  ierr = PetscFree(*dsub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Synchronize the host on this contexts stream */
PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  /* if it isn't setup there is nothing to sync on */
  if (PetscUnlikelyDebug(!dctx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscDeviceContextSetup() first");
  ierr = (*dctx->ops->join)(dctx);CHKERRQ(ierr);
  dctx->idle = PETSC_TRUE;
  PetscFunctionReturn(0);
}
