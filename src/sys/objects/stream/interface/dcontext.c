#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

static PetscInt PetscDeviceContextID = 0;

PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *dctx)
{
  PetscDeviceContext dc;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  ierr = PetscDeviceInitializePackage();CHKERRQ(ierr);
  *dctx = NULL;
  ierr = PetscNew(&dc);CHKERRQ(ierr);
  dc->id   = PetscDeviceContextID++;
  dc->mode = PETSC_STREAM_DEFAULT_BLOCKING;
  *dctx = dc;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*dctx) PetscFunctionReturn(0);
  PetscValidPointer(dctx,1);
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
  ierr = PetscStreamCreate(&dctx->stream);CHKERRQ(ierr);
  ierr = PetscStreamSetType(dctx->stream,dctx->type);CHKERRQ(ierr);
  ierr = PetscStreamSetMode(dctx->stream,dctx->mode);CHKERRQ(ierr);
  ierr = PetscEventCreate(&dctx->event);CHKERRQ(ierr);
  ierr = PetscEventSetType(dctx->event,dctx->type);CHKERRQ(ierr);
  ierr = PetscEventSetUp(dctx->event);CHKERRQ(ierr);
  ierr = (*dctx->ops->setup)(dctx);CHKERRQ(ierr);
  dctx->setup = PETSC_TRUE;
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
      ierr = PetscStreamWaitForStream(dsubTmp[i]->stream,dctx->stream);CHKERRQ(ierr);
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
    ierr = PetscStreamWaitForStream(dctx->stream,(dsub[i])->stream);CHKERRQ(ierr);
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
  ierr = PetscStreamSynchronize(dctx->stream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
