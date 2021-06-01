#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

const char *const PetscStreamModes[] = {"global_blocking","default_blocking","global_nonblocking","MAX_MODE","PetscStreamMode","PETSC_STREAM_",NULL};

/* initial global context will have ID = 0 */
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

/* See PetscDeviceContextMode in petscdevicetypes.h */
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
  ierr = (*dctx->ops->setup)(dctx);CHKERRQ(ierr);
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDuplicate - Duplicates a PetscDeviceContext object

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext object to duplicate

  Output Paramter:
. strmdup - The duplicated PetscDeviceContext

  Level: beginner

  Notes:
  This is a shorthand method for creating a PetscDeviceContext in the immage of another, insofar that the duplicated
  PetscDeviceContext does not share any of the underlying objects with the original.

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
  PetscDeviceContextGetBLASHandle - Retrieves the implementation specific BLAS handle

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext object

  Output Parameter:
. handle - The handle

  Notes:
  This is a borrowed reference, the user should not destroy it themselves

  Level: advanced

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(), PetscDeviceContextSetMode(), PetscDeviceContextRestoreBLASHandle()
@*/
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

/*@C
  PetscDeviceContextRestoreBLASHandle - Restores the implementation specific BLAS handle

  Not Collective, Asynchronous

  Input Parameter:
+ dctx   - The PetscDeviceContext object
- handle - The handle

  Notes:
  The restored handle must be the same handle that was checked out via PetscDeviceContextGetBLASHandle()

  Level: advanced

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(), PetscDeviceContextSetMode(), PetscDeviceContextGetBLASHandle()
@*/
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

PetscErrorCode PetscDeviceContextGetSOLVERHandle(PetscDeviceContext dctx, void *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(handle,2);
  if (PetscUnlikelyDebug(!dctx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscDeviceContext is not setup yet, must call PetscDeviceContextSetUp()");
  ierr = (*dctx->ops->getsolverhandle)(dctx,handle);CHKERRQ(ierr);
  dctx->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDeviceContextRestoreSOLVERHandle(PetscDeviceContext dctx, void *handle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(handle,2);
  ierr = (*dctx->ops->restoresolverhandle)(dctx,handle);CHKERRQ(ierr);
  /* In case the handle is checked out, sync'ed while checked out, then work queued onto stream */
  dctx->idle = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextQueryIdle - Returns whether or not a PetscDeviceContext is idle

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext object

  Output Parameter:
. idle - PETSC_TRUE if PetscDeviceContext has NO work, PETSC_FALSE if it has work

  Notes:
  This routine only refers a singular context and does NOT take any of its children into
  account. That is, if dctx is idle but has dependents who do have work, this routine
  still returns PETSC_TRUE.

  Results of PetscDeviceContextQueryIdle() are cached on return, allowing this function to be
  called repeatedly in an efficient manner.

  Level: advanced

.seealso: PetscDeviceContextCreate(), PetscDeviceContextWaitForContext()
@*/
PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext dctx, PetscBool *idle)
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
  Serializes two PetscDeviceContexts. This routine uses only the state of dctxb at the moment this routine was
  called, so any future work queued will not affect dctxa. It is safe to pass the same context to both arguments.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextQueryIdle()
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
  } else {
    ierr = (*dctxa->ops->waitforctx)(dctxa,dctxb);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextFork - Create a set of child contexts from a parent context

  Not Collective, Asynchronous

  Input Parameters:
+ dctx - The parent PetscDeviceContext
- n    - The number of children to create

  Output Parameter:
. dsub - The created child context(s)

  Notes:
  This routine effectively creates n edges of a DAG from a source node, meaning that work queued on child contexts will
  not start until the parent context finishes its work. This accounts for work queued on the parent up until calling
  this function, any subsequent work enqueued on the parent has no effect on the children.

  Any children created with this routine have their lifetimes bounded by the parent. That is, the parent context expects
  to free all of it's children (and __only__ its children) before itself is freed.

  Level: intermediate

.seealso: PetscDeviceContextJoin(), PetscDeviceContextSynchronize()
@*/
PetscErrorCode PetscDeviceContextFork(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub)
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

/*@C
  PetscDeviceContextJoin() - Converge a set of child contexts

  Not Collective, Asynchronous

  Input Parameters:
+ dctx         - A PetscDeviceContext to converge on
. destroyEdges - Whether to destroy the convergent sub contexts
. n            - The number of sub contexts to converge
- dsub         - The sub contexts to converge

  Notes:
  If PetscDeviceContextFork() creates n edges from a source node which all depend on the source node, then this routine
  is the exact mirror. That is, it creates a source node (represented in dctx) which recieves n edges (and optionally
  destroys them) and is dependent on the completion of all edges.

  If destroyEdges is PETSC_TRUE then all sub contexts must have been created with the dctx passed to this function.

  Level: intermediate

.seealso: PetscDeviceContextFork(), PetscDeviceContextSynchronize()
@*/
PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext dctx, PetscBool destroyEdges, PetscInt n, PetscDeviceContext **dsub)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dsub,4);
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts merged %D < 0",n);
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckValidSameStreamType(dctx,1,(*dsub)[i],4);
    ierr = PetscDeviceContextWaitForContext(dctx,(*dsub)[i]);CHKERRQ(ierr);
  }
  if (destroyEdges) {
    PetscInt i = 0, j = 0;

    if (PetscUnlikelyDebug(n > dctx->numChildren)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy %D children of a parent context that only has %D children, likely trying to restore to wrong parent",n,dctx->numChildren);
    /* update child count while it's still fresh in memory */
    dctx->numChildren -= n;
    while (i != dctx->maxNumChildren) {
      if (dctx->childIDs[i] && (dctx->childIDs[i] == (*dsub)[j]->id)) {
        /* child is one of ours, can destroy it */
        ierr = PetscDeviceContextDestroy((*dsub)+j);CHKERRQ(ierr);
        /* reset the child slot */
        dctx->childIDs[i] = 0;
        if (++j == n) break;
      }
      ++i;
    }
    /* gone through the loop but did not find every child, if this triggers (or well, doesn't) on perf-builds we leak the remaining contexts memory */
    if (PetscUnlikelyDebug(j != n)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"%D contexts still remain after destroy, this may be because you are trying to restore to the wrong parent context, or the device contexts are not in the same order as they were checkout out in.",n-j);
    ierr = PetscFree(*dsub);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSynchronize() - Block the host until all work queued on a PetscDeviceContext has finished

  Not Collective, Synchronous

  Input Parameters:
. dctx - The PetscDeviceContext to synchronize

  Level: beginner

.seealso: PetscDeviceContextFork(), PetscDeviceContextJoin()
@*/
PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  /* if it isn't setup there is nothing to sync on */
  if (PetscUnlikelyDebug(!dctx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscDeviceContextSetup() first");
  ierr = (*dctx->ops->synchronize)(dctx);CHKERRQ(ierr);
  dctx->idle = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscDeviceContext globalContext = NULL;
static PetscBool          globalContextSetup = PETSC_FALSE;
static PetscStreamMode    defaultStreamMode  = PETSC_STREAM_DEFAULT_BLOCKING;
#if PetscDefined(HAVE_CUDA)
static PetscDeviceContextType defaultContextType = PETSCDEVICECONTEXTCUDA;
#elif PetscDefined(HAVE_HIP)
static PetscDeviceContextType defaultContextType = PETSCDEVICECONTEXTHIP;
#else
/* default to cuda if neither? maybe there should be an "invalid" version */
static PetscDeviceContextType defaultContextType = PETSCDEVICECONTEXTCUDA;
#endif

PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyGlobalContext_Internal(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextDestroy(&globalContext);CHKERRQ(ierr);
  /* reset everything to defaults */
  defaultStreamMode  = PETSC_STREAM_DEFAULT_BLOCKING;
#if PetscDefined(HAVE_CUDA)
  defaultContextType = PETSCDEVICECONTEXTCUDA;
#elif PetscDefined(HAVE_HIP)
  defaultContextType = PETSCDEVICECONTEXTHIP;
#else
  defaultContextType = PETSCDEVICECONTEXTCUDA;
#endif
  globalContextSetup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetCurrentContext() - Get the current active PetscDeviceContext

  Not Collective, Asynchronous

  Output Parameter:
. dctx - The PetscDeviceContext

  Notes:
  The user generally should not destroy contexts retrieved with this routine unless they themselves have created
  them. There exists no protection against destroying the root context.

  Level: beginner

.seealso: PetscDeviceContextSetCurrentContext()
@*/
PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  if (!globalContextSetup) {
    PetscErrorCode ierr;

    ierr = PetscRegisterFinalize(PetscDeviceContextDestroyGlobalContext_Internal);CHKERRQ(ierr);
    ierr = PetscDeviceContextCreate(&globalContext);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetMode(globalContext,defaultStreamMode);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetType(globalContext,defaultContextType);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetup(globalContext);CHKERRQ(ierr);
    globalContextSetup = PETSC_TRUE;
  }
  *dctx = globalContext;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetCurrentContext() - Set the current active PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext

  Notes:
  The old context is not stored in any way by this routine; if one is overriding a context that they themselves do not
  control, one should take care to temporarily store it by calling PetscDeviceContextGetCurrentContext() before calling
  this routine.

.seealso: PetscDeviceContextGetCurrentContext()
@*/
PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  globalContext = dctx;
  PetscFunctionReturn(0);
}
