#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/
#include "objpool.hpp"

/* Define the allocator */
struct PetscDeviceContextAllocator : Petsc::Allocator<PetscDeviceContext>
{
  static PetscInt PetscDeviceContextID;

  PETSC_NODISCARD PetscErrorCode create(PetscDeviceContext *dctx) PETSC_NOEXCEPT
  {
    PetscDeviceContext dc;
    PetscErrorCode     ierr;

    PetscFunctionBegin;
    *dctx          = PETSC_NULLPTR;
    ierr           = PetscNew(&dc);CHKERRQ(ierr);
    dc->id         = PetscDeviceContextID++;
    dc->idle       = PETSC_TRUE;
    dc->streamType = PETSC_STREAM_DEFAULT_BLOCKING;
    *dctx          = dc;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode destroy(PetscDeviceContext &dctx) const PETSC_NOEXCEPT
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (PetscUnlikelyDebug(dctx->numChildren)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Device context still has %D un-restored children, must call PetscDeviceContextRestore() on all children before destroying",dctx->numChildren);
    ierr = (*dctx->ops->destroy)(dctx);CHKERRQ(ierr);
    ierr = PetscFree(dctx->type);CHKERRQ(ierr);
    ierr = PetscFree(dctx->childIDs);CHKERRQ(ierr);
    ierr = PetscFree(dctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode reset(PetscDeviceContext &dctx) const PETSC_NOEXCEPT
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    /* don't deallocate the child array, rather just zero it out */
    ierr = PetscArrayzero(dctx->childIDs,dctx->maxNumChildren);CHKERRQ(ierr);
    dctx->setup       = PETSC_FALSE;
    dctx->numChildren = 0;
    dctx->idle        = PETSC_TRUE;
    dctx->streamType  = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscFunctionReturn(0);
  }
};
PetscInt PetscDeviceContextAllocator::PetscDeviceContextID = 0;

static Petsc::objectPool<PetscDeviceContext,PetscDeviceContextAllocator> contextPool;

namespace Petsc {

/* finalizer also needs its specialization */
template <>
PetscErrorCode objectPool<PetscDeviceContext,PetscDeviceContextAllocator>::finalize(void) PETSC_NOEXCEPT
{
#if PetscDefined(HAVE_CXX_DIALECT_CXX11)
  auto         alloc = contextPool.getAllocator();
#else
  allocator_t& alloc = contextPool.getAllocator();
#endif

  PetscFunctionBegin;
  try {
    while (!contextPool._stack.empty()) {
      PetscErrorCode ierr;

      ierr = alloc.destroy(contextPool._stack.top());CHKERRQ(ierr);
      contextPool._stack.pop();
    }
  } catch (std::exception const &ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from std::stack: %s",ex.what());
  }
  /* reset the ID counter, the first PetscDeviceContext created (i.e. the root) should
     always expect to have ID 0 */
  allocator_t::PetscDeviceContextID = 0;
  contextPool._registered = PETSC_FALSE;
  PetscFunctionReturn(0);
}

} /* namespace Petsc */

/*@C
  PetscDeviceContextCreate - Creates a PetscDeviceContext

  Not Collective, Asynchronous

  Ouput Paramemters:
. dctx - The PetscDeviceContext

  Notes:
  Unlike almost every other PETSc class it is advised that most users use
  PetscDeviceContextDuplicate() rather than this routine to create new contexts. Contexts
  of different types are incompatible with one another; using
  PetscDeviceContextDuplicate() ensures compatible types.

  Level: beginner

.seealso: PetscDeviceContextDuplicate(), PetscDeviceContextSetType(),
PetscDeviceContextSetStreamType(), PetscDeviceContextSetUp(), PetscDeviceContextDestroy(),
PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  ierr = PetscDeviceInitializePackage();CHKERRQ(ierr);
  ierr = contextPool.get(*dctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDestroy - Frees a PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameters:
. dctx - The PetscDeviceContext

  Notes:
  No implicit synchronization occurs due to this routine, all resources are released completely asynchronously
  w.r.t. the host. If one needs to guarantee access to the data produced on this contexts stream one should perform the
  appropriate synchronization before calling this routine.

  Developer Notes:
  The context is never actually "destroyed", only returned to an ever growing pool of
  contexts. There are currently no safeguards on the size of the pool, this should perhaps
  be implemented.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(), PetscDeviceContextSetUp(), PetscDeviceContextSynchronize()
@*/
PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*dctx) PetscFunctionReturn(0);
  (*dctx)->setup = PETSC_FALSE;
#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
  /* use move assignment whenever possible */
  ierr = contextPool.reclaim(std::move(*dctx));CHKERRQ(ierr);
#else
  ierr = contextPool.reclaim(*dctx);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetStreamType - Set the implementation type of the underlying stream for a PetscDeviceContext

  Not Collective, Asynchronous

  Input Paramaters:
+ dctx - The PetscDeviceContext
- type - The PetscStreamType

  Notes:
  See PetscStreamType in include/petscdevicetypes.h for more information on the available
  types and their interactions. If the PetscDeviceContext was previously set up and stream
  type was changed, you must call PetscDeviceContextSetUp() again after this routine.

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(),
PetscDeviceContextGetStreamType(), PetscDeviceContextSetUp(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext dctx, PetscStreamType type)
{
  PetscFunctionBegin;
  if (PetscUnlikelyDebug(type >= PETSC_STREAM_MAX_TYPE) || PetscUnlikelyDebug(type < 0)) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscStreamType %d is invalid, out of range of [0,%d)",type,PETSC_STREAM_MAX_TYPE);
  }
  /* only need to do complex swapping if the object has already been setup */
  if (dctx->setup && (dctx->streamType != type)) {
    PetscErrorCode ierr;
    ierr = (*dctx->ops->changestreamtype)(dctx,type);CHKERRQ(ierr);
    dctx->setup = PETSC_FALSE;
  }
  dctx->streamType = type;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetStreamType - Get the implementation type of the underlying stream for a PetscDeviceContext

  Not Collective, Asynchronous

  Input Paramater:
. dctx - The PetscDeviceContext

  Output Parameter:
. type - The PetscStreamType

  Notes:
  See PetscStreamType in include/petscdevicetypes.h for more information on the available types and their interactions

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(),
PetscDeviceContextSetStreamType(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext dctx, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  PetscValidPointer(type,2);
  *type = dctx->streamType;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetUp - Prepares a PetscDeviceContext for use

  Not Collective, Asynchronous

  Intput Parameter:
. dctx - The PetscDeviceContext

  Developer Notes:
  This routine is usually the stage where a PetscDeviceContext acquires device-side data structures such as streams,
  events, and (possibly) handles.

  Level: beginner

.seealso: PetscDeviceContextTypes, PetscDeviceContextCreate(),
PetscDeviceContextSetType(), PetscDeviceContextDestroy(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext dctx)
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
. dctx - The PetscDeviceContext to duplicate

  Output Paramter:
. strmdup - The duplicated PetscDeviceContext

  Notes:
  This is a shorthand method for creating a PetscDeviceContext with the exact same
  settings as another. Note however that the duplicated PetscDeviceContext does not "share"
  any of the underlying data with the original, (including its current stream-state) they
  are completely separate objects.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetType(), PetscDeviceContextSetStreamType()
@*/
PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext dctx, PetscDeviceContext *dctxdup)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dctxdup,2);
  ierr = PetscDeviceContextCreate(dctxdup);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetType(*dctxdup,dctx->type);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetStreamType(*dctxdup,dctx->streamType);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetUp(*dctxdup);CHKERRQ(ierr);
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
  This routine only refers a singular context and does NOT take any of its children into account. That is, if dctx is
  idle but has dependents who do have work, this routine still returns PETSC_TRUE.

  Results of PetscDeviceContextQueryIdle() are cached on return, allowing this function to be called repeatedly in an
  efficient manner. When debug mode is enabled this cache is verified on every call to
  this routine, but is blindly believed when debugging is disabled.

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextWaitForContext(), PetscDeviceContextFork()
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

.seealso: PetscDeviceContextCreate(), PetscDeviceContextQueryIdle(), PetscDeviceContextJoin()
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

#if !defined(PETSC_HAVE_CXX_DIALECT_CXX11)
#include <sstream>

template<typename T>
std::string PetscToString(const T &value) {
  std::ostringstream os;
  os<<value;
  return os.str();
}
#endif

/*@C
  PetscDeviceContextFork - Create a set of dependent child contexts from a parent context

  Not Collective, Asynchronous

  Input Parameters:
+ dctx - The parent PetscDeviceContext
- n    - The number of children to create

  Output Parameter:
. dsub - The created child context(s)

  Notes:
  This routine creates n edges of a DAG from a source node which are causally dependent on the source node, meaning
  that work queued on child contexts will not start until the parent context finishes its work. This accounts for work
  queued on the parent up until calling this function, any subsequent work enqueued on the parent has no effect on the children.

  Any children created with this routine have their lifetimes bounded by the parent. That is, the parent context expects
  to free all of it's children (and ONLY its children) before itself is freed.

  DAG representation:
.vb
  time ->

  -> dctx \----> dctx ------>
           \---> dsub[0] --->
            \--> ... ------->
             \-> dsub[n-1] ->
.ve

  Level: intermediate

.seealso: PetscDeviceContextJoin(), PetscDeviceContextSynchronize(), PetscDeviceContextQueryIdle()
@*/
PetscErrorCode PetscDeviceContextFork(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub)
{
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  const PetscInt      nBefore = n;
  static std::string  idList;
#endif
  PetscDeviceContext *dsubTmp = PETSC_NULLPTR;
  PetscInt            i = 0;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  PetscValidPointer(dsub,3);
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts requested %D < 0",n);
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  idList.reserve(4*n);
#endif
  /* update child totals */
  dctx->numChildren += n;
  /* now to find out if we have room */
  if (dctx->numChildren > dctx->maxNumChildren) {
    /* no room, either from having too many kids or not having any */
    if (dctx->childIDs) {
      /* have existing children, must reallocate them */
      ierr = PetscRealloc(dctx->numChildren*sizeof(*dctx->childIDs),&dctx->childIDs);CHKERRQ(ierr);
      /* clear the extra memory since realloc doesn't do it for us */
      ierr = PetscArrayzero((dctx->childIDs)+(dctx->maxNumChildren),(dctx->numChildren)-(dctx->maxNumChildren));CHKERRQ(ierr);
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
      ierr = PetscDeviceContextDuplicate(dctx,dsubTmp+i);CHKERRQ(ierr);
      ierr = PetscDeviceContextWaitForContext(dsubTmp[i],dctx);CHKERRQ(ierr);
      /* register the child with its parent */
      dctx->childIDs[i] = dsubTmp[i]->id;
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
      idList += std::to_string(dsubTmp[i]->id);
#else
      idList += PetscToString(dsubTmp[i]->id);
#endif
      if (n != 1) idList += ", ";
#endif
      --n;
    }
    ++i;
  }
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  ierr = PetscInfo3(NULL,"Forked %D children from parent %D with IDs: %s\n",nBefore,dctx->id,idList.c_str());CHKERRQ(ierr);
  /* resets the size but doesn't deallocate the memory */
  idList.clear();
#endif
  /* pass the children back to caller */
  *dsub = dsubTmp;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextJoin() - Converge a set of child contexts

  Not Collective, Asynchronous

  Input Parameters:
+ dctx         - A PetscDeviceContext to converge on
. n            - The number of sub contexts to converge
. joinMode     - The type of join to perform
- dsub         - The sub contexts to converge

  Notes:
  If PetscDeviceContextFork() creates n edges from a source node which all depend on the
  source node, then this routine is the exact mirror. That is, it creates a node
  (represented in dctx) which recieves n edges (and optionally destroys them) which is
  dependent on the completion of all incoming edges.

  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_DESTROY all contexts in dsub will be destroyed
  by this routine. Thus all sub contexts must have been created with the dctx passed to
  this routine.

  if joinMode is PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC dctx waits for all sub contexts but the
  sub contexts do not wait for one another afterwards.

  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_SYNC all sub contexts will additionally
  wait on dctx after converging. This has the effect of "synchronizing" the outgoing
  edges.

  DAG representations:
  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_DESTROY
.vb
  time ->

  -> dctx ---------/- dctx ->
  -> dsub[0] -----/
  ->  ... -------/
  -> dsub[n-1] -/
.ve
  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC
.vb
  time ->

  -> dctx ---------/- dctx ->
  -> dsub[0] -----/--------->
  ->  ... -------/---------->
  -> dsub[n-1] -/----------->
.ve
  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_SYNC
.vb
  time ->

  -> dctx ---------/- dctx -\----> dctx ------>
  -> dsub[0] -----/          \---> dsub[0] --->
  ->  ... -------/            \--> ... ------->
  -> dsub[n-1] -/              \-> dsub[n-1] ->
.ve

  Level: intermediate

.seealso: PetscDeviceContextFork(), PetscDeviceContextSynchronize(), PetscDeviceContextJoinMode
@*/
PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext dctx, PetscInt n, PetscDeviceContextJoinMode joinMode, PetscDeviceContext **dsub)
{
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  static std::string idList;
#endif
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* valid stream type of dctx is checked in the wait-for loop */
  PetscValidPointer(dsub,4);
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts merged %D < 0",n);
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  idList.reserve(4*n);
#endif
  /* first dctx waits on all the incoming edges */
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckValidSameStreamType(dctx,1,(*dsub)[i],4);
    ierr = PetscDeviceContextWaitForContext(dctx,(*dsub)[i]);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
#if defined(PETSC_HAVE_CXX_DIALECT_CXX11)
    idList += std::to_string((*dsub)[i]->id);
#else
    idList += PetscToString((*dsub)[i]->id);
#endif
    if (i+1 < n) idList += ", ";
#endif
  }

  /* now we handle the aftermath */
  switch (joinMode) {
  case PETSC_DEVICE_CONTEXT_JOIN_DESTROY:
    {
      PetscInt j = 0;

      if (PetscUnlikelyDebug(n > dctx->numChildren)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy %D children of a parent context that only has %D children, likely trying to restore to wrong parent",n,dctx->numChildren);
      /* update child count while it's still fresh in memory */
      dctx->numChildren -= n;
      for (PetscInt i = 0; i < dctx->maxNumChildren; ++i) {
        if (dctx->childIDs[i] && (dctx->childIDs[i] == (*dsub)[j]->id)) {
          /* child is one of ours, can destroy it */
          ierr = PetscDeviceContextDestroy((*dsub)+j);CHKERRQ(ierr);
          /* reset the child slot */
          dctx->childIDs[i] = 0;
          if (++j == n) break;
        }
      }
      /* gone through the loop but did not find every child, if this triggers (or well, doesn't) on perf-builds we leak the remaining contexts memory */
      if (PetscUnlikelyDebug(j != n)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"%D contexts still remain after destroy, this may be because you are trying to restore to the wrong parent context, or the device contexts are not in the same order as they were checked out out in.",n-j);
      ierr = PetscFree(*dsub);CHKERRQ(ierr);
    }
    break;
  case PETSC_DEVICE_CONTEXT_JOIN_SYNC:
    for (PetscInt i = 0; i < n; ++i) {
      ierr = PetscDeviceContextWaitForContext((*dsub)[i],dctx);CHKERRQ(ierr);
    }
  case PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscDeviceContextJoinMode given");
    break;
  }

#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  ierr = PetscInfo4(NULL,"Joined %D ctxs to ctx %D, mode %s with IDs: %s\n",n,dctx->id,PetscDeviceContextJoinModes[joinMode],idList.c_str());CHKERRQ(ierr);
  idList.clear();
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSynchronize() - Block the host until all work queued on or associated with a PetscDeviceContext has finished

  Not Collective, Synchronous

  Input Parameters:
. dctx - The PetscDeviceContext to synchronize

  Level: beginner

.seealso: PetscDeviceContextFork(), PetscDeviceContextJoin(), PetscDeviceContextQueryIdle()
@*/
PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  /* if it isn't setup there is nothing to sync on */
  if (dctx->setup) {ierr = (*dctx->ops->synchronize)(dctx);CHKERRQ(ierr);}
  dctx->idle = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscDeviceContext     globalContext      = PETSC_NULLPTR;
static PetscBool              globalContextSetup = PETSC_FALSE;
static PetscStreamType        defaultStreamType  = PETSC_STREAM_DEFAULT_BLOCKING;
#if PetscDefined(HAVE_HIP)
static PetscDeviceContextType defaultContextType = PETSCDEVICECONTEXTHIP;
#else
/* default to cuda if neither? maybe there should be an "invalid" version */
static PetscDeviceContextType defaultContextType = PETSCDEVICECONTEXTCUDA;
#endif

/*@C
  PetscDeviceContextSetDefaultRootContextSettings - Set the default settings for the root global context

  Input Parameters:
+ type  - The PetscDeviceContextType, NULL if not needed
- stype - The PetscStreamType, PETSC_STREAM_MAX_TYPE if not needed

  Notes:
  This is one of the few functions that one should only call before PetscInitialize(), as calling this routine after
  the root PetscDeviceContext has already been created will have no effect. It is highly unlikely that the user would
  need to call this routine however, the default values are automatically optimally chosen based on availability.

  Level: advanced

.seealso: PetscDeviceContextGetDefaultRootContextSettings(), PetscDeviceContextGetCurrentContext()
@*/
PetscErrorCode PetscDeviceContextSetDefaultRootContextSettings(PetscDeviceContextType type, PetscStreamType stype)
{
  if (PetscUnlikelyDebug(globalContextSetup)) {
    PetscBool petscInit;

    PetscInitialized(&petscInit);
    if (petscInit) {
      PetscFunctionBegin; /* PetscFunctionBegin so SETERRQ knows where we are */
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Root PetscDeviceContext has already been setup and created, setting default has no effect");
    } else {
      /* Since this may be called before PetscInitialize() we can't use PetscError() */
      printf("Root PetscDeviceContext has already been setup and created, setting default has no effect\n");
    }
  } else {
    if (type)                           defaultContextType = type;
    if (stype != PETSC_STREAM_MAX_TYPE) defaultStreamType  = stype;
  }
  return 0;
}

/*@C
  PetscDeviceContextGetDefaultRootContextSettings - Get the current default root context settings

  Output Parameters:
+ type  - The default PetscDeviceContextType, NULL if not needed
- stype - The default PetscStreamType, NULL if not needed

  Note:
  Can be called prior to PetscInitialize(), but does no error checking

  Level: advanced

.seealso: PetscDeviceContextSetDefaultRootContextSettings(), PetscDeviceContextGetCurrentContext()
@*/
PetscErrorCode PetscDeviceContextGetDefaultRootContextSettings(PetscDeviceContextType *type, PetscStreamType *stype)
{
#if PetscDefined(USE_DEBUG)
  PetscBool petscInit;

  PetscInitialized(&petscInit);
  if (petscInit) {
    PetscFunctionBegin;
    if (type)   *type = defaultContextType;
    if (stype) *stype = defaultStreamType;
    PetscFunctionReturn(0);
  }
#endif
  if (type)   *type = defaultContextType;
  if (stype) *stype = defaultStreamType;
  return 0;
}

/* automatically registered to PetscFinalize() when first context is instantiated, do not
   call */
PETSC_STATIC_INLINE PetscErrorCode PetscDeviceContextDestroyGlobalContext_Internal(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextSynchronize(globalContext);CHKERRQ(ierr);
  ierr = PetscDeviceContextDestroy(&globalContext);CHKERRQ(ierr);
  /* reset everything to defaults */
  defaultStreamType  = PETSC_STREAM_DEFAULT_BLOCKING;
#if PetscDefined(HAVE_HIP)
  defaultContextType = PETSCDEVICECONTEXTHIP;
#else
  defaultContextType = PETSCDEVICECONTEXTCUDA;
#endif
  globalContextSetup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* creates and initializes the root context in PetscInitialize() but does not call
   SetUp() as the user may wish to change types after PetscInitialize() */
PetscErrorCode PetscDeviceContextInitializeRootContext_Internal(MPI_Comm comm, const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo(NULL,"Initializing root PetscDeviceContext\n");CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscDeviceContextDestroyGlobalContext_Internal);CHKERRQ(ierr);
  ierr = PetscDeviceContextCreate(&globalContext);CHKERRQ(ierr);
  if (PetscUnlikelyDebug(globalContext->id != 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"The root current PetscDeviceContext should have id = 0, however it has id = %D",globalContext->id);
  ierr = PetscDeviceContextSetStreamType(globalContext,defaultStreamType);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetType(globalContext,defaultContextType);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetFromOptions(comm,prefix,globalContext);CHKERRQ(ierr);
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

  Developer Notes:
  This routine creates the "root" context the first time it is called, registering its
  destructor to PetscFinalize(). The root context is synchronized before being destroyed.

  Level: beginner

.seealso: PetscDeviceContextSetCurrentContext(), PetscDeviceContextFork(),
PetscDeviceContextJoin(), PetscDeviceContextCreate()
@*/
PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  if (PetscUnlikely(!globalContextSetup)) {
    PetscErrorCode ierr;
    ierr = PetscDeviceContextSetUp(globalContext);CHKERRQ(ierr);
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

  Level: beginner

.seealso: PetscDeviceContextGetCurrentContext(), PetscDeviceContextFork(),
PetscDeviceContextJoin(), PetscDeviceContextCreate()
@*/
PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(dctx,1);
  globalContext = dctx;
  ierr = PetscInfo1(NULL,"Set global device context id %D\n",dctx->id);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
