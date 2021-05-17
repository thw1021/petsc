#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

const char *const PSSCacheTypes[] = {"ZERO","ONE","INF","NAN","PSSCacheType","PSS_",NULL};
/*@C
  PetscStreamScalarCreate - Creates an empty PetscStreamScalar object. The type can then be set with PetscStreamScalarSetType().

  Not Collective, Synchronous only on first call

  Output Parameter:
. pscal  - The allocated PetscStream object

  Notes:
  You must set the stream type before using the PetscStreamScalar object, otherwise an error is generated on debug builds.

  Level: beginner

.seealso: PetscStreamScalarDestroy(), PetscStreamScalarSetType(), PetscStreamScalarSetUp(), PetscStreamScalarDuplicate()
@*/
PetscErrorCode PetscStreamScalarCreate(PetscStreamScalar *pscal)
{
  PetscStreamScalar s;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidPointer(pscal,1);
  ierr = PetscStreamScalarInitializePackage();CHKERRQ(ierr);
  /* Setting to null taken from VecCreate(), why though? */
  *pscal = NULL;
  ierr = PetscNew(&s);CHKERRQ(ierr);
  s->omask = PETSC_OFFLOAD_UNALLOCATED;
  s->poolID = PETSC_DEFAULT;
  *pscal = s;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarDestroy - Destroys a PetscStreamScalar

  Not Collective, Asynchronous

  Input Parameter:
. pscal - The PetscStreamScalar object

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamScalarSetUp(), PetscStreamScalarDuplicate()
@*/
PetscErrorCode PetscStreamScalarDestroy(PetscStreamScalar *pscal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pscal) PetscFunctionReturn(0);
  PetscValidPointer(pscal,1);
  PetscValidStreamType(*pscal,1);
  ierr = (*(*pscal)->ops->destroy)(*pscal);CHKERRQ(ierr);
  ierr = PetscEventDestroy(&(*pscal)->event);CHKERRQ(ierr);
  ierr = PetscFree((*pscal)->type);CHKERRQ(ierr);
  ierr = PetscFree(*pscal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarSetUp - Sets up internal data structures for use

  Not Collective, Asynchronous

  Input Parameter:
. pscal - The PetscStreamScalar object

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamCreate(), PetscStreamScalarSetValue(), PetscStreamScalarDuplicate()
@*/
PetscErrorCode PetscStreamScalarSetUp(PetscStreamScalar pscal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(pscal,1);
  if (pscal->setup) PetscFunctionReturn(0);
  ierr = PetscEventCreate(&pscal->event);CHKERRQ(ierr);
  ierr = PetscEventSetType(pscal->event,pscal->type);CHKERRQ(ierr);
  ierr = PetscEventSetUp(pscal->event);CHKERRQ(ierr);
  ierr = (*pscal->ops->setup)(pscal);CHKERRQ(ierr);
  pscal->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarDuplicate - Duplicates a PetscStreamScalar

  Not Collective, Asynchronous

  Input Parameter:
. pscalref - The PetscStreamScalar object to duplicate

  Output Parameter:
. pscalout - The duplicated PetscStreamScalar

  Notes:
  The duplicated PetscStreamScalar will be of the same type as the reference, but will not share any other feature. It
  is safe to use either independently of the other.

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamScalarSetValue()
@*/
PetscErrorCode PetscStreamScalarDuplicate(PetscStreamScalar pscalref, PetscStreamScalar *pscalout)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(pscalref,1);
  PetscValidPointer(pscalout,2);
  ierr = PetscStreamScalarCreate(pscalout);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetType(*pscalout,pscalref->type);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetUp(*pscalout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarSetValue - Set the value of a PetscStreamScalar

  Not Collective, Asynchronous

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to the value. This may be a host or device pointer. Use NULL for 0
. mtype - The memory type of val, either host or device pointer
- pstream - The PetscStream object to enqueue the operation on

  Notes:
  The user must call PetscStreamScalarSetUp() before using this routine.

  This routine is asynchronous to the host, so the PetscStreamScalar will only represent the value being set once the
  host to device memory copies complete on the attached PetscStream. Normal stream memory semantics apply.

  The device value is always updated by this routine regardless of mtype, while the host value is only updated if mtype
  is PETSC_MEMTYPE_HOST or if val is NULL.

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamCreate(), PetscStreamScalarAwait(), PetscStreamScalarDuplicate()
@*/
PetscErrorCode PetscStreamScalarSetValue(PetscStreamScalar pscal, const PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscal,1,pstream,4);
  if (PetscMemTypeHost(mtype)) {
    if (val) PetscValidScalarPointer(val,2);
  }
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first");
  ierr = (*pscal->ops->setvalue)(pscal,val,mtype,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarUpdateCache_Internal(pscal,val,mtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarAwait - Await completion of asynchronous operation and retrieve the results on the host.

  Not Collective, Synchronous

  Input Parameters:
+ pscal - The PetscStreamScalar object to await
. val - A pointer to hold the host value. This must be host accessible
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. val - pointer containing the result

  Notes:
  In order to guarantee memory coherence this routine will always call PetscStreamSynchronize(), so it is advised to
  delay calling this routine until absolutely necessary.

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamCreate(), PetscStreamScalarGetDeviceRead(), PetscStreamScalarSetValue()
@*/
PetscErrorCode PetscStreamScalarAwait(PetscStreamScalar pscal, PetscScalar *val, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(val,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->await)(pscal,val,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarUpdateCache_Internal(pscal,val,PETSC_MEMTYPE_HOST);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarGetDeviceRead - Get the device pointer of a PetscStreamScalar

  Not Collective, Asynchronous

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to hold the device pointer. This must be device accessible
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. val - pointer containing the device pointer

  Notes:
  See PetscStreamScalarGetDeviceWrite() for this routines stream synchronization behavior.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarAwait(), PetscStreamRestoreDeviceRead(), PetscStreamScalarGetDeviceWrite(), PetscStreamWaitEvent(), PetscStreamRecordEvent()
@*/
PetscErrorCode PetscStreamScalarGetDeviceRead(PetscStreamScalar pscal, const PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first");
  ierr = (*pscal->ops->getdevice)(pscal,(PetscScalar **)ptr,PETSC_TRUE,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarRestoreDeviceRead - Restore the device pointer of a PetscStreamScalar

  Not Collective, Asynchronous

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer checked out via PetscStreamGetDeviceRead()
- pstream - The PetscStream object to enqueue the operation on

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarAwait(), PetscStreamScalarGetDeviceRead(), PetscStreamWaitEvent(), PetscStreamRecordEvent()
@*/
PetscErrorCode PetscStreamScalarRestoreDeviceRead(PetscStreamScalar pscal, const PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first");
  ierr = PetscStreamRecordEvent(pstream,pscal->event);CHKERRQ(ierr);
  *ptr = NULL;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarGetDeviceWrite - Get the device pointer of a PetscStreamScalar

  Not Collective, Asynchronous

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to hold the device pointer. This must be device accessible
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. val - pointer containing the device pointer

  Notes:
  If the host pointer is more up to date this routine will update the device pointer on the attached stream. This
  routine will not synchronize on the stream; if the user intends to use the device pointer in subsequent user code the
  user should either synchronize on the stream used for this routine, or have the other stream wait on the event
  recorded on the PetscStreamScalar by this routine.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamAwait(), PetscStreamScalarGetDeviceRead(),
  PetscStreamWaitEvent(), PetscStreamRecordEvent()
@*/
PetscErrorCode PetscStreamScalarGetDeviceWrite(PetscStreamScalar pscal, PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first");
  ierr = (*pscal->ops->getdevice)(pscal,ptr,PETSC_FALSE,pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarRestoreDeviceWrite - Restores and commits changed device pointer for a PetscStreamScalar

  Not Collective, Asynchronous

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to holding the device pointer. This must be device accessible
- pstream - The PetscStream object to enqueue the operation on

  Notes:
  This routine assumes the user has changed the value of the pointer, but it does not copy the value back to the host
  preferring instead to keep it on device.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamGetDeviceWrite(), PetscStreamScalarGetDeviceRead(), PetscStreamScalarAwait()
@*/
PetscErrorCode PetscStreamScalarRestoreDeviceWrite(PetscStreamScalar pscal, PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first");
  if (PetscUnlikelyDebug(*ptr != pscal->device)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with the same pointer retrieved from PetscStreamScalarGetDeviceWrite()");
  }
  ierr = (*pscal->ops->restoredevice)(pscal,ptr,pstream);CHKERRQ(ierr);
  ierr = PetscStreamRecordEvent(pstream,pscal->event);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarGetInfo - Determines whether a PetscStreamScalar satisfies a particular property.

  Not Collective, Qualified Synchronous

  Input Parameters:
+ pscal - The PetscStreamScalar object
. ctype - The type of property
. compute - Whether the property should be computed if unknown
- pstream - The PetscStream object to enqueue the operation on if needed

  Output Parameters:
. val - Whether the property is true.

  Notes:
  A cache value of "unknown" counts as PETSC_FALSE.

  Should the compute flag be true, and the value be unknown the cache is updated by synchronizing the host value with
  the device value. If the host is out of date this results in a stream-synchronization, so the user should take care to
  only require computation if it __cannot__ be avoided in order to preserve the asynchronicity of the stream. Note that
  in debugging mode this routine __always__ checks the cache for consistency (requiring a synchronization on the
  streamscalars internal event).

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarSetInfo()
@*/
PetscErrorCode PetscStreamScalarGetInfo(PetscStreamScalar pscal, PSSCacheType ctype, PetscBool compute, PetscBool *val, PetscStream pstream)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(val,4);
  if (PetscUnlikelyDebug(ctype >= PSS_CACHE_MAX) || PetscUnlikelyDebug(ctype < 0)) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscStreamScalarCacheType %d is invalid, out of range of [0,%d)",(int)ctype,(int)PSS_CACHE_MAX);
  }
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first");
  if (compute || PetscDefined(USE_DEBUG)) {
#if PetscDefined(USE_DEBUG)
    PSSCacheBool old = pscal->cache[ctype];
#endif
    if ((pscal->cache[ctype] == PSS_UNKNOWN) || PetscDefined(USE_DEBUG)) {
      PetscScalar    host;
      PetscErrorCode ierr;

      PetscCheckValidSameStreamType(pscal,1,pstream,5);
      /* Forces cache to be updated */
      ierr = PetscStreamScalarAwait(pscal,&host,pstream);CHKERRQ(ierr);
    }
#if PetscDefined(USE_DEBUG)
    if (!compute && (old != PSS_UNKNOWN)) {
      if (PetscUnlikely(old != pscal->cache[ctype])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupted or invalid PetscStreamScalar cache: cached %d != returned %d",old,pscal->cache[ctype]);
    }
#endif
  }
  *val = pscal->cache[ctype] == PSS_TRUE ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarSetInfo - Set a known information about a PetscStreamScalar

  Not Collective, Asynchronous

  Input Parameters:
+ pscal - The PetscStreamScalar object
. ctype - The type of property
- val - The value of the property

  Possible Cache Values:
+ PSS_ZERO - The value of the PetscStreamScalar is zero
. PSS_ONE - The value of the PetscStreamScalar is one
. PSS_INF - The value of the PetscStreamScalar is INF
- PSS_NAN - The value of the PetscStreamScalar is NaN

  Notes:
  This routine is a powerful tool to hint at the state of a PetscStreamScalar after a set of operations, but no effort
  is made to check the validity of value being set. It is entirely possible to set completely bogus values using this
  routine so care must be taken to ensure it is correct.

  Many inferences are made possible if val is PETSC_TRUE (e.g. if ctype is PSS_ZERO and val is PETSC_TRUE, then all other
  cache values must be PETSC_FALSE), but the opposite does not apply. Should val be PETSC_FALSE, depending on ctype, this
  routine sets many other cache values to "unknown".

  Level: advanced

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarSetInfo()
@*/
PetscErrorCode PetscStreamScalarSetInfo(PetscStreamScalar pscal, PSSCacheType ctype, PetscBool val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(ctype >= PSS_CACHE_MAX) || PetscUnlikelyDebug(ctype < 0)) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PetscStreamScalarCacheType %d is invalid, out of range of [0,%d)",(int)ctype,(int)PSS_CACHE_MAX);
  }
  ierr = PetscStreamScalarSetCache_Internal(pscal,ctype,val ? PSS_TRUE : PSS_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarRealPart - Replace the value of the PetscStreamScalar with its real component

  Not Collective, Asynchronous

  Input Parameters:
+ pscal   - The PetscStreamScalar object to convert
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. pscal - The converted PetscStreamScalar

  Notes:
  This routine does nothing if PETSc is configured without complex support.

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamAwait(), PetscStreamScalarAXTY(), PetscStreamScalarAYDX()
@*/
PetscErrorCode PetscStreamScalarRealPart(PetscStreamScalar pscal, PetscStream pstream)
{
  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscal,1,pstream,2);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 1 first");
#if PetscDefined(USE_COMPLEX)
  {
    PetscErrorCode ierr;
    ierr = (*pscal->ops->realpart)(pscal,pstream);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarAXTY - Computes x = alpha * x * y

  Not Collective, Asynchronous

  Input Parameters:
+ pscalx,pscaly - The PetscStreamScalars
. alpha         - The scalar
- pstream       - The PetscStream on which to enqueue the operation

  Output Parameter:
. pscalx - The adjusted output PetscStreamScalar

  Notes:
  If pscaly is NULL, it is treated as 1.0, so this routine will scale pscalx by alpha. pscalx and pscaly may be the same
  object, making this routine scale the square of a value. This routine is optimized for alpha = 0.0 and alpha = 1.0
  when pscaly is NULL.

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarAYDX(), PetscStreamScalarRealPart()
@*/
PetscErrorCode PetscStreamScalarAXTY(PetscScalar alpha, PetscStreamScalar pscalx, PetscStreamScalar pscaly, PetscStream pstream)
{
  PetscBool      isYOne = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscalx,2,pstream,4);
  if (PetscUnlikelyDebug(!pscalx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 2 first");
  if (pscaly) {
    PetscCheckValidSameStreamType(pscaly,3,pstream,4);
    if (PetscUnlikelyDebug(!pscaly->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 3 first");
    ierr = PetscStreamScalarGetInfo(pscaly,PSS_ONE,PETSC_FALSE,&isYOne,pstream);CHKERRQ(ierr);
  }
  if (isYOne && (alpha == (PetscScalar)1.0)) PetscFunctionReturn(0);
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscStreamScalarSetValue(pscalx,NULL,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  } else {
    ierr = (*pscalx->ops->axty)(alpha,pscalx,pscaly,pstream);CHKERRQ(ierr);
    if (PetscIsNanScalar(alpha)) {
      ierr = PetscStreamScalarSetInfo(pscalx,PSS_NAN,PETSC_TRUE);CHKERRQ(ierr);
    } else if (PetscIsInfScalar(alpha)) {
      ierr = PetscStreamScalarSetInfo(pscalx,PSS_INF,PETSC_TRUE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarAYDX - Computes x = alpha * y / x

  Not Collective, Asynchronous

  Input Parameters:
+ pscalx,pscaly - The PetscStreamScalars
. alpha - The scalar
- pstream - The PetscStream on which to enqueue the operation

  Output Parameter:
. pscalx - The adjusted output PetscStreamScalar

  Notes:
  If pscaly is NULL, it is treated as 1.0, so this routine will scale the inverse of pscalx by alpha. pscalx and pscaly may be the same
  object, making this routine set pscalx to alpha. This routine is optimized for alpha = 0.0.

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarAXTY(), PetscStreamScalarRealPart()
@*/
PetscErrorCode PetscStreamScalarAYDX(PetscScalar alpha, PetscStreamScalar pscaly, PetscStreamScalar pscalx, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscalx,3,pstream,4);
  if (PetscUnlikelyDebug(!pscalx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 3 first");
  if (pscaly) {
    PetscCheckValidSameStreamType(pscaly,2,pstream,4);
    if (PetscUnlikelyDebug(!pscaly->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 2 first");
  }
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscStreamScalarSetValue(pscalx,NULL,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  } else if (pscalx == pscaly) {
    ierr = PetscStreamScalarSetValue(pscalx,&alpha,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  } else {
    PetscBool isOneBefore,isZero;

    ierr = PetscStreamScalarGetInfo(pscalx,PSS_ONE,PETSC_FALSE,&isOneBefore,pstream);CHKERRQ(ierr);
    ierr = (*pscalx->ops->aydx)(alpha,pscaly,pscalx,pstream);CHKERRQ(ierr);
    ierr = PetscStreamScalarGetInfo(pscalx,PSS_ZERO,PETSC_FALSE,&isZero,pstream);CHKERRQ(ierr);
    if (isOneBefore && !pscaly && (alpha == (PetscScalar)1.0)) {
      ierr = PetscStreamScalarSetInfo(pscalx,PSS_ONE,PETSC_TRUE);CHKERRQ(ierr);
    } else if (isZero) {
      /* anything/0 is NaN */
      ierr = PetscStreamScalarSetInfo(pscalx,PSS_NAN,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (PetscIsNanScalar(alpha)) {
      ierr = PetscStreamScalarSetInfo(pscalx,PSS_NAN,PETSC_TRUE);CHKERRQ(ierr);
    } else if (PetscIsInfScalar(alpha)) {
      ierr = PetscStreamScalarSetInfo(pscalx,PSS_INF,PETSC_TRUE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
