#include <petsc/private/deviceimpl.h>

/*@C
  PetscStreamScalarCreate - Creates an empty PetscStreamScalar object. The type can then be set with PetscStreamScalarSetType().

  Not Collective

  Output Parameter:
. pscal  - The allocated PetscStream object

  Notes:
  You must set the stream type before using the PetscStreamScalar object, otherwise an error is generated on debug builds.

  Level: beginner

.seealso: PetscStreamScalarDestroy(), PetscStreamScalarSetType(), PetscStreamScalarSetUp()
@*/
PetscErrorCode PetscStreamScalarCreate(PetscStreamScalar *pscal)
{
  PetscStreamScalar s;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidPointer(pscal,1);
  ierr = PetscStreamRegisterAll();CHKERRQ(ierr);
  /* Setting to null taken from VecCreate(), why though? */
  *pscal = NULL;
  ierr = PetscNew(&s);CHKERRQ(ierr);
  s->setup = PETSC_FALSE;
  s->omask = PETSC_OFFLOAD_UNALLOCATED;
  s->type = PETSC_STREAM_INVALID;
  s->host = NULL;
  s->device = NULL;
  s->poolID = PETSC_DEFAULT;
  *pscal = s;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarDestroy - Destroys a PetscStreamScalar

  Not Collective

  Input Parameter:
. pscal - The PetscStreamScalar object

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamScalarSetUp()
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
  ierr = PetscFree(*pscal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarSetUp - Sets up internal data structures for use

  Not Collective

  Input Parameter:
. pscal - The PetscStreamScalar object

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamCreate(), PetscStreamScalarSetValue()
@*/
PetscErrorCode PetscStreamScalarSetUp(PetscStreamScalar pscal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(pscal,1);
  if (pscal->setup) PetscFunctionReturn(0);
  ierr = PetscEventCreate(&pscal->event);CHKERRQ(ierr);
  ierr = PetscEventSetType(pscal->event, pscal->type);CHKERRQ(ierr);
  ierr = PetscEventSetUp(pscal->event);CHKERRQ(ierr);
  ierr = (*pscal->ops->setup)(pscal);CHKERRQ(ierr);
  pscal->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarSetValue - Set the value of a PetscStreamScalar

  Not Collective

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

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamCreate(), PetscStreamScalarAwait()
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
  ierr = (*pscal->ops->setvalue)(pscal, val, mtype, pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarUpdateCache_Internal(pscal, val, mtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarAwait - Await completion of asynchronous operation and retrieve the results on the host.

  Not Collective

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
  ierr = (*pscal->ops->await)(pscal, val, pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarUpdateCache_Internal(pscal, val, PETSC_MEMTYPE_HOST);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarGetDeviceRead - Get the device pointer of a PetscStreamScalar

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to hold the device pointer. This must be device accessible
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. val - pointer containing the device pointer

  Notes:
  See PetscStreamScalarGetDeviceWrite() for this routines stream synchronization behavior.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarAwait(), PetscStreamScalarGetDeviceWrite(), PetscStreamWaitEvent(), PetscStreamRecordEvent()
@*/
PetscErrorCode PetscStreamScalarGetDeviceRead(PetscStreamScalar pscal, const PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first");
  ierr = (*pscal->ops->getdevice)(pscal, (PetscScalar **)ptr, PETSC_TRUE, pstream);CHKERRQ(ierr);
  if (pscal->omask != PETSC_OFFLOAD_GPU) {
    ierr = PetscStreamScalarUpdateCache_Internal(pscal, pscal->host, PETSC_MEMTYPE_HOST);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarGetDeviceWrite - Get the device pointer of a PetscStreamScalar

  Not Collective

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
  ierr = (*pscal->ops->getdevice)(pscal, ptr, PETSC_FALSE, pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarUpdateCache_Internal(pscal, *ptr, PETSC_MEMTYPE_DEVICE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarRestoreDeviceWrite - Restores and commits changed device pointer for a PetscStreamScalar

  Not Collective

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
  if (pscal->ops->restoredevice) {
    ierr = (*pscal->ops->restoredevice)(pscal, ptr, pstream);CHKERRQ(ierr);
  } else {
    /*
     This double whammy protects against the possibility that ptr was checked out and used on streamA, "returned" in
     this function on streamB, and then used later with streamC
     */
    ierr = PetscStreamWaitEvent(pstream, pscal->event);CHKERRQ(ierr);
    ierr = PetscStreamRecordEvent(pstream, pscal->event);CHKERRQ(ierr);
    pscal->omask = PETSC_OFFLOAD_GPU;
  }
  ierr = PetscStreamScalarUpdateCache_Internal(pscal, *ptr, PETSC_MEMTYPE_DEVICE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarGetInfo - Determines whether a PetscStreamScalar satisfies a particular property.

  Not Collective

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
  only require computation if it __cannot__ be avoided in order to preserve the asynchronicity of the stream.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarSetInfo()
@*/
PetscErrorCode PetscStreamScalarGetInfo(PetscStreamScalar pscal, PSSCacheType ctype, PetscBool compute, PetscBool *val, PetscStream pstream)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(val,4);
  if (PetscUnlikelyDebug(ctype >= PSSCACHE_MAX)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid CacheType %D requested, larger than maximum value %D\n",ctype,PSSCACHE_MAX-1);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first");
  if (compute) {
    if (pscal->cache[ctype] == PSS_UNKNOWN) {
      PetscScalar    host;
      PetscErrorCode ierr;

      PetscCheckValidSameStreamType(pscal,1,pstream,5);
      /* Forces cache to be updated */
      ierr = PetscStreamScalarAwait(pscal,&host,pstream);CHKERRQ(ierr);
    }
  }
  *val = pscal->cache[ctype] == PSS_TRUE ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarSetInfo - Set a known information about a PetscStreamScalar

  Not Collective

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
  if (PetscUnlikelyDebug(ctype >= PSSCACHE_MAX)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid CacheType %D requested, larger than maximum value %D\n",ctype,PSSCACHE_MAX-1);
  ierr = PetscStreamScalarSetCache_Internal(pscal, ctype, val ? PSS_TRUE : PSS_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarAXTY - Computes x = alpha * x * y

  Not Collective

  Input Parameters:
+ pscalx,pscaly - The PetscStreamScalars
. alpha - The scalar
- pstream - The PetscStream on which to enqueue the operation

  Output Parameter:
. pscalx - The adjusted output PetscStreamScalar

  Notes:
  If pscaly is NULL, it is treated as 1.0, so this routine will scale pscalx by alpha. pscalx and pscaly may be the same
  object, making this routine scale the square of a value. This routine is optimized for alpha = 0.0.

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarAYDX()
@*/
PetscErrorCode PetscStreamScalarAXTY(PetscScalar alpha, PetscStreamScalar pscalx, PetscStreamScalar pscaly, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscalx,2,pstream,4);
  if (PetscUnlikelyDebug(!pscalx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 2 first");
  if (pscaly) {
    PetscCheckValidSameStreamType(pscaly,3,pstream,4);
    if (PetscUnlikelyDebug(!pscaly->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 3 first");
  }
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscStreamScalarSetValue(pscalx,NULL,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  } else {
    ierr = (*pscalx->ops->axty)(alpha, pscalx, pscaly, pstream);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarAYDX - Computes x = alpha * y / x

  Not Collective

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

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamScalarAXTY()
@*/
PetscErrorCode PetscStreamScalarAYDX(PetscScalar alpha, PetscStreamScalar pscalx, PetscStreamScalar pscaly, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscalx,2,pstream,4);
  if (PetscUnlikelyDebug(!pscalx->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 2 first");
  if (pscaly) {
    PetscCheckValidSameStreamType(pscaly,3,pstream,4);
    if (PetscUnlikelyDebug(!pscaly->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() on argument 3 first");
  }
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscStreamScalarSetValue(pscalx, NULL, PETSC_MEMTYPE_DEVICE, pstream);CHKERRQ(ierr);
  } else if (pscalx == pscaly) {
    ierr = PetscStreamScalarSetValue(pscalx, &alpha, PETSC_MEMTYPE_HOST, pstream);CHKERRQ(ierr);
  } else {
    ierr = (*pscalx->ops->aydx)(alpha, pscalx, pscaly, pstream);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarAccumulateOp(PetscStreamScalar pscalacc, PetscInt n, PetscStreamScalar pscal[], PetscStreamComputeOp epiop, PetscStreamComputeOp accop, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscalacc,1,pstream,4);
  if (PetscUnlikelyDebug(n > 7)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Can only accumuate up to 8 scalars at a time\n");
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid number of scalars %D\n",n);
  for (PetscInt i = 0; i < n; ++i) PetscCheckValidSameStreamType(pscal[i],3,pstream,4);
  if (!n) PetscFunctionReturn(0);
  ierr = (*pscalacc->ops->accumop)(pscalacc, n, pscal, epiop, accop, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
