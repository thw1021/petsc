#include <petsc/private/deviceimpl.h>

/*@
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
  *pscal = s;
  PetscFunctionReturn(0);
}

/*@
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

/*@
  PetscStreamScalarSetUp - Sets up internal data structures for use

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
- pstream - The PetscStream object to enqueue the setup operation on

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamCreate(), PetscStreamScalarSetValue()
@*/
PetscErrorCode PetscStreamScalarSetUp(PetscStreamScalar pscal, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscal,1,pstream,4);
  if (pscal->setup) PetscFunctionReturn(0);
  ierr = PetscEventCreate(&pscal->event);CHKERRQ(ierr);
  ierr = PetscEventSetType(pscal->event, pscal->type);CHKERRQ(ierr);
  ierr = PetscEventSetUp(pscal->event);CHKERRQ(ierr);
  ierr = (*pscal->ops->setup)(pscal, pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarUpdateCache_Internal(pscal);CHKERRQ(ierr);
  pscal->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  PetscStreamScalarSetValue - Set the host and device value of a PetscStreamScalar

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to the value. This may be a host or device pointer
. mtype - The memory type of val, either host or device pointer
- pstream - The PetscStream object to enqueue the operation on

  Notes:
  This routine is asynchronous to the host, so the PetscStreamScalar will only represent the value being set once the
  host to device memory copies complete on the attached PetscStream. Normal stream memory semantics apply.

  Level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamCreate(), PetscStreamGetHostWrite()
@*/
PetscErrorCode PetscStreamScalarSetValue(PetscStreamScalar pscal, const PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscal,1,pstream,4);
  if (PetscMemTypeHost(mtype)) {
    if (val) PetscValidScalarPointer(val,2);
  }
  ierr = (*pscal->ops->setvalue)(pscal, val, mtype, pstream);CHKERRQ(ierr);
  if (PetscMemTypeHost(mtype)) {ierr = PetscStreamScalarUpdateCache_Internal(pscal);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscStreamScalarGetHostRead - Get the host pointer containing the up to date value of a PetscStreamScalar

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to hold the host pointer. This must be host accessible
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. val - pointer containing an up to date host pointer

  Notes:
  If the device pointer is more up to date this routine will cause a blocking stream synchronization, so it is advised
  to delay calling this routine until it is absolutely necessary.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType(), PetscStreamCreate(), PetscStreamGetHostWrite(),
  PetscStreamScalarGetDeviceRead()
@*/
PetscErrorCode PetscStreamScalarGetHostRead(PetscStreamScalar pscal, const PetscScalar **val, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(val,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->gethost)(pscal, (PetscScalar**) val, PETSC_TRUE, pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarUpdateCache_Internal(pscal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscStreamScalarGetHostWrite - Get the host pointer of a PetscStreamScalar

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to hold the host pointer. This must be host accessible
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. val - pointer containing the host pointer

  Notes:
  As opposed to PetscStreamScalarGetHostRead(), this routine performs no synchronizations.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamRestoreHostWrite(),
  PetscStreamScalarGetDeviceWrite(), PetscStreamScalarGetHostRead()
@*/
PetscErrorCode PetscStreamScalarGetHostWrite(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(val,2);
  PetscValidScalarPointer(*val,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->gethost)(pscal, val, PETSC_FALSE, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscStreamScalarRestoreHostWrite - Restores and commits the changed host pointer for a PetscStreamScalar

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to holding the host pointer. This must be host accessible
- pstream - The PetscStream object to enqueue the operation on

  Notes:
  This routine assumes the user has changed the value of the pointer, and therefore immediately copies the value to the device.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamGetHostWrite(),
  PetscStreamScalarGetDeviceWrite(), PetscStreamScalarGetHostRead()
@*/
PetscErrorCode PetscStreamScalarRestoreHostWrite(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(val,2);
  PetscValidScalarPointer(*val,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  if (PetscUnlikelyDebug(*val != pscal->host)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with the same pointer retrieved from PetscStreamScalarGetHostWrite()\n");
  }
  ierr = (*pscal->ops->restorehost)(pscal, val, pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarUpdateCache_Internal(pscal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscStreamScalarGetDeviceRead - Get the device pointer of a PetscStreamScalar

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to hold the device pointer. This must be device accessible
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. val - pointer containing the device pointer

  Notes:
  If the host pointer is more up to date this routine will update the device pointer on the attached stream. However,
  unlike PetscStreamScalarGetHostRead() this routine will not synchronize on the stream as it is assumed that the
  returned pointer will be used on the same stream. If the user intends to use the device pointer in subsequent calls on
  a different stream the user should either synchronize on the stream used for this routine, or have the other stream
  wait on an event recorded by the attached stream.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamGetHostRead(), PetscStreamScalarGetDeviceWrite(),
  PetscStreamWaitEvent(), PetscStreamRecordEvent()
@*/
PetscErrorCode PetscStreamScalarGetDeviceRead(PetscStreamScalar pscal, const PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->getdevice)(pscal, (PetscScalar **)ptr, PETSC_TRUE, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscStreamScalarGetDeviceWrite - Get the device pointer of a PetscStreamScalar

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to hold the device pointer. This must be device accessible
- pstream - The PetscStream object to enqueue the operation on

  Output Parameter:
. val - pointer containing the device pointer

  Notes:
  If the host pointer is more up to date this routine will update the device pointer on the attached stream. However,
  unlike PetscStreamScalarGetHostRead() this routine will not synchronize on the stream as it is assumed that the
  returned pointer will be used on the same stream. If the user intends to use the device pointer in subsequent calls on
  a different stream the user should either synchronize on the stream used for this routine, or have the other stream
  wait on an event recorded by the attached stream.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamGetHostRead(), PetscStreamScalarGetDeviceRead(),
  PetscStreamWaitEvent(), PetscStreamRecordEvent()
@*/
PetscErrorCode PetscStreamScalarGetDeviceWrite(PetscStreamScalar pscal, PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->getdevice)(pscal, ptr, PETSC_FALSE, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscStreamScalarRestoreDeviceWrite - Restores and commits changed device pointer for a PetscStreamScalar

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
. val - A pointer to holding the device pointer. This must be device accessible
- pstream - The PetscStream object to enqueue the operation on

  Notes:
  This routine assumes the user has changed the value of the pointer, but as opposed to
  PetscStreamScalarRestoreHostWrite() it does not copy the value back to the host preferring instead to keep it on device.

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamCreate(), PetscStreamGetDeviceWrite(),
  PetscStreamScalarGetHostWrite(), PetscStreamScalarGetDeviceRead()
@*/
PetscErrorCode PetscStreamScalarRestoreDeviceWrite(PetscStreamScalar pscal, PetscScalar **ptr, PetscStream pstream)
{
  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  if (PetscUnlikelyDebug(*ptr != pscal->device)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with the same pointer retrieved from PetscStreamScalarGetDeviceWrite()\n");
  }
  if (pscal->ops->restoredevice) {
    PetscErrorCode ierr;

    ierr = (*pscal->ops->restoredevice)(pscal, ptr, pstream);CHKERRQ(ierr);
  } else {pscal->omask = PETSC_OFFLOAD_GPU;}
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
