#include <petsc/private/deviceimpl.h>

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
  s->cacheValid = PETSC_FALSE;
  *pscal = s;
  PetscFunctionReturn(0);
}

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

PetscErrorCode PetscStreamScalarSetup(PetscStreamScalar pscal, PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscal,1,pstream,4);
  if (PetscMemTypeHost(mtype)) PetscValidScalarPointer(val,2);
  if (pscal->setup) PetscFunctionReturn(0);
  ierr = PetscEventCreate(&pscal->event);CHKERRQ(ierr);
  ierr = PetscEventSetType(pscal->event, pscal->type);CHKERRQ(ierr);
  ierr = PetscEventSetUp(pscal->event);CHKERRQ(ierr);
  ierr = (*pscal->ops->setup)(pscal, val, mtype, pstream);CHKERRQ(ierr);
  pscal->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarGetHostRead(PetscStreamScalar pscal, const PetscScalar **val, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidScalarPointer(val,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->gethost)(pscal, (PetscScalar**) val, pstream);CHKERRQ(ierr);
  /* Can update these since host can't overwrite */
  pscal->isZero = (PetscBool)(*(*val) == (PetscScalar)0.0);
  pscal->isOne  = (PetscBool)(*(*val) == (PetscScalar)1.0);
  pscal->cacheValid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarGetHostWrite(PetscStreamScalar pscal, PetscScalar **val, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(val,2);
  PetscValidScalarPointer(*val,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->gethost)(pscal, val, pstream);CHKERRQ(ierr);
  /* Note we can no longer make assumptions about value of val until it is restored */
  pscal->cacheValid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

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
  pscal->isZero = (PetscBool)(*(*val) == (PetscScalar)0.0);
  pscal->isOne  = (PetscBool)(*(*val) == (PetscScalar)1.0);
  pscal->cacheValid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarGetDeviceRead(PetscStreamScalar pscal, const PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->getdevice)(pscal, (PetscScalar **)ptr, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarGetDeviceWrite(PetscStreamScalar pscal, PetscScalar **ptr, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ptr,2);
  PetscCheckValidSameStreamType(pscal,1,pstream,3);
  if (PetscUnlikelyDebug(!pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamScalarSetUp() first\n");
  ierr = (*pscal->ops->getdevice)(pscal, ptr, pstream);CHKERRQ(ierr);
  /* Assume that value will change and that we cannot access the cache anyore */
  pscal->cacheValid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

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
  }
  pscal->omask = PETSC_OFFLOAD_GPU;
  pscal->cacheValid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarAccumulateOp(PetscStreamScalar pscalacc, PetscInt n, PetscStreamScal pscal[], PetscStreamOp epiop, PetscStreamOp accop, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscal,1,pstream,4);
  if (PetscUnlikelyDebug(n > 7)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Can only accumuate up to 8 scalars at a time\n");
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid number of scalars %D\n",n);
  for (PetscInt i = 0; i < n; ++i) PetscCheckValidSameStreamType(pscal[i],3,pstream,4);
  if (!n) PetscFunctionReturn(0);
  ierr = (*pscal->ops->accumop)(pscalacc, n, pscal, epiop, accop, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
