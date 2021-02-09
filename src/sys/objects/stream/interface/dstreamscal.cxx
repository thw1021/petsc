#include "dstreamscal.hpp"

PetscErrorCode PetscStreamScalarCreate(PetscStreamScalar *pscal)
{
  PetscStreamScalar s;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidPointer(pscal,1);
  ierr = PetscStreamRegisterAll();CHKERRQ(ierr);
  *pscal = NULL;
  ierr = PetscNew(&s);CHKERRQ(ierr);
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
  ierr = PetscFree(*pscal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarSetup(PetscStreamScalar pscal, PetscScalar *val, PetscMemType mtype, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscal,1,pstream,4);
  if (PetscMemTypeHost(mtype)) {
    if (val) PetscValidScalarPointer(val,2);
  }
  if (pscal->setup) PetscFunctionReturn(0);
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
  ierr = (*pscal->ops->gethost)(pscal, (PetscScalar**) val, PETSC_TRUE, pstream);CHKERRQ(ierr);
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
  ierr = (*pscal->ops->gethost)(pscal, val, PETSC_FALSE, pstream);CHKERRQ(ierr);
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
  ierr = ((PetscDeviceScalar<PetscScalar> *)(pscal->dscal)).restoreDevice(ptr, pstream);CHKERRQ(ierr);
  pscal->cacheValid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarAdd(PetscStreamScalar res, PetscStreamScalar left, PetscStreamScalar right, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(res,1,pstream,4);
  PetscCheckValidSameStreamType(left,2,pstream,4);
  PetscCheckValidSameStreamType(right,3,pstream,4);
  if (res == left) {
    *(static_cast<PetscDeviceContainer<PetscScalar>*>(res->dscal)) += *(static_cast<PetscDeviceContainer<PetscScalar>*>(right->dscal));
  } else if (res == right) {
    *(static_cast<PetscDeviceContainer<PetscScalar>*>(res->dscal)) += *(static_cast<PetscDeviceContainer<PetscScalar>*>(left->dscal));
  } else {
    *(static_cast<PetscDeviceContainer<PetscScalar>*>(res->dscal)) = *(static_cast<PetscDeviceContainer<PetscScalar>*>(left->dscal))+*(static_cast<PetscDeviceContainer<PetscScalar>*>(right->dscal));
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

PetscErrorCode PetscStreamScalarHostOp(PetscStreamScalar pscal, void (*hostfunc)(void*), void *ctx, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscal,1,pstream,4);
  ierr = (*pscal->op->hostop)(pscal, hostfunc, ctx, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
