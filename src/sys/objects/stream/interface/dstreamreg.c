#include <petsc/private/deviceimpl.h>

PETSC_INTERN PetscErrorCode PetscStreamCreate_CUDA(PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamCreate_HIP(PetscStream);
PETSC_INTERN PetscErrorCode PetscEventCreate_CUDA(PetscEvent);
PETSC_INTERN PetscErrorCode PetscEventCreate_HIP(PetscEvent);
PETSC_INTERN PetscErrorCode PetscStreamScalarCreate_CUDA(PetscStreamScalar);
PETSC_INTERN PetscErrorCode PetscStreamScalarCreate_HIP(PetscStreamScalar);

PetscFunctionList PetscStreamList              = NULL;
PetscFunctionList PetscEventList               = NULL;
PetscFunctionList PetscStreamScalarList        = NULL;
PetscBool         PetscStreamRegisterAllCalled = PETSC_FALSE;

const char *PetscStreamTypes[] = {"INVALID","CUDA","HIP",NULL};

PetscErrorCode PetscStreamSetType(PetscStream strm, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscStream);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(strm->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change type on already setup PetscStream\n");
  ierr = PetscFunctionListFind(PetscStreamList, PetscStreamTypes[type], &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStream type: %d", type);
  if (strm->ops->destroy) {ierr = (*strm->ops->destroy)(strm);CHKERRQ(ierr);}
  ierr = PetscMemzero(strm->ops, sizeof(struct _StreamOps));CHKERRQ(ierr);
  ierr = (*create)(strm);CHKERRQ(ierr);
  strm->type = type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGetType(PetscStream strm, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(type,2);
  *type = strm->type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventSetType(PetscEvent event, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscEvent);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(event->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change type on already setup PetscEvent\n");
  ierr = PetscFunctionListFind(PetscEventList, PetscStreamTypes[type], &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscEvent type: %d", type);
  if (event->ops->destroy) {ierr = (*event->ops->destroy)(event);CHKERRQ(ierr);}
  ierr = PetscMemzero(event->ops, sizeof(struct _EventOps));CHKERRQ(ierr);
  ierr = (*create)(event);CHKERRQ(ierr);
  event->type = type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventGetType(PetscEvent event, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  PetscValidPointer(type,2);
  *type = event->type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarSetType(PetscStreamScalar pscal, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscStreamScalar);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(pscal->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change type on already setup PetscEvent\n");
  ierr = PetscFunctionListFind(PetscStreamScalarList, PetscStreamTypes[type], &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStreamScalar type: %d", type);
  if (pscal->ops->destroy) {ierr = (*pscal->ops->destroy)(pscal);CHKERRQ(ierr);}
  ierr = PetscMemzero(pscal->ops, sizeof(struct _ScalOps));CHKERRQ(ierr);
  ierr = (*create)(pscal);CHKERRQ(ierr);
  pscal->type = type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarGetType(PetscStreamScalar pscal, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(pscal,1);
  PetscValidPointer(type,2);
  *type = pscal->type;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscStreamRegisterAllCalled) PetscFunctionReturn(0);
  PetscStreamRegisterAllCalled = PETSC_TRUE;
  ierr = PetscFunctionListAdd(&PetscStreamList, PetscStreamTypes[PETSC_STREAM_CUDA], PetscStreamCreate_CUDA);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscStreamList, PetscStreamTypes[PETSC_STREAM_HIP], PetscStreamCreate_HIP);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscEventList, PetscStreamTypes[PETSC_STREAM_CUDA], PetscEventCreate_CUDA);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscEventList, PetscStreamTypes[PETSC_STREAM_HIP], PetscEventCreate_HIP);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscStreamScalarList, PetscStreamTypes[PETSC_STREAM_CUDA], PetscStreamScalarCreate_CUDA);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscStreamScalarList, PetscStreamTypes[PETSC_STREAM_HIP], PetscStreamScalarCreate_HIP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
