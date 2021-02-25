#include <petsc/private/deviceimpl.h>

PETSC_INTERN PetscErrorCode PetscStreamCreate_CUDA(PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamCreate_HIP(PetscStream);
PETSC_INTERN PetscErrorCode PetscEventCreate_CUDA(PetscEvent);
PETSC_INTERN PetscErrorCode PetscEventCreate_HIP(PetscEvent);
PETSC_INTERN PetscErrorCode PetscStreamScalarCreate_CUDA(PetscStreamScalar);
PETSC_INTERN PetscErrorCode PetscStreamScalarCreate_HIP(PetscStreamScalar);
PETSC_INTERN PetscErrorCode PetscStreamGraphCreate_CUDA(PetscStreamGraph);

PetscFunctionList PetscStreamList              = NULL;
PetscFunctionList PetscEventList               = NULL;
PetscFunctionList PetscStreamScalarList        = NULL;
PetscFunctionList PetscStreamGraphList         = NULL;
PetscBool         PetscStreamRegisterAllCalled = PETSC_FALSE;

const char *PetscStreamTypes[] = {"INVALID","CUDA","HIP",NULL};

/*@C
  PetscStreamSetType - Builds a PetscStream for a particular stream implementation

  Not Collective

  Input Parameters:
+ strm - The PetscStream object
- type - The PetscStream type

  Notes:
  See "petsc/include/petscdevice.h" for available stream types

  Level: intermediate

.seealso: PetscStreamCreate(), PetscStreamGetType()
@*/
PetscErrorCode PetscStreamSetType(PetscStream strm, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscStream);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(type == PETSC_STREAM_INVALID)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscStream to type %s",PetscStreamTypes[type]);
  if (strm->type == type) PetscFunctionReturn(0);
  if (PetscUnlikelyDebug(strm->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change type on already setup PetscStream");
  ierr = PetscFunctionListFind(PetscStreamList, PetscStreamTypes[type], &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStream type: %d", type);
  if (strm->ops->destroy) {ierr = (*strm->ops->destroy)(strm);CHKERRQ(ierr);}
  ierr = PetscMemzero(strm->ops, sizeof(struct _StreamOps));CHKERRQ(ierr);
  ierr = (*create)(strm);CHKERRQ(ierr);
  strm->type = type;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGetType - Gets the typename of a PetscStream

  Not Collective

  Input Parameter:
. strm - The PetscStream object

  Output Parameter:
. type - The PetscStream type

  Level: intermediate

.seealso: PetscStreamCreate(), PetscStreamSetType()
@*/
PetscErrorCode PetscStreamGetType(PetscStream strm, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(strm,1);
  PetscValidPointer(type,2);
  *type = strm->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventSetType - Builds a PetscEvent for a particular stream implementation

  Not Collective

  Input Parameters:
+ event - The PetscEvent object
- type - The PetscStream type

  Notes:
  See "petsc/include/petscdevice.h" for available stream types

  Level: intermediate

.seealso: PetscEventCreate(), PetscEventGetType()
@*/
PetscErrorCode PetscEventSetType(PetscEvent event, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscEvent);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(type == PETSC_STREAM_INVALID)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscEvent to type %s",PetscStreamTypes[type]);
  if (event->type == type) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscEventList, PetscStreamTypes[type], &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscEvent type: %d", type);
  if (event->ops->destroy) {ierr = (*event->ops->destroy)(event);CHKERRQ(ierr);}
  ierr = PetscMemzero(event->ops, sizeof(struct _EventOps));CHKERRQ(ierr);
  ierr = (*create)(event);CHKERRQ(ierr);
  event->type = type;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventGetType - Gets the typename of a PetscEvent

  Not Collective

  Input Parameter:
. event - The PetscEvent object

  Output Parameter:
. type - The PetscStream type

  Level: intermediate

.seealso: PetscEventCreate(), PetscEventSetType()
@*/
PetscErrorCode PetscEventGetType(PetscEvent event, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(event,1);
  PetscValidPointer(type,2);
  *type = event->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarSetType - Builds a PetscStreamScalar for a particular stream implementation

  Not Collective

  Input Parameters:
+ pscal - The PetscStreamScalar object
- type - The PetscStream type

  Notes:
  See "petsc/include/petscdevice.h" for available stream types

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamScalarGetType()
@*/
PetscErrorCode PetscStreamScalarSetType(PetscStreamScalar pscal, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscStreamScalar);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(type == PETSC_STREAM_INVALID)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscEvent to type %s",PetscStreamTypes[type]);
  if (pscal->type == type) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscStreamScalarList, PetscStreamTypes[type], &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStreamScalar type: %d", type);
  if (pscal->ops->destroy) {ierr = (*pscal->ops->destroy)(pscal);CHKERRQ(ierr);}
  ierr = PetscMemzero(pscal->ops, sizeof(struct _ScalOps));CHKERRQ(ierr);
  ierr = (*create)(pscal);CHKERRQ(ierr);
  pscal->type = type;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamScalarGetType - Gets the typename of a PetscStreamScalar

  Not Collective

  Input Parameter:
. pscal - The PetscStreamScalar object

  Output Parameter:
. type - The PetscStream type

  Level: intermediate

.seealso: PetscStreamScalarCreate(), PetscStreamScalarSetType()
@*/
PetscErrorCode PetscStreamScalarGetType(PetscStreamScalar pscal, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(pscal,1);
  PetscValidPointer(type,2);
  *type = pscal->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphSetType - Builds a PetscStreamGraph for a particular stream implementation

  Not Collective

  Input Parameters:
+ sgraph - The PetscStreamGraph object
- type - The PetscStream type

  Notes:
  See "petsc/include/petscdevice.h" for available stream types

  Level: intermediate

.seealso: PetscStreamGraphCreate(), PetscStreamGraphGetType()
@*/
PetscErrorCode PetscStreamGraphSetType(PetscStreamGraph sgraph, PetscStreamType type)
{
  PetscErrorCode (*create)(PetscStreamGraph);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(type == PETSC_STREAM_INVALID)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set PetscStreamGraph to type %s",PetscStreamTypes[type]);
  if (sgraph->type == type) PetscFunctionReturn(0);
  ierr = PetscFunctionListFind(PetscStreamGraphList, PetscStreamTypes[type], &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscStreamGraph type: %d", type);
  if (sgraph->ops->destroy) {ierr = (*sgraph->ops->destroy)(sgraph);CHKERRQ(ierr);}
  ierr = PetscMemzero(sgraph->ops, sizeof(struct _GraphOps));CHKERRQ(ierr);
  ierr = (*create)(sgraph);CHKERRQ(ierr);
  sgraph->type = type;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphGetType - Gets the typename of a PetscStreamGraph

  Not Collective

  Input Parameter:
. sgraph - The PetscStreamGraph object

  Output Parameter:
. type - The PetscStream type

  Level: intermediate

.seealso: PetscStreamGraphCreate(), PetscStreamGraphSetType()
@*/
PetscErrorCode PetscStreamGraphGetType(PetscStreamGraph sgraph, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidStreamType(sgraph,1);
  PetscValidPointer(type,2);
  *type = sgraph->type;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamRegisterAll - Registers all of the stream components in the PetscStream package.

  Not Collective

  Level: advanced

.seealso:  PetscStreamCreate(), PetscEventCreate(), PetscStreamScalarCreate()
@*/
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
  ierr = PetscFunctionListAdd(&PetscStreamGraphList, PetscStreamTypes[PETSC_STREAM_CUDA], PetscStreamGraphCreate_CUDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
