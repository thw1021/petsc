/*
 This file should contain all "core" ops that every petsc impls is expected to provide a function for, i.e. every
 function in _IMOps
 */
#include <petsc/private/imimpl.h>
#include <petscim.h>

PetscErrorCode IMCreate(MPI_Comm comm, IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(m,3);
  ierr = IMInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*m,IM_CLASSID,"IM","Mapping","IM",comm,IMDestroy,IMView);CHKERRQ(ierr);
  ierr = IMResetBase_Private(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy(IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*m) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*m,IM_CLASSID,1);
  if ((*m)->ops->destroy) {
    ierr = (*(*m)->ops->destroy)(m);CHKERRQ(ierr);
  }
  ierr = IMResetBase_Private(m);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetType(IM m, IMType type)
{
  PetscErrorCode (*create)(IM);
  PetscBool      sametype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) m, type, &sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr = IMRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(IMList, type, &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PetscObjectComm((PetscObject) m),PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown IM type: %s", type);

  if (m->ops->destroy) {
    ierr = (*m->ops->destroy)(&m);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(m->ops, sizeof(*m->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) m, type);CHKERRQ(ierr);
  ierr = (*create)(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetType(IM m, IMType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = IMRegisterAll();CHKERRQ(ierr);
  PetscValidType(m,1);
  *type = ((PetscObject) m)->type_name;
  PetscFunctionReturn(0);
}

PetscErrorCode IMView(IM m, PetscViewer vwr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (m->ops->view) {
    ierr = (*m->ops->view)(m,vwr);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectView((PetscObject) m, vwr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetUp(IM m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  if (m->setup) PetscFunctionReturn(0);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->kstorage)) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Map must be set as contiguous or non contiguous before setup");
  }
  if (m->ops->setup) {
    PetscErrorCode ierr;
    ierr = (*m->ops->setup)(m);CHKERRQ(ierr);
  }
  m->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetFromOptions(IM m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (m->ops->setfromoptions) {
    PetscErrorCode ierr;
    ierr = (*m->ops->setfromoptions)(m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetKeyState(IM m, IMState *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidPointer(state,2);
  *state = m->kstorage;
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetKeysContiguous(IM m, PetscInt keyStart, PetscInt keyEnd)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setup) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on already setup map");
    if (m->kstorage == IM_DISCONTIG) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Mapping is already of type discontiguous");
    if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"KeyEnd %D < KeyStart %D contiguous keys must be increasing", keyEnd, keyStart);
  }
  ierr = PetscFree(m->contig);CHKERRQ(ierr);
  ierr = PetscNewLog(m, &(m->contig));CHKERRQ(ierr);
  m->kstorage = IM_CONTIG;
  m->contig->keyStart = keyStart;
  m->contig->keyEnd = keyEnd;
  m->nKeys[IM_LOCAL] = keyEnd-keyStart;
  ierr = MPIU_Allreduce(&m->nKeys[IM_LOCAL],&(m->nKeys[IM_GLOBAL]),1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetKeysContiguous(IM m, PetscInt *keyStart, PetscInt *keyEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage == IM_DISCONTIG) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping is not contiguous");
    if (m->kstorage == IM_INVALID) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping has no valid keys");
  }
  if (keyStart) *keyStart = m->contig->keyStart;
  if (keyEnd) *keyEnd = m->contig->keyEnd;
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetKeysDiscontiguous(IM m, PetscInt n, const PetscInt keys[], PetscCopyMode mode)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,3);
  ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setup) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on already setup map");
    if (m->kstorage == IM_CONTIG) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Mapping is already of type contiguous");
    if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of keys %D < 0", n);
  }
  if (m->discontig) {
    if (m->discontig->alloced) {
      ierr = PetscFree(m->discontig->keys);CHKERRQ(ierr);
      ierr = PetscFree(m->discontig->keyIndexGlobal);CHKERRQ(ierr);
      ierr = PetscFree(m->discontig);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"CONTIG FREED WITH OWN POINTER");
    }
  }
  ierr = PetscNewLog(m, &(m->discontig));CHKERRQ(ierr);
  switch (mode) {
  case PETSC_COPY_VALUES:
    ierr = PetscMalloc1(n, &(m->discontig));CHKERRQ(ierr);
    ierr = PetscArraycpy(m->discontig->keys, keys, n);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)m,n*sizeof(PetscInt));CHKERRQ(ierr);
    m->discontig->alloced = PETSC_TRUE;
    break;
  case PETSC_OWN_POINTER:
    m->discontig->keys = (PetscInt *)keys;
    ierr = PetscLogObjectMemory((PetscObject)m,n*sizeof(PetscInt));CHKERRQ(ierr);
    m->discontig->alloced = PETSC_TRUE;
    break;
  case PETSC_USE_POINTER:
    m->discontig->keys = (PetscInt *)keys;
    m->discontig->alloced = PETSC_FALSE;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscCopyMode");
    break;
  }
  m->nKeys[IM_LOCAL] = n;
  ierr = MPIU_Allreduce(&m->nKeys[IM_LOCAL],&(m->nKeys[IM_GLOBAL]),1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  m->kstorage = IM_DISCONTIG;
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetKeysDiscontiguous(IM m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage == IM_CONTIG) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping is of type contiguous");
    if (m->kstorage == IM_INVALID) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping has no valid keys");
  }
  *keys = m->discontig->keys;
  PetscFunctionReturn(0);
}

PetscErrorCode IMRestoreKeysDiscontiguous(IM m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage == IM_CONTIG) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping is of type contiguous");
    if (*keys != m->discontig->keys) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from IMGetKeysDiscontiguous()");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IMSort(IM m, IMOpMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  PetscValidLogicalCollectiveEnum(m,mode,2);
  ierr = (*m->ops->sort)(m,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSorted(IM m, IMOpMode mode, PetscBool *sorted)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(m,mode,2);
  PetscValidBoolPointer(sorted,3);
  *sorted = m->sorted[mode];
  PetscFunctionReturn(0);
}
