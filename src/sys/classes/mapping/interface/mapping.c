/*
 This file should contain all "core" ops that every petsc impls is expected to provide a function for, i.e. every
 function in _PetscMappingOps
 */
#include <petsc/private/mappingimpl.h>
#include <petscmapping.h>

PETSC_STATIC_INLINE PetscErrorCode PetscMappingClear_Base(PetscMapping *m)
{
  PetscFunctionBegin;
  (*m)->map = NULL;
  (*m)->permutation = NULL;
  (*m)->nKeysLocal = PETSC_DEFAULT;
  (*m)->nKeysGlobal = PETSC_DEFAULT;
  (*m)->sorted = PETSC_FALSE;
  if ((*m)->kstorage) {
    PetscErrorCode ierr;
    if ((*m)->kstorage == IM_CONTIG) {ierr = PetscFree((*m)->contig);CHKERRQ(ierr);}
    else {ierr = PetscFree((*m)->discontig);CHKERRQ(ierr);}
  }
  (*m)->kstorage = IM_INVALID;
  (*m)->contig = NULL;
  (*m)->discontig = NULL;
  (*m)->data = NULL;
  (*m)->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingCreate(MPI_Comm comm, PetscMapping *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(m,2);
  ierr = PetscMappingInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*m,PETSC_MAPPING_CLASSID,"PetscMapping","Mapping","IS",comm,PetscMappingDestroy,PetscMappingView);CHKERRQ(ierr);
  ierr = PetscMappingClear_Base(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingDestroy(PetscMapping *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*m) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*m,PETSC_MAPPING_CLASSID,1);
  if ((*m)->ops->destroy) {
    ierr = (*(*m)->ops->destroy)(m);CHKERRQ(ierr);
  }
  ierr = PetscMappingClear_Base(m);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetType(PetscMapping m, PetscMappingType type)
{
  PetscErrorCode (*create)(PetscMapping);
  PetscBool      sametype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) m, type, &sametype);CHKERRQ(ierr);
  if (sametype) PetscFunctionReturn(0);

  ierr = PetscMappingRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(PetscMappingList, type, &create);CHKERRQ(ierr);
  if (!create) SETERRQ1(PetscObjectComm((PetscObject) m),PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscMapping type: %s", type);

  if (m->ops->destroy) {
    ierr = (*m->ops->destroy)(&m);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(m->ops, sizeof(*m->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) m, type);CHKERRQ(ierr);
  ierr = (*create)(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingGetType(PetscMapping m, PetscMappingType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = PetscMappingRegisterAll();CHKERRQ(ierr);
  *type = ((PetscObject) m)->type_name;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingView(PetscMapping m, PetscViewer vwr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (m->ops->view) {
    ierr = (*m->ops->view)(m,vwr);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectView((PetscObject) m, vwr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetUp(PetscMapping m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (m->setup) PetscFunctionReturn(0);
  if (!(m->kstorage)) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Map must be set as contiguous or non contiguous before setup");
  if (m->ops->setup) {
    PetscErrorCode ierr;
    ierr = (*m->ops->setup)(m);CHKERRQ(ierr);
  }
  m->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetFromOptions(PetscMapping m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (m->ops->setfromoptions) {
    PetscErrorCode ierr;
    ierr = (*m->ops->setfromoptions)(m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingGetKeyState(PetscMapping m, PetscMappingState *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  PetscValidPointer(state,2);
  *state = m->kstorage;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetKeysContiguous(PetscMapping m, PetscInt keyStart, PetscInt keyEnd)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setup) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on already setup map");
    if (m->kstorage == IM_DISCONTIG) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Mapping is already of type discontiguous");
    if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"KeyEnd %D < KeyStart %D contiguous keys must be increasing", keyEnd, keyStart);
  }
  ierr = PetscFree(m->contig);CHKERRQ(ierr);
  ierr = PetscNewLog(m, &(m->contig));CHKERRQ(ierr);
  m->contig->keyStart = keyStart;
  m->contig->keyEnd = keyEnd;
  m->nKeysLocal = keyEnd-keyStart;
  ierr = MPIU_Allreduce(&m->nKeysLocal,&(m->nKeysGlobal),1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  m->kstorage = IM_CONTIG;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingGetKeysContiguous(PetscMapping m, PetscInt *keyStart, PetscInt *keyEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage == IM_DISCONTIG) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping is not contiguous");
    if (m->kstorage == IM_INVALID) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping has no valid keys");
  }
  if (keyStart) *keyStart = m->contig->keyStart;
  if (keyEnd) *keyEnd = m->contig->keyEnd;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetKeysDiscontiguous(PetscMapping m, PetscInt n, const PetscInt keys[], PetscCopyMode mode)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  PetscValidIntPointer(keys,3);
  ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setup) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on already setup map");
    if (m->kstorage == IM_CONTIG) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Mapping is already of type contiguous");
    if (n < 0) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of keys %D < 0", n);
  }
  if (m->discontig->alloced) {
    ierr = PetscFree(m->discontig->keys);CHKERRQ(ierr);
    ierr = PetscFree(m->discontig->keyIndexGlobal);CHKERRQ(ierr);
  }
  ierr = PetscFree(m->discontig);CHKERRQ(ierr);
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
  m->nKeysLocal = n;
  ierr = MPIU_Allreduce(&m->nKeysLocal,&(m->nKeysGlobal),1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  m->kstorage = IM_DISCONTIG;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingGetKeysDiscontiguous(PetscMapping m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage == IM_CONTIG) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping is of type contiguous");
    if (m->kstorage == IM_INVALID) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping has no valid keys");
  }
  *keys = m->discontig->keys;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingRestoreKeysDiscontiguous(PetscMapping m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage == IM_CONTIG) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Mapping is of type contiguous");
    if (*keys != m->discontig->keys) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from PetscMappingGetKeysDiscontiguous()");
  }
  PetscFunctionReturn(0);
}
