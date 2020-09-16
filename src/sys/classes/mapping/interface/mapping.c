/*
 This file should contain all "core" ops that every petsc impls is expected to provide a function for, i.e. every
 function in _PetscMappingOps
 */
#include <petsc/private/mappingimpl.h>
#include <petscmapping.h>

PETSC_STATIC_INLINE PetscErrorCode PetscMappingClear_Base(PetscMapping *m)
{
  PetscFunctionBegin;
  (*m)->maps = NULL;
  (*m)->keys = NULL;
  (*m)->cidx = NULL;
  (*m)->dof = NULL;
  (*m)->nidx = -1;
  (*m)->nblade = -1;
  (*m)->valid = NONE_VALID;
  (*m)->iallocated = PETSC_FALSE;
  (*m)->mallocated = PETSC_FALSE;
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
  ierr = PetscMappingClear_Base(m);CHKERRQ(ierr);
  if ((*m)->ops->destroy) {
    ierr = (*(*m)->ops->destroy)(m);CHKERRQ(ierr);
  }
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

/*
PetscErrorCode PetscMappingDestroy(PetscMapping *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*m) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*m,PETSC_MAPPING_CLASSID,1);
  if ((*m)->mallocated) {
    PetscInt i, nblade = (*m)->nblade;

    for (i = 0; i < nblade; ++i) {ierr = PetscMappingDestroy(&((*m)->maps[i]));CHKERRQ(ierr);}
  }
  if ((*m)->iallocated) {
    ierr = PetscFree((*m)->cidx);CHKERRQ(ierr);
    ierr = PetscFree((*m)->dof);CHKERRQ(ierr);
  }
  ierr = PetscFree((*m)->keys);CHKERRQ(ierr);
  ierr = PetscMappingClear(m);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetKeys(PetscMapping m, PetscInt n, const PetscInt keys[], PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of keys %D < 0",n);
  PetscValidIntPointer(keys,3);
  switch(mode) {
  case PETSC_COPY_VALUES:
    if (m->kallocated) {ierr = PetscFree(m->keys);CHKERRQ(ierr);}
    ierr = PetscMalloc1(n, &(m->keys));CHKERRQ(ierr);
    ierr = PetscArraycpy(m->keys, keys, n);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject) m, n*sizeof(PetscInt));CHKERRQ(ierr);
    m->kallocated = PETSC_TRUE;
    break;
  case PETSC_OWN_POINTER:
    if (m->kallocated) {ierr = PetscFree(m->keys);CHKERRQ(ierr);}
    m->keys = (PetscInt *)keys;
    ierr = PetscLogObjectMemory((PetscObject) m, n*sizeof(PetscInt));CHKERRQ(ierr);
    m->kallocated = PETSC_TRUE;
    break;
  case PETSC_USE_POINTER:
    m->keys = (PetscInt *)keys;
    m->kallocated = PETSC_FALSE;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PetscCopyMode %D invalid",(PetscInt)mode);
    break;
  }
  m->nblade = n;
  m->valid = KEY_VALID;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetIndices(PetscMapping m, PetscInt n, const PetscInt dof[], PetscInt ni, const PetscInt indices[], PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length %D < 0",n);
  PetscValidIntPointer(dof,3);
  PetscValidIntPointer(indices,5);
  switch(mode) {
  case PETSC_COPY_VALUES:
    if (m->iallocated) {
      ierr = PetscFree(m->cidx);CHKERRQ(ierr);
      ierr = PetscFree(m->dof);CHKERRQ(ierr);
    }
    ierr = PetscMalloc2(n, &(m->dof), ni, &(m->cidx));CHKERRQ(ierr);
    ierr = PetscArraycpy(m->dof, dof, n);CHKERRQ(ierr);
    ierr = PetscArraycpy(m->cidx, indices, ni);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject) m, (n+ni)*sizeof(PetscInt));CHKERRQ(ierr);
    m->iallocated = PETSC_TRUE;
    break;
  case PETSC_OWN_POINTER:
    if (m->iallocated) {
      ierr = PetscFree(m->cidx);CHKERRQ(ierr);
      ierr = PetscFree(m->dof);CHKERRQ(ierr);
    }
    m->cidx = (PetscInt *)indices;
    m->dof = (PetscInt *)dof;
    ierr = PetscLogObjectMemory((PetscObject) m, (n+ni)*sizeof(PetscInt));CHKERRQ(ierr);
    m->iallocated = PETSC_TRUE;
    break;
  case PETSC_USE_POINTER:
    m->cidx = (PetscInt *)indices;
    m->dof = (PetscInt *)dof;
    m->iallocated = PETSC_FALSE;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PetscCopyMode %D invalid",(PetscInt)mode);
    break;
  }
  m->nblade = n;
  m->nidx   = ni;
  m->valid  = INDICES_VALID;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetMaps(PetscMapping m, PetscInt n, const PetscMapping maps[], PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of maps %D < 0",n);
  PetscValidPointer(maps,3);
  PetscValidHeaderSpecific(*maps,PETSC_MAPPING_CLASSID,3);
  switch(mode) {
  case PETSC_COPY_VALUES:
    if (m->mallocated) {ierr = PetscFree(m->maps);CHKERRQ(ierr);}
    ierr = PetscMalloc1(n, &(m->maps));CHKERRQ(ierr);
    ierr = PetscArraycpy(m->maps, maps, n);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject) m, n*sizeof(PetscMapping));CHKERRQ(ierr);
    m->mallocated = PETSC_TRUE;
    break;
  case PETSC_OWN_POINTER:
    if (m->mallocated) {ierr = PetscFree(m->maps);CHKERRQ(ierr);}
    m->maps = (PetscMapping *)maps;
    ierr = PetscLogObjectMemory((PetscObject) m, n*sizeof(PetscMapping));CHKERRQ(ierr);
    m->mallocated = PETSC_TRUE;
    break;
  case PETSC_USE_POINTER:
    m->maps = (PetscMapping *)maps;
    m->mallocated = PETSC_FALSE;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PetscCopyMode %D invalid",(PetscInt)mode);
    break;
  }
  m->nblade = n;
  m->valid = MAPS_VALID;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingValidate(PetscMapping m)
{
  PetscInt       i, nblade, offset = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  nblade = m->nblade;
  switch(m->valid) {
  case ALL_VALID:
    PetscFunctionReturn(0);
    break;
  case KEY_VALID:
    PetscFunctionReturn(0);
    break;
  case NONE_VALID:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscMapping leaf map not set up");CHKERRQ(ierr);
    break;
  case MAPS_VALID:
  {
    PetscInt nidx = 0;
 // maps are more up to date, assume entire recursive structure is ok too
// so we read the level 1 keys from every local map and populate keys and dof
    if (m->iallocated) {
      ierr = PetscFree(m->cidx);CHKERRQ(ierr);
      ierr = PetscFree(m->dof);CHKERRQ(ierr);
    }
    for (i = 0; i < nblade; ++i) {nidx += m->maps[i]->nblade;}
    ierr = PetscMalloc2(nidx, &(m->cidx), nblade, &(m->dof));CHKERRQ(ierr);
    for (i = 0; i < nblade; ++i) {
      ierr = PetscArraycpy(m->cidx+offset, m->maps[i]->keys, m->maps[i]->nblade);CHKERRQ(ierr);
      m->dof[i] = m->maps[i]->nblade;
      offset += m->maps[i]->nblade;
    }
    m->nidx = nidx;
    break;
  }
  case INDICES_VALID:
  {
    // cidx is more up to date, clear entire recursive map structure
    // repopulate from cidx and dof
    MPI_Comm     comm;
    PetscMapping *newm;
    PetscInt     offset = 0;

    ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
    ierr = PetscMalloc1(nblade, &newm);CHKERRQ(ierr);
 // For now clear the recursive map structure, TODO handle remapping maybe?
    for (i = 0; i < nblade; ++i) {
      ierr = PetscMappingDestroy(&(m->maps[i]));CHKERRQ(ierr);
      ierr = PetscMappingCreate(comm, &newm[i]);CHKERRQ(ierr);
      ierr = PetscMappingSetKeys(newm[i], m->dof[i], &(m->cidx[offset]), PETSC_COPY_VALUES);CHKERRQ(ierr);
      offset += m->dof[i];
    }
    ierr = PetscMappingSetMaps(m, nblade, newm, PETSC_OWN_POINTER);CHKERRQ(ierr);
    break;
  }
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid PetscMappingState %D",(PetscInt)m->valid);
    break;
  }
  m->valid = ALL_VALID;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingGetSize(PetscMapping m, PetscInt *nidx, PetscInt *nblade)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (nidx) {
    PetscValidIntPointer(nidx,2);
    *nidx = m->nidx;
  }
  if (nblade) {
    PetscValidIntPointer(nblade,3);
    *nblade = m->nblade;
  }
  PetscFunctionReturn(0);
}
*/
