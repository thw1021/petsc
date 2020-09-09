#include <petsc/private/mappingimpl.h>
#include <petscmapping.h>

PetscClassId PETSC_MAPPING_CLASSID;

static PetscBool PetscMappingPackageInitialized = PETSC_FALSE;

PetscErrorCode PetscMappingFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscMappingPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscMappingPackageInitialized) PetscFunctionReturn(0);
  PetscMappingPackageInitialized = PETSC_TRUE;
  ierr = PetscClassIdRegister("Mapping",&PETSC_MAPPING_CLASSID);CHKERRQ(ierr);
  {
    PetscClassId classids[1];

    classids[0] = PETSC_MAPPING_CLASSID;
    ierr = PetscInfoProcessClass("mapping", 1, classids);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    PetscBool pkg;

    ierr = PetscStrInList("mapping",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSC_MAPPING_CLASSID);CHKERRQ(ierr);}
  }
  ierr = PetscRegisterFinalize(PetscMappingFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingClear(PetscMapping *m)
{
  PetscFunctionBegin;
  (*m)->maps            = NULL;
  (*m)->indices         = NULL;
  (*m)->offsets         = NULL;
  (*m)->idx2mapidx      = NULL;
  (*m)->maxNumChildMaps = 0;
  (*m)->numChildMaps    = 0;
  (*m)->size            = -1;
  (*m)->allocated       = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingCreate(MPI_Comm comm, PetscMapping *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(m,2);
  ierr = PetscMappingInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*m,PETSC_MAPPING_CLASSID,"PetscMapping","Mapping","IS",comm,PetscMappingDestroy,PetscMappingView);CHKERRQ(ierr);
  ierr = PetscMappingClear(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingDestroy(PetscMapping *m)
{
  PetscErrorCode ierr;
  PetscInt       i, nmap;

  PetscFunctionBegin;
  if (!*m) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*m,PETSC_MAPPING_CLASSID,1);
  ierr = PetscMappingGetSize(*m, &nmap);CHKERRQ(ierr);
  for (i = 0; i < nmap; ++i) {
    ierr = PetscMappingDestroy(&((*m)->maps[i]));CHKERRQ(ierr);
  }
  if ((*m)->allocated) {ierr = PetscFree((*m)->indices);CHKERRQ(ierr);}
  ierr = PetscMappingClear(m);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingView(PetscMapping m, PetscViewer vwr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  ierr = PetscObjectView((PetscObject) m, vwr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscMappingClearChildMaps(PetscMapping m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  ierr = PetscFree(m->maps);CHKERRQ(ierr);
  ierr = PetscFree(m->idx2mapidx);CHKERRQ(ierr);
  m->numChildMaps = 0;
  m->maxNumChildMaps = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingGrowChildMapsAutomatic(PetscMapping m)
{
  PetscInt       allocSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (m->numChildMaps <= m->maxNumChildMaps) PetscFunctionReturn(0);
  if (m->numChildMaps >= m->size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot grow child map array of size %D larger than index size %D",m->numChildMaps,m->size);
  if (!(m->maxNumChildMaps)) {
    allocSize = PetscMax(m->size/10, 2);
    /* Split up both because we can't realloc both */
    ierr = PetscMalloc1(allocSize, m->maps);CHKERRQ(ierr);
    ierr = PetscCalloc1(allocSize, &m->idx2mapidx);CHKERRQ(ierr);
  } else {
    allocSize = PetscMax(m->maxNumChildMaps*m->maxNumChildMaps, m->size);
    ierr = PetscRealloc(allocSize*sizeof(*(m->maps)), m->maps);CHKERRQ(ierr);
    ierr = PetscRealloc(allocSize*sizeof(*(m->idx2mapidx)), m->idx2mapidx);CHKERRQ(ierr);
  }
  m->maxNumChildMaps = allocSize;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingSetIndices(PetscMapping m, PetscInt n, const PetscInt indices[], PetscInt no, const PetscInt offsets[], PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length %D < 0",n);
  PetscValidIntPointer(indices,3);
  PetscValidIntPointer(offsets,5);
  switch(mode) {
  case PETSC_COPY_VALUES:
    if (m->allocated) {
      ierr = PetscFree(m->indices);CHKERRQ(ierr);
      ierr = PetscFree(m->offsets);CHKERRQ(ierr);
    }
    ierr = PetscMalloc2(n, &m->indices, no, &m->offsets);CHKERRQ(ierr);
    ierr = PetscArraycpy(m->indices, indices, n);CHKERRQ(ierr);
    ierr = PetscArraycpy(m->offsets, offsets, no);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject) m, n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject) m, no*sizeof(PetscInt));CHKERRQ(ierr);
    m->allocated = PETSC_TRUE;
    break;
  case PETSC_OWN_POINTER:
    if (m->allocated) {
      ierr = PetscFree(m->indices);CHKERRQ(ierr);
      ierr = PetscFree(m->offsets);CHKERRQ(ierr);
    }
    m->indices = (PetscInt *)indices;
    m->offsets = (PetscInt *)offsets;
    ierr = PetscLogObjectMemory((PetscObject) m, n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject) m, no*sizeof(PetscInt));CHKERRQ(ierr);
    m->allocated = PETSC_TRUE;
    break;
  case PETSC_USE_POINTER:
    m->indices = (PetscInt *)indices;
    m->offsets = (PetscInt *)offsets;
    m->allocated = PETSC_FALSE;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"PetscCopyMode %D invalid",(PetscInt)mode);
    break;
  }
  m->size = n;
  ierr = PetscMappingClearChildMaps(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingAddMap(PetscMapping mParent, PetscInt index, PetscMapping *mChild)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mParent,PETSC_MAPPING_CLASSID,1);
  ierr = PetscMappingGrowChildMapsAutomatic(mParent);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) mParent, &comm);CHKERRQ(ierr);
  ++(mParent->numChildMaps);
  ierr = PetscMappingCreate(comm, &(mParent->maps[mParent->numChildMaps]));CHKERRQ(ierr);
  mParent->idx2mapidx[mParent->numChildMaps] = index;
  if (mChild) *mChild = mParent->maps[mParent->numChildMaps];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingGetSize(PetscMapping m, PetscInt *size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,PETSC_MAPPING_CLASSID,1);
  PetscValidIntPointer(size,2);
  *size = m->size;
  PetscFunctionReturn(0);
}
