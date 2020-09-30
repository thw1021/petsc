/*
 This file should contain all "core" ops that every petsc impls is expected to provide a function for, i.e. every
 function in _IMOps
 */
#include <petsc/private/imimpl.h>
#include <petscim.h>

const char* const IMStates[] = {"invalid", "contiguous", "discontiguous", "IMState", "IM_", NULL};

PetscErrorCode IMCreate(MPI_Comm comm, IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(m,2);
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
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidHeaderSpecific(vwr,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject) m, vwr);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) vwr, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscInt    i, nloc = m->nKeys[IM_LOCAL], nglob = m->nKeys[IM_GLOBAL];
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) m), &rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "key storage: %s\n", IMStates[m->kstorage]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "global n keys: %D\n", nglob);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(vwr);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(vwr, "[%d] local n keys: %D\n", rank, nloc);CHKERRQ(ierr);
    if (m->kstorage == IM_CONTIGUOUS) {
      ierr = PetscViewerASCIISynchronizedPrintf(vwr, "[%d] [%D, %D) \n", rank, m->contig->keyStart, m->contig->keyEnd);CHKERRQ(ierr);
    } else {
      for (i = 0; i < nloc; ++i) {
        ierr = PetscViewerASCIISynchronizedPrintf(vwr, "[%d] %D: %D\n", rank, i, m->discontig->keys[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(vwr);CHKERRQ(ierr);
    if (m->ops->view) {
      ierr = (*m->ops->view)(m,vwr);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopSynchronized(vwr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetUp(IM m)
{

  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  if (m->setup) PetscFunctionReturn(0);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->kstorage)) SETERRQ(PetscObjectComm((PetscObject)m),PETSC_ERR_ARG_WRONGSTATE,"Map must be set as contiguous or non contiguous before setup");
  }
  if (m->ops->setup) {
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

/* JUST SET NUMBER OF KEYS */
/* NOT COLLECTIVE */
PetscErrorCode IMGetKeyState(IM m, IMState *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidPointer(state,2);
  *state = m->kstorage;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMGetNumKeys(IM m, IMOpMode mode, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(n,3);
  *n = m->nKeys[mode];
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetNumKeys(IM m, IMOpMode mode, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  m->nKeys[mode] = n;
  PetscFunctionReturn(0);
}

/* SET/GET KEYS DIRECTLY */
/* NOT COLLECTIVE, keyStart = keyEnd = PETSC_DETERMINE */
PetscErrorCode IMSetKeysContiguous(IM m, PetscInt keyStart, PetscInt keyEnd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  ierr = IMCheckKeyState_Private(m, IM_CONTIGUOUS);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"KeyEnd %D < KeyStart %D contiguous keys must be increasing", keyEnd, keyStart);
    if (m->nKeys[IM_LOCAL] && ((keyEnd-keyStart) != m->nKeys[IM_LOCAL])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of keys previously set %D != %D keyrange given",m->nKeys[IM_LOCAL],keyEnd-keyStart);
  }
  if (!(m->contig)) {
    ierr = PetscNewLog(m, &(m->contig));CHKERRQ(ierr);
  }
  m->kstorage = IM_CONTIGUOUS;
  m->contig->keyStart = keyStart;
  m->contig->keyEnd = keyEnd;
  m->contig->keyStartGlobal = PETSC_DETERMINE;
  m->nKeys[IM_LOCAL] = keyEnd-keyStart ? keyEnd-keyStart : PETSC_DETERMINE;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMGetKeysContiguous(IM m, PetscInt *keyStart, PetscInt *keyEnd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  ierr = IMCheckKeyState_Private(m, IM_CONTIGUOUS);CHKERRQ(ierr);
  if (keyStart) *keyStart = m->contig->keyStart;
  if (keyEnd) *keyEnd = m->contig->keyEnd;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMSetKeysDiscontiguous(IM m, PetscInt n, const PetscInt keys[], PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,3);
  ierr = IMCheckKeyState_Private(m, IM_DISCONTIGUOUS);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of keys %D < 0", n);
    if (m->nKeys[IM_LOCAL] && (n != m->nKeys[IM_LOCAL])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of keys previously set %D != %D keyrange given",m->nKeys[IM_LOCAL],n);
  }
  if (m->discontig) {
    if (m->discontig->alloced) {
      ierr = PetscFree(m->discontig->keys);CHKERRQ(ierr);
      ierr = PetscFree(m->discontig->keyIndexGlobal);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscNewLog(m, &(m->discontig));CHKERRQ(ierr);
  }
  switch (mode) {
  case PETSC_COPY_VALUES:
    ierr = PetscMalloc1(n, &(m->discontig->keys));CHKERRQ(ierr);
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
  m->kstorage = IM_DISCONTIGUOUS;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMGetKeysDiscontiguous(IM m, const PetscInt *keys[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  ierr = IMCheckKeyState_Private(m, IM_DISCONTIGUOUS);CHKERRQ(ierr);
  *keys = m->discontig->keys;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMRestoreKeysDiscontiguous(IM m, const PetscInt *keys[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  ierr = IMCheckKeyState_Private(m, IM_DISCONTIGUOUS);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (*keys != m->discontig->keys) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from IMGetKeysDiscontiguous()");
  }
  PetscFunctionReturn(0);
}

/* UTIL FUNCTIONS */
/* POSSIBLY COLLECTIVE */
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
