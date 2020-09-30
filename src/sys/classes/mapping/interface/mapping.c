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

/* COLLECTIVE, EVERYONE MUST HAVE SAME TYPE OF STORAGE */
PetscErrorCode IMSetUp(IM m)
{
  PetscMPIInt    size, rank;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  PetscValidLogicalCollectiveEnum(m,m->kstorage,1);
  if (m->setup) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)m, &comm);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->kstorage)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be set as contiguous or non contiguous before setup");
  }
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(comm, &(m->nKeys[IM_LOCAL]), &(m->nKeys[IM_GLOBAL]));CHKERRQ(ierr);
  switch (m->kstorage) {
  case IM_CONTIGUOUS:
  {
    PetscInt chunk[2];

    /* Interval not explicitly set, only sizes */
    if (!(m->keySet)) m->contig->keyEnd = m->nKeys[IM_LOCAL];
    chunk[0] = m->contig->keyStart, chunk[1] = m->contig->keyEnd;
    if (!(m->contig->globalRanges)) {
      ierr = PetscMalloc1(2*size, &(m->contig->globalRanges));CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)m, 2*size*sizeof(PetscInt));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Inside setup -> contig, keystart %D keyend %D\n", chunk[0], chunk[1]);CHKERRQ(ierr);
    ierr = MPI_Allgather(chunk, 2, MPIU_INT, m->contig->globalRanges, 2, MPIU_INT, comm);CHKERRQ(ierr);
    if (!(m->keySet)) {
      PetscInt i, n = 0;
      for (i = 1; i <= size-1; ++i) {
        n += m->contig->globalRanges[2*i-1]-m->contig->globalRanges[2*i-2];
        ierr = PetscPrintf(PETSC_COMM_WORLD,"n %D\n", n);CHKERRQ(ierr);
        m->contig->globalRanges[2*i] += n;
        m->contig->globalRanges[2*i+1] += n;
      }
      m->contig->keyStart = m->contig->globalRanges[2*rank];
      m->contig->keyEnd = m->contig->globalRanges[2*rank+1];
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Printing\n");CHKERRQ(ierr);
    ierr = PetscIntView(2*size, m->contig->globalRanges, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    break;
  }
  case IM_ARRAY:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid IMState");
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

/* LOGICALLY COLLECTIVE */
PetscErrorCode IMSetKeyState(IM m, IMState state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(m,state,2);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change key state on setup map");
  }
  ierr = IMReplaceKeyStateWithNew_Private(m, state);CHKERRQ(ierr);
  m->keySet = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE, MUST BE SETUP */
PetscErrorCode IMGetNumKeys(IM m, IMOpMode mode, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(n,3);
  if (PetscDefined(USE_DEBUG)) {
    if ((mode == IM_GLOBAL) && !(m->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping must be setup to retrieve global number of keys");
  }
  *n = m->nKeys[mode];
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMSetNumKeys(IM m, PetscInt n, PetscInt N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on setup map");
  }
  m->nKeys[IM_LOCAL] = n;
  m->nKeys[IM_GLOBAL] = N;
  m->keySet = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* SET/GET KEYS DIRECTLY */
/* LOGICALLY COLLECTIVE */
PetscErrorCode IMContiguousSetKeyInterval(IM m, PetscInt keyStart, PetscInt keyEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on setup map");
    if (m->kstorage && (m->kstorage != IM_CONTIGUOUS)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping has already been set to %s, use IMConvertKeys() to change",IMStates[m->kstorage]);
    if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"KeyEnd %D < KeyStart %D contiguous keys must be increasing", keyEnd, keyStart);
  }
  if (!(m->contig)) {
    PetscErrorCode ierr;
    ierr = PetscNewLog(m, &(m->contig));CHKERRQ(ierr);
  }
  m->contig->keyStart = keyStart;
  m->contig->keyEnd = keyEnd;
  m->nKeys[IM_LOCAL] = keyEnd-keyStart;
  m->kstorage = IM_CONTIGUOUS;
  m->sorted[IM_LOCAL] = PETSC_TRUE;
  m->keySet = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMContiguousGetKeyInterval(IM m, PetscInt *keyStart, PetscInt *keyEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    const IMState state = IM_CONTIGUOUS;
    if (m->kstorage && (m->kstorage != state)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is of type %d not %s",IMStates[m->kstorage], IMStates[state]);
  }
  if (keyStart) *keyStart = m->contig->keyStart;
  if (keyEnd) *keyEnd = m->contig->keyEnd;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMArraySetKeyArray(IM m, PetscInt n, const PetscInt keys[], PetscBool issorted, PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,3);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setup) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on setup map");
    if (m->kstorage && (m->kstorage != IM_ARRAY)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping has already been set to %s, use IMConvertKeys() to change",IMStates[m->kstorage]);
    if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of keys %D < 0", n);
  }
  if (m->discontig) {
    if (m->discontig->alloced) {
      ierr = PetscFree(m->discontig->keys);CHKERRQ(ierr);
      ierr = PetscFree(m->discontig->keyIndexGlobal);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)m, -(m->nKeys[IM_LOCAL])*sizeof(PetscInt));CHKERRQ(ierr);
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
  m->kstorage = IM_ARRAY;
  m->sorted[IM_LOCAL] = issorted;
  m->keySet = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMArrayGetKeyArray(IM m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    const IMState state = IM_ARRAY;
    if (m->kstorage && (m->kstorage != state)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is not %s", IMStates[state]);
  }
  *keys = m->discontig->keys;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMArrayRestoreKeyArray(IM m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage && (m->kstorage != IM_ARRAY)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is not discontiguous");
    if (*keys != m->discontig->keys) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from IMGetKeysDiscontiguous()");
  }
  PetscFunctionReturn(0);
}

/* UTIL FUNCTIONS */
/* POSSIBLY COLLECTIVE */
PetscErrorCode IMConvertKeyState(IM m, IMState newstate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (m->kstorage == newstate) PetscFunctionReturn(0);
  if (!(m->kstorage)) {
    m->kstorage = newstate;
    PetscFunctionReturn(0);
  }
  /* not equal and not null, so old state must be discontiguous */
  if (newstate == IM_CONTIGUOUS) {
    PetscInt keyStart, keyEnd;

    if (!(m->sorted[IM_LOCAL])) {
      ierr = PetscIntSortSemiOrdered(m->nKeys[IM_LOCAL], m->discontig->keys);CHKERRQ(ierr);
      m->sorted[IM_LOCAL] = PETSC_TRUE;
    }
    keyStart = m->discontig->keys[0];
    keyEnd = m->discontig->keys[m->nKeys[IM_LOCAL]-1];
    if (PetscDefined(USE_DEBUG)) {
      if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"KeyEnd %D < KeyStart %D from discontiguous but contiguous keys must be increasing",keyEnd,keyStart);
      if (m->nKeys[IM_LOCAL] && ((keyEnd-keyStart) != m->nKeys[IM_LOCAL])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of keys previously set %D != %D keyrange given",m->nKeys[IM_LOCAL],keyEnd-keyStart);
    }
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

PetscErrorCode IMSetIsSorted(IM m, PetscBool sorted)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
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
