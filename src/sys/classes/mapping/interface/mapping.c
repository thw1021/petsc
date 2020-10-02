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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  if (!vwr) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)m), &vwr);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(vwr,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(m,1,vwr,2);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must setup map before viewing");
  }
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject) m, vwr);CHKERRQ(ierr);
  if (m->ops->view) {
    ierr = (*m->ops->view)(m,vwr);CHKERRQ(ierr);
  } else {
    PetscBool iascii;

    ierr = PetscObjectTypeCompare((PetscObject) vwr, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
    if (iascii) {
      PetscInt    i;
      PetscMPIInt rank, size;

      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) m), &rank);CHKERRQ(ierr);
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject) m), &size);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(vwr, "IMState: %s\n", IMStates[m->kstorage]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(vwr, "N keys:  %D\n", m->nKeys[IM_GLOBAL]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(vwr, "Globally sorted: %s\n", m->sorted[IM_GLOBAL] ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(vwr);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(vwr, "Locally sorted:  %s\n", m->sorted[IM_LOCAL] ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
      if (m->kstorage == IM_CONTIGUOUS) {
        for (i = 0; i < size; ++i) {
          const PetscInt n = m->contig->globalRanges[2*i+1]-m->contig->globalRanges[2*i];
          ierr = PetscViewerASCIIPrintf(vwr, "[%D] %D: [%D, %D) \n", i, n, m->contig->globalRanges[2*i], m->contig->globalRanges[2*i+1]);CHKERRQ(ierr);
        }
      } else {
        for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) {
          ierr = PetscViewerASCIISynchronizedPrintf(vwr, "[%d] %D: %D\n", rank, i, m->discontig->keys[i]);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerFlush(vwr);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(vwr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IMViewFromOptions(IM m, PetscObject obj, const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)m, obj, name);CHKERRQ(ierr);
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
  PetscValidLogicalCollectiveEnum(m,m->keysetcalled,1);
  if (m->setupcalled) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)m, &comm);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->kstorage)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be set as contiguous or non contiguous before setup");
  }
  m->setupcalled = PETSC_TRUE;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  /* If user supplied local size generate global size, if user gave global generate local */
  ierr = PetscSplitOwnership(comm, &(m->nKeys[IM_LOCAL]), &(m->nKeys[IM_GLOBAL]));CHKERRQ(ierr);
  switch (m->kstorage) {
  case IM_CONTIGUOUS:
  {
    PetscInt chunk[2];

    /* Interval not explicitly set, only sizes */
    if (!(m->keysetcalled)) m->contig->keyEnd = m->nKeys[IM_LOCAL];
    chunk[0] = m->contig->keyStart, chunk[1] = m->contig->keyEnd;
    if (!(m->contig->globalRanges)) {
      ierr = PetscMalloc1(2*size, &(m->contig->globalRanges));CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)m, 2*size*sizeof(PetscInt));CHKERRQ(ierr);
    }
    ierr = MPI_Allgather(chunk, 2, MPIU_INT, m->contig->globalRanges, 2, MPIU_INT, comm);CHKERRQ(ierr);
    if (!(m->keysetcalled)) {
      PetscInt i, n = 0;
      for (i = 1; i <= size-1; ++i) {
        n += m->contig->globalRanges[2*i-1]-m->contig->globalRanges[2*i-2];
        m->contig->globalRanges[2*i] += n;
        m->contig->globalRanges[2*i+1] += n;
      }
      m->contig->keyStart = m->contig->globalRanges[2*rank];
      m->contig->keyEnd = m->contig->globalRanges[2*rank+1];
    }
    break;
  }
  case IM_ARRAY:
  {
    PetscInt i, sum = 0;

    ierr = MPI_Exscan(&(m->nKeys[IM_LOCAL]), &sum, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
    if (!(m->discontig->keyIndexGlobal)) {
      ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &(m->discontig->keyIndexGlobal));CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)m, (m->nKeys[IM_LOCAL])*sizeof(PetscInt));CHKERRQ(ierr);
    }
    for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) m->discontig->keyIndexGlobal[i] = sum+i;
    if (!(m->keysetcalled)) {
      ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &(m->discontig->keys));CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)m, (m->nKeys[IM_LOCAL])*sizeof(PetscInt));CHKERRQ(ierr);
      for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) m->discontig->keys[i] = sum+i;
    }
    break;
  }
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid IMState");
    break;
  }
  if (m->ops->setup) {
    ierr = (*m->ops->setup)(m);CHKERRQ(ierr);
  }
  ierr = IMViewFromOptions(m, NULL, "-im_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetFromOptions(IM m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)m);CHKERRQ(ierr);
  if (m->ops->setfromoptions) {
    ierr = (*m->ops->setfromoptions)(m);CHKERRQ(ierr);
  }
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) m);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* LOGICALLY COLLECTIVE, CANNOT OVERRIDE */
PetscErrorCode IMSetKeyStateAndSizes(IM m, IMState state, PetscInt n, PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(m,state,2);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change key state or number of keys on setup map");
    //if (m->keysetcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Keys already explicitly set cannot override");
  }
  ierr = IMSetupKeyState_Private(m, state);CHKERRQ(ierr);
  m->nKeys[IM_LOCAL] = n;
  m->nKeys[IM_GLOBAL] = N;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE, MUST BE SETUP */
PetscErrorCode IMGetNumKeys(IM m, IMOpMode mode, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(n,3);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be setup first call IMSetup()");
  }
  *n = m->nKeys[mode];
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

/* SET/GET KEYS DIRECTLY */
/* LOGICALLY COLLECTIVE, CAN OVERRIDE */
PetscErrorCode IMSetKeyInterval(IM m, PetscInt keyStart, PetscInt keyEnd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on setup map");
    if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"KeyEnd %D < KeyStart %D contiguous keys must be increasing", keyEnd, keyStart);
  }
  ierr = IMSetupKeyState_Private(m, IM_CONTIGUOUS);CHKERRQ(ierr);
  m->contig->keyStart = keyStart;
  m->contig->keyEnd = keyEnd;
  m->nKeys[IM_LOCAL] = keyEnd-keyStart;
  m->keysetcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMGetKeyInterval(IM m, PetscInt *keyStart, PetscInt *keyEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    const IMState state = IM_CONTIGUOUS;
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be setup first call IMSetup()");
    if (m->kstorage && (m->kstorage != state)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is of type %d not %s",IMStates[m->kstorage], IMStates[state]);
  }
  if (keyStart) *keyStart = m->contig->keyStart;
  if (keyEnd) *keyEnd = m->contig->keyEnd;
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetGlobalKeyIntervals(IM m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    const IMState state = IM_CONTIGUOUS;
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be setup first call IMSetup()");
    if (m->kstorage && (m->kstorage != state)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is of type %d not %s",IMStates[m->kstorage], IMStates[state]);
  }
  *keys = m->contig->globalRanges;
  PetscFunctionReturn(0);
}

PetscErrorCode IMRestoreGlobalKeyIntervals(IM m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    const IMState state = IM_CONTIGUOUS;
    if (m->kstorage && (m->kstorage != state)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is of type %d not %s",IMStates[m->kstorage], IMStates[state]);
    if (*keys != m->contig->globalRanges) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from IMGetGlobalKeyIntervals()");
  }
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE, CAN OVERRIDE */
PetscErrorCode IMSetKeyArray(IM m, PetscInt n, const PetscInt keys[], PetscBool issorted, PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,3);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on setup map");
    if (m->kstorage && (m->kstorage != IM_ARRAY)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping has already been set to %s, use IMConvertKeys() to change",IMStates[m->kstorage]);
    if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of keys %D < 0", n);
  }
  ierr = IMSetupKeyState_Private(m, IM_ARRAY);CHKERRQ(ierr);
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
  m->sorted[IM_LOCAL] = issorted;
  m->keysetcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMGetKeyArray(IM m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    const IMState state = IM_ARRAY;
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be setup first call IMSetup()");
    if (m->kstorage && (m->kstorage != state)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is not %s", IMStates[state]);
  }
  *keys = m->discontig->keys;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMRestoreKeyArray(IM m, const PetscInt *keys[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,2);
  if (PetscDefined(USE_DEBUG)) {
    const IMState state = IM_ARRAY;
    if (m->kstorage && (m->kstorage != state)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is of type %d not %s",IMStates[m->kstorage], IMStates[state]);
    if (*keys != m->discontig->keys) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from IMGetKeysDiscontiguous()");
  }
  PetscFunctionReturn(0);
}

/* UTIL FUNCTIONS */
/* LOGICALLY COLLECTIVE, ONE OF THE ONLY FUNCTIONS THAT INVALIDATES SETUP */
PetscErrorCode IMConvertKeyState(IM m, IMState newstate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->keysetcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for maps built without explicit keys set");
  }
  if (m->kstorage == newstate) PetscFunctionReturn(0);
  m->setupcalled = PETSC_FALSE;
  if (!(m->kstorage)) {
    ierr = IMSetupKeyState_Private(m, newstate);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  switch (newstate) {
  case IM_CONTIGUOUS:
  {
    PetscInt keyStart, keyEnd, nKeysSaved[2] = {m->nKeys[IM_LOCAL], m->nKeys[IM_GLOBAL]};

    if (!(m->sorted[IM_LOCAL])) {
      ierr = PetscIntSortSemiOrdered(m->nKeys[IM_LOCAL], m->discontig->keys);CHKERRQ(ierr);
    }
    keyStart = m->discontig->keys[0];
    keyEnd = m->discontig->keys[m->nKeys[IM_LOCAL]-1];
    if (PetscDefined(USE_DEBUG)) {
      if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"KeyEnd %D < KeyStart %D from discontiguous but contiguous keys must be increasing",keyEnd,keyStart);
      if (m->nKeys[IM_LOCAL] && ((keyEnd-keyStart) != m->nKeys[IM_LOCAL])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of keys previously set %D != %D keyrange given",m->nKeys[IM_LOCAL],keyEnd-keyStart);
    }
    ierr = IMSetupKeyState_Private(m, IM_CONTIGUOUS);CHKERRQ(ierr);
    /* Need to save and set these since they are reset in setupkeystate */
    m->nKeys[IM_LOCAL] = nKeysSaved[0];
    m->nKeys[IM_GLOBAL] = nKeysSaved[1];
    m->contig->keyStart = keyStart;
    m->contig->keyEnd = keyEnd;
    break;
  }
  case IM_ARRAY:
  {
    PetscInt i, keyStart = m->contig->keyStart, nKeysSaved[2] = {m->nKeys[IM_LOCAL], m->nKeys[IM_GLOBAL]};

    if (PetscDefined(USE_DEBUG)) {
      PetscInt keyEnd = m->contig->keyEnd;
      if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"KeyEnd %D < KeyStart %D from discontiguous but contiguous keys must be increasing",keyEnd,keyStart);
      if (m->nKeys[IM_LOCAL] && ((keyEnd-keyStart) != m->nKeys[IM_LOCAL])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of keys previously set %D != %D keyrange given",m->nKeys[IM_LOCAL],keyEnd-keyStart);
    }
    ierr = IMSetupKeyState_Private(m, IM_ARRAY);CHKERRQ(ierr);
    m->nKeys[IM_LOCAL] = nKeysSaved[0];
    m->nKeys[IM_GLOBAL] = nKeysSaved[0];
    /* Interval always locally sorted so conversion means also locally sorted */
    m->sorted[IM_LOCAL] = PETSC_TRUE;
    m->discontig->alloced = PETSC_TRUE;
    ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &(m->discontig->keys));CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject) m, (m->nKeys[IM_LOCAL])*sizeof(PetscInt));CHKERRQ(ierr);
    for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) m->discontig->keys[i] = keyStart+i;
    break;
  }
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown IMState");
    break;
  }
  if (m->ops->convertkeys) {
    ierr = (*m->ops->convertkeys)(m, newstate);CHKERRQ(ierr);
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
  if (m->sorted[mode]) PetscFunctionReturn(0);
  ierr = (*m->ops->sort)(m,mode);CHKERRQ(ierr);
  m->sorted[mode] = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode IMPermute(IM m, IM pm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  PetscValidHeaderSpecific(m,IM_CLASSID,2);
  PetscValidType(pm,2);
  if (PetscDefined(USE_DEBUG)) {
    PetscBool isbasic;

    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map to be permuted must be setup first");
    if (!(pm->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Permuation map must be setup first");
    if (m->nKeys[IM_LOCAL] > pm->nKeys[IM_LOCAL]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot permute map with local size %D with smaller map of local size %D",m->nKeys[IM_LOCAL],pm->nKeys[IM_LOCAL]);
    ierr = PetscObjectTypeCompare((PetscObject) m, IMBASIC, &isbasic);CHKERRQ(ierr);
    if (!(isbasic)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Permuation map must be of type %s", IMBASIC);
  }
  if (m->ops->permute) {
    ierr = (*m->ops->permute)(m,pm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetSorted(IM m, IMOpMode mode, PetscBool sorted)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(m,mode,2);
  m->sorted[mode] = sorted;
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
