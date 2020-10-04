/*
 This file should contain all "core" ops that every petsc impls is expected to provide a function for, i.e. every
 function in _IMOps
 */
#include <petsc/private/imimpl.h>
#include <petscim.h>

const char* const IMStates[] = {"interval", "array", "state_max", "IMState", "IM_", NULL};

PetscErrorCode IMCreate(MPI_Comm comm, IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(m,2);
  ierr = IMInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*m,IM_CLASSID,"IM","Mapping","IM",comm,IMDestroy,IMView);CHKERRQ(ierr);
  ierr = IMInitializeBase_Private(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy(IM *m)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*m) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*m,IM_CLASSID,1);
  if ((*m)->ops->destroy) {
    ierr = (*(*m)->ops->destroy)(*m);CHKERRQ(ierr);
  }
  ierr = IMDestroyBase_Private(*m);CHKERRQ(ierr);
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
    ierr = (*m->ops->destroy)(m);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(m->ops, sizeof(*m->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) m, type);CHKERRQ(ierr);
  ierr = (*create)(m);CHKERRQ(ierr);
  PetscObjectStateIncrease((PetscObject) m);
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
  PetscValidType(m,1);
  if (!vwr) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)m), &vwr);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(vwr,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(m,1,vwr,2);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must setup map before viewing");
  }
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject) m, vwr);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) vwr, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii){
    PetscMPIInt rank, size;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) m), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject) m), &size);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "IMState: %s\n", IMStates[m->kstorage]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "N keys:  %D\n", m->nKeys[IM_GLOBAL]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "Globally sorted: %s\n", m->sorted[IM_GLOBAL] ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(vwr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "Locally sorted:  %s\n", m->sorted[IM_LOCAL] ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
    if (m->kstorage == IM_INTERVAL) {
      ierr = PetscViewerASCIISynchronizedPrintf(vwr, "[%d] [%D, %D]\n", rank, m->interval->keyStart, m->interval->keyEnd);CHKERRQ(ierr);
    } else {
      PetscInt i;

      for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) {
        ierr = PetscViewerASCIISynchronizedPrintf(vwr, "[%d] %D: %D\n", rank, i, m->array->keys[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(vwr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(vwr);CHKERRQ(ierr);
  }
  if (m->ops->view) {
    ierr = (*m->ops->view)(m,vwr);CHKERRQ(ierr);
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
  PetscInt         sum = 0;
  PetscObjectState state;
  MPI_Comm         comm;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  PetscValidLogicalCollectiveEnum(m,m->kstorage,1);
  PetscValidLogicalCollectiveEnum(m,m->generated,1);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be set as contiguous or non contiguous before setup");
  }
  if (m->setupcalled) PetscFunctionReturn(0);
  m->setupcalled = PETSC_TRUE;
  PetscObjectStateIncrease((PetscObject) m);
  ierr = PetscObjectStateGet((PetscObject) m, &state);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)m, &comm);CHKERRQ(ierr);
  /* If user supplied local size generate global size, if user gave global generate local */
  ierr = PetscSplitOwnership(comm, &(m->nKeys[IM_LOCAL]), &(m->nKeys[IM_GLOBAL]));CHKERRQ(ierr);
  switch (m->kstorage) {
  case IM_INTERVAL:
    if (m->generated) {
      ierr = MPI_Exscan(&(m->nKeys[IM_LOCAL]), &sum, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
      m->interval->keyStart = sum;
      m->interval->keyEnd = sum + m->nKeys[IM_LOCAL]-1;
    }
    m->interval->state = state;
    break;
  case IM_ARRAY:
    if (m->generated) {
      PetscInt i;

      ierr = MPI_Exscan(&(m->nKeys[IM_LOCAL]), &sum, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
      ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &(m->array->keys));CHKERRQ(ierr);
      for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) m->array->keys[i] = sum+i;
      m->array->alloced = PETSC_TRUE;
    }
    m->array->state = state;
    break;
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
  ierr = IMSetUp(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* COLLECTIVE, PROVIDES READY TO USE KEY MAP */
PetscErrorCode IMGenerateDefault(MPI_Comm comm, IMType type, IMState state, PetscInt n, PetscInt N, IM *m)
{
  IM             m_;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMCreate(comm, &m_);CHKERRQ(ierr);
  PetscValidLogicalCollectiveEnum(m_,state,3);
  ierr = IMSetType(m_, type);CHKERRQ(ierr);
  /* Generated maps are always increasing linear maps and therefore sorted */
  m_->sorted[IM_LOCAL] = PETSC_TRUE;
  m_->sorted[IM_GLOBAL] = PETSC_TRUE;
  m_->kstorage = state;
  m_->nKeys[IM_LOCAL] = n;
  m_->nKeys[IM_GLOBAL] = N;
  m_->generated = PETSC_TRUE;
  ierr = IMSetUp(m_);CHKERRQ(ierr);
  *m = m_;
  PetscFunctionReturn(0);
}

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
  if (PetscDefined(USE_DEBUG)) {
    if ((mode <= IM_MIN_MODE) || (mode >= IM_MAX_MODE)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"IMOpMode %D provided not within (%D, %D)",mode,IM_MIN_MODE,IM_MAX_MODE);
  }
  *n = m->nKeys[mode];
  PetscFunctionReturn(0);
}

/* SET/GET KEYS DIRECTLY */
/* LOGICALLY COLLECTIVE, CAN OVERRIDE */
PetscErrorCode IMSetKeyInterval(IM m, PetscInt keyStart, PetscInt keyEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on setup map");
    if (keyEnd < keyStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"KeyEnd %D < KeyStart %D contiguous keys must be increasing", keyEnd, keyStart);
  }
  m->interval->keyStart = keyStart;
  m->interval->keyEnd = keyEnd;
  m->nKeys[IM_LOCAL] = keyEnd-keyStart;
  m->generated = PETSC_FALSE;
  PetscObjectStateIncrease((PetscObject) m);
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE */
PetscErrorCode IMGetKeyInterval(IM m, PetscInt *keyStart, PetscInt *keyEnd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    const IMState state = IM_INTERVAL;
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be setup first call IMSetup()");
    if (m->kstorage >= 0  && (m->kstorage != state)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is of type %d not %s use IMConvertKeys() to swap",IMStates[m->kstorage], IMStates[state]);
  }
  if (keyStart) *keyStart = m->interval->keyStart;
  if (keyEnd) *keyEnd = m->interval->keyEnd;
  PetscFunctionReturn(0);
}

/* NOT COLLECTIVE, CAN OVERRIDE */
PetscErrorCode IMSetKeyArray(IM m, PetscInt n, const PetscInt keys[], PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(keys,3);
  if (PetscDefined(USE_DEBUG)) {
    if (m->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change keys on setup map");
    if (m->kstorage >= 0 && (m->kstorage != IM_ARRAY)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping has already been set to %s, use IMConvertKeys() to change",IMStates[m->kstorage]);
    if (n < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of keys %D < 0", n);
  }
  switch (mode) {
  case PETSC_COPY_VALUES:
    ierr = PetscMalloc1(n, &(m->array->keys));CHKERRQ(ierr);
    ierr = PetscArraycpy(m->array->keys, keys, n);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)m,n*sizeof(PetscInt));CHKERRQ(ierr);
    m->array->alloced = PETSC_TRUE;
    break;
  case PETSC_OWN_POINTER:
    m->array->keys = (PetscInt *)keys;
    ierr = PetscLogObjectMemory((PetscObject)m,n*sizeof(PetscInt));CHKERRQ(ierr);
    m->array->alloced = PETSC_TRUE;
    break;
  case PETSC_USE_POINTER:
    m->array->keys = (PetscInt *)keys;
    m->array->alloced = PETSC_FALSE;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscCopyMode");
    break;
  }
  m->kstorage = IM_ARRAY;
  m->nKeys[IM_LOCAL] = n;
  m->generated = PETSC_FALSE;
  PetscObjectStateIncrease((PetscObject) m);
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
    if (m->kstorage >= 0 && (m->kstorage != state)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is not %s", IMStates[state]);
  }
  *keys = m->array->keys;
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
    if (m->kstorage >= 0 && (m->kstorage != state)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping keystorage is of type %d not %s",IMStates[m->kstorage], IMStates[state]);
    if (*keys != m->array->keys) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from IMGetKeysDiscontiguous()");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetIndices(IM m, const PetscInt *idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  PetscValidIntPointer(idx,1);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map must be setup first call IMSetup()");
  }
  ierr = (*m->ops->getindices)(m, idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* UTIL FUNCTIONS */
/* TODO */
PetscErrorCode IMConvertKeyState(IM m, IMState newstate)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(m,newstate,2);
  if (PetscDefined(USE_DEBUG)) {
    if (m->kstorage < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map has no valid key state to convert");
  }
  if (m->kstorage == newstate) PetscFunctionReturn(0);
  /* Since this can potentially throw more key-dependant impls out of whack defer to impls */
  if (m->ops->convertkeys) {
    ierr = (*m->ops->convertkeys)(m, newstate);CHKERRQ(ierr);
  } else {
    PetscObjectState state;

    ierr = PetscObjectStateGet((PetscObject)m, &state);CHKERRQ(ierr);
    switch (newstate) {
    case IM_INTERVAL:
      /* TODO Should check state? */
      if (m->sorted[IM_LOCAL] && (state == m->array->state)) {
        m->interval->keyStart = m->array->keys[0];
        m->interval->keyEnd = m->array->keys[m->nKeys[IM_LOCAL]-1];
      } else {
        PetscInt *keycopy;

        ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &keycopy);CHKERRQ(ierr);
        ierr = PetscArraycpy(keycopy, m->array->keys, m->nKeys[IM_LOCAL]);CHKERRQ(ierr);
        ierr = PetscIntSortSemiOrdered(m->nKeys[IM_LOCAL], keycopy);CHKERRQ(ierr);
        m->interval->keyStart = keycopy[0];
        m->interval->keyEnd = keycopy[m->nKeys[IM_LOCAL]-1];
        ierr = PetscFree(keycopy);CHKERRQ(ierr);
      }
      m->interval->state = state;
      /* Local sort is tricky here, since it really only applies to the key array */
      break;
    case IM_ARRAY:
    {
      const PetscInt keyStart = m->interval->keyStart, keyEnd = m->interval->keyEnd;
      PetscInt       i;

      if (PetscDefined(USE_DEBUG)) {
        if (m->nKeys[IM_LOCAL] >= 0 && ((keyEnd-keyStart) != m->nKeys[IM_LOCAL])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Number of keys previously set %D > %D keyrange given",m->nKeys[IM_LOCAL],keyEnd-keyStart);
      }
      /* Interval always locally sorted so conversion means also locally sorted */
      if (m->array->alloced) {ierr = PetscFree(m->array->keys);CHKERRQ(ierr);}
      ierr = PetscMalloc1(m->nKeys[IM_LOCAL], &(m->array->keys));CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject) m, (m->nKeys[IM_LOCAL])*sizeof(PetscInt));CHKERRQ(ierr);
      for (i = 0; i < m->nKeys[IM_LOCAL]; ++i) m->array->keys[i] = keyStart+i;
      m->sorted[IM_LOCAL] = PETSC_TRUE;
      m->array->alloced = PETSC_TRUE;
      m->array->state = state;
      break;
    }
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown IMState");
      break;
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
  if (PetscDefined(USE_DEBUG)) {
    if ((mode <= IM_MIN_MODE) || (mode >= IM_MAX_MODE)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"IMOpMode %D provided not within (%D, %D)",mode,IM_MIN_MODE,IM_MAX_MODE);
  }
  if (m->sorted[mode]) PetscFunctionReturn(0);
  ierr = (*m->ops->sort)(m,mode);CHKERRQ(ierr);
  m->sorted[mode] = PETSC_TRUE;
  PetscObjectStateIncrease((PetscObject) m);
  PetscFunctionReturn(0);
}

PetscErrorCode IMPermute(IM m, IM pm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  PetscValidHeaderSpecificType(m,IM_CLASSID,2,IMBASIC);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Map to be permuted must be setup first");
    if (!(pm->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Permuation map must be setup first");
    if (m->nKeys[IM_LOCAL] > pm->nKeys[IM_LOCAL]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot permute map with local size %D with smaller map of local size %D",m->nKeys[IM_LOCAL],pm->nKeys[IM_LOCAL]);
  }
  ierr = IMConvertKeyState(pm, IM_ARRAY);CHKERRQ(ierr);
  if (m->ops->permute) {
    ierr = (*m->ops->permute)(m,pm);CHKERRQ(ierr);
  }
  PetscObjectStateIncrease((PetscObject) m);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetSorted(IM m, IMOpMode mode, PetscBool sorted)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(m,mode,2);
  if (PetscDefined(USE_DEBUG)) {
    if ((mode <= IM_MIN_MODE) || (mode >= IM_MAX_MODE)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"IMOpMode %D provided not within (%D, %D)",mode,IM_MIN_MODE,IM_MAX_MODE);
  }
  m->sorted[mode] = sorted;
  PetscObjectStateIncrease((PetscObject) m);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSorted(IM m, IMOpMode mode, PetscBool *sorted)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidLogicalCollectiveEnum(m,mode,2);
  PetscValidBoolPointer(sorted,3);
  if (PetscDefined(USE_DEBUG)) {
    if ((mode <= IM_MIN_MODE) || (mode >= IM_MAX_MODE)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"IMOpMode %D provided not within (%D, %D)",mode,IM_MIN_MODE,IM_MAX_MODE);
  }
  *sorted = m->sorted[mode];
  PetscFunctionReturn(0);
}
