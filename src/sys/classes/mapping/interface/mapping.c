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
  PetscValidPointer(m,2);
  ierr = IMInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*m,IM_CLASSID,"IM","Mapping","IM",comm,IMDestroy,IMView);CHKERRQ(ierr);
  ierr = IMInitializeBase_Private(*m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMCreateChild(IM m, IM *c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  ierr = IMCreate(PetscObjectComm((PetscObject) m), c);CHKERRQ(ierr);
  (*c)->echelon = m->echelon+1;
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetChild(IM m, IM c)
{
  PetscFunctionBegin;

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
    PetscInt    i;
    PetscMPIInt rank, size;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) m), &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject) m), &size);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "N keys:  %D\n", m->nIdx[IM_GLOBAL]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "Globally sorted: %s\n", m->sorted[IM_GLOBAL] ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(vwr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(vwr, "Locally sorted:  %s\n", m->sorted[IM_LOCAL] ? "PETSC_TRUE" : "PETSC_FALSE");CHKERRQ(ierr);
    for (i = 0; i < m->nIdx[IM_LOCAL]; ++i) {
      ierr = PetscViewerASCIISynchronizedPrintf(vwr, "[%d] %D: %D\n", rank, i, m->idx[i]);CHKERRQ(ierr);
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
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidType(m,1);
  if (m->setupcalled) PetscFunctionReturn(0);
  /* If user supplied local size generate global size, if user gave global generate local */
  ierr = PetscObjectGetComm((PetscObject) m, &comm);CHKERRQ(ierr);
  ierr = PetscSplitOwnership(comm, &m->nIdx[IM_LOCAL], &m->nIdx[IM_GLOBAL]);CHKERRQ(ierr);
  if (!(m->echelon) && !(m->map)) {
    ierr = IMCreateChild(m, &m->map);CHKERRQ(ierr);
    ierr = IMSetType(m->map, IMLAYOUT);CHKERRQ(ierr);
    ierr = IMLayoutSetFromSizes(m->map, m->nIdx[IM_LOCAL], m->nIdx[IM_GLOBAL]);CHKERRQ(ierr);
    ierr = IMSetUp(m->map);CHKERRQ(ierr);
  }
  if (m->ops->setup) {
    ierr = (*m->ops->setup)(m);CHKERRQ(ierr);
  }
  ierr = IMViewFromOptions(m, NULL, "-im_view");CHKERRQ(ierr);
  m->setupcalled = PETSC_TRUE;
  PetscObjectStateIncrease((PetscObject)m);
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

PetscErrorCode IMSetIndices(IM m, PetscInt n, const PetscInt idx[], PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(idx,2);
  switch (mode) {
  case PETSC_COPY_VALUES:
    ierr = PetscMalloc1(n, &(m->idx));CHKERRQ(ierr);
    ierr = PetscArraycpy(m->idx, idx, n);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)m,n*sizeof(PetscInt));CHKERRQ(ierr);
    m->alloced = PETSC_TRUE;
    break;
  case PETSC_OWN_POINTER:
    m->idx = (PetscInt *)idx;
    ierr = PetscLogObjectMemory((PetscObject)m,n*sizeof(PetscInt));CHKERRQ(ierr);
    m->alloced = PETSC_TRUE;
    break;
  case PETSC_USE_POINTER:
    m->idx = (PetscInt *)idx;
    m->alloced = PETSC_FALSE;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscCopyMode");
    break;
  }
  m->nIdx[IM_LOCAL] = n;
  PetscObjectStateIncrease((PetscObject)m);
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetIndices(IM m, const PetscInt *idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(idx,2);
  ierr = (m->ops->getindices)(m, idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetSizes(IM m, PetscInt *n, PetscInt *N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  if (n) {PetscValidIntPointer(n,2);}
  if (N) {PetscValidIntPointer(N,3);}
  ierr = (m->ops->getsizes)(m,n,N);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject) m, &m->cstate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetLayout(IM m, IM *ml)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidPointer(ml,2);
  if (PetscDefined(USE_DEBUG)) {
    if (!(m->setupcalled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call IMSetUp() before accessing layout");
  }
  *ml = m->map;
  ierr = PetscObjectStateGet((PetscObject) m->map, &m->map->cstate);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMRestoreLayout(IM m, IM *ml)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidPointer(ml,2);
  PetscValidHeaderSpecific(*ml,IM_CLASSID,2);
  if (PetscDefined(USE_DEBUG)) {
    PetscObjectState state;
    PetscErrorCode   ierr;

    ierr = PetscObjectStateGet((PetscObject) *ml, &state);CHKERRQ(ierr);
    if (state != (*ml)->cstate) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Must restore map in its original state");
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IMRestoreIndices(IM m, const PetscInt *idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,IM_CLASSID,1);
  PetscValidIntPointer(idx,2);
  PetscFunctionReturn(0);
}
