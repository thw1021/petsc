#include <petsc/private/deviceimpl.h>

PetscErrorCode PetscStreamGraphCreate(PetscStreamGraph *sgraph)
{
  PetscStreamGraph sg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidPointer(sgraph,1);
  ierr = PetscStreamRegisterAll();CHKERRQ(ierr);
  /* Setting to null taken from VecCreate(), why though? */
  *sgraph = NULL;
  ierr = PetscNew(&sg);CHKERRQ(ierr);
  sg->setup = PETSC_FALSE;
  sg->assembled = PETSC_FALSE;
  sg->type = PETSC_STREAM_INVALID;
  *sgraph = sg;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphDestroy(PetscStreamGraph *sgraph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sgraph) PetscFunctionReturn(0);
  PetscValidPointer(sgraph,1);
  if ((*sgraph)->ops->destroy) {ierr = (*(*sgraph)->ops->destroy)(*sgraph);CHKERRQ(ierr);}
  ierr = PetscFree(*sgraph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphSetUp(PetscStreamGraph sgraph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(sgraph,1);
  if (sgraph->setup) PetscFunctionReturn(0);
  ierr = (*sgraph->ops->setup)(sgraph);CHKERRQ(ierr);
  sgraph->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphAssemble(PetscStreamGraph sgraph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(sgraph,1);
  if (PetscUnlikelyDebug(!sgraph->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamGraphSetUp() first");
  if (sgraph->assembled) PetscFunctionReturn(0);
  ierr = (*sgraph->ops->assemble)(sgraph);CHKERRQ(ierr);
  sgraph->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphExecute(PetscStreamGraph sgraph, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(sgraph,1,pstream,2);
  if (PetscUnlikelyDebug(!sgraph->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamGraphSetUp() first");
  if (PetscUnlikelyDebug(!sgraph->assembled)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamGraphAssemble() first");
  ierr = (*sgraph->ops->exec)(sgraph, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphGetGraph(PetscStreamGraph sgraph, void *gptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(sgraph,1);
  PetscValidPointer(gptr,2);
  if (PetscUnlikelyDebug(!sgraph->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamGraphSetUp() first");
  ierr = (*sgraph->ops->getgraph)(sgraph, gptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphRestoreGraph(PetscStreamGraph sgraph, void *gptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(sgraph,1);
  PetscValidPointer(gptr,2);
  if (PetscUnlikelyDebug(!sgraph->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamGraphSetUp() first");
  ierr = (*sgraph->ops->restoregraph)(sgraph, gptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
