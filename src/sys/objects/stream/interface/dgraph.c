#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

/*@C
  PetscStreamGraphCreate - Creates an empty PetscStreamGraph object. The type can then be set with PetscStreamSetType().

  Not Collective

  Output Parameter:
. sgraph  - A pointer to the allocated PetscStreamGraph object

  Notes:
  You must set the stream type before using the returned object, otherwise an error is generatedon debug builds.

  Level: beginner

.seealso: PetscStreamGraphDestroy(), PetscStreamGraphSetType(), PetscStreamGraphSetUp()
@*/
PetscErrorCode PetscStreamGraphCreate(PetscStreamGraph *sgraph)
{
  PetscStreamGraph sg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidPointer(sgraph,1);
  ierr = PetscStreamGraphRegisterAll();CHKERRQ(ierr);
  /* Setting to null taken from VecCreate(), why though? */
  *sgraph = NULL;
  ierr = PetscNew(&sg);CHKERRQ(ierr);
  sg->capStrmId = PETSC_DEFAULT;
  *sgraph = sg;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphDestroy - Destroys a PetscStreamGraph.

  Not Collective

  Input Parameter:
. sgraph  - A pointer to the PetscStreamGraph to destroy

  Level: beginner

.seealso: PetscStreamGraphCreate(), PetscStreamGraphSetType(), PetscStreamGraphSetUp()
@*/
PetscErrorCode PetscStreamGraphDestroy(PetscStreamGraph *sgraph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sgraph) PetscFunctionReturn(0);
  PetscValidPointer(sgraph,1);
  if ((*sgraph)->ops->destroy) {ierr = (*(*sgraph)->ops->destroy)(*sgraph);CHKERRQ(ierr);}
  ierr = PetscFree((*sgraph)->type);CHKERRQ(ierr);
  ierr = PetscFree(*sgraph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphSetUp - Sets up internal data structures for use.

  Not Collective

  Input Parameter:
. sgraph  - The PetscStreamGraph object

  Level: beginner

.seealso: PetscStreamGraphCreate(), PetscStreamGraphSetType(), PetscStreamGraphCaptureBegin(), PetscStreamGraphCaptureEnd()
@*/
PetscErrorCode PetscStreamGraphSetUp(PetscStreamGraph sgraph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(sgraph,1);
  if (sgraph->setup) PetscFunctionReturn(0);
  if (sgraph->ops->setup) {ierr = (*sgraph->ops->setup)(sgraph);CHKERRQ(ierr);}
  sgraph->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphCaptureBegin - Begins capture of actions on a stream

  Not Collective

  Input Parameters:
+ sgraph - The PetscStreamGraph object
- pstream - The PetscStream to capture

  developer notes:
  This routine doesn't actually call anything on the graph object, but since the "End" variant actually creates the
  graph it lives here.

  Level: beginner

.seealso: PetscStreamGraphCreate(), PetscStreamGraphSetType(), PetscStreamGraphSetUp(), PetscStreamGraphCaptureEnd()
@*/
PetscErrorCode PetscStreamGraphCaptureBegin(PetscStreamGraph sgraph, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(sgraph,1,pstream,2);
  if (PetscUnlikelyDebug(!sgraph->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamGraphSetUp() first");
  if (PetscUnlikelyDebug(!pstream->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamSetUp() first");
  if (PetscUnlikelyDebug(sgraph->capStrmId != PETSC_DEFAULT)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Graph is already capturing another stream, must call PetscStreamGraphCaptureEnd() first");
  if (PetscLikely(pstream->ops->capturebegin)) {
    ierr = (*pstream->ops->capturebegin)(pstream);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"PetscStream of type %s has no support for graph capture",pstream->type);
  }
  sgraph->capStrmId = pstream->id;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphCaptureEnd - Ends capture of a PetscStream and creates a graph blueprint from it

  Not Collective

  Input Parameters:
+ sgraph - The PetscStreamGraph object
- pstream - The PetscStream being captured

  Notes:
  The PetscStream passed to this routine must be the same PetscStream passed to a previous PetscStreamGraphCaptureBegin()

  Level: beginner

.seealso: PetscStreamGraphCreate(), PetscStreamGraphSetType(), PetscStreamGraphSetUp(), PetscStreamGraphCaptureBegin(),
PetscStreamGraphAssemble(), PetscStreamGraphExecute()
@*/
PetscErrorCode PetscStreamGraphCaptureEnd(PetscStreamGraph sgraph, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(sgraph,1,pstream,2);
  if (PetscUnlikelyDebug(!sgraph->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamGraphSetUp() first");
  if (PetscUnlikelyDebug(!pstream->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamSetUp() first");
  if (PetscUnlikely(sgraph->capStrmId != pstream->id)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Graph is already capturing another stream, must call PetscStreamGraphCaptureEnd() first");
  if (PetscLikely(pstream->ops->captureend)) {
    ierr = (*pstream->ops->captureend)(pstream, sgraph);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"PetscStream of type %s has no support for graph capture",pstream->type);
  }
  sgraph->capStrmId = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphAssemble - Assembles a graph from a graph blueprint

  Not Collective

  Input Parameters:
+ sgraph - The PetscStreamGraph object
- type - The type of assembly for the graph

  Notes:
  Must have called PetscStreamGraphCaptureBegin()/PetscStreamGraphCaptureEnd() on the graph first.

  See PetscGraphAssemblyType for further information on the graph assembly types.

  Level: beginner

.seealso: PetscStreamGraphCaptureBegin(), PetscStreamGraphCaptureEnd(), PetscStreamGraphExecute(), PetscGraphAssemblyType
@*/
PetscErrorCode PetscStreamGraphAssemble(PetscStreamGraph sgraph, PetscGraphAssemblyType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(sgraph,1);
  if (PetscUnlikelyDebug(!sgraph->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscStreamGraphSetUp() first");
  if (PetscUnlikely(sgraph->capStrmId != PETSC_DEFAULT)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Graph is already capturing another stream, must call PetscStreamGraphCaptureEnd() first");
  if (sgraph->assembled && type == PETSC_GRAPH_INIT_ASSEMBLY) PetscFunctionReturn(0);
  ierr = (*sgraph->ops->assemble)(sgraph, type);CHKERRQ(ierr);
  sgraph->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphExecute - Launches an assembled graph

  Not Collective

  Input Parameters:
+ sgraph - The PetscStreamGraph object
- pstream - The PetscStream on which to launch the graph

  Notes:
  The stream used to capture the graph and the one passed to this routine need not be the same. The graph execution
  obeys regular stream ordering semantics. Only a single stream may execute a graph at a time, you must duplicate the
  graph using PetscStreamGraphDuplicate().

  Level: beginner

.seealso: PetscStreamGraphCaptureBegin(), PetscStreamGraphCaptureEnd(), PetscStreamGraphDuplicate()
@*/
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

/*@C
  PetscStreamGraphDuplicate - Duplicates a PetscStreamGraph

  Not Collective

  Input Parameter:
. sgraphref - The PetscStreamGraph object to duplicate

  Output Parameter:
. sgraphdup - The duplicate PetscStreamGraph

  Notes:
  If the reference graph as assembled the duplicate graph will also be assembled.

  Level: beginner

.seealso: PetscStreamGraphCreate(), PetscStreamGraphCaptureBegin(), PetscStreamGraphCaptureEnd(), PetscStreamGraphExecute()
@*/
PetscErrorCode PetscStreamGraphDuplicate(PetscStreamGraph sgraphref, PetscStreamGraph *sgraphdup)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidStreamType(sgraphref,1);
  ierr = PetscStreamGraphCreate(sgraphdup);CHKERRQ(ierr);
  ierr = PetscStreamGraphSetType(*sgraphdup,sgraphref->type);CHKERRQ(ierr);
  ierr = PetscStreamGraphSetUp(*sgraphdup);CHKERRQ(ierr);
  ierr = (*sgraphref->ops->duplicate)(sgraphref,*sgraphdup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscStreamGraphGetGraph - Retrieves the implementation specific graph

  Not Collective

  Input Parameter:
. sgraph - The PetscStreamGraph object

  Output Parameter:
. gptr - A pointer to the graph object

  Notes:
  This is a borrowed reference, the user should not free it

  Level: advanced

.seealso: PetscStreamGraphRestoreGraph(), PetscStreamGraphCreate()
@*/
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

/*@C
  PetscStreamGraphRestoreGraph - Restores the implementation specific graph

  Not Collective

  Input Parameters:
+ sgraph - The PetscStreamGraph object
- gptr - A pointer to the graph object

  Notes:
  The restored graph must be the same graph that was checked out via PetscStreamGraphGetGraph() (but may be altered)

  Level: advanced

.seealso: PetscStreamGraphGetGraph(), PetscStreamGraphCreate()
@*/
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
