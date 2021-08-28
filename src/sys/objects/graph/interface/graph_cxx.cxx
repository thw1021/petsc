#include <petsc/private/graphimpl.h>
#if 0
PetscClassId PETSCCALLGRAPH_CLASSID;
PetscClassId PETSCCALLNODE_CLASSID;

PetscErrorCode PetscCallNodeCreate(MPI_Comm comm, PetscCallNode *node)
{
  static PetscInt counter = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(node,2);
  ierr = PetscHeaderCreate(node,PETSCCALLNODE_CLASSID,"PetscCallNode","PetscCallNode","Sys",comm,PetscCallNodeDestroy,PetscCallNodeView);CHKERRQ(ierr);
  (*node)->id = counter++;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeDestroy(PetscCallNode *node)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*node) PetscFunctionReturn(0);
  ierr = PetscHeaderDestroy(node);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeView(PetscCallNode node, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeExecute(PetscCallNode node, PetscExecutionContext exec)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"TODO");
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallGraphCreate(MPI_Comm comm, PetscCallGraph *graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(graph,2);
  ierr = PetscHeaderCreate(graph,PETSCCALLGRAPH_CLASSID,"PetscCallGraph","PetscCallGraph","Sys",comm,PetscCallGraphDestroy,PetscCallGraphView);CHKERRQ(ierr);
  ierr = PetscHMapICreate(&(*graph)->idMap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallGraphDestroy(PetscCallGraph *graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*graph) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*graph),PETSCCALLGRAPH_CLASSID,1);
  if ((*graph)->ops->destroy) {ierr = (*(*graph)->ops->destroy)(*graph);CHKERRQ(ierr);}
  ierr = PetscFree((*graph)->nodes);CHKERRQ(ierr);
  ierr = PetscFree((*graph)->exec);CHKERRQ(ierr);
  ierr = PetscCallNodeDestroy(&(*graph)->begin);CHKERRQ(ierr);
  ierr = PetscCallNodeDestroy(&(*graph)->end);CHKERRQ(ierr);
  ierr = PetscHMapIDestroy(&(*graph)->idMap);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(graph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallGraphView(PetscCallGraph graph, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallGraphAddNode(PetscCallGraph graph, PetscCallNode node)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(graph,PETSCCALLGRAPH_CLASSID,1);
  PetscValidHeaderSpecific(graph,PETSCCALLNODE_CLASSID,2);
  ierr = PetscCallGraphPushBackNode_Private(graph,node);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallGraphSetUp(PetscCallGraph graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (graph->setup) PetscFunctionReturn(0);
  ierr = PetscCallNodeCreate(PetscObjectComm((PetscObject)graph),&graph->begin);CHKERRQ(ierr);
  ierr = PetscCallNodeCreate(PetscObjectComm((PetscObject)graph),&graph->end);CHKERRQ(ierr);
  graph->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCallGraphTopologicalSort_Private(PetscCallGraph graph, PetscInt v, PetscInt *execIdx, PetscBool *visited)
{
  PetscFunctionBegin;
  visited[v] = PETSC_TRUE;
  for (PetscInt i = 0; i < graph->nodes[v]->sizeIn; ++i) {
    const PetscGraphEdge edge = graph->nodes[v]->inedges[i];
    PetscErrorCode       ierr;
    PetscInt             idx;

    ierr = PetscHMapIGet(graph->idMap,edge->begin->id,&idx);CHKERRQ(ierr);
    if (!visited[idx]) {
      ierr = PetscCallGraphTopologicalSort_Private(graph,idx,execIdx,visited);CHKERRQ(ierr);
    }
  }
  graph->exec[(*execIdx)++] = graph->nodes[v];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCallGraphFinalize_Private(PetscCallGraph graph)
{
  PetscFunctionBegin;
  if (!graph->assembled) {
    PetscErrorCode ierr;
    PetscInt       execIdx = 0;
    PetscBool      *visited;

    if (graph->execCapacity < graph->size+2) {
      /* no need to resize if we have already have the capacity */
      graph->execCapacity = graph->size+2;
      ierr = PetscFree(graph->exec);CHKERRQ(ierr);
      ierr = PetscMalloc1(graph->execCapacity,&graph->exec);CHKERRQ(ierr);
    }
    /* install the closures */
    graph->exec[0] = graph->begin;
    graph->exec[graph->size+1] = graph->end;
    ierr = PetscCalloc1(graph->size,&visited);CHKERRQ(ierr);
    for (PetscInt v = 0; v < graph->size; ++v) {
      if (!visited[v]) {
        ierr = PetscCallGraphTopologicalSort_Private(graph,v,&execIdx,visited);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(visited);CHKERRQ(ierr);
    if (graph->ops->assemble) {ierr = (*graph->ops->assemble)(graph);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallGraphExecute(PetscCallGraph graph)
{
  PetscExecutionContext exec;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(graph,PETSCCALLGRAPH_CLASSID,1);
  ierr = PetscNew(&exec);CHKERRQ(ierr);
  exec->ctx = graph->userCtx;
  ierr = PetscCallGraphFinalize_Private(graph);CHKERRQ(ierr);
  for (PetscInt i = 0; i < graph->size; ++i) {
    ierr = PetscCallNodeExecute(graph->exec[i],exec);CHKERRQ(ierr);
  }
  ierr = PetscFree(exec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
