#include <petsc/private/graphimpl.h>
#include <petscviewer.h>

PetscClassId PETSCCALLGRAPH_CLASSID;
PetscClassId PETSCCALLNODE_CLASSID;

/* PetscGraphOperator */
PetscErrorCode PetscGraphOperatorCreate(PetscGraphOperator *operator)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(operator,1);
  ierr = PetscNew(operator);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGraphOperatorDestroy(PetscGraphOperator *operator)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*operator) PetscFunctionReturn(0);
  if ((*operator)->ops->destroy) {ierr = (*(*operator)->ops->destroy)(*operator);CHKERRQ(ierr);}
  ierr = PetscFree(*operator);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGraphOperatorExecute(PetscGraphOperator operator, PetscExecutionContext exec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (operator->ops->execute)(operator,exec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGraphOperatorGetDefaultNodeState(PetscGraphOperator operator, PetscCallNodeState *state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(state,2);
  ierr = (*operator->ops->defaultstate)(operator,state);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGraphOperatorNodeAfter(PetscGraphOperator operator, PetscGraphEdge edge)
{
  PetscFunctionBegin;
  if (operator->ops->nodeafter) {
    PetscErrorCode ierr;

    ierr = (*operator->ops->nodeafter)(operator,edge);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* PetscGraphEdge */
PetscErrorCode PetscGraphEdgeCreate(PetscCallNode begin, PetscCallNode end, PetscGraphEdge *edge)
{
  static PetscInt counter = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(begin,PETSCCALLNODE_CLASSID,1);
  PetscValidHeaderSpecific(end,PETSCCALLNODE_CLASSID,2);
  PetscValidPointer(edge,3);
  ierr = PetscNew(edge);CHKERRQ(ierr);
  (*edge)->begin  = begin;
  (*edge)->end    = end;
  (*edge)->id     = counter++;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscGraphEdgeDestroy(PetscGraphEdge *edge)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*edge) PetscFunctionReturn(0);
  if (--(*edge)->refcnt < 1) {
    ierr = PetscFree(*edge);CHKERRQ(ierr);
  } else {
    *edge = NULL;
  }
  PetscFunctionReturn(0);
}

/* PetscCallNode */
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
  PetscValidHeaderSpecific(*node,PETSCCALLNODE_CLASSID,1);
  if ((*node)->ops->destroy) {ierr = (*(*node)->ops->destroy)(*node);CHKERRQ(ierr);}
  for (PetscInt i = 0; i < (*node)->sizeIn; ++i) {
    ierr = PetscGraphEdgeDestroy(((*node)->inedges)+i);CHKERRQ(ierr);
  }
  ierr = PetscFree((*node)->inedges);CHKERRQ(ierr);
  for (PetscInt i = 0; i < (*node)->sizeOut; ++i) {
    ierr = PetscGraphEdgeDestroy(((*node)->outedges)+i);CHKERRQ(ierr);
  }
  ierr = PetscFree((*node)->outedges);CHKERRQ(ierr);
  ierr = PetscGraphOperatorDestroy(&(*node)->operator);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(node);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeView(PetscCallNode node, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(node,PETSCCALLNODE_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeSetOperator(PetscCallNode node, PetscGraphOperator operator)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(node,PETSCCALLNODE_CLASSID,1);
  ierr = PetscGraphOperatorDestroy(&node->operator);CHKERRQ(ierr);
  node->operator = operator;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeGetOperator(PetscCallNode node, PetscGraphOperator *operator)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(node,PETSCCALLNODE_CLASSID,1);
  PetscValidPointer(operator,2);
  *operator = node->operator;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeResetState(PetscCallNode node)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(node,PETSCCALLNODE_CLASSID,1);
  if (node->operator) {
    PetscErrorCode ierr;

    ierr = PetscGraphOperatorGetDefaultNodeState(node->operator,&node->state);CHKERRQ(ierr);
  } else {
    node->state = NODE_STATE_DISABLED;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeAfter(PetscCallNode node, PetscCallNode other)
{
  PetscGraphEdge edge;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(node,PETSCCALLNODE_CLASSID,1);
  PetscValidHeaderSpecific(other,PETSCCALLNODE_CLASSID,2);
  ierr = PetscGraphEdgeCreate(other,node,&edge);CHKERRQ(ierr);
  ierr = PetscCallNodePushBackInEdge_Private(node,edge);CHKERRQ(ierr);
  ierr = PetscCallNodePushBackOutEdge_Private(other,edge);CHKERRQ(ierr);
  if (other->operator) {ierr = PetscGraphOperatorNodeAfter(other->operator,edge);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeBefore(PetscCallNode node, PetscCallNode other)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(node,PETSCCALLNODE_CLASSID,1);
  PetscValidHeaderSpecific(other,PETSCCALLNODE_CLASSID,2);
  ierr = PetscCallNodeAfter(other,node);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallNodeExecute(PetscCallNode node, PetscExecutionContext exec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(node,PETSCCALLNODE_CLASSID,1);
  if (node->state != NODE_STATE_DISABLED) {
    PetscErrorCode ierr;

    ierr = PetscGraphOperatorExecute(node->operator,exec);CHKERRQ(ierr);
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"TODO");
  PetscFunctionReturn(0);
}

/* PetscCallGraph */
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
  PetscValidHeaderSpecific(*graph,PETSCCALLGRAPH_CLASSID,1);
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

static PetscErrorCode PetscCallGraphJoinAncestors_Private(PetscCallNode node, PetscExecutionContext exec)
{
  PetscFunctionBegin;
  if (node->sizeIn) {
    PetscInt numCancelled = 0;

    for (PetscInt i = 0; i < node->sizeIn; ++i) {
      const PetscGraphEdge edge = node->inedges[i];

      if (i) {
        PetscErrorCode ierr;

        ierr = PetscDeviceContextWaitForContext(exec->dctx,edge->dctx);CHKERRQ(ierr);
        ierr = PetscDeviceContextDestroy(&edge->dctx);CHKERRQ(ierr);
      } else {
        exec->dctx = edge->dctx;
      }
      numCancelled += (edge->begin->state == NODE_STATE_DISABLED);
    }
    if (numCancelled == node->sizeIn) {node->state = NODE_STATE_DISABLED;}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCallGraphForkDescendants_Private(const PetscCallNode node, const PetscExecutionContext exec)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < node->sizeOut; ++i) {
    const PetscGraphEdge edge = node->outedges[i];

    if (i) {
      PetscErrorCode ierr;

      ierr = PetscDeviceContextDuplicate(exec->dctx,&edge->dctx);CHKERRQ(ierr);
      ierr = PetscDeviceContextWaitForContext(edge->dctx,exec->dctx);CHKERRQ(ierr);
    } else {
      edge->dctx = exec->dctx;
    }
  }
  PetscFunctionReturn(0);
}

/* does a BFS */
static PetscErrorCode PetscCallGraphTopologicalSort_Private(PetscCallGraph graph, PetscInt v, PetscInt *execIdx, PetscBool *visited)
{
  const PetscCallNode node = graph->nodes[v];

  PetscFunctionBegin;
  visited[v] = PETSC_TRUE;
  for (PetscInt i = 0; i < node->sizeIn; ++i) {
    const PetscInt edgeBeginId = node->inedges[i]->begin->id;
    PetscErrorCode ierr;
    PetscInt       idx;

    ierr = PetscHMapIGet(graph->idMap,edgeBeginId,&idx);CHKERRQ(ierr);
    if (!visited[idx]) {
      ierr = PetscCallGraphTopologicalSort_Private(graph,idx,execIdx,visited);CHKERRQ(ierr);
    }
  }
  graph->exec[(*execIdx)++] = node;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCallGraphResetClosure_Private(PetscCallGraph graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (graph->execCapacity < graph->size+2) {
    graph->execCapacity = graph->size+2;
    ierr = PetscFree(graph->exec);CHKERRQ(ierr);
    ierr = PetscMalloc1(graph->execCapacity,&graph->exec);CHKERRQ(ierr);
  }
  /* install the closures */
  graph->exec[0]             = graph->begin;
  graph->exec[graph->size+1] = graph->end;
  /*
   PetscGraphEdge are effectively std::shared_ptr's, so in order to remove the bogus connections required to form the
   closures we not only need to clear graph->begin->outedges and graph->end->inedges but also find their counterparts
   within the nodes of the graph and manually destroy them.
   */

  /* search for and destroy the INEDGES connected to graph->begin */
  for (PetscInt i = 0; i < graph->begin->sizeOut; ++i) {
    const PetscGraphEdge edge   = graph->begin->outedges[i];
    const PetscInt       edgeId = edge->id;
    PetscInt             id;

    ierr = PetscHMapIGet(graph->idMap,edge->end->id,&id);CHKERRQ(ierr);
    if (PetscUnlikelyDebug(id == graph->begin->id)) SETERRQ1(PetscObjectComm((PetscObject)graph),PETSC_ERR_PLIB,"Graph closure begin node (id %D) is connected to itself",id);
    {
      const PetscCallNode node         = graph->nodes[id];
      PetscGraphEdge      *nodeInedges = node->inedges;

      for (PetscInt j = 0; j < node->sizeIn; ++j) {
        if (nodeInedges[j]->id == edgeId) {
          /* employ destroy-and-swap, whereby we destroy the matching inedge, and just overwrite it with the final
           entry */
          ierr = PetscGraphEdgeDestroy(nodeInedges+j);CHKERRQ(ierr);
          nodeInedges[j] = nodeInedges[--node->sizeIn];
          break;
        }
      }
    }

    /* search for and destroy the OUTEDGES connected to graph->end */
    for (PetscInt i = 0; i < graph->end->sizeIn; ++i) {
      const PetscGraphEdge edge   = graph->end->inedges[i];
      const PetscInt       edgeId = edge->id;
      PetscInt             id;

      ierr = PetscHMapIGet(graph->idMap,edge->begin->id,&id);CHKERRQ(ierr);
      if (PetscUnlikelyDebug(id == graph->end->id)) SETERRQ1(PetscObjectComm((PetscObject)graph),PETSC_ERR_PLIB,"Graph closure begin node (id %D) is connected to itself",id);
      {
        const PetscCallNode node         = graph->nodes[id];
        PetscGraphEdge      *nodeOutedges = node->outedges;

        for (PetscInt j = 0; j < node->sizeOut; ++j) {
          if (nodeOutedges[j]->id == edgeId) {
            ierr = PetscGraphEdgeDestroy(nodeOutedges+j);CHKERRQ(ierr);
            nodeOutedges[j] = nodeOutedges[--node->sizeOut];
            break;
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCallGraphFinalize_Private(PetscCallGraph graph)
{
  PetscFunctionBegin;
  if (!graph->assembled) {
    PetscErrorCode ierr;
    PetscInt       execIdx = 1; /* idx to push back into for graph->exec, note that
                                   graph->exec[0] is always graph->begin */
    PetscBool      *visited;

    ierr = PetscCallGraphResetClosure_Private(graph);CHKERRQ(ierr);
    ierr = PetscCalloc1(graph->size,&visited);CHKERRQ(ierr);
    for (PetscInt v = 0; v < graph->size; ++v) {
      PetscCallNode node = graph->nodes[v];

      if (!visited[v]) {
        ierr = PetscCallGraphTopologicalSort_Private(graph,v,&execIdx,visited);CHKERRQ(ierr);
      }
      if (!node->sizeIn) {ierr = PetscCallNodeAfter(node,graph->begin);CHKERRQ(ierr);}
      if (!node->sizeOut) {ierr = PetscCallNodeBefore(node,graph->end);CHKERRQ(ierr);}
    }
    ierr = PetscFree(visited);CHKERRQ(ierr);
    if (graph->ops->assemble) {ierr = (*graph->ops->assemble)(graph);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCallGraphExecute(PetscCallGraph graph, PetscDeviceContext ctx)
{
  const PetscBool enclosed = ctx ? PETSC_TRUE : PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(graph,PETSCCALLGRAPH_CLASSID,1);
  if (enclosed) {
    PetscValidDeviceContext(ctx,2);
  } else {
    ierr = PetscDeviceContextCreate(&ctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetStreamType(ctx,PETSC_STREAM_DEFAULT_BLOCKING);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetUp(ctx);CHKERRQ(ierr);
  }
  ierr = PetscCallGraphFinalize_Private(graph);CHKERRQ(ierr);
  {
    /* allocate this on the stack using the private stack type */
    _s_PetscExecutionContext exec = {
      .userctx  = graph->userctx,
      .dctx     = ctx,
      .enclosed = enclosed
    };

    for (PetscInt i = 0; i < graph->size; ++i) {
      ierr = PetscCallGraphJoinAncestors_Private(graph->exec[i],&exec);CHKERRQ(ierr);
      ierr = PetscCallNodeExecute(graph->exec[i],&exec);CHKERRQ(ierr);
      ierr = PetscCallGraphForkDescendants_Private(graph->exec[i],&exec);CHKERRQ(ierr);
    }
  }
  for (PetscInt i = 0; i < graph->size; ++i) {
    ierr = PetscCallNodeResetState(graph->nodes[i]);CHKERRQ(ierr);
  }
  if (!enclosed) {
    ierr = PetscDeviceContextSynchronize(ctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextDestroy(&ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
