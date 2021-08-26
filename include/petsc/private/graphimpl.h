#ifndef PETSCGRAPHIMPL_H
#define PETSCGRAPHIMPL_H

#include <petscgraph.hpp>
#include <petsc/private/petscimpl.h>
#include <petsc/private/hashmapi.h>

struct _n_PetscExecutionContext {
  void *ctx;
};
typedef struct _n_PetscExecutionContext *PetscExecutionContext;

struct _n_PetscGraphEdge {
  PetscCallNode begin;
  PetscCallNode end;
  PetscInt      stream;
  PetscInt      refcnt;
};
typedef struct _n_PetscGraphEdge *PetscGraphEdge;

struct _NodeOps {
  PetscErrorCode (*create)(PetscCallNode);
  PetscErrorCode (*destroy)(PetscCallGraph);
};

struct _p_PetscCallNode {
  PETSCHEADER(struct _NodeOps);
  PetscInt       id;
  PetscGraphEdge *inedges;
  PetscGraphEdge *outedges;
  PetscInt       sizeIn,sizeOut;
  PetscInt       capacityIn,capacityOut;
  void           *data;
};

struct _GraphOps {
  PetscErrorCode (*create)(PetscCallGraph);
  PetscErrorCode (*destroy)(PetscCallGraph);
  PetscErrorCode (*assemble)(PetscCallGraph);
};

struct _p_PetscCallGraph {
  PETSCHEADER(struct _GraphOps);
  PetscCallNode *exec;
  PetscCallNode *nodes;
  PetscInt      size;
  PetscInt      capacity,execCapacity;
  PetscCallNode begin;
  PetscCallNode end;
  PetscHMapI    idMap;
  PetscBool     setup;
  PetscBool     assembled;
  void          *userCtx;
  void          *data;
};

PetscErrorCode PetscCallGraphGrowArray_Private(MPI_Comm comm, PetscInt size, PetscInt *capacity, size_t sizeThing, void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (size == *capacity) {
    if (size) {
      *capacity *= 2;
      ierr = PetscRealloc(sizeThing*(*capacity),array);CHKERRQ(ierr);
    } else {
      *capacity = 10;
      /* we call PetscMallocA direclty since we already know the size_t (and don't want it calculated from
       sizeof(**void)) */
      ierr = PetscMallocA(1,PETSC_FALSE,__LINE__,PETSC_FUNCTION_NAME,__FILE__,(size_t)(*capacity)*sizeThing,array);CHKERRQ(ierr);
    }
  } else if (PetscUnlikelyDebug(size > *capacity)) {
    SETERRQ2(comm,PETSC_ERR_PLIB,"Array size %D > capacity %D during push back",size,*capacity);
  }
  PetscFunctionReturn(0);
}

#define PetscCallGraphPushBackThing_Private(comm,thing,array,size,capacity) (PetscCallGraphGrowArray_Private((comm),(size),&(capacity),sizeof(**(array)),array) || ((*array)[(size)++] = (thing),0))

PETSC_STATIC_INLINE PetscErrorCode PetscCallNodePushBackInEdge_Private(PetscCallNode node, PetscGraphEdge edge)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCallGraphPushBackThing_Private(PetscObjectComm((PetscObject)node),edge,&node->inedges,node->sizeIn,node->capacityIn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscCallNodePushBackOutEdge_Private(PetscCallNode node, PetscGraphEdge edge)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCallGraphPushBackThing_Private(PetscObjectComm((PetscObject)node),edge,&node->outedges,node->sizeOut,node->capacityOut);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscCallGraphPushBackNode_Private(PetscCallGraph graph, PetscCallNode node)
{
  PetscErrorCode ierr;
  PetscBool      notInMap;

  PetscFunctionBegin;
  ierr = PetscHMapIQuerySet(graph->idMap,node->id,graph->size+1,&notInMap);CHKERRQ(ierr);
  /* silly double negation here, but this checks if it IS in the map */
  if (PetscUnlikelyDebug(!notInMap)) SETERRQ(PetscObjectComm((PetscObject)graph),PETSC_ERR_ARG_WRONG,"Cannot insert duplicate node into the graph as a DAG containing duplicate nodes is ill-formed");
  for (PetscInt i = 0; i < graph->size; ++i) {
    if (PetscUnlikelyDebug(graph->nodes[i]->id == node->id)) SETERRQ(PetscObjectComm((PetscObject)graph),PETSC_ERR_ARG_WRONG,"Cannot insert duplicate node into the graph as a DAG containing duplicate nodes is ill-formed");
  }
  ierr = PetscCallGraphPushBackThing_Private(PetscObjectComm((PetscObject)graph),node,&graph->nodes,graph->size,graph->capacity);CHKERRQ(ierr);
  graph->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#endif /* PETSCGRAPHIMPL_H */
