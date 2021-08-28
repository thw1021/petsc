#ifndef PETSCGRAPHTYPES_H
#define PETSCGRAPHTYPES_H

typedef struct _p_PetscCallGraph *PetscCallGraph;

typedef struct _p_PetscCallNode  *PetscCallNode;

typedef struct _n_PetscGraphOperator *PetscGraphOperator;

typedef struct _n_PetscGraphEdge *PetscGraphEdge;

typedef struct _n_PetscExecutionContext *PetscExecutionContext;

typedef enum {
  NODE_STATE_DISABLED,
  NODE_STATE_ENABLED,
  NODE_STATE_PLACEHOLDER
} PetscCallNodeState;
#endif
