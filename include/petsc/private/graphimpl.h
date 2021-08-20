#ifndef PETSCGRAPHIMPL_H
#define PETSCGRAPHIMPL_H

#include <petscgraph.h>
#include <petsc/private/petscimpl.h>

struct _GraphOps {
  PetscErrorCode (*create)(PetscCallGraph);
};

struct _p_PetscCallGraph {
  PETSCHEADER(struct _GraphOps);
  void *data;
};

#endif /* PETSCGRAPHIMPL_H */
