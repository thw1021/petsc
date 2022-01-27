#ifndef COO_H_
#define COO_H_

#include <petscmat.h>

typedef struct {
  PetscInt Nv;           /* number of vertices */
  PetscInt Ne;           /* number of elements */
  PetscInt n;            /* dimension of the resulting linear system; size of the Jacobian */
  PetscInt *vertices;    /* list of vertices for each element */
  PetscInt *coo;         /* offset into the matrices COO array for the start of each element stiffness */
} FEStruct;

PETSC_EXTERN PetscErrorCode FillMatrixKokkosCOO(FEStruct *,Mat);

#endif // COO_H_
