#pragma once

#include <petscmat.h>

/* SUBMANSEC = Mat */

PETSC_EXTERN PetscFunctionList MatCoarsenList;

/*S
     MatCoarsen - Object for managing the coarsening of a graph (symmetric matrix)

   Level: advanced

  Note:
    This is used by the `PCGAMG` to generate coarser representations of an algebraic problem

.seealso: [](ch_matrices), [](sec_graph), `Mat`, `MatCoarsenCreate()`, `MatCoarsenType`, `MatColoringType`, `MatPartitioningType`, `MatOrderingType`
          `MatColoring`, `MatPartitioning`
S*/
typedef struct _p_MatCoarsen *MatCoarsen;

/*J
    MatCoarsenType - String with the name of a PETSc matrix coarsening algorithm

   Level: beginner

.seealso: [](ch_matrices), [](sec_graph), `Mat`, `MatCoarsenCreate()`, `MatCoarsen`, `MatColoringType`, `MatPartitioningType`, `MatOrderingType`
J*/
typedef const char *MatCoarsenType;
#define MATCOARSENMIS  "mis"
#define MATCOARSENHEM  "hem"
#define MATCOARSENMISK "misk"

PETSC_EXTERN PetscErrorCode MatCoarsenMISKSetDistance(MatCoarsen, PetscInt);
PETSC_EXTERN PetscErrorCode MatCoarsenMISKGetDistance(MatCoarsen, PetscInt *);
PETSC_EXTERN PetscErrorCode MatCoarsenSetMaximumIterations(MatCoarsen, PetscInt);
PETSC_EXTERN PetscErrorCode MatCoarsenSetThreshold(MatCoarsen, PetscReal);
PETSC_EXTERN PetscErrorCode MatCoarsenSetStrengthIndex(MatCoarsen, PetscInt, PetscInt[]);
