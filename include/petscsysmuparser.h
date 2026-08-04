#pragma once

#include <petscsys.h>

/* SUBMANSEC = Sys */

/*S
     PetscMuParserCoordFunc - A muParser generated coordinate function for use in PETSc

     Level: intermediate

     Notes:
     This object requires --download-muparser

     The function variables are named x, y, z

.seealso: `PetscMuParserCoordFuncCreate()`, `PetscMuParserCoordFuncDestroy()`
S*/
typedef struct _n_PetscMuParserCoordFunc *PetscMuParserCoordFunc;

PETSC_EXTERN PetscErrorCode PetscMuParserCoordFuncCreate(const char[], PetscMuParserCoordFunc *);
PETSC_EXTERN PetscErrorCode PetscMuParserCoordFuncDestroy(PetscMuParserCoordFunc *);
