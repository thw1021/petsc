#pragma once

#include <petscsys.h>

/* SUBMANSEC = Sys */

#if PetscDefined(HAVE_MUPARSER)
  #include "muParserDLL.h"
#endif

/*S
     PetscMuParserCoordFunc - A muParser generated coordinate function for use in PETSc

     Level: advanced

.seealso: `PetscMuParserCoordFuncCreate()`, `PetscMuParserCoordFuncDestroy()`
S*/
struct _n_PetscMuParserCoordFunc {
#if PetscDefined(HAVE_MUPARSER)
  muParserHandle_t parser; // muParser object
  muFloat_t        x[3];   // Coordinate variables
#else
  PetscReal x[3];
#endif
};
typedef struct _n_PetscMuParserCoordFunc *PetscMuParserCoordFunc;

PETSC_EXTERN PetscErrorCode PetscMuParserCoordFuncCreate(const char[], PetscMuParserCoordFunc *);
PETSC_EXTERN PetscErrorCode PetscMuParserCoordFuncDestroy(PetscMuParserCoordFunc *);
