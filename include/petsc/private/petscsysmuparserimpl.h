#pragma once

#include <petscsysmuparser.h>

#if PetscDefined(HAVE_MUPARSER)
  #include "muParserDLL.h"
#endif

struct _n_PetscMuParserCoordFunc {
#if PetscDefined(HAVE_MUPARSER)
  muParserHandle_t parser; // muParser object
  muFloat_t        x[3];   // Coordinate variables
#else
  PetscReal x[3];
#endif
};
