#include <petscmacros.h>

#if !PetscDefined(HAVE_CUDA) && !PetscDefined(HAVE_HIP)
  #include "cg_async.hpp"
#endif
