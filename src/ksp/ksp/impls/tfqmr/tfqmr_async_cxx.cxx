#include <petscmacros.h>

#if !PetscDefined(HAVE_CUDA) && !PetscDefined(HAVE_HIP)
  #include "tfqmr_async.hpp"
#endif
