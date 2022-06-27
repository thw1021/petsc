#include <petscconf.h>

#if !defined(PETSC_HAVE_CUDA) && !defined(PETSC_HAVE_CUDA_DIALECT_CXX11)
#  error "This file should only be compiled when PETSc has a CUDA compiler!"
#endif
