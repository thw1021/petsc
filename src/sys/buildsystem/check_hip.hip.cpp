#include <petscconf.h>

#if !defined(PETSC_HAVE_HIP) && !defined(PETSC_HAVE_HIP_DIALECT_CXX11)
#  error "This file should only be compiled when PETSc has a HIP compiler!"
#endif
