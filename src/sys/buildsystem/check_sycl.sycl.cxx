#include <petscconf.h>

#if !defined(PETSC_HAVE_SYCL) && !defined(PETSC_HAVE_SYCL_DIALECT_CXX11)
#  error "This file should only be compiled when PETSc has a SYCL compiler!"
#endif
