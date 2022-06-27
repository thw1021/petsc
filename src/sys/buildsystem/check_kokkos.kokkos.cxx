#include <petscconf.h>

#ifndef PETSC_HAVE_KOKKOS
#  error "This file should only be compiled when PETSc has a KOKKOS compiler!"
#endif
