#include <petscconf.h>

#ifndef NAGFOR
#ifndef PETSC_HAVE_FORTRAN
error "This file should only be compiled when PETSc has a FORTRAN compiler!"
#endif
#endif
