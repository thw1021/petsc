#include <petscconf.h>

#ifndef NAGFOR
#ifndef PETSC_HAVE_FORTRAN90
error "This file should only be compiled when PETSc has a FORTRAN90 compiler!"
#endif
#endif
