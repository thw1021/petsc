/*
  This is the include file for source code that needs to know if the code is being run under valgrind
*/
#if !defined(PETSCVALGRIND_H)
#define PETSCVALGRIND_H

#if !defined(PETSC_HAVE_WINDOWS_COMPILERS)
#  include <petsc/private/kernels/valgrind.h>
#  define PETSC_RUNNING_ON_VALGRIND RUNNING_ON_VALGRIND
#else
#  define PETSC_RUNNING_ON_VALGRIND PETSC_FALSE
#endif

#endif
