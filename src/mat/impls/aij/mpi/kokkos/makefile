-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_KOKKOS'
ALL: lib

CFLAGS        =
FFLAGS        =
SOURCECU      =
SOURCEKOKKOS  = mpiaijkok.kokkos.cxx
SOURCEF	      =
SOURCEH	      =
LIBBASE	      = libpetscmat
DIRS	        =
MANSEC	      = Mat
LOCDIR	      = src/mat/impls/aij/mpi/kokkos/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
