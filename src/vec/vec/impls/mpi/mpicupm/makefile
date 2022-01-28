#requirespackage 'PETSC_HAVE_CUPM'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEF  =
SOURCEH  = vecmpicupm.hpp
LIBBASE  = libpetscvec
DIRS     = cuda hip
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/mpi/mpicupm/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
