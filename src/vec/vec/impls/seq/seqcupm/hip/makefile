#requirespackage 'PETSC_HAVE_HIP'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = veccupm.hip.cpp
SOURCEF  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqcupm/hip/
DIRS     =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

