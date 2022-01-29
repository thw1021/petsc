#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = vecseqcupm.cu
SOURCEF  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/mpi/mpicupm/cuda/
DIRS     =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
